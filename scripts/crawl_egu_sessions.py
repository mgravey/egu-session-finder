#!/usr/bin/env python3
"""Respectfully discover, cache, and parse EGU session pages."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup


HOST = "meetingorganizer.copernicus.org"
EVENTS: dict[int, tuple[str, str]] = {
    2020: ("EGU2020", "https://meetingorganizer.copernicus.org/EGU2020/sessionprogramme"),
    2021: ("EGU21", "https://meetingorganizer.copernicus.org/EGU21/sessionprogramme"),
    2022: ("EGU22", "https://meetingorganizer.copernicus.org/EGU22/sessionprogramme"),
    2023: ("EGU23", "https://meetingorganizer.copernicus.org/EGU23/sessionprogramme"),
    2024: ("EGU24", "https://meetingorganizer.copernicus.org/EGU24/sessionprogramme"),
    2025: ("EGU25", "https://meetingorganizer.copernicus.org/EGU25/sessionprogramme"),
    2026: ("EGU26", "https://meetingorganizer.copernicus.org/EGU26/sessionprogramme/pg-selection"),
}
SESSION_PATH_PATTERN = re.compile(r"/(?P<event>EGU(?:\d{2}|\d{4}))/session/(?P<id>\d+)", re.I)
GENERIC_SESSION_LINK_TEXT = re.compile(
    r"^(enter live session|orals?|posters?(?: on site| virtual)?|pico|visit presentation)?$",
    re.I,
)


class CrawlError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def canonical_url(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, "", "", ""))


def validate_url(url: str, event_code: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc.lower() != HOST:
        raise CrawlError(f"refusing off-site URL: {url}")
    event_prefix = f"/{event_code.lower()}/"
    path = parsed.path.lower()
    if not path.startswith(event_prefix):
        raise CrawlError(f"URL is outside event {event_code}: {url}")
    allowed = (
        "/sessionprogramme" in path
        or re.search(r"/session/\d+/?$", path) is not None
        or "/sessions-of-special-interest/" in path
    )
    if not allowed:
        raise CrawlError(f"URL path is outside the crawler allowlist: {url}")


class CacheClient:
    def __init__(
        self,
        *,
        delay: float,
        jitter: float,
        max_requests: int,
        contact: str,
    ) -> None:
        if delay < 0.2:
            raise CrawlError("--delay must be at least 0.2 seconds (maximum 5 requests/second)")
        self.delay = delay
        self.jitter = max(jitter, 0.0)
        self.max_requests = max_requests
        self.contact = contact
        self.requests = 0
        self.cache_hits = 0
        self.last_request_finished = 0.0

    def get(self, url: str, cache_path: Path, event_code: str) -> bytes:
        validate_url(url, event_code)
        if cache_path.is_file():
            self.cache_hits += 1
            return cache_path.read_bytes()
        if self.requests >= self.max_requests:
            raise CrawlError(
                f"request budget exhausted at {self.max_requests}; cached progress is preserved"
            )

        retries = (5, 20, 60, 180)
        for attempt in range(len(retries) + 1):
            elapsed = time.monotonic() - self.last_request_finished
            wait = self.delay + random.uniform(0, self.jitter) - elapsed
            if wait > 0:
                time.sleep(wait)
            request = Request(
                url,
                headers={
                    "Accept": "text/html,application/xhtml+xml",
                    "User-Agent": f"egu-session-matcher/0.1 (contact: {self.contact})",
                },
            )
            self.requests += 1
            try:
                with urlopen(request, timeout=60) as response:
                    content = response.read()
                    final_url = response.geturl()
                    status = response.status
                    headers = dict(response.headers.items())
                self.last_request_finished = time.monotonic()
                atomic_write(cache_path, content)
                atomic_json(
                    cache_path.with_suffix(".meta.json"),
                    {
                        "requested_url": url,
                        "final_url": final_url,
                        "status": status,
                        "fetched_at": utc_now(),
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "content_type": headers.get("Content-Type"),
                        "etag": headers.get("ETag"),
                        "last_modified": headers.get("Last-Modified"),
                    },
                )
                return content
            except HTTPError as exc:
                self.last_request_finished = time.monotonic()
                if exc.code not in {429, 500, 502, 503, 504} or attempt == len(retries):
                    raise CrawlError(f"HTTP {exc.code} for {url}") from exc
                retry_after = exc.headers.get("Retry-After")
                pause = int(retry_after) if retry_after and retry_after.isdigit() else retries[attempt]
                failure = f"HTTP {exc.code}"
            except URLError as exc:
                self.last_request_finished = time.monotonic()
                if attempt == len(retries):
                    raise CrawlError(f"network error for {url}: {exc}") from exc
                pause = retries[attempt]
                failure = f"network error: {exc.reason}"
            print(
                f"Temporary {failure}; pausing {pause}s before retrying {url}",
                file=sys.stderr,
            )
            time.sleep(pause)
        raise CrawlError("unreachable retry state")


def soup_from_bytes(content: bytes) -> BeautifulSoup:
    return BeautifulSoup(content, "html.parser")


def session_links(soup: BeautifulSoup, event_code: str) -> dict[str, dict[str, str]]:
    sessions: dict[str, dict[str, str]] = {}
    for anchor in soup.find_all("a", href=True):
        url = canonical_url(urljoin(f"https://{HOST}/", str(anchor["href"])))
        match = SESSION_PATH_PATTERN.search(urlparse(url).path)
        if match is None or match.group("event").lower() != event_code.lower():
            continue
        session_id = match.group("id")
        text = " ".join(anchor.get_text(" ", strip=True).split())
        existing = sessions.setdefault(session_id, {"url": url, "title_hint": ""})
        if text and not GENERIC_SESSION_LINK_TEXT.fullmatch(text):
            if len(text) > len(existing["title_hint"]):
                existing["title_hint"] = text
    return sessions


def discovery_links(
    soup: BeautifulSoup,
    *,
    event_code: str,
    current_url: str,
) -> dict[str, str]:
    links: dict[str, str] = {}
    base = f"https://{HOST}/{event_code}/"

    for element in soup.select('input[name="pg_checkbox[]"][value]'):
        value = str(element.get("value", ""))
        if not value.isdigit():
            continue
        label = soup.find("label", attrs={"for": element.get("id")})
        label_text = " ".join(label.get_text(" ", strip=True).split()) if label else value
        links[canonical_url(urljoin(base, f"sessionprogramme/{value}"))] = label_text

    for element in soup.find_all(["a", "input"], href=True):
        candidate = canonical_url(urljoin(current_url, str(element.get("href"))))
        path = urlparse(candidate).path.lower()
        if (
            urlparse(candidate).netloc.lower() == HOST
            and path.startswith(f"/{event_code.lower()}/")
            and ("/sessionprogramme/" in path or "/sessions-of-special-interest/" in path)
            and candidate != canonical_url(current_url)
        ):
            links.setdefault(candidate, " ".join(element.get_text(" ", strip=True).split()))

    url_pattern = re.compile(r"https://meetingorganizer\.copernicus\.org/[^'\"\s;]+", re.I)
    for element in soup.find_all(attrs={"onclick": True}):
        for raw_url in url_pattern.findall(str(element.get("onclick"))):
            candidate = canonical_url(raw_url)
            path = urlparse(candidate).path.lower()
            if (
                path.startswith(f"/{event_code.lower()}/")
                and ("/sessionprogramme/" in path or "/sessions-of-special-interest/" in path)
            ):
                label = soup.find("label", attrs={"for": element.get("id")})
                links.setdefault(
                    candidate,
                    " ".join(label.get_text(" ", strip=True).split()) if label else "",
                )
    return links


def discover_event(
    year: int,
    *,
    data_dir: Path,
    client: CacheClient,
) -> dict[str, Any]:
    event_code, seed_url = EVENTS[year]
    raw_dir = data_dir / "raw" / event_code
    root_content = client.get(seed_url, raw_dir / "discovery" / "root.html", event_code)
    root_soup = soup_from_bytes(root_content)
    sessions = session_links(root_soup, event_code)
    session_groups: defaultdict[str, set[str]] = defaultdict(set)
    pages = [{"url": seed_url, "label": "root"}]

    groups = discovery_links(root_soup, event_code=event_code, current_url=seed_url)
    for index, (group_url, group_label) in enumerate(sorted(groups.items()), start=1):
        digest = hashlib.sha256(group_url.encode("utf-8")).hexdigest()[:16]
        content = client.get(
            group_url,
            raw_dir / "discovery" / f"group_{index:03d}_{digest}.html",
            event_code,
        )
        pages.append({"url": group_url, "label": group_label})
        found = session_links(soup_from_bytes(content), event_code)
        for session_id, details in found.items():
            existing = sessions.setdefault(session_id, details)
            if len(details["title_hint"]) > len(existing["title_hint"]):
                existing["title_hint"] = details["title_hint"]
            if group_label:
                session_groups[session_id].add(group_label)

    manifest = {
        "event_code": event_code,
        "year": year,
        "seed_url": seed_url,
        "generated_at": utc_now(),
        "discovery_pages": pages,
        "session_count": len(sessions),
        "sessions": [
            {
                "session_id": session_id,
                **sessions[session_id],
                "discovery_groups": sorted(session_groups[session_id]),
            }
            for session_id in sorted(sessions, key=int)
        ],
    }
    atomic_json(data_dir / "manifests" / f"{event_code}.json", manifest)
    print(
        f"{event_code}: {len(pages)} discovery pages, {len(sessions)} sessions",
        file=sys.stderr,
    )
    return manifest


def load_manifest(data_dir: Path, event_code: str) -> dict[str, Any]:
    path = data_dir / "manifests" / f"{event_code}.json"
    if not path.is_file():
        raise CrawlError(f"missing manifest; run discover first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_event(year: int, *, data_dir: Path, client: CacheClient) -> None:
    event_code, _ = EVENTS[year]
    manifest = load_manifest(data_dir, event_code)
    sessions = manifest["sessions"]
    for index, session in enumerate(sessions, start=1):
        client.get(
            session["url"],
            data_dir / "raw" / event_code / "sessions" / f"{session['session_id']}.html",
            event_code,
        )
        if index % 25 == 0 or index == len(sessions):
            print(
                f"{event_code}: cached {index}/{len(sessions)} session pages "
                f"(network requests this run: {client.requests})",
                file=sys.stderr,
            )


def description_from_soup(soup: BeautifulSoup) -> tuple[str | None, str | None]:
    modal = soup.select_one('[id^="modalSessionDescription"]')
    if modal is not None:
        content = modal.select_one(".modal-content-container")
        heading = modal.select_one(".co-modal-default-modal-headline")
        if content is not None:
            return (
                " ".join(content.get_text(" ", strip=True).split()) or None,
                " ".join(heading.get_text(" ", strip=True).split()) if heading else None,
            )
        # Version 7 and newer use a custom <co-modal-default> element whose
        # text children are the description and whose title attribute carries
        # the session code.
        if modal.name == "co-modal-default":
            description = " ".join(modal.get_text(" ", strip=True).split()) or None
            modal_title = " ".join(str(modal.get("title", "")).split()) or None
            return description, modal_title
    for selector in ("#session-description", ".session-description", ".session_description"):
        element = soup.select_one(selector)
        if element is not None:
            return " ".join(element.get_text(" ", strip=True).split()) or None, None
    return None, None


def parse_session_page(
    content: bytes,
    *,
    event_code: str,
    year: int,
    session: dict[str, Any],
) -> dict[str, Any]:
    soup = soup_from_bytes(content)
    description, description_heading = description_from_soup(soup)
    code = None
    if description_heading:
        code = re.sub(r"\s*\|\s*Session description\s*$", "", description_heading, flags=re.I)
    if not code:
        heading = soup.select_one("#sessiondetails h1")
        if heading is not None:
            candidate = " ".join(heading.get_text(" ", strip=True).split())
            if candidate and candidate.lower() != "asset":
                code = candidate
    if not code:
        block = soup.select_one(
            f'.co_mto_programme-session-block[data-id="{session["session_id"]}"] '
            ".co_mto_programme-session-block-number"
        )
        if block is not None:
            candidate = " ".join(block.get_text(" ", strip=True).split())
            if candidate:
                code = candidate
    if not code:
        title_text = soup.title.get_text(" ", strip=True) if soup.title else ""
        match = re.search(r"\bSession\s+([A-Z0-9.]+)", title_text, re.I)
        if match:
            code = match.group(1).strip()
    if not code:
        heading = soup.find(["h1", "h2"], string=re.compile(r"Session\s+", re.I))
        if heading:
            match = re.search(r"Session\s+(.+)", heading.get_text(" ", strip=True), re.I)
            code = match.group(1).strip() if match else None

    title = session.get("title_hint") or None
    if not title:
        title_element = soup.select_one("#sessiondetails strong.link-coloured")
        if title_element is not None:
            title = " ".join(title_element.get_text(" ", strip=True).split()) or None

    text = soup.get_text(" ", strip=True)
    abstract_pattern = re.compile(rf"\b{re.escape(event_code)}-\d+\b", re.I)
    abstract_codes = sorted(
        {match.upper() for match in abstract_pattern.findall(text)},
        key=lambda item: int(item.rsplit("-", 1)[1]),
    )
    return {
        "event_code": event_code,
        "event_year": year,
        "session_id": session["session_id"],
        "session_code": code,
        "title": title,
        "description": description,
        "has_description": bool(description),
        "description_missing": not bool(description),
        "abstract_codes": abstract_codes,
        "abstract_count": len(abstract_codes),
        "discovery_groups": session.get("discovery_groups") or [],
        "source_url": session["url"],
        "raw_sha256": hashlib.sha256(content).hexdigest(),
    }


def parse_events(years: list[int], data_dir: Path) -> dict[str, Any]:
    combined: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    for year in years:
        event_code, _ = EVENTS[year]
        manifest = load_manifest(data_dir, event_code)
        parsed: list[dict[str, Any]] = []
        for session in manifest["sessions"]:
            path = data_dir / "raw" / event_code / "sessions" / f"{session['session_id']}.html"
            if not path.is_file():
                raise CrawlError(f"missing cached session page: {path}")
            parsed.append(
                parse_session_page(
                    path.read_bytes(),
                    event_code=event_code,
                    year=year,
                    session=session,
                )
            )
        output = data_dir / "processed" / f"{event_code}_sessions.jsonl.gz"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            for record in parsed:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary.replace(output)
        combined.extend(parsed)
        report[event_code] = {
            "sessions": len(parsed),
            "with_description": sum(item["has_description"] for item in parsed),
            "without_description": sum(not item["has_description"] for item in parsed),
            "abstract_assignments": sum(item["abstract_count"] for item in parsed),
        }

    combined_output = data_dir / "processed" / "egu_sessions_2020_2026.jsonl.gz"
    temporary = combined_output.with_name(f".{combined_output.name}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        for record in combined:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(combined_output)
    atomic_json(data_dir / "processed" / "coverage_report.json", report)
    print(json.dumps(report, indent=2))
    return report


def parse_years(value: str) -> list[int]:
    years: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            years.update(range(int(start), int(end) + 1))
        elif part:
            years.add(int(part))
    unsupported = sorted(year for year in years if year not in EVENTS)
    if unsupported:
        raise argparse.ArgumentTypeError(f"unsupported years: {unsupported}")
    return sorted(years)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("discover", "fetch"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--years", type=parse_years, default=parse_years("2020-2026"))
        subparser.add_argument("--data-dir", type=Path, default=Path("data/sessions"))
        subparser.add_argument("--delay", type=float, default=2.0)
        subparser.add_argument("--jitter", type=float, default=0.5)
        subparser.add_argument("--max-requests", type=int, default=10000)
        subparser.add_argument("--contact", required=True)
    parse_parser = subparsers.add_parser("parse")
    parse_parser.add_argument("--years", type=parse_years, default=parse_years("2020-2026"))
    parse_parser.add_argument("--data-dir", type=Path, default=Path("data/sessions"))
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "parse":
            parse_events(args.years, args.data_dir)
            return 0
        client = CacheClient(
            delay=args.delay,
            jitter=args.jitter,
            max_requests=args.max_requests,
            contact=args.contact,
        )
        if args.command == "discover":
            for year in args.years:
                discover_event(year, data_dir=args.data_dir, client=client)
        else:
            for year in args.years:
                fetch_event(year, data_dir=args.data_dir, client=client)
        print(
            json.dumps(
                {
                    "network_requests": client.requests,
                    "cache_hits": client.cache_hits,
                },
                indent=2,
            )
        )
    except CrawlError as exc:
        print(json.dumps({"event": "error", "message": str(exc)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
