#!/usr/bin/env python3
"""Fetch one EGU year's current Crossref records with a persistent page cache."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from crossref_prefix_scan import json_dumps, json_loads
from extract_egu_records import normalize_egu_record, parse_egu_doi


API_URL = "https://api.crossref.org/v1/prefixes/10.5194/works"


def atomic_gzip_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with gzip.open(temporary, "wb") as stream:
        stream.write(data)
    temporary.replace(path)


def load_page(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as stream:
        document = json_loads(stream.read())
    if not isinstance(document, dict) or document.get("status") != "ok":
        raise RuntimeError(f"invalid cached Crossref response: {path}")
    return document


def fetch_url(url: str, email: str, retries: int = 5) -> bytes:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": f"egu-session-matcher/0.1 (mailto:{email})",
        },
    )
    delays = (5, 15, 60, 120, 300)
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=60) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt + 1 == retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = int(retry_after) if retry_after and retry_after.isdigit() else delays[attempt]
        except URLError:
            if attempt + 1 == retries:
                raise
            delay = delays[attempt]
        print(f"Temporary API failure; retrying in {delay} seconds", file=sys.stderr)
        time.sleep(delay)
    raise RuntimeError("unreachable retry state")


def validate_manifest(cache_dir: Path, year: int, rows: int) -> None:
    manifest_path = cache_dir / "manifest.json"
    expected = {
        "api_url": API_URL,
        "doi_prefix": "10.5194/",
        "record_type": "posted-content",
        "year": year,
        "rows": rows,
    }
    if manifest_path.exists():
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise RuntimeError(
                f"cache manifest does not match this query: {manifest_path}"
            )
        return
    if list(cache_dir.glob("page_*.json.gz")):
        raise RuntimeError("cached pages exist without a manifest")
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)


def download_pages(cache_dir: Path, year: int, rows: int, email: str, delay: float) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    validate_manifest(cache_dir, year, rows)
    pages = sorted(cache_dir.glob("page_*.json.gz"))

    if pages:
        last = load_page(pages[-1])["message"]
        last_items = last.get("items") or []
        if len(last_items) < rows or not last.get("next-cursor"):
            print(f"Cache is already complete with {len(pages)} pages.", file=sys.stderr)
            return
        cursor = last["next-cursor"]
        page_number = len(pages) + 1
    else:
        cursor = "*"
        page_number = 1

    while True:
        parameters = {
            "filter": (
                "type:posted-content,"
                f"from-pub-date:{year}-01-01,"
                f"until-pub-date:{year}-12-31"
            ),
            "rows": rows,
            "cursor": cursor,
            "mailto": email,
        }
        url = f"{API_URL}?{urlencode(parameters)}"
        raw = fetch_url(url, email)
        document = json_loads(raw)
        if not isinstance(document, dict) or document.get("status") != "ok":
            raise RuntimeError("Crossref returned a non-success response")
        message = document.get("message") or {}
        items = message.get("items") or []
        page_path = cache_dir / f"page_{page_number:05d}.json.gz"
        atomic_gzip_write(page_path, raw)
        print(
            f"Cached page {page_number}: {len(items)} records "
            f"(reported total {message.get('total-results')})",
            file=sys.stderr,
        )
        if len(items) < rows or not message.get("next-cursor"):
            break
        cursor = message["next-cursor"]
        page_number += 1
        time.sleep(delay)


def export_year(cache_dir: Path, output: Path, year: int) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    seen: set[str] = set()
    total_candidates = 0
    abstracts = 0

    try:
        with gzip.open(temporary, "wb") as destination:
            for page_path in sorted(cache_dir.glob("page_*.json.gz")):
                message = load_page(page_path)["message"]
                for item_index, record in enumerate(message.get("items") or []):
                    parsed = parse_egu_doi(record.get("DOI"))
                    if parsed is None or parsed[1] != year:
                        continue
                    total_candidates += 1
                    normalized = normalize_egu_record(
                        record,
                        source_file=page_path.name,
                        source_line=0,
                        item_index=item_index,
                    )
                    if normalized is None:
                        continue
                    doi = str(normalized["doi"])
                    if doi in seen:
                        continue
                    seen.add(doi)
                    abstracts += int(normalized["has_abstract"])
                    destination.write(json_dumps(normalized) + b"\n")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "year": year,
        "records": len(seen),
        "duplicate_candidates": total_candidates - len(seen),
        "with_abstract": abstracts,
        "without_abstract": len(seen) - abstracts,
        "abstract_coverage": round(abstracts / len(seen), 6) if seen else 0,
        "output": str(output),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--email", required=True, help="contact for Crossref's polite pool")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1000, choices=range(1, 1001))
    parser.add_argument("--delay", type=float, default=1.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.year < 2000 or args.year > 2100:
        print("--year must be between 2000 and 2100", file=sys.stderr)
        return 2
    try:
        download_pages(
            args.cache_dir,
            args.year,
            args.rows,
            args.email,
            args.delay,
        )
        summary = export_year(args.cache_dir, args.output, args.year)
    except (OSError, RuntimeError, HTTPError, URLError, ValueError) as exc:
        print(json.dumps({"event": "error", "message": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
