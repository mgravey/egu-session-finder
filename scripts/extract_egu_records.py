#!/usr/bin/env python3
"""Extract complete EGU Crossref records from located dump files."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

from crossref_prefix_scan import (
    ScanError,
    first_text,
    iter_records,
    json_dumps,
    json_loads,
    normalize_doi,
    source_name,
)


EGU_DOI_PATTERN = re.compile(
    r"^10\.5194/egusphere-egu(?P<event_year>\d{2}|\d{4})-(?P<abstract_id>\d+)$",
    re.IGNORECASE,
)


class TextExtractor(HTMLParser):
    BLOCK_TAGS = {
        "br",
        "div",
        "li",
        "p",
        "section",
        "table",
        "td",
        "th",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        local_tag = tag.lower().rsplit(":", 1)[-1]
        if local_tag in self.BLOCK_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        local_tag = tag.lower().rsplit(":", 1)[-1]
        if local_tag in self.BLOCK_TAGS:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def clean_abstract(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    parser = TextExtractor()
    parser.feed(value)
    parser.close()
    text = unescape("".join(parser.parts))
    return " ".join(text.split()) or None


def parse_egu_doi(doi: Any) -> tuple[str, int, str] | None:
    normalized = normalize_doi(doi)
    if normalized is None:
        return None
    match = EGU_DOI_PATTERN.fullmatch(normalized)
    if match is None:
        return None
    event_year_text = match.group("event_year")
    event_year = (
        int(event_year_text)
        if len(event_year_text) == 4
        else 2000 + int(event_year_text)
    )
    event_code = f"EGU{event_year_text}"
    return event_code, event_year, match.group("abstract_id")


def normalize_egu_record(
    record: dict[str, Any],
    *,
    source_file: str,
    source_line: int,
    item_index: int,
) -> dict[str, Any] | None:
    parsed = parse_egu_doi(record.get("DOI"))
    if parsed is None or record.get("type") != "posted-content":
        return None
    event_code, event_year, abstract_id = parsed
    abstract_raw = record.get("abstract")
    abstract_text = clean_abstract(abstract_raw)
    return {
        "source_file": source_file,
        "source_line": source_line,
        "item_index": item_index,
        "doi": normalize_doi(record.get("DOI")),
        "event_code": event_code,
        "event_year": event_year,
        "abstract_id": abstract_id,
        "title": first_text(record.get("title")),
        "abstract_text": abstract_text,
        "abstract_raw": abstract_raw if isinstance(abstract_raw, str) else None,
        "has_abstract": abstract_text is not None,
        "authors": record.get("author") or [],
        "container_title": first_text(record.get("container-title")),
        "publisher": record.get("publisher"),
        "published": record.get("published"),
        "created": record.get("created"),
        "deposited": record.get("deposited"),
        "indexed": record.get("indexed"),
        "url": record.get("URL"),
        "event": record.get("event"),
        "crossref_record": record,
    }


def list_relevant_files(preview: Path, dump_root: Path, output: Any) -> int:
    files: set[str] = set()
    try:
        with gzip.open(preview, "rt", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("type") != "posted-content":
                    continue
                if parse_egu_doi(record.get("doi")) is None:
                    continue
                source_file = record.get("source_file")
                if not isinstance(source_file, str):
                    raise ScanError("preview record is missing source_file")
                files.add(str(dump_root / source_file))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScanError(f"cannot read preview {preview}: {exc}") from exc

    for file_name in sorted(files):
        output.write(file_name.encode("utf-8") + b"\0")
    return len(files)


def extract_file(input_path: Path, dump_root: Path, output: Any) -> dict[str, Any]:
    relative_source = source_name(input_path, dump_root)
    lines = 0
    records = 0
    matches = 0
    abstracts = 0
    try:
        with gzip.open(input_path, "rb") as stream:
            for source_line, raw_line in enumerate(stream, start=1):
                lines = source_line
                if not raw_line.strip():
                    continue
                document = json_loads(raw_line)
                for item_index, record in iter_records(document):
                    records += 1
                    normalized = normalize_egu_record(
                        record,
                        source_file=relative_source,
                        source_line=source_line,
                        item_index=item_index,
                    )
                    if normalized is None:
                        continue
                    matches += 1
                    abstracts += int(normalized["has_abstract"])
                    output.write(json_dumps(normalized) + b"\n")
    except Exception as exc:
        if isinstance(exc, ScanError):
            raise
        raise ScanError(f"cannot extract {relative_source}: {exc}") from exc
    return {
        "event": "extract_summary",
        "source_file": relative_source,
        "lines": lines,
        "records": records,
        "matches": matches,
        "abstracts": abstracts,
    }


def summarize(path: Path) -> dict[str, Any]:
    events: Counter[str] = Counter()
    total = 0
    abstracts = 0
    seen: set[str] = set()
    duplicates = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            total += 1
            abstracts += int(bool(record.get("has_abstract")))
            events[str(record.get("event_code"))] += 1
            doi = str(record.get("doi"))
            if doi in seen:
                duplicates += 1
            seen.add(doi)
    return {
        "records": total,
        "unique_dois": len(seen),
        "duplicates": duplicates,
        "with_abstract": abstracts,
        "without_abstract": total - abstracts,
        "abstract_coverage": round(abstracts / total, 6) if total else 0,
        "events": dict(sorted(events.items())),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-files")
    list_parser.add_argument("preview", type=Path)
    list_parser.add_argument("--dump-root", type=Path, required=True)

    extract_parser = subparsers.add_parser("extract-file")
    extract_parser.add_argument("input", type=Path)
    extract_parser.add_argument("--dump-root", type=Path, required=True)
    extract_parser.add_argument("--quiet", action="store_true")

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("input", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list-files":
            count = list_relevant_files(
                args.preview,
                args.dump_root,
                sys.stdout.buffer,
            )
            print(
                json.dumps({"event": "file_list_summary", "files": count}),
                file=sys.stderr,
            )
        elif args.command == "extract-file":
            summary_data = extract_file(
                args.input,
                args.dump_root,
                sys.stdout.buffer,
            )
            if not args.quiet:
                print(json.dumps(summary_data), file=sys.stderr)
        else:
            print(json.dumps(summarize(args.input), indent=2))
    except ScanError as exc:
        print(json.dumps({"event": "error", "message": str(exc)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
