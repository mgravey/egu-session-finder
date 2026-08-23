#!/usr/bin/env python3
"""Find Crossref records whose DOI starts with a configured prefix.

The program is intentionally a single-input-file worker so GNU Parallel can
run one independent process per compressed Crossref dump file. Matching
records are written as JSON Lines to stdout; progress and summaries go to
stderr so stdout can be safely concatenated and compressed.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Iterator


try:
    import orjson  # type: ignore[import-not-found]
except ImportError:
    orjson = None


DOI_URL_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
)


class ScanError(RuntimeError):
    """A fatal error carrying enough context for a concise CLI message."""


def json_loads(raw: bytes) -> Any:
    if orjson is not None:
        return orjson.loads(raw)
    return json.loads(raw)


def json_dumps(value: Any) -> bytes:
    if orjson is not None:
        return orjson.dumps(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def normalize_doi(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    doi = value.strip().lower()
    for prefix in DOI_URL_PREFIXES:
        if doi.startswith(prefix):
            doi = doi[len(prefix) :].strip()
            break
    return doi or None


def normalize_prefix(value: str) -> str:
    prefix = normalize_doi(value)
    if not prefix:
        raise argparse.ArgumentTypeError("DOI prefix must not be empty")
    return prefix


def iter_records(document: Any) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield records from common Crossref JSONL shapes.

    Most public-dump lines are individual records. Supporting arrays and
    message/items envelopes makes the scanner safe for other Crossref exports
    without changing the source-line locator semantics.
    """

    if isinstance(document, dict) and "DOI" in document:
        yield 0, document
        return

    candidates: Any = None
    if isinstance(document, list):
        candidates = document
    elif isinstance(document, dict):
        message = document.get("message")
        if isinstance(message, dict) and isinstance(message.get("items"), list):
            candidates = message["items"]
        elif isinstance(document.get("items"), list):
            candidates = document["items"]

    if candidates is not None:
        for item_index, record in enumerate(candidates):
            if isinstance(record, dict):
                yield item_index, record


def first_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                return item
    return None


def preview_fields(record: dict[str, Any]) -> dict[str, Any]:
    published = (
        record.get("published")
        or record.get("published-online")
        or record.get("published-print")
        or record.get("issued")
    )
    return {
        "type": record.get("type"),
        "subtype": record.get("subtype"),
        "publisher": record.get("publisher"),
        "title": first_text(record.get("title")),
        "container_title": first_text(record.get("container-title")),
        "published": published,
    }


def source_name(input_path: Path, dump_root: Path | None) -> str:
    resolved_input = input_path.resolve()
    if dump_root is not None:
        try:
            return str(resolved_input.relative_to(dump_root.resolve()))
        except ValueError as exc:
            raise ScanError(
                f"input file is outside --dump-root: {resolved_input}"
            ) from exc
    return str(resolved_input)


def make_output_record(
    *,
    source_file: str,
    source_line: int,
    item_index: int,
    doi: str,
    record: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "source_file": source_file,
        "source_line": source_line,
        "item_index": item_index,
        "doi": doi,
    }
    if mode in {"preview", "full"}:
        output.update(preview_fields(record))
    if mode == "full":
        output["record"] = record
    return output


def scan_file(
    input_path: Path,
    *,
    dump_root: Path | None,
    doi_prefix: str,
    mode: str,
    on_json_error: str,
    output: Any = None,
) -> dict[str, Any]:
    if output is None:
        output = sys.stdout.buffer
    if not input_path.is_file():
        raise ScanError(f"input is not a file: {input_path}")

    relative_source = source_name(input_path, dump_root)
    started = time.monotonic()
    line_count = 0
    record_count = 0
    match_count = 0
    json_error_count = 0
    decompressed_bytes = 0

    try:
        with gzip.open(input_path, "rb") as stream:
            for source_line, raw_line in enumerate(stream, start=1):
                line_count = source_line
                decompressed_bytes += len(raw_line)
                if not raw_line.strip():
                    continue
                try:
                    document = json_loads(raw_line)
                except Exception as exc:  # parser exceptions vary by backend
                    json_error_count += 1
                    if on_json_error == "fail":
                        raise ScanError(
                            f"invalid JSON in {relative_source}:{source_line}: {exc}"
                        ) from exc
                    continue

                for item_index, record in iter_records(document):
                    record_count += 1
                    doi = normalize_doi(record.get("DOI"))
                    if doi is None or not doi.startswith(doi_prefix):
                        continue
                    match_count += 1
                    match = make_output_record(
                        source_file=relative_source,
                        source_line=source_line,
                        item_index=item_index,
                        doi=doi,
                        record=record,
                        mode=mode,
                    )
                    output.write(json_dumps(match) + b"\n")
    except OSError as exc:
        raise ScanError(f"cannot read gzip file {relative_source}: {exc}") from exc

    elapsed = max(time.monotonic() - started, 1e-9)
    return {
        "event": "scan_summary",
        "source_file": relative_source,
        "lines": line_count,
        "records": record_count,
        "matches": match_count,
        "json_errors": json_error_count,
        "decompressed_bytes": decompressed_bytes,
        "elapsed_seconds": round(elapsed, 6),
        "decompressed_mib_per_second": round(
            decompressed_bytes / (1024 * 1024) / elapsed,
            3,
        ),
        "json_backend": "orjson" if orjson is not None else "stdlib",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan one gzip-compressed Crossref JSONL file for a DOI prefix; "
            "write matching records as JSONL to stdout."
        )
    )
    parser.add_argument("input", type=Path, help="one .json.gz Crossref dump file")
    parser.add_argument(
        "--dump-root",
        type=Path,
        help="store source_file relative to this directory",
    )
    parser.add_argument(
        "--prefix",
        type=normalize_prefix,
        default="10.5194/",
        help="normalized DOI prefix to match (default: 10.5194/)",
    )
    parser.add_argument(
        "--mode",
        choices=("locator", "preview", "full"),
        default="preview",
        help="amount of Crossref metadata to emit (default: preview)",
    )
    parser.add_argument(
        "--on-json-error",
        choices=("fail", "skip"),
        default="fail",
        help="fail the worker or skip malformed JSON lines (default: fail)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not write the per-file JSON summary to stderr",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = scan_file(
            args.input,
            dump_root=args.dump_root,
            doi_prefix=args.prefix,
            mode=args.mode,
            on_json_error=args.on_json_error,
        )
    except ScanError as exc:
        print(
            json.dumps({"event": "scan_error", "message": str(exc)}),
            file=sys.stderr,
        )
        return 2
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except BrokenPipeError:
            pass
        return 1

    if not args.quiet:
        print(json_dumps(summary).decode("utf-8"), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
