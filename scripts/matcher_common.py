"""Shared, dependency-light helpers for matcher data and evaluation scripts."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Iterator


DIVISION_CODE = re.compile(r"\(([A-Z][A-Z0-9-]{0,7})\)\s*$")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc


def write_jsonl_gz(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    count = 0
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
            count += 1
    temporary.replace(path)
    return count


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def division_codes(groups: Iterable[str]) -> list[str]:
    codes = {
        match.group(1)
        for group in groups
        if (match := DIVISION_CODE.search(clean_text(group))) is not None
    }
    return sorted(codes)


def primary_division_code(session_code: Any, known_codes: Iterable[str]) -> str | None:
    """Return the host programme encoded at the start of an EGU session code."""
    value = clean_text(session_code).upper()
    for code in sorted({clean_text(item).upper() for item in known_codes}, key=len, reverse=True):
        if code and re.match(rf"^{re.escape(code)}(?=\d|\.|$)", value):
            return code
    return None


def visual_division_codes(
    session_code: Any,
    known_codes: Iterable[str],
    co_organizer_codes: Iterable[str] = (),
) -> list[str]:
    """Order divisions for presentation: host, named co-organisers, then others."""
    known = sorted({clean_text(item).upper() for item in known_codes if clean_text(item)})
    primary = primary_division_code(session_code, known)
    co_organizers = [
        clean_text(item).upper()
        for item in co_organizer_codes
        if clean_text(item).upper() in known
    ]
    ordered = ([primary] if primary else []) + co_organizers + known
    return list(dict.fromkeys(item for item in ordered if item))


def abstract_text(title: str, abstract: str) -> str:
    title = clean_text(title)
    abstract = clean_text(abstract)
    return f"Title: {title}\nAbstract: {abstract}" if title else f"Abstract: {abstract}"


def session_text(session: dict[str, Any], *, include_divisions: bool = True) -> str:
    code = clean_text(session.get("session_code"))
    title = clean_text(session.get("title"))
    description = clean_text(session.get("description"))
    groups = [clean_text(item) for item in session.get("discovery_groups", []) if clean_text(item)]
    parts = [f"Session {code}: {title}" if code else f"Session: {title}"]
    if include_divisions and groups:
        parts.append("Divisions: " + "; ".join(groups))
    if description:
        parts.append("Description: " + description)
    return "\n".join(parts)


def l2_normalize(values: Iterable[float]) -> list[float]:
    vector = [float(value) for value in values]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
