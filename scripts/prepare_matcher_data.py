#!/usr/bin/env python3
"""Join EGU Crossref abstracts to cached programme sessions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Any, Iterable

from matcher_common import (
    abstract_text,
    clean_text,
    division_codes,
    iter_jsonl,
    primary_division_code,
    session_text,
    visual_division_codes,
    write_json,
    write_jsonl_gz,
)


SPLITS = {
    # The public post-conference programmes do not expose representative
    # abstract assignments for 2020 or 2021. Keep their session catalogues for
    # auxiliary analysis, but never treat the sparse labels as training truth.
    2020: "auxiliary_unlabeled",
    2021: "auxiliary_unlabeled",
    2022: "train",
    2023: "train",
    2024: "validation",
    2025: "test",
    2026: "forward",
}


def canonical_abstract_code(event_code: str, abstract_id: str | int) -> str:
    return f"{clean_text(event_code).upper()}-{clean_text(abstract_id)}"


def prepare(
    *,
    crossref_paths: list[Path],
    sessions_path: Path,
    output_dir: Path,
    review_size: int = 100,
    seed: int = 20260823,
) -> dict[str, Any]:
    sessions: list[dict[str, Any]] = []
    session_by_key: dict[str, dict[str, Any]] = {}
    assignments_by_code: defaultdict[str, list[str]] = defaultdict(list)

    for raw in iter_jsonl(sessions_path):
        event_code = clean_text(raw.get("event_code")).upper()
        year = int(raw["event_year"])
        session_id = clean_text(raw.get("session_id"))
        key = f"{event_code}/{session_id}"
        if key in session_by_key:
            raise ValueError(f"duplicate session key: {key}")
        groups = [clean_text(item) for item in raw.get("discovery_groups", []) if clean_text(item)]
        codes = division_codes(groups)
        co_organizer_codes = [
            clean_text(item).upper()
            for item in raw.get("co_organizer_codes", [])
            if clean_text(item).upper() in codes
        ]
        record = {
            **raw,
            "event_code": event_code,
            "event_year": year,
            "split": SPLITS.get(year, "unassigned"),
            "session_key": key,
            "session_code": clean_text(raw.get("session_code")) or None,
            "title": clean_text(raw.get("title")) or None,
            "description": clean_text(raw.get("description")) or None,
            "description_missing": not bool(clean_text(raw.get("description"))),
            "discovery_groups": groups,
            "division_codes": codes,
            "primary_division_code": primary_division_code(raw.get("session_code"), codes),
            "co_organizer_label": clean_text(raw.get("co_organizer_label")) or None,
            "co_organizer_codes": list(dict.fromkeys(co_organizer_codes)),
            "visual_division_codes": visual_division_codes(
                raw.get("session_code"), codes, co_organizer_codes
            ),
        }
        record["session_text"] = session_text(record, include_divisions=True)
        record["session_text_without_divisions"] = session_text(
            record, include_divisions=False
        )
        sessions.append(record)
        session_by_key[key] = record
        for code in raw.get("abstract_codes", []):
            normalized = clean_text(code).upper()
            if key not in assignments_by_code[normalized]:
                assignments_by_code[normalized].append(key)

    abstracts: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    seen_dois: set[str] = set()
    seen_codes: set[str] = set()
    unmatched_by_year: Counter[int] = Counter()
    matched_by_year: Counter[int] = Counter()
    abstract_count_by_year: Counter[int] = Counter()
    abstracts_with_multiple_sessions = 0

    for path in crossref_paths:
        for raw in iter_jsonl(path):
            doi = clean_text(raw.get("doi")).lower()
            if doi in seen_dois:
                raise ValueError(f"duplicate DOI across Crossref inputs: {doi}")
            seen_dois.add(doi)
            event_code = clean_text(raw.get("event_code")).upper()
            year = int(raw["event_year"])
            code = canonical_abstract_code(event_code, raw.get("abstract_id"))
            if code in seen_codes:
                raise ValueError(f"duplicate abstract code across Crossref inputs: {code}")
            seen_codes.add(code)
            session_keys = sorted(assignments_by_code.get(code, []))
            if len(session_keys) > 1:
                abstracts_with_multiple_sessions += 1
            if session_keys:
                matched_by_year[year] += 1
            else:
                unmatched_by_year[year] += 1
            abstract_count_by_year[year] += 1
            title = clean_text(raw.get("title"))
            body = clean_text(raw.get("abstract_text"))
            abstract = {
                "abstract_key": code,
                "event_code": event_code,
                "event_year": year,
                "split": SPLITS.get(year, "unassigned"),
                "abstract_id": clean_text(raw.get("abstract_id")),
                "doi": doi,
                "title": title,
                "abstract_text": body,
                "has_abstract": bool(body),
                "query_text": abstract_text(title, body),
                "session_keys": session_keys,
            }
            abstracts.append(abstract)
            for session_key in session_keys:
                session = session_by_key[session_key]
                assignments.append(
                    {
                        "abstract_key": code,
                        "session_key": session_key,
                        "event_code": event_code,
                        "event_year": year,
                        "split": SPLITS.get(year, "unassigned"),
                        "division_codes": session["division_codes"],
                    }
                )

    unknown_assignment_codes = sorted(set(assignments_by_code) - seen_codes)
    years = sorted(set(abstract_count_by_year) | {item["event_year"] for item in sessions})
    coverage: dict[str, Any] = {
        "totals": {
            "abstracts": len(abstracts),
            "abstracts_with_text": sum(item["has_abstract"] for item in abstracts),
            "sessions": len(sessions),
            "assignments": len(assignments),
            "matched_abstracts": sum(bool(item["session_keys"]) for item in abstracts),
            "unmatched_abstracts": sum(not item["session_keys"] for item in abstracts),
            "abstracts_with_multiple_sessions": abstracts_with_multiple_sessions,
            "programme_abstract_codes_not_in_crossref": len(unknown_assignment_codes),
        },
        "years": {},
        "unknown_programme_abstract_codes": unknown_assignment_codes,
    }
    for year in years:
        year_sessions = [item for item in sessions if item["event_year"] == year]
        year_assignments = [item for item in assignments if item["event_year"] == year]
        total = abstract_count_by_year[year]
        coverage["years"][str(year)] = {
            "split": SPLITS.get(year, "unassigned"),
            "crossref_abstracts": total,
            "matched_abstracts": matched_by_year[year],
            "unmatched_abstracts": unmatched_by_year[year],
            "match_coverage": round(matched_by_year[year] / total, 6) if total else None,
            "supervised_eligible": SPLITS.get(year) in {"train", "validation", "test"},
            "sessions": len(year_sessions),
            "sessions_with_description": sum(
                not item["description_missing"] for item in year_sessions
            ),
            "title_only_sessions": sum(item["description_missing"] for item in year_sessions),
            "assignments": len(year_assignments),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_gz(output_dir / "sessions.jsonl.gz", sessions)
    write_jsonl_gz(output_dir / "abstracts.jsonl.gz", abstracts)
    write_jsonl_gz(output_dir / "assignments.jsonl.gz", assignments)
    write_json(output_dir / "coverage_report.json", coverage)

    rng = random.Random(seed)
    eligible = [
        item
        for item in abstracts
        if item["has_abstract"]
        and item["session_keys"]
        and item["split"] != "auxiliary_unlabeled"
    ]
    strata: defaultdict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for item in eligible:
        first = session_by_key[item["session_keys"][0]]
        division = first["division_codes"][0] if first["division_codes"] else "OTHER"
        strata[(item["event_year"], division)].append(item)
    for values in strata.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    queues = [values for _, values in sorted(strata.items())]
    while queues and len(selected) < min(review_size, len(eligible)):
        next_queues = []
        for values in queues:
            if values and len(selected) < review_size:
                item = values.pop()
                selected.append(
                    {
                        **item,
                        "assigned_sessions": [
                            {
                                "session_key": key,
                                "session_code": session_by_key[key]["session_code"],
                                "title": session_by_key[key]["title"],
                                "division_codes": session_by_key[key]["division_codes"],
                            }
                            for key in item["session_keys"]
                        ],
                    }
                )
            if values:
                next_queues.append(values)
        queues = next_queues
    write_jsonl_gz(output_dir / "human_review_sample.jsonl.gz", selected)
    return coverage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--crossref",
        type=Path,
        action="append",
        required=True,
        help="Crossref EGU JSONL(.gz); repeat for multiple files",
    )
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/matcher/prepared"))
    parser.add_argument("--review-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = prepare(
        crossref_paths=args.crossref,
        sessions_path=args.sessions,
        output_dir=args.output_dir,
        review_size=args.review_size,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
