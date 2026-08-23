#!/usr/bin/env python3
"""Calibrate dynamic top-score-gap prediction sets from persisted score matrices."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from export_dense_rank_distribution import rank_relevant_candidates
from matcher_common import iter_jsonl, write_json
from session_categories import is_typical_scientific_session


def conformal_higher_quantile(values: Sequence[float], target_coverage: float) -> float:
    """Return the finite-sample higher quantile used by split conformal sets."""
    if not 0 < target_coverage < 1:
        raise ValueError("target coverage must be between zero and one")
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.size == 0:
        raise ValueError("at least one calibration value is required")
    order_statistic = min(
        ordered.size,
        int(math.ceil((ordered.size + 1) * target_coverage)),
    )
    return float(ordered[order_statistic - 1])


def dynamic_set_size(scores: np.ndarray, max_score_gap: float, min_results: int) -> int:
    """Count candidates within the calibrated score gap, with a minimum display set."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a non-empty vector")
    threshold = float(values.max()) - max_score_gap
    within_gap = int(np.count_nonzero(values >= threshold - 1e-7))
    return min(values.size, max(min_results, within_gap))


def scored_examples(
    *,
    year: int,
    manifest: dict[str, Any],
    matrix_dir: Path,
    abstracts: dict[str, dict[str, Any]],
    sessions: dict[str, dict[str, Any]],
    typical_sessions_only: bool,
) -> Iterator[tuple[np.ndarray, list[int]]]:
    year_manifest = manifest["years"][str(year)]
    matrix = np.load(matrix_dir / year_manifest["matrix"], mmap_mode="r")
    all_session_keys = year_manifest["session_keys"]
    kept_columns = np.asarray(
        [
            index
            for index, key in enumerate(all_session_keys)
            if not typical_sessions_only or is_typical_scientific_session(sessions[key])
        ],
        dtype=np.int64,
    )
    kept_keys = [all_session_keys[index] for index in kept_columns]
    candidate_lookup = {key: index for index, key in enumerate(kept_keys)}
    for row_index, abstract_key in enumerate(year_manifest["abstract_keys"]):
        abstract = abstracts[abstract_key]
        original_labels = abstract["session_keys"]
        relevant = [candidate_lookup[key] for key in original_labels if key in candidate_lookup]
        if typical_sessions_only and original_labels and not relevant:
            continue
        if not abstract["has_abstract"] or not relevant:
            continue
        yield np.asarray(matrix[row_index])[kept_columns], relevant


def evaluate_year(
    examples: Iterable[tuple[np.ndarray, Sequence[int]]],
    *,
    max_score_gap: float,
    min_results: int,
    max_results: int,
) -> dict[str, Any]:
    sizes: list[int] = []
    covered = 0
    covered_after_cap = 0
    for scores, relevant in examples:
        size = dynamic_set_size(scores, max_score_gap, min_results)
        rank = rank_relevant_candidates(scores, relevant)
        sizes.append(size)
        covered += int(rank is not None and rank <= size)
        covered_after_cap += int(rank is not None and rank <= min(size, max_results))
    values = np.asarray(sizes, dtype=np.int64)
    queries = len(sizes)
    return {
        "queries": queries,
        "empiricalCoverage": round(covered / queries, 8) if queries else 0.0,
        "displayCappedCoverage": round(covered_after_cap / queries, 8) if queries else 0.0,
        "meanSetSize": round(float(values.mean()), 4) if queries else 0.0,
        "medianSetSize": float(np.median(values)) if queries else 0.0,
        "p90SetSize": float(np.quantile(values, 0.9, method="higher")) if queries else 0.0,
        "maximumSetSize": int(values.max()) if queries else 0,
        "fractionTruncatedAtDisplayCap": (
            round(float(np.mean(values > max_results)), 8) if queries else 0.0
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads((args.matrix_dir / "manifest.json").read_text(encoding="utf-8"))
    abstracts = {
        row["abstract_key"]: row
        for row in iter_jsonl(args.prepared_dir / "abstracts.jsonl.gz")
    }
    sessions = {
        row["session_key"]: row
        for row in iter_jsonl(args.prepared_dir / "sessions.jsonl.gz")
    }
    calibration_examples = list(
        scored_examples(
            year=args.calibration_year,
            manifest=manifest,
            matrix_dir=args.matrix_dir,
            abstracts=abstracts,
            sessions=sessions,
            typical_sessions_only=args.typical_sessions_only,
        )
    )
    gaps = [
        float(scores.max() - scores[np.asarray(relevant, dtype=np.int64)].max())
        for scores, relevant in calibration_examples
    ]
    max_score_gap = conformal_higher_quantile(gaps, args.target_coverage)
    years = list(dict.fromkeys([args.calibration_year, *args.evaluation_years]))
    evaluations = {}
    for year in years:
        examples = (
            calibration_examples
            if year == args.calibration_year
            else scored_examples(
                year=year,
                manifest=manifest,
                matrix_dir=args.matrix_dir,
                abstracts=abstracts,
                sessions=sessions,
                typical_sessions_only=args.typical_sessions_only,
            )
        )
        evaluations[str(year)] = evaluate_year(
            examples,
            max_score_gap=max_score_gap,
            min_results=args.min_results,
            max_results=args.max_results,
        )
    result = {
        "schemaVersion": 1,
        "method": "top-score-gap",
        "score": manifest["score"],
        "candidatePolicy": (
            "typical-scientific-only" if args.typical_sessions_only else "all-sessions"
        ),
        "calibrationYear": args.calibration_year,
        "calibrationExamples": len(calibration_examples),
        "targetCoverage": args.target_coverage,
        "maxScoreGap": max_score_gap,
        "minResults": args.min_results,
        "maxResults": args.max_results,
        "evaluations": evaluations,
        "interpretation": (
            "Coverage measures inclusion of at least one recorded historical assignment; "
            "it is not the probability that a session is scientifically correct."
        ),
    }
    write_json(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/matcher/prepared"))
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--calibration-year", type=int, default=2024)
    parser.add_argument("--evaluation-years", type=int, nargs="+", default=[2025, 2026])
    parser.add_argument("--target-coverage", type=float, default=0.8)
    parser.add_argument("--min-results", type=int, default=3)
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--typical-sessions-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/matcher/experiments/multi-vector/prediction-set.json"),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_results < 1 or args.max_results < args.min_results:
        raise SystemExit("require 1 <= min-results <= max-results")
    result = run(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
