#!/usr/bin/env python3
"""Summarize rolling-origin backtests into the website's track-record asset.

Each test year Y was scored by a model trained only on earlier years and
calibrated on Y-1, exactly as the deployed model is prepared for the next
conference. Statistics use typical scientific sessions only, as the site does.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from matcher_common import write_json


def cumulative(curve: list[dict[str, Any]], max_rank: int) -> list[float]:
    return [round(point["cumulative_fraction"], 6) for point in curve[:max_rank]]


def run(args: argparse.Namespace) -> dict[str, Any]:
    years = []
    for year in args.years:
        directory = args.rolling_dir / str(year)
        ranks = json.loads((directory / "ranks-typical.json").read_text(encoding="utf-8"))
        prediction = json.loads((directory / "prediction-set.json").read_text(encoding="utf-8"))
        ranker = json.loads((directory / "ranker" / "report.json").read_text(encoding="utf-8"))
        summary = ranks["years"][str(year)]
        evaluation = prediction["evaluations"][str(year)]
        years.append(
            {
                "year": year,
                "trainingYears": ranker["training_years"],
                "calibrationYear": prediction["calibrationYear"],
                "abstracts": summary["denominator_rows"],
                "candidateSessions": summary["candidate_sessions"],
                "model": cumulative(summary["curve"], args.max_rank),
                "shortlist": {
                    "coverage": round(evaluation["empiricalCoverage"], 6),
                    "medianSize": evaluation["medianSetSize"],
                },
            }
        )
    result = {
        "schemaVersion": 1,
        "method": "rolling-origin backtest: train on earlier years, calibrate on the previous year",
        "candidatePolicy": "typical-scientific-only",
        "maxRank": args.max_rank,
        "years": years,
    }
    write_json(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rolling-dir", type=Path, default=Path("data/matcher/egu27/rolling"))
    parser.add_argument("--years", type=int, nargs="+", default=[2024, 2025, 2026])
    parser.add_argument("--max-rank", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("web/src/track-record.json"))
    return parser


def main() -> int:
    result = run(build_parser().parse_args())
    for year in result["years"]:
        model = year["model"]
        print(year["year"], {f"top{k}": model[k - 1] for k in (1, 2, 5, 10, 20)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
