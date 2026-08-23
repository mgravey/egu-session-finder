#!/usr/bin/env python3
"""Select a browser baseline from validation metrics and compare tuned runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from matcher_common import write_json


def validation_candidates(benchmark: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for model_name, result in benchmark["models"].items():
        variants = result["variants"]
        for variant_name in ("dense_with_divisions", "dense_without_divisions"):
            metrics = variants[variant_name]["2024"]
            candidates.append(
                {
                    "model": model_name,
                    "variant": variant_name,
                    "dense_weight": 1.0,
                    "recall_at_10": metrics["recall_at_10"],
                    "mrr": metrics["mrr"],
                }
            )
        hybrid = variants["hybrid"]
        metrics = hybrid["years"]["2024"]
        candidates.append(
            {
                "model": model_name,
                "variant": "hybrid",
                "dense_weight": hybrid["dense_weight"],
                "recall_at_10": metrics["recall_at_10"],
                "mrr": metrics["mrr"],
            }
        )
    return candidates


def select_baseline(benchmark: dict[str, Any]) -> dict[str, Any]:
    candidates = validation_candidates(benchmark)
    if not candidates:
        raise ValueError("benchmark contains no validation candidates")
    selected = max(
        candidates,
        key=lambda item: (item["recall_at_10"], item["mrr"], -item["dense_weight"]),
    )
    model = benchmark["models"][selected["model"]]
    return {
        "selected": selected,
        "model_config": model["model"],
        "confidence_bands": model["confidence_bands"],
        "selection_metric": "2024 recall_at_10, then MRR",
        "all_validation_candidates": sorted(
            candidates, key=lambda item: (-item["recall_at_10"], -item["mrr"])
        ),
    }


def tuned_acceptance(
    baseline: dict[str, Any],
    tuned: dict[str, Any],
    *,
    variant: str,
) -> dict[str, Any]:
    def metrics(source: dict[str, Any], year: str) -> dict[str, Any]:
        value = next(iter(source["models"].values()))["variants"][variant]
        return value["years"][year] if variant == "hybrid" else value[year]

    validation_gain = metrics(tuned, "2024")["recall_at_10"] - metrics(
        baseline, "2024"
    )["recall_at_10"]
    test_regression = metrics(baseline, "2025")["recall_at_10"] - metrics(
        tuned, "2025"
    )["recall_at_10"]
    return {
        "accepted": validation_gain >= 0.02 and test_regression <= 0.01,
        "validation_recall_at_10_gain": round(validation_gain, 6),
        "test_recall_at_10_regression": round(test_regression, 6),
        "rules": {"minimum_validation_gain": 0.02, "maximum_test_regression": 0.01},
    }


def quantized_acceptance(
    baseline: dict[str, Any],
    quantized: dict[str, Any],
    *,
    model_name: str,
    variant: str,
) -> dict[str, Any]:
    def metrics(source: dict[str, Any], year: str, name: str | None = None) -> dict[str, Any]:
        models = source["models"]
        model = models[name] if name is not None else next(iter(models.values()))
        value = model["variants"][variant]
        return value["years"][year] if variant == "hybrid" else value[year]

    drops = {
        year: round(
            metrics(baseline, year, model_name)["recall_at_10"]
            - metrics(quantized, year)["recall_at_10"],
            6,
        )
        for year in ("2024", "2025")
    }
    return {
        "accepted": max(drops.values()) <= 0.01,
        "recall_at_10_drops": drops,
        "maximum_allowed_drop": 0.01,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path("config/models.json"))
    parser.add_argument("--output", type=Path, default=Path("data/matcher/selection.json"))
    parser.add_argument("--tuned-benchmark", type=Path)
    parser.add_argument("--quantized-benchmark", type=Path)
    parser.add_argument("--variant", default="dense_with_divisions")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    baseline = json.loads(args.benchmark.read_text(encoding="utf-8"))
    result = select_baseline(baseline)
    current_models = json.loads(args.models.read_text(encoding="utf-8"))["models"]
    current = next(
        (item for item in current_models if item["name"] == result["selected"]["model"]),
        None,
    )
    if current is None:
        raise ValueError(f"selected model is absent from {args.models}")
    result["model_config"] = current
    if args.tuned_benchmark:
        tuned = json.loads(args.tuned_benchmark.read_text(encoding="utf-8"))
        result["tuned_acceptance"] = tuned_acceptance(
            baseline, tuned, variant=args.variant
        )
    if args.quantized_benchmark:
        quantized = json.loads(args.quantized_benchmark.read_text(encoding="utf-8"))
        result["quantized_acceptance"] = quantized_acceptance(
            baseline,
            quantized,
            model_name=result["selected"]["model"],
            variant=args.variant,
        )
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
