#!/usr/bin/env python3
"""Persist complete same-year score matrices for a learned multi-vector model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

from matcher_common import iter_jsonl, write_json
from train_multivector_ranker import (
    FEATURE_NAMES,
    MultiVectorWeights,
    batch_features,
    load_arrays,
)


def ordered_weights(path: Path) -> np.ndarray:
    values = json.loads(path.read_text(encoding="utf-8"))
    weights = np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float32)
    if np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0, atol=1e-5):
        raise ValueError("browser weights must be positive and sum to one")
    return weights


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    sessions = list(iter_jsonl(args.prepared_dir / "sessions.jsonl.gz"))
    abstracts = list(iter_jsonl(args.prepared_dir / "abstracts.jsonl.gz"))
    array_args = SimpleNamespace(
        baseline_embedding_dir=args.baseline_embedding_dir,
        multivector_dir=args.multivector_dir,
        metric_head=args.metric_head,
        abstract_count=len(abstracts),
    )
    arrays = load_arrays(array_args)
    weights = ordered_weights(args.weights)
    model = MultiVectorWeights.create(weights).to(args.device)
    with torch.no_grad():
        model.raw_weight.copy_(torch.log(torch.as_tensor(weights, device=args.device)))
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "score": "learned_metric_plus_multivector",
        "dtype": "float32",
        "features": list(FEATURE_NAMES),
        "weights": {name: float(value) for name, value in zip(FEATURE_NAMES, weights)},
        "years": {},
    }
    for year in args.years:
        session_indices = np.asarray(
            [index for index, row in enumerate(sessions) if row["event_year"] == year],
            dtype=np.int64,
        )
        abstract_indices = np.asarray(
            [index for index, row in enumerate(abstracts) if row["event_year"] == year],
            dtype=np.int64,
        )
        matrix_path = args.output_dir / f"scores-{year}.npy"
        matrix = np.lib.format.open_memmap(
            matrix_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(abstract_indices), len(session_indices)),
        )
        with torch.inference_mode():
            for start in range(0, len(abstract_indices), args.block_size):
                block = abstract_indices[start : start + args.block_size]
                features = batch_features(arrays, block, session_indices, args.device)
                matrix[start : start + len(block)] = model(features).cpu().numpy()
        matrix.flush()
        del matrix
        manifest["years"][str(year)] = {
            "matrix": matrix_path.name,
            "shape": [len(abstract_indices), len(session_indices)],
            "abstract_keys": [abstracts[index]["abstract_key"] for index in abstract_indices],
            "session_keys": [sessions[index]["session_key"] for index in session_indices],
        }
        print(f"{year}: wrote {len(abstract_indices)} x {len(session_indices)} scores", flush=True)
    write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--baseline-embedding-dir", type=Path, required=True)
    parser.add_argument("--multivector-dir", type=Path, required=True)
    parser.add_argument("--metric-head", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+", default=[2024, 2025, 2026])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--block-size", type=int, default=128)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
