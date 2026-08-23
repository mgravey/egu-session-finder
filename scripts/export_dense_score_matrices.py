#!/usr/bin/env python3
"""Persist complete same-year abstract-by-session cosine matrices."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from export_browser_catalogue import (
    load_metric_head,
    transform_query_vectors,
    transform_session_vectors,
)
from matcher_common import iter_jsonl, write_json


def export_score_matrices(
    *,
    sessions: list[dict[str, Any]],
    abstracts: list[dict[str, Any]],
    session_vectors: np.ndarray,
    abstract_vectors: np.ndarray,
    years: Sequence[int],
    output_dir: Path,
    block_size: int = 512,
    metric: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one full float32 matrix per year and return its row/column manifest."""
    if len(sessions) != len(session_vectors):
        raise ValueError("session rows and vectors differ")
    if len(abstracts) != len(abstract_vectors):
        raise ValueError("abstract rows and vectors differ")
    if block_size < 1:
        raise ValueError("block_size must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "score": "learned_cosine" if metric is not None else "dense_cosine",
        "dtype": "float32",
        "layout": "one same-event-year matrix; rows are abstracts and columns are sessions",
        "years": {},
    }
    for year in years:
        session_indices = [
            index for index, session in enumerate(sessions) if session["event_year"] == year
        ]
        abstract_indices = [
            index for index, abstract in enumerate(abstracts) if abstract["event_year"] == year
        ]
        matrix_path = output_dir / f"cosine-{year}.npy"
        matrix = np.lib.format.open_memmap(
            matrix_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(abstract_indices), len(session_indices)),
        )
        candidate_vectors = np.asarray(session_vectors[session_indices], dtype=np.float32)
        if metric is not None:
            candidate_vectors = transform_session_vectors(candidate_vectors, metric)
        for start in range(0, len(abstract_indices), block_size):
            block = abstract_indices[start : start + block_size]
            query_vectors = np.asarray(abstract_vectors[block], dtype=np.float32)
            if metric is not None:
                query_vectors = transform_query_vectors(query_vectors, metric)
            matrix[start : start + len(block)] = (
                query_vectors @ candidate_vectors.T
            )
        matrix.flush()
        del matrix

        manifest["years"][str(year)] = {
            "matrix": matrix_path.name,
            "shape": [len(abstract_indices), len(session_indices)],
            "abstract_keys": [abstracts[index]["abstract_key"] for index in abstract_indices],
            "session_keys": [sessions[index]["session_key"] for index in session_indices],
        }
        size_mb = matrix_path.stat().st_size / 1024**2
        print(
            f"{year}: wrote {len(abstract_indices)} x {len(session_indices)} "
            f"cosine scores ({size_mb:.1f} MiB)",
            flush=True,
        )
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-dir", type=Path, default=Path("data/matcher/prepared")
    )
    parser.add_argument(
        "--embedding-dir", type=Path, required=True, help="Model embedding cache directory"
    )
    parser.add_argument(
        "--abstract-vectors", default="abstracts_121971.npy", help="Filename in embedding-dir"
    )
    parser.add_argument(
        "--session-vectors",
        default="sessions_with_divisions.npy",
        help="Filename in embedding-dir",
    )
    parser.add_argument(
        "--years", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024, 2025, 2026]
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/matcher/evaluation/minilm-dense-matrices"),
    )
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument(
        "--metric-head",
        type=Path,
        help="Optional metric-head.npz for learned query/session similarity",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sessions = list(iter_jsonl(args.prepared_dir / "sessions.jsonl.gz"))
    abstracts = list(iter_jsonl(args.prepared_dir / "abstracts.jsonl.gz"))
    export_score_matrices(
        sessions=sessions,
        abstracts=abstracts,
        session_vectors=np.load(
            args.embedding_dir / args.session_vectors, mmap_mode="r"
        ),
        abstract_vectors=np.load(
            args.embedding_dir / args.abstract_vectors, mmap_mode="r"
        ),
        years=args.years,
        output_dir=args.output_dir,
        block_size=args.block_size,
        metric=load_metric_head(args.metric_head) if args.metric_head else None,
    )
    print(f"Wrote {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
