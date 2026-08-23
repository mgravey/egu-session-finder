#!/usr/bin/env python3
"""Export the rank distribution of historically assigned EGU sessions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from matcher_common import iter_jsonl, write_json
from session_categories import is_typical_scientific_session, session_code_prefix


def rank_relevant_candidates(
    scores: np.ndarray,
    relevant_indices: Sequence[int],
) -> int | None:
    """Return the stable 1-based rank of the first relevant candidate."""
    if len(relevant_indices) == 0:
        return None
    relevant = np.asarray(relevant_indices, dtype=np.int64)
    relevant_scores = scores[relevant]
    best_score = float(relevant_scores.max())
    best_relevant_index = int(relevant[relevant_scores == best_score].min())
    better = int(np.count_nonzero(scores > best_score))
    tied_before = int(
        np.count_nonzero(
            (scores[:best_relevant_index] == best_score)
        )
    )
    return 1 + better + tied_before


def dense_rank_distribution(
    *,
    sessions: list[dict[str, Any]],
    abstracts: list[dict[str, Any]],
    session_vectors: np.ndarray,
    abstract_vectors: np.ndarray,
    years: Sequence[int],
    block_size: int = 512,
) -> dict[str, Any]:
    if len(sessions) != len(session_vectors):
        raise ValueError("session rows and vectors differ")
    if len(abstracts) != len(abstract_vectors):
        raise ValueError("abstract rows and vectors differ")

    result: dict[str, Any] = {"score": "dense_cosine", "years": {}}
    for year in years:
        session_indices = [
            index for index, session in enumerate(sessions) if session["event_year"] == year
        ]
        abstract_indices = [
            index
            for index, abstract in enumerate(abstracts)
            if abstract["event_year"] == year
            and abstract["has_abstract"]
            and abstract["session_keys"]
        ]
        candidate_lookup = {
            sessions[index]["session_key"]: position
            for position, index in enumerate(session_indices)
        }
        candidate_vectors = np.asarray(session_vectors[session_indices], dtype=np.float32)
        ranks: list[int | None] = []

        for start in range(0, len(abstract_indices), block_size):
            block = abstract_indices[start : start + block_size]
            score_rows = np.asarray(abstract_vectors[block], dtype=np.float32) @ candidate_vectors.T
            for abstract_index, scores in zip(block, score_rows, strict=True):
                relevant = [
                    candidate_lookup[key]
                    for key in abstracts[abstract_index]["session_keys"]
                    if key in candidate_lookup
                ]
                ranks.append(rank_relevant_candidates(scores, relevant))

        valid_ranks = [rank for rank in ranks if rank is not None]
        exact = Counter(valid_ranks)
        cumulative = 0
        curve = []
        for rank in range(1, len(session_indices) + 1):
            cumulative += exact.get(rank, 0)
            curve.append(
                {
                    "rank": rank,
                    "exact_count": exact.get(rank, 0),
                    "cumulative_count": cumulative,
                    "cumulative_fraction": round(cumulative / len(ranks), 8) if ranks else 0.0,
                }
            )
        result["years"][str(year)] = {
            "queries": len(ranks),
            "candidate_sessions": len(session_indices),
            "queries_with_relevant_candidate": len(valid_ranks),
            "curve": curve,
        }
        print(
            f"{year}: ranked {len(ranks)} abstracts against {len(session_indices)} sessions",
            flush=True,
        )
    return result


def rank_distribution_from_matrices(
    *,
    abstracts: list[dict[str, Any]],
    matrix_dir: Path,
    years: Sequence[int],
    sessions: list[dict[str, Any]] | None = None,
    typical_sessions_only: bool = False,
) -> dict[str, Any]:
    """Calculate cumulative ranks from persisted same-year score matrices."""
    manifest = json.loads((matrix_dir / "manifest.json").read_text(encoding="utf-8"))
    abstract_lookup = {item["abstract_key"]: item for item in abstracts}
    session_lookup = {item["session_key"]: item for item in sessions or []}
    if typical_sessions_only and not sessions:
        raise ValueError("sessions are required for the typical-session filter")
    result: dict[str, Any] = {"score": manifest["score"], "years": {}}
    for year in years:
        year_manifest = manifest["years"][str(year)]
        matrix = np.load(matrix_dir / year_manifest["matrix"], mmap_mode="r")
        all_candidate_keys = year_manifest["session_keys"]
        if typical_sessions_only:
            kept_columns = np.asarray(
                [
                    index
                    for index, key in enumerate(all_candidate_keys)
                    if is_typical_scientific_session(session_lookup[key])
                ],
                dtype=np.int64,
            )
            candidate_keys = [all_candidate_keys[index] for index in kept_columns]
        else:
            kept_columns = np.arange(len(all_candidate_keys), dtype=np.int64)
            candidate_keys = all_candidate_keys
        candidate_lookup = {key: index for index, key in enumerate(candidate_keys)}
        ranks: list[int | None] = []
        excluded_abstracts = 0
        excluded_by_prefix: Counter[str] = Counter()
        for row_index, abstract_key in enumerate(year_manifest["abstract_keys"]):
            abstract = abstract_lookup[abstract_key]
            original_labels = abstract["session_keys"]
            relevant_labels = [key for key in original_labels if key in candidate_lookup]
            if typical_sessions_only and original_labels and not relevant_labels:
                excluded_abstracts += 1
                excluded_by_prefix.update(
                    session_code_prefix(session_lookup[key].get("session_code"))
                    for key in original_labels
                    if key in session_lookup
                )
                continue
            if not abstract["has_abstract"] or not relevant_labels:
                continue
            relevant = [
                candidate_lookup[key]
                for key in relevant_labels
            ]
            scores = np.asarray(matrix[row_index])[kept_columns]
            ranks.append(rank_relevant_candidates(scores, relevant))

        valid_ranks = [rank for rank in ranks if rank is not None]
        total_rows = len(year_manifest["abstract_keys"]) - excluded_abstracts
        exact = Counter(valid_ranks)
        cumulative = 0
        curve = []
        for rank in range(1, len(candidate_keys) + 1):
            cumulative += exact.get(rank, 0)
            curve.append(
                {
                    "rank": rank,
                    "exact_count": exact.get(rank, 0),
                    "cumulative_count": cumulative,
                    "cumulative_fraction": round(cumulative / total_rows, 8) if total_rows else 0.0,
                }
            )
        result["years"][str(year)] = {
            "queries": len(ranks),
            "all_abstract_rows": len(year_manifest["abstract_keys"]),
            "denominator_rows": total_rows,
            "candidate_sessions": len(candidate_keys),
            "excluded_nonstandard_sessions": len(all_candidate_keys) - len(candidate_keys),
            "excluded_nonstandard_abstracts": excluded_abstracts,
            "excluded_abstracts_by_prefix": dict(sorted(excluded_by_prefix.items())),
            "queries_with_relevant_candidate": len(valid_ranks),
            "curve": curve,
        }
        print(f"{year}: derived ranks for {len(ranks)} labeled abstracts", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-dir", type=Path, default=Path("data/matcher/prepared")
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--embedding-dir", type=Path, help="Model embedding cache directory"
    )
    parser.add_argument(
        "--typical-sessions-only",
        action="store_true",
        help="Exclude non-standard programme categories from candidates and statistics",
    )
    source.add_argument(
        "--matrix-dir", type=Path, help="Directory made by export_dense_score_matrices.py"
    )
    parser.add_argument(
        "--abstract-vectors", default="abstracts_121971.npy", help="Filename in embedding-dir"
    )
    parser.add_argument(
        "--session-vectors",
        default="sessions_with_divisions.npy",
        help="Filename in embedding-dir",
    )
    parser.add_argument("--years", type=int, nargs="+", default=[2024, 2025, 2026])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/matcher/evaluation/dense-rank-distribution.json"),
    )
    parser.add_argument("--block-size", type=int, default=512)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    abstracts = list(iter_jsonl(args.prepared_dir / "abstracts.jsonl.gz"))
    if args.matrix_dir:
        sessions = list(iter_jsonl(args.prepared_dir / "sessions.jsonl.gz"))
        report = rank_distribution_from_matrices(
            abstracts=abstracts,
            matrix_dir=args.matrix_dir,
            years=args.years,
            sessions=sessions,
            typical_sessions_only=args.typical_sessions_only,
        )
    else:
        sessions = list(iter_jsonl(args.prepared_dir / "sessions.jsonl.gz"))
        report = dense_rank_distribution(
            sessions=sessions,
            abstracts=abstracts,
            session_vectors=np.load(
                args.embedding_dir / args.session_vectors, mmap_mode="r"
            ),
            abstract_vectors=np.load(
                args.embedding_dir / args.abstract_vectors, mmap_mode="r"
            ),
            years=args.years,
            block_size=args.block_size,
        )
    write_json(args.output, report)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
