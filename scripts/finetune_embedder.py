#!/usr/bin/env python3
"""Fine-tune one selected sentence encoder with EGU hard-negative triplets."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np

from matcher_common import iter_jsonl, write_json


def select_hard_negative(
    *,
    query_vector: np.ndarray,
    sessions: list[dict[str, Any]],
    session_vectors: np.ndarray,
    valid_session_keys: set[str],
    year: int,
    positive_divisions: set[str],
) -> int:
    candidates = [
        index
        for index, session in enumerate(sessions)
        if session["event_year"] == year
        and session["session_key"] not in valid_session_keys
        and (
            not positive_divisions
            or positive_divisions.intersection(session.get("division_codes", []))
        )
    ]
    if not candidates:
        candidates = [
            index
            for index, session in enumerate(sessions)
            if session["event_year"] == year and session["session_key"] not in valid_session_keys
        ]
    if not candidates:
        raise ValueError(f"no negative session available for year {year}")
    scores = np.asarray(session_vectors[candidates]) @ np.asarray(query_vector)
    return candidates[int(np.argmax(scores))]


def build_triplets(
    *,
    sessions: list[dict[str, Any]],
    abstracts: list[dict[str, Any]],
    session_vectors: np.ndarray,
    abstract_vectors: np.ndarray,
    max_examples: int,
    seed: int,
    query_prefix: str = "",
) -> list[tuple[str, str, str]]:
    session_lookup = {item["session_key"]: item for item in sessions}
    rows = []
    for abstract_index, abstract in enumerate(abstracts):
        if abstract["split"] != "train" or not abstract["has_abstract"] or not abstract["session_keys"]:
            continue
        positive_keys = set(abstract["session_keys"])
        divisions = {
            code
            for key in positive_keys
            for code in session_lookup[key].get("division_codes", [])
        }
        negative_index = select_hard_negative(
            query_vector=abstract_vectors[abstract_index],
            sessions=sessions,
            session_vectors=session_vectors,
            valid_session_keys=positive_keys,
            year=abstract["event_year"],
            positive_divisions=divisions,
        )
        negative_text = sessions[negative_index]["session_text"]
        for positive_key in sorted(positive_keys):
            rows.append(
                (
                    query_prefix + abstract["query_text"],
                    session_lookup[positive_key]["session_text"],
                    negative_text,
                )
            )
    random.Random(seed).shuffle(rows)
    return rows[:max_examples] if max_examples else rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("Install requirements-ml.txt in a Python 3.12 environment") from exc

    sessions = list(iter_jsonl(args.prepared_dir / "sessions.jsonl.gz"))
    abstracts = list(iter_jsonl(args.prepared_dir / "abstracts.jsonl.gz"))
    session_vectors = np.load(args.embedding_dir / "sessions_with_divisions.npy", mmap_mode="r")
    abstract_vectors = np.load(
        args.embedding_dir / f"abstracts_{len(abstracts)}.npy", mmap_mode="r"
    )
    triplets = build_triplets(
        sessions=sessions,
        abstracts=abstracts,
        session_vectors=session_vectors,
        abstract_vectors=abstract_vectors,
        max_examples=args.max_examples,
        seed=args.seed,
        query_prefix=args.query_prefix,
    )
    if not triplets:
        raise ValueError("no training triplets were produced")
    examples = [InputExample(texts=list(row)) for row in triplets]
    generator = None
    try:
        import torch

        generator = torch.Generator().manual_seed(args.seed)
    except ImportError:
        pass
    loader = DataLoader(
        examples,
        shuffle=True,
        batch_size=args.batch_size,
        generator=generator,
    )
    model = SentenceTransformer(args.base_model, device=args.device)
    loss = losses.TripletLoss(model=model, triplet_margin=args.margin)
    warmup_steps = max(1, int(len(loader) * args.epochs * 0.1))
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=str(args.output_dir),
        show_progress_bar=True,
    )
    manifest = {
        "base_model": args.base_model,
        "training_examples": len(triplets),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "margin": args.margin,
        "seed": args.seed,
        "device": args.device,
        "query_prefix": args.query_prefix,
        "selection_rule": {
            "minimum_validation_recall_at_10_gain": 0.02,
            "maximum_test_recall_at_10_regression": 0.01,
        },
    }
    write_json(args.output_dir / "egu_training_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/matcher/prepared"))
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--epochs", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--query-prefix", default="")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
