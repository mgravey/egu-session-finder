#!/usr/bin/env python3
"""Learn a compact listwise scorer over frozen multi-vector EGU features."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

import numpy as np

from export_browser_catalogue import (
    load_metric_head,
    transform_query_vectors,
    transform_session_vectors,
)
from matcher_common import iter_jsonl, write_json
from matching_metrics import retrieval_metrics
from train_embedding_metric import build_year_training_data


FEATURE_NAMES = (
    "joint",
    "title_to_title",
    "title_to_description",
    "chunks_to_title",
    "chunks_to_description",
)


def masked_max(values: Any, mask: Any, *, dim: int, fallback: Any) -> Any:
    import torch

    masked = values.masked_fill(~mask, -torch.inf)
    maximum = masked.amax(dim=dim)
    available = mask.any(dim=dim)
    return torch.where(available, maximum, fallback)


def score_features(
    *,
    query_joint: Any,
    query_title: Any,
    query_chunks: Any,
    query_chunk_mask: Any,
    session_joint: Any,
    session_title: Any,
    session_description: Any,
    session_description_mask: Any,
) -> Any:
    """Return [queries, sessions, features], falling back to title when needed."""
    import torch

    joint = query_joint @ session_joint.T
    title_title = query_title @ session_title.T
    title_description_all = torch.einsum("bd,cld->bcl", query_title, session_description)
    description_mask = session_description_mask[None, :, :].expand_as(title_description_all)
    title_description = masked_max(
        title_description_all,
        description_mask,
        dim=2,
        fallback=title_title,
    )
    chunk_title_all = torch.einsum("bkd,cd->bck", query_chunks, session_title)
    chunk_mask = query_chunk_mask[:, None, :].expand_as(chunk_title_all)
    chunk_title = masked_max(chunk_title_all, chunk_mask, dim=2, fallback=title_title)
    chunk_description_all = torch.einsum(
        "bkd,cld->bckl", query_chunks, session_description
    )
    pair_mask = (
        query_chunk_mask[:, None, :, None]
        & session_description_mask[None, :, None, :]
    ).expand_as(chunk_description_all)
    chunk_description = masked_max(
        chunk_description_all.flatten(2),
        pair_mask.flatten(2),
        dim=2,
        fallback=chunk_title,
    )
    return torch.stack(
        (joint, title_title, title_description, chunk_title, chunk_description),
        dim=-1,
    )


class MultiVectorWeights:
    @staticmethod
    def create(initial: Sequence[float]) -> Any:
        import torch
        from torch import nn

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                values = torch.as_tensor(initial, dtype=torch.float32)
                self.raw_weight = nn.Parameter(torch.log(values))

            def weights(self) -> Any:
                return torch.softmax(self.raw_weight, dim=0)

            def forward(self, features: Any) -> Any:
                return features @ self.weights()

        return Model()


def load_arrays(args: argparse.Namespace) -> dict[str, np.ndarray]:
    arrays = {
        "query_joint": np.load(
            args.baseline_embedding_dir / f"abstracts_{args.abstract_count}.npy",
            mmap_mode="r",
        ),
        "session_joint": np.load(
            args.baseline_embedding_dir / "sessions_with_divisions.npy", mmap_mode="r"
        ),
        "query_title": np.load(args.multivector_dir / "query_title.npy", mmap_mode="r")[:, 0],
        "query_chunks": np.load(args.multivector_dir / "query_chunks.npy", mmap_mode="r"),
        "query_chunk_mask": np.load(
            args.multivector_dir / "query_chunks_mask.npy", mmap_mode="r"
        ),
        "session_title": np.load(
            args.multivector_dir / "session_title.npy", mmap_mode="r"
        )[:, 0],
        "session_description": np.load(
            args.multivector_dir / "session_description.npy", mmap_mode="r"
        ),
        "session_description_mask": np.load(
            args.multivector_dir / "session_description_mask.npy", mmap_mode="r"
        ),
    }
    if args.metric_head:
        metric = load_metric_head(args.metric_head)
        for name in ("query_joint", "query_title", "query_chunks"):
            shape = arrays[name].shape
            arrays[name] = transform_query_vectors(
                np.asarray(arrays[name]).reshape(-1, shape[-1]), metric
            ).reshape(shape)
        for name in ("session_joint", "session_title", "session_description"):
            shape = arrays[name].shape
            arrays[name] = transform_session_vectors(
                np.asarray(arrays[name]).reshape(-1, shape[-1]), metric
            ).reshape(shape)
    return arrays


def tensor_rows(values: np.ndarray, indices: np.ndarray, device: str) -> Any:
    import torch

    return torch.as_tensor(np.asarray(values[indices]), device=device)


def batch_features(
    arrays: dict[str, np.ndarray],
    query_indices: np.ndarray,
    session_indices: np.ndarray,
    device: str,
) -> Any:
    return score_features(
        query_joint=tensor_rows(arrays["query_joint"], query_indices, device),
        query_title=tensor_rows(arrays["query_title"], query_indices, device),
        query_chunks=tensor_rows(arrays["query_chunks"], query_indices, device),
        query_chunk_mask=tensor_rows(arrays["query_chunk_mask"], query_indices, device).bool(),
        session_joint=tensor_rows(arrays["session_joint"], session_indices, device),
        session_title=tensor_rows(arrays["session_title"], session_indices, device),
        session_description=tensor_rows(
            arrays["session_description"], session_indices, device
        ),
        session_description_mask=tensor_rows(
            arrays["session_description_mask"], session_indices, device
        ).bool(),
    )


def evaluate(
    *,
    model: Any,
    data: Any,
    arrays: dict[str, np.ndarray],
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    import torch

    ranks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(data.abstract_indices), batch_size):
            features = batch_features(
                arrays,
                data.abstract_indices[start : start + batch_size],
                data.session_indices,
                device,
            )
            scores = model(features).cpu().numpy()
            masks = data.positive_mask[start : start + batch_size]
            for score_row, mask in zip(scores, masks, strict=True):
                relevant = np.flatnonzero(mask)
                best_positive = float(np.max(score_row[relevant]))
                ranks.append(1 + int(np.sum(score_row > best_positive)))
    return retrieval_metrics(ranks)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sessions = list(iter_jsonl(args.prepared_dir / "sessions.jsonl.gz"))
    abstracts = list(iter_jsonl(args.prepared_dir / "abstracts.jsonl.gz"))
    args.abstract_count = len(abstracts)
    arrays = load_arrays(args)
    years = {
        year: build_year_training_data(
            sessions=sessions, abstracts=abstracts, year=year
        )
        for year in (2022, 2023, 2024, 2025, 2026)
    }
    model = MultiVectorWeights.create((0.60, 0.10, 0.10, 0.10, 0.10)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_state = None
    best_key = (-1.0, -1.0)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for year in (2022, 2023):
            data = years[year]
            order = np.random.permutation(len(data.abstract_indices))
            for start in range(0, len(order), args.batch_size):
                selected = order[start : start + args.batch_size]
                features = batch_features(
                    arrays,
                    data.abstract_indices[selected],
                    data.session_indices,
                    args.device,
                )
                logits = model(features) / args.temperature
                mask = torch.as_tensor(data.positive_mask[selected], device=args.device)
                denominator = torch.logsumexp(logits, dim=1)
                numerator = torch.logsumexp(logits.masked_fill(~mask, -torch.inf), dim=1)
                loss = (denominator - numerator).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
        validation = evaluate(
            model=model,
            data=years[2024],
            arrays=arrays,
            device=args.device,
            batch_size=args.eval_batch_size,
        )
        record = {
            "epoch": epoch,
            "loss": round(float(np.mean(losses)), 6),
            "weights": {
                name: round(float(value), 6)
                for name, value in zip(
                    FEATURE_NAMES,
                    model.weights().detach().cpu().numpy(),
                    strict=True,
                )
            },
            "validation": validation,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        key = (validation["recall_at_10"], validation["mrr"])
        if key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)

    learned = {
        str(year): evaluate(
            model=model,
            data=years[year],
            arrays=arrays,
            device=args.device,
            batch_size=args.eval_batch_size,
        )
        for year in (2024, 2025, 2026)
    }
    baseline_model = MultiVectorWeights.create((1.0, 1e-8, 1e-8, 1e-8, 1e-8)).to(
        args.device
    )
    baseline = {
        str(year): evaluate(
            model=baseline_model,
            data=years[year],
            arrays=arrays,
            device=args.device,
            batch_size=args.eval_batch_size,
        )
        for year in (2024, 2025, 2026)
    }
    weights = model.weights().detach().cpu().numpy()
    report = {
        "features": list(FEATURE_NAMES),
        "training_years": [2022, 2023],
        "validation_year": 2024,
        "sealed_test_year": 2025,
        "forward_year": 2026,
        "input_metric_head": str(args.metric_head) if args.metric_head else None,
        "selected_weights": {
            name: round(float(value), 8)
            for name, value in zip(FEATURE_NAMES, weights, strict=True)
        },
        "history": history,
        "baseline": baseline,
        "learned": learned,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "report.json", report)
    write_json(args.output_dir / "browser-weights.json", report["selected_weights"])
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--baseline-embedding-dir", type=Path, required=True)
    parser.add_argument("--multivector-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--metric-head",
        type=Path,
        help="Optionally transform every query/session field with an existing learned head",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
