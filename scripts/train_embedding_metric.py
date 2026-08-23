#!/usr/bin/env python3
"""Learn a listwise ranking metric on cached EGU abstract/session embeddings."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

import numpy as np

from export_dense_rank_distribution import rank_relevant_candidates
from matcher_common import iter_jsonl, write_json
from matching_metrics import confidence_features, fit_confidence_bands, retrieval_metrics


@dataclass
class YearTrainingData:
    year: int
    abstract_indices: np.ndarray
    session_indices: np.ndarray
    positive_mask: np.ndarray


def build_year_training_data(
    *,
    sessions: Sequence[dict[str, Any]],
    abstracts: Sequence[dict[str, Any]],
    year: int,
) -> YearTrainingData:
    session_indices = np.asarray(
        [index for index, row in enumerate(sessions) if row["event_year"] == year],
        dtype=np.int64,
    )
    candidate_lookup = {
        sessions[index]["session_key"]: position
        for position, index in enumerate(session_indices)
    }
    abstract_indices = []
    positives = []
    for index, row in enumerate(abstracts):
        if row["event_year"] != year or not row["has_abstract"] or not row["session_keys"]:
            continue
        relevant = [candidate_lookup[key] for key in row["session_keys"] if key in candidate_lookup]
        if relevant:
            abstract_indices.append(index)
            positives.append(relevant)
    positive_mask = np.zeros((len(abstract_indices), len(session_indices)), dtype=np.bool_)
    for row_index, relevant in enumerate(positives):
        positive_mask[row_index, relevant] = True
    return YearTrainingData(
        year=year,
        abstract_indices=np.asarray(abstract_indices, dtype=np.int64),
        session_indices=session_indices,
        positive_mask=positive_mask,
    )


def multi_positive_listwise_loss(logits: Any, positive_mask: Any) -> Any:
    """Negative log probability assigned to any valid session label."""
    import torch

    if logits.shape != positive_mask.shape:
        raise ValueError("logits and positive mask must have the same shape")
    if not bool(torch.all(positive_mask.any(dim=1))):
        raise ValueError("every query must have at least one positive session")
    denominator = torch.logsumexp(logits, dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~positive_mask, -torch.inf), dim=1)
    return (denominator - numerator).mean()


def make_metric_head(kind: str, dimension: int, rank: int) -> Any:
    import torch
    from torch import nn
    from torch.nn import functional as F

    class DiagonalMetric(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.raw_log_weight = nn.Parameter(torch.zeros(dimension))

        def bounded_log_weight(self) -> Any:
            return 3.0 * torch.tanh(self.raw_log_weight / 3.0)

        def _transform(self, values: Any) -> Any:
            scale = torch.exp(0.5 * self.bounded_log_weight())
            return F.normalize(values * scale, p=2, dim=-1)

        def transform_query(self, values: Any) -> Any:
            return self._transform(values)

        def transform_session(self, values: Any) -> Any:
            return self._transform(values)

        def identity_regularization(self) -> Any:
            return self.bounded_log_weight().square().mean()

        def export_arrays(self) -> dict[str, np.ndarray]:
            return {
                "log_weight": self.bounded_log_weight().detach().cpu().numpy(),
            }

    class LowRankMetric(DiagonalMetric):
        def __init__(self) -> None:
            super().__init__()
            generator = torch.Generator().manual_seed(20260823)
            self.query_down = nn.Parameter(
                torch.randn(dimension, rank, generator=generator) * 0.02
            )
            self.query_up = nn.Parameter(torch.zeros(rank, dimension))
            self.session_down = nn.Parameter(
                torch.randn(dimension, rank, generator=generator) * 0.02
            )
            self.session_up = nn.Parameter(torch.zeros(rank, dimension))

        def _base(self, values: Any) -> Any:
            return values * torch.exp(0.5 * self.bounded_log_weight())

        def transform_query(self, values: Any) -> Any:
            base = self._base(values)
            projected = base + (base @ self.query_down) @ self.query_up
            return F.normalize(projected, p=2, dim=-1)

        def transform_session(self, values: Any) -> Any:
            base = self._base(values)
            projected = base + (base @ self.session_down) @ self.session_up
            return F.normalize(projected, p=2, dim=-1)

        def identity_regularization(self) -> Any:
            return (
                super().identity_regularization()
                + self.query_up.square().mean()
                + self.session_up.square().mean()
            )

        def export_arrays(self) -> dict[str, np.ndarray]:
            return {
                **super().export_arrays(),
                "query_down": self.query_down.detach().cpu().numpy(),
                "query_up": self.query_up.detach().cpu().numpy(),
                "session_down": self.session_down.detach().cpu().numpy(),
                "session_up": self.session_up.detach().cpu().numpy(),
            }

    if kind == "diagonal":
        return DiagonalMetric()
    if kind == "lowrank":
        return LowRankMetric()
    raise ValueError(f"unknown metric kind: {kind}")


def transform_numpy(
    head: Any,
    values: np.ndarray,
    *,
    query: bool,
    device: str,
    block_size: int = 4096,
) -> np.ndarray:
    import torch

    transform = head.transform_query if query else head.transform_session
    rows = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(values), block_size):
            tensor = torch.as_tensor(
                np.asarray(values[start : start + block_size], dtype=np.float32),
                device=device,
            )
            rows.append(transform(tensor).cpu().numpy())
    return np.concatenate(rows) if rows else np.empty_like(values)


def evaluate_metric(
    *,
    head: Any,
    year_data: YearTrainingData,
    abstract_vectors: np.ndarray,
    session_vectors: np.ndarray,
    total_abstract_rows: int,
    device: str,
    block_size: int = 512,
) -> tuple[dict[str, Any], list[int | None]]:
    queries = transform_numpy(
        head,
        np.asarray(abstract_vectors[year_data.abstract_indices], dtype=np.float32),
        query=True,
        device=device,
    )
    candidates = transform_numpy(
        head,
        np.asarray(session_vectors[year_data.session_indices], dtype=np.float32),
        query=False,
        device=device,
    )
    ranks: list[int | None] = []
    for start in range(0, len(queries), block_size):
        scores = queries[start : start + block_size] @ candidates.T
        masks = year_data.positive_mask[start : start + block_size]
        for score_row, mask in zip(scores, masks, strict=True):
            ranks.append(rank_relevant_candidates(score_row, np.flatnonzero(mask)))
    metrics = retrieval_metrics(ranks)
    valid = [rank_value for rank_value in ranks if rank_value is not None]
    metrics["mean_rank"] = round(float(np.mean(valid)), 4) if valid else None
    metrics["median_rank"] = float(np.median(valid)) if valid else None
    metrics["all_abstract_rows"] = total_abstract_rows
    for cutoff in (1, 3, 5, 10, 20, 50, 100):
        metrics[f"all_rows_recall_at_{cutoff}"] = (
            round(sum(rank_value <= cutoff for rank_value in valid) / total_abstract_rows, 6)
            if total_abstract_rows
            else 0.0
        )
    for target in (0.50, 0.75, 0.90, 0.95):
        threshold = next(
            (
                cutoff
                for cutoff in range(1, len(year_data.session_indices) + 1)
                if sum(rank_value <= cutoff for rank_value in valid) / total_abstract_rows
                >= target
            ),
            None,
        )
        metrics[f"rank_for_{int(target * 100)}_percent_all_rows"] = threshold
    return metrics, ranks


def validation_confidence_bands(
    *,
    head: Any,
    year_data: YearTrainingData,
    abstract_vectors: np.ndarray,
    session_vectors: np.ndarray,
    device: str,
    block_size: int = 512,
) -> dict[str, float]:
    """Calibrate empirical confidence features after the ranking metric is fixed."""
    queries = transform_numpy(
        head,
        np.asarray(abstract_vectors[year_data.abstract_indices], dtype=np.float32),
        query=True,
        device=device,
    )
    candidates = transform_numpy(
        head,
        np.asarray(session_vectors[year_data.session_indices], dtype=np.float32),
        query=False,
        device=device,
    )
    observations = []
    for start in range(0, len(queries), block_size):
        score_rows = queries[start : start + block_size] @ candidates.T
        masks = year_data.positive_mask[start : start + block_size]
        for scores, mask in zip(score_rows, masks, strict=True):
            top, margin = confidence_features(scores)
            observations.append((top, margin, bool(mask[int(np.argmax(scores))])))
    return fit_confidence_bands(observations)


def train_candidate(
    *,
    kind: str,
    dimension: int,
    rank: int,
    train_data: Sequence[YearTrainingData],
    validation_data: YearTrainingData,
    sessions: Sequence[dict[str, Any]],
    abstracts: Sequence[dict[str, Any]],
    session_vectors: np.ndarray,
    abstract_vectors: np.ndarray,
    output_dir: Path,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    temperature: float,
    regularization: float,
    patience: int,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    head = make_metric_head(kind, dimension, rank).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-5)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    history = []
    best_key = (-1.0, -1.0)
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        head.train()
        losses = []
        year_order = list(train_data)
        rng.shuffle(year_order)
        for year_data in year_order:
            candidates = torch.as_tensor(
                np.asarray(session_vectors[year_data.session_indices], dtype=np.float32),
                device=device,
            )
            permutation = rng.permutation(len(year_data.abstract_indices))
            for start in range(0, len(permutation), batch_size):
                local_rows = permutation[start : start + batch_size]
                query_indices = year_data.abstract_indices[local_rows]
                queries = torch.as_tensor(
                    np.asarray(abstract_vectors[query_indices], dtype=np.float32),
                    device=device,
                )
                positive_mask = torch.as_tensor(
                    year_data.positive_mask[local_rows], dtype=torch.bool, device=device
                )
                optimizer.zero_grad(set_to_none=True)
                query_output = head.transform_query(queries)
                session_output = head.transform_session(candidates)
                logits = (query_output @ session_output.T) / temperature
                loss = multi_positive_listwise_loss(logits, positive_mask)
                loss = loss + regularization * head.identity_regularization()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

        validation_metrics, _ = evaluate_metric(
            head=head,
            year_data=validation_data,
            abstract_vectors=abstract_vectors,
            session_vectors=session_vectors,
            total_abstract_rows=sum(
                row["event_year"] == validation_data.year for row in abstracts
            ),
            device=device,
        )
        selection_key = (
            float(validation_metrics["recall_at_10"]),
            float(validation_metrics["mrr"]),
        )
        history.append(
            {
                "epoch": epoch,
                "mean_training_loss": round(float(np.mean(losses)), 6),
                "validation": validation_metrics,
            }
        )
        print(
            f"{kind} epoch {epoch}: loss={np.mean(losses):.5f} "
            f"validation R@10={selection_key[0]:.4f} MRR={selection_key[1]:.4f}",
            flush=True,
        )
        if selection_key > best_key:
            best_key = selection_key
            stale_epochs = 0
            torch.save(head.state_dict(), best_path)
        else:
            stale_epochs += 1
            if patience and stale_epochs >= patience:
                print(f"{kind}: early stopping after {epoch} epochs", flush=True)
                break

    head.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    best_metrics, _ = evaluate_metric(
        head=head,
        year_data=validation_data,
        abstract_vectors=abstract_vectors,
        session_vectors=session_vectors,
        total_abstract_rows=sum(row["event_year"] == validation_data.year for row in abstracts),
        device=device,
    )
    candidate_report = {
        "kind": kind,
        "parameters": sum(parameter.numel() for parameter in head.parameters()),
        "epochs_completed": len(history),
        "best_validation": best_metrics,
        "history": history,
    }
    write_json(output_dir / "training.json", candidate_report)
    return head, candidate_report


def save_metric_head(path: Path, head: Any, *, kind: str, dimension: int, rank: int) -> None:
    arrays = head.export_arrays()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        kind=np.asarray(kind),
        dimension=np.asarray(dimension, dtype=np.int32),
        rank=np.asarray(rank, dtype=np.int32),
        **arrays,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS is unavailable; falling back to CPU", flush=True)
        args.device = "cpu"
    sessions = list(iter_jsonl(args.prepared_dir / "sessions.jsonl.gz"))
    abstracts = list(iter_jsonl(args.prepared_dir / "abstracts.jsonl.gz"))
    session_vectors = np.load(args.embedding_dir / args.session_vectors, mmap_mode="r")
    abstract_vectors = np.load(args.embedding_dir / args.abstract_vectors, mmap_mode="r")
    dimension = int(session_vectors.shape[1])

    train_data = [
        build_year_training_data(sessions=sessions, abstracts=abstracts, year=year)
        for year in args.train_years
    ]
    validation_data = build_year_training_data(
        sessions=sessions, abstracts=abstracts, year=args.validation_year
    )
    baseline_head = make_metric_head("diagonal", dimension, args.rank).to(args.device)
    baseline_validation, _ = evaluate_metric(
        head=baseline_head,
        year_data=validation_data,
        abstract_vectors=abstract_vectors,
        session_vectors=session_vectors,
        total_abstract_rows=sum(
            row["event_year"] == args.validation_year for row in abstracts
        ),
        device=args.device,
    )

    candidates = {}
    candidate_heads = {}
    for kind in args.metric_types:
        learning_rate = (
            args.diagonal_learning_rate if kind == "diagonal" else args.learning_rate
        )
        head, report = train_candidate(
            kind=kind,
            dimension=dimension,
            rank=args.rank,
            train_data=train_data,
            validation_data=validation_data,
            sessions=sessions,
            abstracts=abstracts,
            session_vectors=session_vectors,
            abstract_vectors=abstract_vectors,
            output_dir=args.output_dir / kind,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=learning_rate,
            temperature=args.temperature,
            regularization=args.regularization,
            patience=args.patience,
            seed=args.seed,
        )
        candidates[kind] = report
        candidate_heads[kind] = head

    selected_kind = max(
        candidates,
        key=lambda kind: (
            candidates[kind]["best_validation"]["recall_at_10"],
            candidates[kind]["best_validation"]["mrr"],
        ),
    )
    selected_head = candidate_heads[selected_kind]
    confidence_bands = validation_confidence_bands(
        head=selected_head,
        year_data=validation_data,
        abstract_vectors=abstract_vectors,
        session_vectors=session_vectors,
        device=args.device,
    )
    write_json(args.output_dir / "confidence-bands.json", confidence_bands)
    final_years = [args.test_year, *args.forward_years]
    final_metrics = {}
    baseline_final = {}
    rank_files = {}
    for year in final_years:
        year_data = build_year_training_data(
            sessions=sessions, abstracts=abstracts, year=year
        )
        total_rows = sum(row["event_year"] == year for row in abstracts)
        baseline_final[str(year)], _ = evaluate_metric(
            head=baseline_head,
            year_data=year_data,
            abstract_vectors=abstract_vectors,
            session_vectors=session_vectors,
            total_abstract_rows=total_rows,
            device=args.device,
        )
        final_metrics[str(year)], ranks = evaluate_metric(
            head=selected_head,
            year_data=year_data,
            abstract_vectors=abstract_vectors,
            session_vectors=session_vectors,
            total_abstract_rows=total_rows,
            device=args.device,
        )
        rank_path = args.output_dir / f"ranks-{year}.npy"
        np.save(
            rank_path,
            np.asarray([-1 if value is None else value for value in ranks], dtype=np.int32),
        )
        rank_files[str(year)] = rank_path.name

    save_metric_head(
        args.output_dir / "metric-head.npz",
        selected_head,
        kind=selected_kind,
        dimension=dimension,
        rank=args.rank,
    )
    report = {
        "objective": "multi-positive listwise softmax over every same-year session",
        "selection_rule": "maximum 2024 Recall@10, then MRR",
        "train_years": args.train_years,
        "validation_year": args.validation_year,
        "sealed_test_year": args.test_year,
        "forward_years": args.forward_years,
        "embedding_dimension": dimension,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "regularization": args.regularization,
        "baseline": {
            "validation": baseline_validation,
            "final_years": baseline_final,
        },
        "candidates": candidates,
        "selected_metric": selected_kind,
        "selected_validation": candidates[selected_kind]["best_validation"],
        "selected_final_years": final_metrics,
        "rank_files": rank_files,
        "metric_head": "metric-head.npz",
        "confidence_bands": confidence_bands,
        "confidence_bands_file": "confidence-bands.json",
    }
    write_json(args.output_dir / "report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/matcher/prepared"))
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--abstract-vectors", default="abstracts_121971.npy")
    parser.add_argument("--session-vectors", default="sessions_with_divisions.npy")
    parser.add_argument("--output-dir", type=Path, default=Path("data/matcher/metric-learning"))
    parser.add_argument("--metric-types", nargs="+", choices=("diagonal", "lowrank"), default=["diagonal", "lowrank"])
    parser.add_argument("--train-years", nargs="+", type=int, default=[2022, 2023])
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--forward-years", nargs="*", type=int, default=[2026])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--diagonal-learning-rate", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    report = run(build_parser().parse_args(argv))
    print(json.dumps({
        "selected_metric": report["selected_metric"],
        "selected_validation": report["selected_validation"],
        "selected_final_years": report["selected_final_years"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
