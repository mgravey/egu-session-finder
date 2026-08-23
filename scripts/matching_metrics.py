"""Retrieval metrics, BM25 scoring, and confidence calibration."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Iterable, Sequence

import numpy as np


TOKEN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", re.I)


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN.finditer(text)]


@dataclass
class BM25Index:
    documents: list[list[str]]
    frequencies: list[Counter[str]]
    document_frequency: Counter[str]
    postings: dict[str, list[tuple[int, int]]]
    document_lengths: np.ndarray
    average_length: float
    k1: float = 1.2
    b: float = 0.75

    @classmethod
    def build(cls, documents: Iterable[str]) -> "BM25Index":
        tokenized = [tokenize(document) for document in documents]
        frequencies = [Counter(document) for document in tokenized]
        document_frequency: Counter[str] = Counter()
        postings: dict[str, list[tuple[int, int]]] = {}
        for index, frequency in enumerate(frequencies):
            document_frequency.update(frequency.keys())
            for term, occurrences in frequency.items():
                postings.setdefault(term, []).append((index, occurrences))
        average = sum(map(len, tokenized)) / len(tokenized) if tokenized else 0.0
        return cls(
            tokenized,
            frequencies,
            document_frequency,
            postings,
            np.asarray([len(document) for document in tokenized], dtype=np.float32),
            average,
        )

    def scores(self, query: str) -> np.ndarray:
        count = len(self.documents)
        result = np.zeros(count, dtype=np.float32)
        if not count or not self.average_length:
            return result
        for term in set(tokenize(query)):
            document_frequency = self.document_frequency.get(term, 0)
            if not document_frequency:
                continue
            inverse_frequency = math.log(
                1 + (count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            for index, occurrences in self.postings[term]:
                denominator = occurrences + self.k1 * (
                    1 - self.b + self.b * self.document_lengths[index] / self.average_length
                )
                result[index] += inverse_frequency * occurrences * (self.k1 + 1) / denominator
        return result


def minmax(values: np.ndarray) -> np.ndarray:
    if not values.size:
        return values.copy()
    lower = float(values.min())
    upper = float(values.max())
    if upper <= lower:
        return np.zeros_like(values)
    return (values - lower) / (upper - lower)


def hybrid_scores(dense: np.ndarray, lexical: np.ndarray, dense_weight: float) -> np.ndarray:
    if not 0 <= dense_weight <= 1:
        raise ValueError("dense_weight must be between zero and one")
    return dense_weight * minmax(dense) + (1 - dense_weight) * minmax(lexical)


def ranks_for_scores(
    scores: np.ndarray,
    candidate_keys: Sequence[str],
    relevant_keys: Iterable[str],
) -> tuple[int | None, list[int]]:
    relevant = set(relevant_keys)
    order = np.argsort(-scores, kind="stable")
    ranks = [rank for rank, index in enumerate(order, start=1) if candidate_keys[index] in relevant]
    return (min(ranks) if ranks else None), order.tolist()


def retrieval_metrics(ranks: Sequence[int | None]) -> dict[str, float | int]:
    valid = [rank for rank in ranks if rank is not None]
    count = len(ranks)
    result: dict[str, float | int] = {
        "queries": count,
        "queries_with_relevant_candidate": len(valid),
        "coverage": round(len(valid) / count, 6) if count else 0.0,
    }
    for cutoff in (1, 3, 5, 10):
        result[f"recall_at_{cutoff}"] = (
            round(sum(rank <= cutoff for rank in valid) / count, 6) if count else 0.0
        )
    result["mrr"] = round(sum(1 / rank for rank in valid) / count, 6) if count else 0.0
    return result


def confidence_features(scores: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-scores, kind="stable")
    top = float(scores[order[0]]) if len(order) else 0.0
    second = float(scores[order[1]]) if len(order) > 1 else top
    return top, top - second


def fit_confidence_bands(
    observations: Sequence[tuple[float, float, bool]],
) -> dict[str, float]:
    """Fit deterministic tercile thresholds without pretending scores are probabilities."""
    if not observations:
        return {"top_score_high": 1.0, "margin_high": 1.0, "top_score_low": -1.0}
    correct = sorted((top, margin) for top, margin, is_correct in observations if is_correct)
    all_scores = sorted(top for top, _, _ in observations)
    if not correct:
        return {
            "top_score_high": max(all_scores),
            "margin_high": 1.0,
            "top_score_low": all_scores[len(all_scores) // 3],
        }
    high_index = max(0, int(len(correct) * 2 / 3) - 1)
    low_index = max(0, int(len(all_scores) / 3) - 1)
    return {
        "top_score_high": float(correct[high_index][0]),
        "margin_high": float(sorted(item[1] for item in correct)[high_index]),
        "top_score_low": float(all_scores[low_index]),
    }


def confidence_label(top: float, margin: float, bands: dict[str, float]) -> str:
    if top >= bands["top_score_high"] and margin >= bands["margin_high"]:
        return "high"
    if top < bands["top_score_low"]:
        return "low"
    return "medium"
