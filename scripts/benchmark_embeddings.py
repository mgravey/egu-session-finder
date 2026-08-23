#!/usr/bin/env python3
"""Cache candidate embeddings and evaluate temporal EGU session retrieval."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import numpy as np

from matcher_common import iter_jsonl, write_json, write_jsonl_gz
from matching_metrics import (
    BM25Index,
    confidence_features,
    fit_confidence_bands,
    hybrid_scores,
    ranks_for_scores,
    retrieval_metrics,
)


def load_model_configs(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    models = value.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"no models found in {path}")
    return models


def limited_evaluation_sample(
    abstracts: Sequence[dict[str, Any]], years: Sequence[int], limit: int
) -> list[dict[str, Any]]:
    """Select a deterministic, year-balanced pilot sample.

    A chronological prefix can silently contain no validation or test rows.
    Round-robin selection keeps every requested evaluation year represented.
    """
    if limit <= 0:
        return list(abstracts)
    queues = {
        year: [
            item
            for item in abstracts
            if item["event_year"] == year and item["has_abstract"] and item["session_keys"]
        ]
        for year in years
    }
    selected: list[dict[str, Any]] = []
    while len(selected) < limit:
        added = False
        for year in years:
            queue = queues[year]
            if queue and len(selected) < limit:
                selected.append(queue.pop(0))
                added = True
        if not added:
            break
    return selected


def input_digest(keys: Sequence[str], texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for key, text in zip(keys, texts, strict=True):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def token_chunks(tokenizer: Any, text: str, max_tokens: int, overlap: int) -> list[tuple[str, int]]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    token_ids = encoded["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    capacity = max(8, max_tokens - 2)
    if len(token_ids) <= capacity:
        return [(text, max(1, len(token_ids)))]
    step = max(1, capacity - overlap)
    chunks = []
    for start in range(0, len(token_ids), step):
        ids = token_ids[start : start + capacity]
        if not ids:
            break
        chunks.append((tokenizer.decode(ids, skip_special_tokens=True), len(ids)))
        if start + capacity >= len(token_ids):
            break
    return chunks


def encode_chunked(
    model: Any,
    texts: Sequence[str],
    *,
    max_tokens: int,
    overlap: int,
    batch_size: int,
    block_size: int = 512,
) -> np.ndarray:
    results: list[np.ndarray] = []
    for block_start in range(0, len(texts), block_size):
        block = texts[block_start : block_start + block_size]
        chunks: list[str] = []
        owners: list[int] = []
        weights: list[int] = []
        for owner, text in enumerate(block):
            for chunk, weight in token_chunks(model.tokenizer, text, max_tokens, overlap):
                chunks.append(chunk)
                owners.append(owner)
                weights.append(weight)
        encoded = np.asarray(
            model.encode(
                chunks,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        combined = np.zeros((len(block), encoded.shape[1]), dtype=np.float32)
        totals = np.zeros(len(block), dtype=np.float32)
        for vector, owner, weight in zip(encoded, owners, weights, strict=True):
            combined[owner] += vector * weight
            totals[owner] += weight
        combined /= np.maximum(totals[:, None], 1)
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        combined /= np.maximum(norms, 1e-12)
        results.append(combined)
        print(f"Embedded {min(block_start + len(block), len(texts))}/{len(texts)}", flush=True)
    return np.concatenate(results) if results else np.empty((0, 0), dtype=np.float32)


def cached_embeddings(
    *,
    cache_dir: Path,
    cache_name: str,
    model: Any,
    model_config: dict[str, Any],
    keys: Sequence[str],
    texts: Sequence[str],
    max_tokens: int,
    overlap: int,
    batch_size: int,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    vector_path = cache_dir / f"{cache_name}.npy"
    metadata_path = cache_dir / f"{cache_name}.json"
    digest = input_digest(keys, texts)
    expected = {
        "model_id": model_config["python_model_id"],
        "max_tokens": max_tokens,
        "overlap": overlap,
        "input_sha256": digest,
        "rows": len(texts),
    }
    if model_config.get("runtime_backend"):
        expected["runtime_backend"] = model_config["runtime_backend"]
        expected["runtime_model_file"] = model_config.get("runtime_model_file")
    if vector_path.is_file() and metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in expected.items()):
            return np.load(vector_path, mmap_mode="r")
    started = time.monotonic()
    vectors = encode_chunked(
        model,
        texts,
        max_tokens=max_tokens,
        overlap=overlap,
        batch_size=batch_size,
    )
    temporary = vector_path.with_name(f".{vector_path.name}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, vectors)
    temporary.replace(vector_path)
    write_json(
        metadata_path,
        {
            **expected,
            "dimension": int(vectors.shape[1]) if vectors.size else 0,
            "seconds": round(time.monotonic() - started, 3),
        },
    )
    return vectors


def evaluate_year(
    *,
    year: int,
    sessions: list[dict[str, Any]],
    abstracts: list[dict[str, Any]],
    session_vectors: np.ndarray,
    abstract_vectors: np.ndarray,
    dense_weight: float,
    use_bm25: bool,
    lexical_index: BM25Index | None = None,
    lexical_cache: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[tuple[float, float, bool]]]:
    session_indices = [index for index, value in enumerate(sessions) if value["event_year"] == year]
    abstract_indices = [
        index
        for index, value in enumerate(abstracts)
        if value["event_year"] == year and value["has_abstract"] and value["session_keys"]
    ]
    candidate_keys = [sessions[index]["session_key"] for index in session_indices]
    candidate_vectors = np.asarray(session_vectors[session_indices])
    lexical = lexical_index or BM25Index.build(
        [sessions[index]["session_text"] for index in session_indices]
    )
    ranks: list[int | None] = []
    predictions: list[dict[str, Any]] = []
    observations: list[tuple[float, float, bool]] = []
    division_hits = 0
    by_division: defaultdict[str, list[int | None]] = defaultdict(list)
    by_description: defaultdict[str, list[int | None]] = defaultdict(list)
    session_lookup = {item["session_key"]: item for item in sessions}
    for block_start in range(0, len(abstract_indices), 512):
        block_indices = abstract_indices[block_start : block_start + 512]
        dense_rows = np.asarray(abstract_vectors[block_indices]) @ candidate_vectors.T
        for abstract_index, dense in zip(block_indices, dense_rows, strict=True):
            abstract = abstracts[abstract_index]
            if use_bm25:
                lexical_scores = (
                    lexical_cache.get(abstract["abstract_key"])
                    if lexical_cache is not None
                    else None
                )
                if lexical_scores is None:
                    lexical_scores = lexical.scores(abstract["query_text"])
                    if lexical_cache is not None:
                        lexical_cache[abstract["abstract_key"]] = lexical_scores
                scores = hybrid_scores(dense, lexical_scores, dense_weight)
            else:
                scores = dense
            rank, order = ranks_for_scores(scores, candidate_keys, abstract["session_keys"])
            ranks.append(rank)
            top, margin = confidence_features(scores)
            observations.append((top, margin, rank == 1))
            true_sessions = [session_lookup[key] for key in abstract["session_keys"]]
            true_divisions = {
                code for session in true_sessions for code in session.get("division_codes", [])
            }
            predicted_divisions = set(
                sessions[session_indices[order[0]]].get("division_codes", [])
            )
            division_hits += bool(true_divisions & predicted_divisions)
            for division in true_divisions or {"OTHER"}:
                by_division[division].append(rank)
            description_group = (
                "with_description"
                if any(not session["description_missing"] for session in true_sessions)
                else "title_only"
            )
            by_description[description_group].append(rank)
            predictions.append(
                {
                    "abstract_key": abstract["abstract_key"],
                    "event_year": year,
                    "true_session_keys": abstract["session_keys"],
                    "rank": rank,
                    "top_score": top,
                    "top_margin": margin,
                    "top10_session_keys": [candidate_keys[index] for index in order[:10]],
                }
            )
    metrics = retrieval_metrics(ranks)
    metrics["division_recall_at_1"] = (
        round(division_hits / len(abstract_indices), 6) if abstract_indices else 0.0
    )
    metrics["by_division"] = {
        division: retrieval_metrics(values) for division, values in sorted(by_division.items())
    }
    metrics["by_description"] = {
        name: retrieval_metrics(values) for name, values in sorted(by_description.items())
    }
    return metrics, predictions, observations


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("Install requirements-ml.txt in a Python 3.12 environment") from exc

    sessions = list(iter_jsonl(args.prepared_dir / "sessions.jsonl.gz"))
    abstracts = list(iter_jsonl(args.prepared_dir / "abstracts.jsonl.gz"))
    if args.limit:
        abstracts = limited_evaluation_sample(abstracts, args.years, args.limit)
    configurations = load_model_configs(args.models)
    if args.model != "all":
        configurations = [item for item in configurations if item["name"] == args.model]
        if not configurations:
            raise ValueError(f"unknown model: {args.model}")
    if args.model_path and len(configurations) != 1:
        raise ValueError("--model-path requires selecting exactly one --model")
    all_results: dict[str, Any] = {"models": {}}
    lexical_indices = {
        year: BM25Index.build(
            [item["session_text"] for item in sessions if item["event_year"] == year]
        )
        for year in args.years
    }
    lexical_caches: dict[int, dict[str, np.ndarray]] = {
        year: {} for year in args.years
    }
    for config in configurations:
        print(f"Loading {config['python_model_id']}", flush=True)
        model_source = str(args.model_path) if args.model_path else config["python_model_id"]
        runtime_config = {**config, "python_model_id": model_source}
        model_kwargs = {"file_name": args.model_file} if args.model_file else None
        model = SentenceTransformer(
            model_source,
            device=args.device,
            backend=args.backend,
            model_kwargs=model_kwargs,
        )
        if args.backend != "torch" or args.model_file:
            runtime_config["runtime_backend"] = args.backend
            runtime_config["runtime_model_file"] = args.model_file
        max_tokens = min(int(config["max_tokens"]), int(model.max_seq_length))
        model_cache = args.cache_dir / config["name"]
        session_keys = [item["session_key"] for item in sessions]
        abstract_keys = [item["abstract_key"] for item in abstracts]
        session_with = cached_embeddings(
            cache_dir=model_cache,
            cache_name="sessions_with_divisions",
            model=model,
            model_config=runtime_config,
            keys=session_keys,
            texts=[item["session_text"] for item in sessions],
            max_tokens=max_tokens,
            overlap=args.overlap,
            batch_size=args.batch_size,
        )
        session_without = cached_embeddings(
            cache_dir=model_cache,
            cache_name="sessions_without_divisions",
            model=model,
            model_config=runtime_config,
            keys=session_keys,
            texts=[item["session_text_without_divisions"] for item in sessions],
            max_tokens=max_tokens,
            overlap=args.overlap,
            batch_size=args.batch_size,
        )
        query_prefix = config.get("query_prefix", "")
        abstract_vectors = cached_embeddings(
            cache_dir=model_cache,
            cache_name=f"abstracts_{len(abstracts)}",
            model=model,
            model_config=runtime_config,
            keys=abstract_keys,
            texts=[query_prefix + item["query_text"] for item in abstracts],
            max_tokens=max_tokens,
            overlap=args.overlap,
            batch_size=args.batch_size,
        )
        variants: dict[str, Any] = {}
        for name, vectors in (
            ("dense_with_divisions", session_with),
            ("dense_without_divisions", session_without),
        ):
            variants[name] = {}
            for year in args.years:
                metrics, _, _ = evaluate_year(
                    year=year,
                    sessions=sessions,
                    abstracts=abstracts,
                    session_vectors=vectors,
                    abstract_vectors=abstract_vectors,
                    dense_weight=1.0,
                    use_bm25=False,
                    lexical_index=lexical_indices[year],
                    lexical_cache=lexical_caches[year],
                )
                variants[name][str(year)] = metrics

        candidate_weights = (
            [args.hybrid_dense_weight]
            if args.hybrid_dense_weight is not None
            else [0.5, 0.65, 0.75, 0.85, 0.95]
        )
        validation = []
        for weight in candidate_weights:
            metrics, _, _ = evaluate_year(
                year=2024,
                sessions=sessions,
                abstracts=abstracts,
                session_vectors=session_with,
                abstract_vectors=abstract_vectors,
                dense_weight=weight,
                use_bm25=True,
                lexical_index=lexical_indices[2024],
                lexical_cache=lexical_caches[2024],
            )
            validation.append((metrics["recall_at_10"], metrics["mrr"], weight))
        selected_weight = max(validation)[2]
        variants["hybrid"] = {"dense_weight": selected_weight, "years": {}}
        validation_observations = []
        all_predictions = []
        for year in args.years:
            metrics, predictions, observations = evaluate_year(
                year=year,
                sessions=sessions,
                abstracts=abstracts,
                session_vectors=session_with,
                abstract_vectors=abstract_vectors,
                dense_weight=selected_weight,
                use_bm25=True,
                lexical_index=lexical_indices[year],
                lexical_cache=lexical_caches[year],
            )
            variants["hybrid"]["years"][str(year)] = metrics
            all_predictions.extend(predictions)
            if year == 2024:
                validation_observations = observations
        bands = fit_confidence_bands(validation_observations)
        write_json(model_cache / "confidence_bands.json", bands)
        write_jsonl_gz(model_cache / "predictions.jsonl.gz", all_predictions)
        result = {
            "model": runtime_config,
            "max_tokens": max_tokens,
            "overlap": args.overlap,
            "rows": {"sessions": len(sessions), "abstracts": len(abstracts)},
            "variants": variants,
            "confidence_bands": bands,
        }
        write_json(model_cache / "metrics.json", result)
        all_results["models"][config["name"]] = result
    write_json(args.output, all_results)
    return all_results


def parse_years(value: str) -> list[int]:
    return sorted({int(item) for item in value.split(",") if item.strip()})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/matcher/prepared"))
    parser.add_argument("--models", type=Path, default=Path("config/models.json"))
    parser.add_argument("--model", default="all")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--backend", choices=("torch", "onnx"), default="torch")
    parser.add_argument("--model-file")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/matcher/embeddings"))
    parser.add_argument("--output", type=Path, default=Path("data/matcher/benchmark.json"))
    parser.add_argument("--years", type=parse_years, default=parse_years("2024,2025,2026"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--hybrid-dense-weight", type=float)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_benchmark(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
