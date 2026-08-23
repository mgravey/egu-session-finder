#!/usr/bin/env python3
"""Build fixed-width field and chunk embeddings for multi-vector retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import numpy as np

from benchmark_embeddings import token_chunks
from matcher_common import iter_jsonl, write_json


def digest_rows(keys: Sequence[str], texts: Sequence[Sequence[str]]) -> str:
    digest = hashlib.sha256()
    for key, parts in zip(keys, texts, strict=True):
        digest.update(key.encode("utf-8"))
        for part in parts:
            digest.update(b"\0")
            digest.update(part.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def fixed_chunks(
    tokenizer: Any,
    text: str,
    *,
    max_tokens: int,
    overlap: int,
    limit: int,
) -> list[str]:
    if not text.strip():
        return []
    return [
        chunk
        for chunk, _ in token_chunks(tokenizer, text, max_tokens, overlap)[:limit]
    ]


def encode_fixed_parts(
    *,
    model: Any,
    rows: Sequence[Sequence[str]],
    width: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    flattened: list[str] = []
    positions: list[tuple[int, int]] = []
    for row_index, values in enumerate(rows):
        for part_index, value in enumerate(values[:width]):
            if value.strip():
                flattened.append(value)
                positions.append((row_index, part_index))
    if not flattened:
        raise ValueError("no non-empty text parts were supplied")
    encoded = np.asarray(
        model.encode(
            flattened,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )
    vectors = np.zeros((len(rows), width, encoded.shape[1]), dtype=np.float32)
    mask = np.zeros((len(rows), width), dtype=np.bool_)
    for vector, (row_index, part_index) in zip(encoded, positions, strict=True):
        vectors[row_index, part_index] = vector
        mask[row_index, part_index] = True
    return vectors, mask


def cached_parts(
    *,
    output_dir: Path,
    name: str,
    model: Any,
    model_id: str,
    keys: Sequence[str],
    rows: Sequence[Sequence[str]],
    width: int,
    batch_size: int,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    vector_path = output_dir / f"{name}.npy"
    mask_path = output_dir / f"{name}_mask.npy"
    metadata_path = output_dir / f"{name}.json"
    expected = {
        "model_id": model_id,
        "rows": len(rows),
        "width": width,
        "input_sha256": digest_rows(keys, rows),
        **settings,
    }
    if vector_path.is_file() and mask_path.is_file() and metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in expected.items()):
            return np.load(vector_path, mmap_mode="r"), np.load(mask_path, mmap_mode="r")
    started = time.monotonic()
    vectors, mask = encode_fixed_parts(
        model=model,
        rows=rows,
        width=width,
        batch_size=batch_size,
    )
    np.save(vector_path, vectors)
    np.save(mask_path, mask)
    write_json(
        metadata_path,
        {
            **expected,
            "dimension": int(vectors.shape[-1]),
            "populated_parts": int(mask.sum()),
            "seconds": round(time.monotonic() - started, 3),
        },
    )
    return vectors, mask


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("Install requirements-ml.txt") from exc

    sessions = list(iter_jsonl(args.prepared_dir / "sessions.jsonl.gz"))
    abstracts = list(iter_jsonl(args.prepared_dir / "abstracts.jsonl.gz"))
    model = SentenceTransformer(args.model, device=args.device)
    max_tokens = min(args.max_tokens, int(model.max_seq_length))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "pooling": "attention_mask_mean",
        "max_tokens": max_tokens,
        "overlap": args.overlap,
    }

    query_title_rows = [[row["title"]] for row in abstracts]
    query_chunk_rows = [
        fixed_chunks(
            model.tokenizer,
            row.get("abstract_text") or "",
            max_tokens=max_tokens,
            overlap=args.overlap,
            limit=args.query_chunks,
        )
        for row in abstracts
    ]
    session_title_rows = [[row["title"]] for row in sessions]
    session_description_rows = [
        fixed_chunks(
            model.tokenizer,
            row.get("description") or "",
            max_tokens=max_tokens,
            overlap=args.overlap,
            limit=args.session_chunks,
        )
        for row in sessions
    ]

    outputs = {}
    for name, keys, rows, width in (
        (
            "query_title",
            [row["abstract_key"] for row in abstracts],
            query_title_rows,
            1,
        ),
        (
            "query_chunks",
            [row["abstract_key"] for row in abstracts],
            query_chunk_rows,
            args.query_chunks,
        ),
        (
            "session_title",
            [row["session_key"] for row in sessions],
            session_title_rows,
            1,
        ),
        (
            "session_description",
            [row["session_key"] for row in sessions],
            session_description_rows,
            args.session_chunks,
        ),
    ):
        vectors, mask = cached_parts(
            output_dir=args.output_dir,
            name=name,
            model=model,
            model_id=args.model,
            keys=keys,
            rows=rows,
            width=width,
            batch_size=args.batch_size,
            settings=settings,
        )
        outputs[name] = {
            "shape": list(vectors.shape),
            "populated_parts": int(mask.sum()),
        }
    report = {
        "model": args.model,
        "max_tokens": max_tokens,
        "overlap": args.overlap,
        "query_chunks": args.query_chunks,
        "session_chunks": args.session_chunks,
        "outputs": outputs,
    }
    write_json(args.output_dir / "multivector-embedding-report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--query-chunks", type=int, default=4)
    parser.add_argument("--session-chunks", type=int, default=2)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
