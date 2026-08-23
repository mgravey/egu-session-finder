#!/usr/bin/env python3
"""Export one year's session metadata and dense vectors for the browser app."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from matcher_common import iter_jsonl, sha256_file, write_json


def load_metric_head(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as stored:
        metric = {name: np.asarray(stored[name]) for name in stored.files}
    metric["kind"] = str(metric["kind"].item())
    metric["dimension"] = int(metric["dimension"].item())
    metric["rank"] = int(metric["rank"].item())
    return metric


def transform_session_vectors(
    vectors: np.ndarray, metric: dict[str, Any]
) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    dimension = int(metric["dimension"])
    if values.shape[1] != dimension:
        raise ValueError("metric and session embedding dimensions differ")
    log_weight = np.asarray(metric["log_weight"], dtype=np.float32)
    transformed = values * np.exp(0.5 * log_weight)
    if metric["kind"] == "lowrank":
        transformed = transformed + (
            transformed @ np.asarray(metric["session_down"], dtype=np.float32)
        ) @ np.asarray(metric["session_up"], dtype=np.float32)
    elif metric["kind"] != "diagonal":
        raise ValueError(f"unsupported metric kind: {metric['kind']}")
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    return np.asarray(transformed / np.maximum(norms, 1e-12), dtype=np.float32)


def transform_query_vectors(
    vectors: np.ndarray, metric: dict[str, Any]
) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    dimension = int(metric["dimension"])
    if values.shape[1] != dimension:
        raise ValueError("metric and query embedding dimensions differ")
    log_weight = np.asarray(metric["log_weight"], dtype=np.float32)
    transformed = values * np.exp(0.5 * log_weight)
    if metric["kind"] == "lowrank":
        transformed = transformed + (
            transformed @ np.asarray(metric["query_down"], dtype=np.float32)
        ) @ np.asarray(metric["query_up"], dtype=np.float32)
    elif metric["kind"] != "diagonal":
        raise ValueError(f"unsupported metric kind: {metric['kind']}")
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    return np.asarray(transformed / np.maximum(norms, 1e-12), dtype=np.float32)


def write_query_metric(
    *, metric: dict[str, Any], output_dir: Path
) -> tuple[Path, dict[str, Any]]:
    arrays = [("logWeight", np.asarray(metric["log_weight"], dtype="<f4"))]
    if metric["kind"] == "lowrank":
        arrays.extend(
            [
                ("queryDown", np.asarray(metric["query_down"], dtype="<f4")),
                ("queryUp", np.asarray(metric["query_up"], dtype="<f4")),
            ]
        )
    layout = []
    offset = 0
    for name, values in arrays:
        size = int(values.size)
        layout.append(
            {
                "name": name,
                "shape": list(values.shape),
                "offsetFloats": offset,
                "lengthFloats": size,
            }
        )
        offset += size
    output = output_dir / "query-metric.bin"
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("wb") as stream:
        for _, values in arrays:
            values.tofile(stream)
    temporary.replace(output)
    return output, {
        "kind": metric["kind"],
        "dimension": int(metric["dimension"]),
        "rank": int(metric["rank"]),
        "sessionTransformApplied": True,
        "queryLayout": layout,
    }


def export_catalogue(
    *,
    prepared_dir: Path,
    vectors_path: Path,
    model_config: dict[str, Any],
    confidence_bands: dict[str, float],
    colors: dict[str, str],
    year: int,
    output_dir: Path,
    dense_weight: float,
    metric_head_path: Path | None = None,
) -> dict[str, Any]:
    sessions = list(iter_jsonl(prepared_dir / "sessions.jsonl.gz"))
    vectors = np.load(vectors_path, mmap_mode="r")
    if len(vectors) != len(sessions):
        raise ValueError(
            f"vector/session row mismatch: {len(vectors)} vectors for {len(sessions)} sessions"
        )
    indices = [index for index, session in enumerate(sessions) if session["event_year"] == year]
    if not indices:
        raise ValueError(f"no prepared sessions for {year}")
    selected_vectors = np.asarray(vectors[indices], dtype=np.float32)
    metric = load_metric_head(metric_head_path) if metric_head_path else None
    if metric is not None:
        selected_vectors = transform_session_vectors(selected_vectors, metric)
    selected_vectors = np.asarray(selected_vectors, dtype="<f4")
    norms = np.linalg.norm(selected_vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("session embeddings are not L2-normalized")
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors_output = output_dir / "session-vectors.bin"
    temporary = vectors_output.with_name(f".{vectors_output.name}.tmp")
    selected_vectors.tofile(temporary)
    temporary.replace(vectors_output)
    metadata = []
    for row, index in enumerate(indices):
        session = sessions[index]
        codes = session.get("division_codes", [])
        metadata.append(
            {
                "vectorRow": row,
                "eventCode": session["event_code"],
                "eventYear": year,
                "sessionId": session["session_id"],
                "code": session.get("session_code"),
                "title": session.get("title"),
                "description": session.get("description"),
                "descriptionMissing": session["description_missing"],
                "divisions": session.get("discovery_groups", []),
                "divisionCodes": codes,
                "color": next((colors[code] for code in codes if code in colors), None),
                "url": session["source_url"],
                **(
                    {"searchText": session["session_text"]}
                    if dense_weight < 1
                    else {}
                ),
            }
        )
    sessions_output = output_dir / "sessions.json"
    write_json(sessions_output, metadata)
    generated = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schemaVersion": 2 if metric is not None else 1,
        "catalogueYear": year,
        "catalogueVersion": f"EGU{str(year)[-2:]}-{generated[:10]}",
        "generatedAt": generated,
        "modelId": model_config["python_model_id"],
        "browserModelId": model_config["browser_model_id"],
        "browserModelPath": model_config.get("browser_model_path"),
        "modelRevision": model_config.get("revision", "main-unpinned"),
        "dimension": int(selected_vectors.shape[1]),
        "maxTokens": int(model_config["max_tokens"]),
        "overlapTokens": int(model_config.get("overlap", 32)),
        "pooling": "mean",
        "normalized": True,
        "queryPrefix": model_config.get("query_prefix", ""),
        "vectorDtype": "float32-le",
        "sessionCount": len(metadata),
        "scoring": {
            "denseWeight": dense_weight,
            "bm25": {"enabled": dense_weight < 1, "k1": 1.2, "b": 0.75},
        },
        "confidenceBands": confidence_bands,
        "assets": {
            "sessions": {"path": "sessions.json"},
            "vectors": {"path": "session-vectors.bin"},
        },
    }
    manifest["assets"]["sessions"]["sha256"] = sha256_file(sessions_output)
    manifest["assets"]["vectors"]["sha256"] = sha256_file(vectors_output)
    if metric is not None:
        metric_output, metric_manifest = write_query_metric(
            metric=metric, output_dir=output_dir
        )
        manifest["metric"] = metric_manifest
        manifest["assets"]["queryMetric"] = {
            "path": metric_output.name,
            "sha256": sha256_file(metric_output),
        }
    browser_model_path = model_config.get("browser_model_path")
    if browser_model_path:
        relative_model_path = str(browser_model_path).lstrip("/")
        model_directory = output_dir.parent / relative_model_path
        if model_directory.is_dir():
            model_files = []
            for path in sorted(model_directory.rglob("*")):
                relative_parts = path.relative_to(model_directory).parts
                if (
                    not path.is_file()
                    or any(part.startswith(".") for part in relative_parts)
                    or path.name == "egu_model_source.json"
                ):
                    continue
                model_files.append(
                    {
                        "path": path.relative_to(output_dir.parent).as_posix(),
                        "sha256": sha256_file(path),
                    }
                )
            if model_files:
                manifest["assets"]["model"] = model_files
    write_json(output_dir / "model-manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/matcher/prepared"))
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=Path("config/models.json"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--confidence-bands", type=Path, required=True)
    parser.add_argument("--division-colors", type=Path, default=Path("config/division_colors.json"))
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("web/public/catalogue"))
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--model-revision")
    parser.add_argument("--browser-model-path")
    parser.add_argument(
        "--metric-head",
        type=Path,
        help="Optional metric-head.npz; transforms sessions offline and exports the query side",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configs = json.loads(args.model_config.read_text(encoding="utf-8"))["models"]
    model = next((item for item in configs if item["name"] == args.model), None)
    if model is None:
        raise SystemExit(f"model not found in {args.model_config}: {args.model}")
    model = dict(model)
    if args.model_revision:
        model["revision"] = args.model_revision
    if args.browser_model_path:
        model["browser_model_path"] = args.browser_model_path
    colors_value = json.loads(args.division_colors.read_text(encoding="utf-8"))
    colors = colors_value.get("colors", colors_value)
    bands = json.loads(args.confidence_bands.read_text(encoding="utf-8"))
    manifest = export_catalogue(
        prepared_dir=args.prepared_dir,
        vectors_path=args.vectors,
        model_config=model,
        confidence_bands=bands,
        colors=colors,
        year=args.year,
        output_dir=args.output_dir,
        dense_weight=args.dense_weight,
        metric_head_path=args.metric_head,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
