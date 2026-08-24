#!/usr/bin/env python3
"""Export an EGU multi-vector catalogue for fully static browser inference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from export_browser_catalogue import (
    browser_prediction_set,
    load_metric_head,
    transform_session_vectors,
    write_query_metric,
)
from export_multivector_score_matrices import ordered_weights
from matcher_common import iter_jsonl, sha256_file, write_json
from session_categories import is_typical_scientific_session
from train_multivector_ranker import FEATURE_NAMES


def transformed_parts(values: np.ndarray, metric: dict[str, Any]) -> np.ndarray:
    shape = values.shape
    return transform_session_vectors(
        np.asarray(values, dtype=np.float32).reshape(-1, shape[-1]), metric
    ).reshape(shape)


def export_multivector_catalogue(
    *,
    prepared_dir: Path,
    baseline_embedding_dir: Path,
    multivector_dir: Path,
    metric_head_path: Path,
    weights_path: Path,
    prediction_set_value: dict[str, Any],
    model_config: dict[str, Any],
    colors: dict[str, str],
    year: int,
    output_dir: Path,
) -> dict[str, Any]:
    sessions = list(iter_jsonl(prepared_dir / "sessions.jsonl.gz"))
    prediction_set = browser_prediction_set(prediction_set_value)
    if prediction_set["score"] != "learned_metric_plus_multivector":
        raise ValueError("prediction-set scores do not match the multi-vector browser scorer")
    indices = [
        index
        for index, session in enumerate(sessions)
        if session["event_year"] == year
        and (
            prediction_set["candidatePolicy"] != "typical-scientific-only"
            or is_typical_scientific_session(session)
        )
    ]
    if not indices:
        raise ValueError(f"no browser candidates for {year}")

    joint = np.load(
        baseline_embedding_dir / "sessions_with_divisions.npy", mmap_mode="r"
    )
    title = np.load(multivector_dir / "session_title.npy", mmap_mode="r")[:, 0]
    description = np.load(
        multivector_dir / "session_description.npy", mmap_mode="r"
    )
    description_mask = np.load(
        multivector_dir / "session_description_mask.npy", mmap_mode="r"
    )
    if not all(len(values) == len(sessions) for values in (joint, title, description)):
        raise ValueError("session metadata and multi-vector arrays differ")

    metric = load_metric_head(metric_head_path)
    selected_joint = transform_session_vectors(np.asarray(joint[indices]), metric)
    selected_title = transform_session_vectors(np.asarray(title[indices]), metric)
    selected_description = transformed_parts(np.asarray(description[indices]), metric)
    selected_mask = np.asarray(description_mask[indices], dtype=np.bool_)
    packed = np.concatenate(
        (selected_joint[:, None, :], selected_title[:, None, :], selected_description),
        axis=1,
    ).astype("<f2", copy=False)
    session_chunks = int(selected_description.shape[1])
    stride = int(packed.shape[1])
    dimension = int(packed.shape[2])

    embedding_report = json.loads(
        (multivector_dir / "multivector-embedding-report.json").read_text(encoding="utf-8")
    )
    weights = ordered_weights(weights_path)
    weight_values = {
        name: float(value) for name, value in zip(FEATURE_NAMES, weights, strict=True)
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    vector_output = output_dir / "session-vectors.bin"
    temporary = vector_output.with_name(f".{vector_output.name}.tmp")
    packed.tofile(temporary)
    temporary.replace(vector_output)

    palette = (
        "#2563a7", "#0f766e", "#7c3aed", "#c2410c", "#a21caf",
        "#047857", "#b45309", "#be123c", "#0369a1", "#4d7c0f",
    )
    metadata = []
    for vector_row, index in enumerate(indices):
        session = sessions[index]
        division_codes = session.get("division_codes", [])
        fallback_index = sum(ord(character) for code in division_codes for character in code)
        color = next((colors[code] for code in division_codes if code in colors), None)
        if color is None:
            color = palette[fallback_index % len(palette)] if division_codes else "#52656f"
        metadata.append(
            {
                "vectorRow": vector_row,
                "eventCode": session["event_code"],
                "eventYear": year,
                "sessionId": session["session_id"],
                "code": session.get("session_code"),
                "title": session.get("title"),
                "description": session.get("description"),
                "descriptionMissing": session["description_missing"],
                "descriptionVectorCount": int(selected_mask[vector_row].sum()),
                "divisions": session.get("discovery_groups", []),
                "divisionCodes": division_codes,
                "color": color,
                "url": session["source_url"],
            }
        )
    sessions_output = output_dir / "sessions.json"
    write_json(sessions_output, metadata)

    metric_output, metric_manifest = write_query_metric(
        metric=metric, output_dir=output_dir
    )
    generated = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "schemaVersion": 4,
        "catalogueYear": year,
        "catalogueVersion": f"EGU{str(year)[-2:]}-multivector-{generated[:10]}",
        "generatedAt": generated,
        "modelId": model_config["python_model_id"],
        "browserModelId": model_config["browser_model_id"],
        "browserModelPath": model_config.get("browser_model_path"),
        "modelRevision": model_config.get("revision", "main-unpinned"),
        "dimension": dimension,
        "maxTokens": int(model_config["max_tokens"]),
        "overlapTokens": int(embedding_report["overlap"]),
        "pooling": "mean",
        "normalized": True,
        "queryPrefix": model_config.get("query_prefix", ""),
        "vectorDtype": "float16-le",
        "sessionCount": len(metadata),
        "scoring": {
            "kind": "multivector",
            "denseWeight": 1.0,
            "bm25": {"enabled": False, "k1": 1.2, "b": 0.75},
            "multiVector": {
                "weights": weight_values,
                "queryChunks": int(embedding_report["query_chunks"]),
                "sessionChunks": session_chunks,
                "sessionVectorStride": stride,
            },
        },
        "confidenceBands": {
            "top_score_high": 1.0,
            "margin_high": 1.0,
            "top_score_low": -1.0,
        },
        "predictionSet": prediction_set,
        "metric": metric_manifest,
        "assets": {
            "sessions": {"path": sessions_output.name, "sha256": sha256_file(sessions_output)},
            "vectors": {"path": vector_output.name, "sha256": sha256_file(vector_output)},
            "queryMetric": {"path": metric_output.name, "sha256": sha256_file(metric_output)},
        },
    }
    browser_model_path = model_config.get("browser_model_path")
    if browser_model_path:
        model_directory = output_dir.parent / str(browser_model_path).lstrip("/")
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
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--baseline-embedding-dir", type=Path, required=True)
    parser.add_argument("--multivector-dir", type=Path, required=True)
    parser.add_argument("--metric-head", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--prediction-set", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=Path("config/models.json"))
    parser.add_argument("--model", default="minilm-l6")
    parser.add_argument(
        "--division-colors", type=Path, default=Path("config/division_colors.json")
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configs = json.loads(args.model_config.read_text(encoding="utf-8"))["models"]
    model = next((item for item in configs if item["name"] == args.model), None)
    if model is None:
        raise SystemExit(f"model not found: {args.model}")
    colors_value = json.loads(args.division_colors.read_text(encoding="utf-8"))
    manifest = export_multivector_catalogue(
        prepared_dir=args.prepared_dir,
        baseline_embedding_dir=args.baseline_embedding_dir,
        multivector_dir=args.multivector_dir,
        metric_head_path=args.metric_head,
        weights_path=args.weights,
        prediction_set_value=json.loads(args.prediction_set.read_text(encoding="utf-8")),
        model_config=model,
        colors=colors_value.get("colors", colors_value),
        year=args.year,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
