#!/usr/bin/env python3
"""Export an accepted local Sentence Transformer to quantized ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from matcher_common import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--quantization",
        choices=("arm64", "avx2", "avx512", "avx512_vnni"),
        default="arm64",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.backend import export_dynamic_quantized_onnx_model
    except ImportError as exc:
        raise SystemExit("Install requirements-ml.txt in a Python 3.12 environment") from exc
    model = SentenceTransformer(str(args.model), backend="onnx", device="cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_dynamic_quantized_onnx_model(
        model,
        args.quantization,
        str(args.output_dir),
        file_suffix="quantized",
    )
    onnx_files = sorted(str(path.relative_to(args.output_dir)) for path in args.output_dir.rglob("*.onnx"))
    if not onnx_files:
        raise RuntimeError("ONNX export completed without producing an ONNX file")
    manifest = {
        "source_model": str(args.model),
        "quantization": args.quantization,
        "onnx_files": onnx_files,
    }
    write_json(args.output_dir / "egu_onnx_export.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
