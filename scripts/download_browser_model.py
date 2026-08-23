#!/usr/bin/env python3
"""Download only the files needed for one quantized Transformers.js model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Xenova/bge-small-en-v1.5")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--name", default="bge-small-en-v1.5")
    parser.add_argument("--output-root", type=Path, default=Path("web/public/models"))
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from huggingface_hub import model_info, snapshot_download
    except ImportError as exc:
        raise SystemExit("Install requirements-ml.txt in a Python 3.12 environment") from exc
    destination = args.output_root / args.name
    info = model_info(args.model_id, revision=args.revision)
    downloaded = snapshot_download(
        repo_id=args.model_id,
        revision=info.sha,
        local_dir=destination,
        allow_patterns=[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "modules.json",
            "sentence_bert_config.json",
            "config_sentence_transformers.json",
            "1_Pooling/*",
            "onnx/model_quantized.onnx*",
        ],
    )
    destination = Path(downloaded)
    required = [
        destination / "config.json",
        destination / "tokenizer.json",
        destination / "onnx" / "model_quantized.onnx",
    ]
    missing = [str(path.relative_to(destination)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "browser model snapshot lacks required Transformers.js assets: "
            + ", ".join(missing)
        )
    result = {
        "model_id": args.model_id,
        "revision": info.sha,
        "path": str(destination),
    }
    (destination / "egu_model_source.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
