from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_multivector_browser_catalogue import export_multivector_catalogue  # noqa: E402


class MultiVectorBrowserExportTests(unittest.TestCase):
    def test_exports_packed_typical_session_vectors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared"
            baseline = root / "baseline"
            multivector = root / "multi"
            output = root / "public" / "catalogue"
            for directory in (prepared, baseline, multivector, output.parent / "models" / "mini"):
                directory.mkdir(parents=True, exist_ok=True)
            sessions = [
                {
                    "event_code": "EGU26", "event_year": 2026, "session_id": "1",
                    "session_key": "EGU26:1", "session_code": "HS1", "title": "Water",
                    "description": "Hydrology", "description_missing": False,
                    "discovery_groups": ["Hydrological Sciences (HS)"],
                    "division_codes": ["HS"], "source_url": "https://example.test/1",
                },
                {
                    "event_code": "EGU26", "event_year": 2026, "session_id": "2",
                    "session_key": "EGU26:2", "session_code": "SC1", "title": "Short course",
                    "description": "Training", "description_missing": False,
                    "discovery_groups": ["Short Courses (SC)"], "division_codes": ["SC"],
                    "source_url": "https://example.test/2",
                },
            ]
            with gzip.open(prepared / "sessions.jsonl.gz", "wt", encoding="utf-8") as stream:
                for session in sessions:
                    stream.write(json.dumps(session) + "\n")
            np.save(baseline / "sessions_with_divisions.npy", np.eye(2, dtype=np.float32))
            np.save(multivector / "session_title.npy", np.eye(2, dtype=np.float32)[:, None, :])
            np.save(
                multivector / "session_description.npy",
                np.asarray([[[1, 0], [0, 0]], [[0, 1], [0, 0]]], dtype=np.float32),
            )
            np.save(
                multivector / "session_description_mask.npy",
                np.asarray([[True, False], [True, False]]),
            )
            (multivector / "multivector-embedding-report.json").write_text(
                json.dumps({"overlap": 1, "query_chunks": 2}), encoding="utf-8"
            )
            metric = root / "metric.npz"
            np.savez_compressed(
                metric,
                kind=np.asarray("diagonal"), dimension=np.asarray(2), rank=np.asarray(1),
                log_weight=np.zeros(2, dtype=np.float32),
            )
            weights = root / "weights.json"
            weights.write_text(
                json.dumps({
                    "joint": 0.6, "title_to_title": 0.1,
                    "title_to_description": 0.1, "chunks_to_title": 0.1,
                    "chunks_to_description": 0.1,
                }),
                encoding="utf-8",
            )
            manifest = export_multivector_catalogue(
                prepared_dir=prepared,
                baseline_embedding_dir=baseline,
                multivector_dir=multivector,
                metric_head_path=metric,
                weights_path=weights,
                prediction_set_value={
                    "method": "top-score-gap", "score": "learned_metric_plus_multivector",
                    "candidatePolicy": "typical-scientific-only", "calibrationYear": 2024,
                    "calibrationExamples": 10, "targetCoverage": 0.8, "maxScoreGap": 0.1,
                    "minResults": 1, "maxResults": 2,
                },
                model_config={
                    "python_model_id": "fixture", "browser_model_id": "fixture",
                    "browser_model_path": "/models/mini", "revision": "abc", "max_tokens": 8,
                },
                colors={"HS": "#123456"}, year=2026, output_dir=output,
            )
            metadata = json.loads((output / "sessions.json").read_text())
            self.assertEqual(manifest["schemaVersion"], 4)
            self.assertEqual(manifest["sessionCount"], 1)
            self.assertEqual(manifest["scoring"]["multiVector"]["sessionVectorStride"], 4)
            self.assertEqual(metadata[0]["descriptionVectorCount"], 1)
            self.assertEqual(metadata[0]["color"], "#123456")
            self.assertEqual(manifest["vectorDtype"], "float16-le")
            self.assertEqual((output / "session-vectors.bin").stat().st_size, 4 * 2 * 2)


if __name__ == "__main__":
    unittest.main()
