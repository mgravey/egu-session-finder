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

from export_browser_catalogue import (  # noqa: E402
    export_catalogue,
    transform_query_vectors,
    transform_session_vectors,
)


class ExportBrowserCatalogueTests(unittest.TestCase):
    def test_query_and_session_transformations_can_be_asymmetric(self):
        metric = {
            "kind": "lowrank",
            "dimension": 2,
            "rank": 1,
            "log_weight": np.zeros(2, dtype=np.float32),
            "query_down": np.asarray([[1.0], [0.0]], dtype=np.float32),
            "query_up": np.asarray([[0.0, 1.0]], dtype=np.float32),
            "session_down": np.asarray([[0.0], [1.0]], dtype=np.float32),
            "session_up": np.asarray([[1.0, 0.0]], dtype=np.float32),
        }
        query = transform_query_vectors(np.asarray([[1.0, 0.0]]), metric)
        session = transform_session_vectors(np.asarray([[0.0, 1.0]]), metric)
        np.testing.assert_allclose(query, [[2 ** -0.5, 2 ** -0.5]], atol=1e-6)
        np.testing.assert_allclose(session, [[2 ** -0.5, 2 ** -0.5]], atol=1e-6)
    def test_exports_ordered_metadata_and_little_endian_vectors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared"
            prepared.mkdir()
            session = {
                "event_code": "EGU26",
                "event_year": 2026,
                "session_id": "1",
                "session_code": "HS1",
                "title": "Water",
                "description": "Hydrology",
                "description_missing": False,
                "discovery_groups": ["Hydrological Sciences (HS)"],
                "division_codes": ["HS"],
                "source_url": "https://example.test/session/1",
                "session_text": "Session HS1: Water",
            }
            with gzip.open(prepared / "sessions.jsonl.gz", "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(session) + "\n")
            vector_path = root / "vectors.npy"
            np.save(vector_path, np.array([[0.6, 0.8]], dtype=np.float32))
            output = root / "out"
            manifest = export_catalogue(
                prepared_dir=prepared,
                vectors_path=vector_path,
                model_config={
                    "python_model_id": "test/model",
                    "browser_model_id": "test/model-onnx",
                    "max_tokens": 512,
                },
                confidence_bands={
                    "top_score_high": 0.8,
                    "margin_high": 0.1,
                    "top_score_low": 0.3,
                },
                colors={"HS": "#123456"},
                year=2026,
                output_dir=output,
                dense_weight=1.0,
                prediction_set={
                    "schemaVersion": 1,
                    "method": "top-score-gap",
                    "score": "fixture",
                    "candidatePolicy": "typical-scientific-only",
                    "calibrationYear": 2024,
                    "calibrationExamples": 100,
                    "targetCoverage": 0.8,
                    "maxScoreGap": 0.12,
                    "minResults": 3,
                    "maxResults": 20,
                    "evaluations": {"2024": {}},
                },
            )
            metadata = json.loads((output / "sessions.json").read_text())
            vector = np.fromfile(output / "session-vectors.bin", dtype="<f4")
            self.assertEqual(manifest["sessionCount"], 1)
            self.assertEqual(len(manifest["assets"]["sessions"]["sha256"]), 64)
            self.assertEqual(len(manifest["assets"]["vectors"]["sha256"]), 64)
            self.assertEqual(metadata[0]["color"], "#123456")
            self.assertEqual(manifest["schemaVersion"], 4)
            self.assertEqual(manifest["predictionSet"]["maxScoreGap"], 0.12)
            self.assertNotIn("evaluations", manifest["predictionSet"])
            np.testing.assert_allclose(vector, [0.6, 0.8])

    def test_includes_local_browser_model_checksums_when_available(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared"
            prepared.mkdir()
            public = root / "public"
            model_dir = public / "models" / "fixture"
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            session = {
                "event_code": "EGU26",
                "event_year": 2026,
                "session_id": "1",
                "session_code": "HS1",
                "title": "Water",
                "description": "Hydrology",
                "description_missing": False,
                "discovery_groups": ["Hydrological Sciences (HS)"],
                "division_codes": ["HS"],
                "source_url": "https://example.test/session/1",
                "session_text": "Session HS1: Water",
            }
            with gzip.open(prepared / "sessions.jsonl.gz", "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(session) + "\n")
            vector_path = root / "vectors.npy"
            np.save(vector_path, np.array([[1.0, 0.0]], dtype=np.float32))
            manifest = export_catalogue(
                prepared_dir=prepared,
                vectors_path=vector_path,
                model_config={
                    "python_model_id": "fixture",
                    "browser_model_id": "fixture",
                    "browser_model_path": "/models/fixture",
                    "max_tokens": 8,
                },
                confidence_bands={
                    "top_score_high": 0.8,
                    "margin_high": 0.1,
                    "top_score_low": 0.2,
                },
                colors={},
                year=2026,
                output_dir=public / "catalogue",
                dense_weight=1.0,
            )
            model_asset = manifest["assets"]["model"][0]
            self.assertEqual(model_asset["path"], "models/fixture/config.json")
            self.assertEqual(len(model_asset["sha256"]), 64)

    def test_exports_query_metric_and_pretransforms_sessions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared"
            prepared.mkdir()
            session = {
                "event_code": "EGU26",
                "event_year": 2026,
                "session_id": "1",
                "session_code": "HS1",
                "title": "Water",
                "description": "Hydrology",
                "description_missing": False,
                "discovery_groups": ["Hydrological Sciences (HS)"],
                "division_codes": ["HS"],
                "source_url": "https://example.test/session/1",
                "session_text": "Session HS1: Water",
            }
            with gzip.open(prepared / "sessions.jsonl.gz", "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(session) + "\n")
            vectors = root / "vectors.npy"
            np.save(vectors, np.asarray([[1.0, 0.0]], dtype=np.float32))
            metric_path = root / "metric.npz"
            np.savez_compressed(
                metric_path,
                kind=np.asarray("lowrank"),
                dimension=np.asarray(2, dtype=np.int32),
                rank=np.asarray(1, dtype=np.int32),
                log_weight=np.zeros(2, dtype=np.float32),
                query_down=np.asarray([[1.0], [0.0]], dtype=np.float32),
                query_up=np.asarray([[0.0, 1.0]], dtype=np.float32),
                session_down=np.asarray([[1.0], [0.0]], dtype=np.float32),
                session_up=np.asarray([[0.0, 1.0]], dtype=np.float32),
            )
            output = root / "out"
            manifest = export_catalogue(
                prepared_dir=prepared,
                vectors_path=vectors,
                model_config={
                    "python_model_id": "fixture",
                    "browser_model_id": "fixture",
                    "max_tokens": 8,
                },
                confidence_bands={
                    "top_score_high": 0.8,
                    "margin_high": 0.1,
                    "top_score_low": 0.2,
                },
                colors={},
                year=2026,
                output_dir=output,
                dense_weight=1.0,
                metric_head_path=metric_path,
            )
            transformed = np.fromfile(output / "session-vectors.bin", dtype="<f4")
            np.testing.assert_allclose(
                transformed, [2 ** -0.5, 2 ** -0.5], atol=1e-6
            )
            self.assertEqual(manifest["schemaVersion"], 3)
            self.assertEqual(manifest["metric"]["kind"], "lowrank")
            self.assertEqual(manifest["assets"]["queryMetric"]["path"], "query-metric.bin")
            self.assertEqual(
                (output / "query-metric.bin").stat().st_size,
                (2 + 2 + 2) * 4,
            )


if __name__ == "__main__":
    unittest.main()
