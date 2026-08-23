from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from finetune_embedder import select_hard_negative  # noqa: E402


class FinetuneEmbedderTests(unittest.TestCase):
    def test_hard_negative_prefers_same_division_and_excludes_positive(self):
        sessions = [
            {"session_key": "E/1", "event_year": 2020, "division_codes": ["HS"]},
            {"session_key": "E/2", "event_year": 2020, "division_codes": ["HS"]},
            {"session_key": "E/3", "event_year": 2020, "division_codes": ["AS"]},
        ]
        vectors = np.array([[1.0, 0], [0.8, 0.2], [0.99, 0.01]], dtype=np.float32)
        selected = select_hard_negative(
            query_vector=np.array([1.0, 0], dtype=np.float32),
            sessions=sessions,
            session_vectors=vectors,
            valid_session_keys={"E/1"},
            year=2020,
            positive_divisions={"HS"},
        )
        self.assertEqual(selected, 1)


if __name__ == "__main__":
    unittest.main()
