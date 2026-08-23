import json
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_dense_score_matrices import export_score_matrices  # noqa: E402
from export_dense_rank_distribution import rank_distribution_from_matrices  # noqa: E402


class DenseScoreMatrixTests(unittest.TestCase):
    def test_export_and_rank_from_complete_matrix(self):
        sessions = [
            {"event_year": 2024, "session_key": "s1"},
            {"event_year": 2024, "session_key": "s2"},
        ]
        abstracts = [
            {
                "event_year": 2024,
                "abstract_key": "a1",
                "has_abstract": True,
                "session_keys": ["s2"],
            },
            {
                "event_year": 2024,
                "abstract_key": "a2",
                "has_abstract": True,
                "session_keys": ["s1"],
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            export_score_matrices(
                sessions=sessions,
                abstracts=abstracts,
                session_vectors=np.eye(2, dtype=np.float32),
                abstract_vectors=np.asarray([[0.8, 0.9], [0.1, 0.8]], dtype=np.float32),
                years=[2024],
                output_dir=output,
                block_size=1,
            )
            matrix = np.load(output / "cosine-2024.npy")
            np.testing.assert_allclose(matrix, [[0.8, 0.9], [0.1, 0.8]])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["years"]["2024"]["shape"], [2, 2])

            ranks = rank_distribution_from_matrices(
                abstracts=abstracts, matrix_dir=output, years=[2024]
            )
            curve = ranks["years"]["2024"]["curve"]
            self.assertEqual([point["cumulative_count"] for point in curve], [1, 2])

    def test_rank_filter_removes_special_candidates_and_abstracts(self):
        sessions = [
            {"event_year": 2024, "session_key": "science", "session_code": "HS1"},
            {"event_year": 2024, "session_key": "medal", "session_code": "MAL1"},
        ]
        abstracts = [
            {
                "event_year": 2024,
                "abstract_key": "regular",
                "has_abstract": True,
                "session_keys": ["science"],
            },
            {
                "event_year": 2024,
                "abstract_key": "special",
                "has_abstract": True,
                "session_keys": ["medal"],
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            export_score_matrices(
                sessions=sessions,
                abstracts=abstracts,
                session_vectors=np.eye(2, dtype=np.float32),
                abstract_vectors=np.asarray([[0.8, 0.9], [0.1, 0.8]], dtype=np.float32),
                years=[2024],
                output_dir=output,
            )
            ranks = rank_distribution_from_matrices(
                abstracts=abstracts,
                sessions=sessions,
                matrix_dir=output,
                years=[2024],
                typical_sessions_only=True,
            )["years"]["2024"]
            self.assertEqual(ranks["candidate_sessions"], 1)
            self.assertEqual(ranks["excluded_nonstandard_abstracts"], 1)
            self.assertEqual(ranks["denominator_rows"], 1)
            self.assertEqual(ranks["curve"][0]["cumulative_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
