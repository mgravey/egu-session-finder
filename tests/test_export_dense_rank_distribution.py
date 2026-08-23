import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_dense_rank_distribution import (  # noqa: E402
    dense_rank_distribution,
    rank_relevant_candidates,
)


class DenseRankDistributionTests(unittest.TestCase):
    def test_relevant_rank_respects_stable_ties(self):
        scores = np.asarray([0.8, 0.9, 0.8, 0.2], dtype=np.float32)
        self.assertEqual(rank_relevant_candidates(scores, [2]), 3)
        self.assertEqual(rank_relevant_candidates(scores, [0, 2]), 2)
        self.assertIsNone(rank_relevant_candidates(scores, []))
        self.assertEqual(rank_relevant_candidates(scores, np.asarray([2])), 3)

    def test_distribution_counts_best_valid_assignment(self):
        sessions = [
            {"event_year": 2024, "session_key": "s1"},
            {"event_year": 2024, "session_key": "s2"},
            {"event_year": 2024, "session_key": "s3"},
        ]
        abstracts = [
            {
                "event_year": 2024,
                "has_abstract": True,
                "session_keys": ["s2", "s3"],
            },
            {"event_year": 2024, "has_abstract": True, "session_keys": ["s1"]},
        ]
        session_vectors = np.eye(3, dtype=np.float32)
        abstract_vectors = np.asarray(
            [
                [0.9, 0.8, 0.7],
                [0.1, 0.8, 0.9],
            ],
            dtype=np.float32,
        )
        report = dense_rank_distribution(
            sessions=sessions,
            abstracts=abstracts,
            session_vectors=session_vectors,
            abstract_vectors=abstract_vectors,
            years=[2024],
            block_size=2,
        )
        curve = report["years"]["2024"]["curve"]
        self.assertEqual([point["exact_count"] for point in curve], [0, 1, 1])
        self.assertEqual([point["cumulative_count"] for point in curve], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
