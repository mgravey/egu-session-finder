import sys
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_embedding_metric import (  # noqa: E402
    build_year_training_data,
    make_metric_head,
    multi_positive_listwise_loss,
    validation_confidence_bands,
)


class MetricLearningTests(unittest.TestCase):
    def test_year_data_preserves_multiple_positive_sessions(self):
        sessions = [
            {"event_year": 2024, "session_key": "s1"},
            {"event_year": 2024, "session_key": "s2"},
            {"event_year": 2025, "session_key": "s3"},
        ]
        abstracts = [
            {
                "event_year": 2024,
                "has_abstract": True,
                "session_keys": ["s1", "s2"],
            },
            {"event_year": 2024, "has_abstract": True, "session_keys": []},
        ]
        data = build_year_training_data(
            sessions=sessions, abstracts=abstracts, year=2024
        )
        self.assertEqual(data.abstract_indices.tolist(), [0])
        self.assertEqual(data.session_indices.tolist(), [0, 1])
        self.assertEqual(data.positive_mask.tolist(), [[True, True]])

    def test_listwise_loss_rewards_positive_logit(self):
        mask = torch.tensor([[True, False, False]])
        good = multi_positive_listwise_loss(torch.tensor([[3.0, 0.0, -1.0]]), mask)
        bad = multi_positive_listwise_loss(torch.tensor([[0.0, 3.0, -1.0]]), mask)
        self.assertLess(float(good), float(bad))

    def test_metric_heads_are_identity_at_initialization(self):
        values = torch.nn.functional.normalize(torch.randn(5, 8), dim=1)
        for kind in ("diagonal", "lowrank"):
            head = make_metric_head(kind, dimension=8, rank=3)
            np.testing.assert_allclose(
                head.transform_query(values).detach().numpy(),
                values.numpy(),
                atol=1e-6,
            )
            np.testing.assert_allclose(
                head.transform_session(values).detach().numpy(),
                values.numpy(),
                atol=1e-6,
            )

    def test_confidence_bands_are_calibrated_from_transformed_scores(self):
        sessions = [
            {"event_year": 2024, "session_key": "s1"},
            {"event_year": 2024, "session_key": "s2"},
        ]
        abstracts = [
            {"event_year": 2024, "has_abstract": True, "session_keys": ["s1"]},
            {"event_year": 2024, "has_abstract": True, "session_keys": ["s2"]},
        ]
        data = build_year_training_data(
            sessions=sessions, abstracts=abstracts, year=2024
        )
        head = make_metric_head("diagonal", dimension=2, rank=1)
        bands = validation_confidence_bands(
            head=head,
            year_data=data,
            abstract_vectors=np.eye(2, dtype=np.float32),
            session_vectors=np.eye(2, dtype=np.float32),
            device="cpu",
        )
        self.assertEqual(set(bands), {"top_score_high", "margin_high", "top_score_low"})


if __name__ == "__main__":
    unittest.main()
