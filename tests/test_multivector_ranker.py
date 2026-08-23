from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_multivector_embeddings import fixed_chunks  # noqa: E402
from train_multivector_ranker import score_features  # noqa: E402
from export_multivector_score_matrices import ordered_weights  # noqa: E402


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split())))}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(value) for value in ids)


class MultiVectorTests(unittest.TestCase):
    def test_ordered_weights_uses_feature_order(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            path.write_text(
                json.dumps(
                    {
                        "chunks_to_description": 0.2,
                        "joint": 0.2,
                        "chunks_to_title": 0.2,
                        "title_to_description": 0.2,
                        "title_to_title": 0.2,
                    }
                )
            )
            np.testing.assert_allclose(ordered_weights(path), np.full(5, 0.2))

    def test_fixed_chunks_respects_limit(self) -> None:
        chunks = fixed_chunks(
            FakeTokenizer(),
            " ".join(f"word{index}" for index in range(30)),
            max_tokens=10,
            overlap=2,
            limit=2,
        )
        self.assertEqual(len(chunks), 2)

    def test_feature_shape_and_missing_description_fallback(self) -> None:
        query_joint = torch.tensor([[1.0, 0.0]])
        query_title = torch.tensor([[1.0, 0.0]])
        query_chunks = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
        query_mask = torch.tensor([[True, True]])
        session_joint = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        session_title = session_joint.clone()
        descriptions = torch.zeros((2, 1, 2))
        description_mask = torch.zeros((2, 1), dtype=torch.bool)
        features = score_features(
            query_joint=query_joint,
            query_title=query_title,
            query_chunks=query_chunks,
            query_chunk_mask=query_mask,
            session_joint=session_joint,
            session_title=session_title,
            session_description=descriptions,
            session_description_mask=description_mask,
        )
        self.assertEqual(tuple(features.shape), (1, 2, 5))
        np.testing.assert_allclose(features[0, :, 2].numpy(), [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
