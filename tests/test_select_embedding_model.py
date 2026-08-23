from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from select_embedding_model import quantized_acceptance, select_baseline  # noqa: E402


class SelectEmbeddingModelTests(unittest.TestCase):
    def test_selects_recall_then_mrr(self):
        def model(recall: float, mrr: float):
            metric = {"recall_at_10": recall, "mrr": mrr}
            return {
                "model": {"name": "model"},
                "confidence_bands": {},
                "variants": {
                    "dense_with_divisions": {"2024": metric},
                    "dense_without_divisions": {"2024": metric},
                    "hybrid": {"dense_weight": 0.75, "years": {"2024": metric}},
                },
            }

        result = select_baseline({"models": {"a": model(0.8, 0.5), "b": model(0.8, 0.6)}})
        self.assertEqual(result["selected"]["model"], "b")

    def test_quantized_model_must_stay_within_one_point(self):
        def benchmark(validation: float, test: float):
            return {
                "models": {
                    "selected": {
                        "variants": {
                            "dense_with_divisions": {
                                "2024": {"recall_at_10": validation},
                                "2025": {"recall_at_10": test},
                            }
                        }
                    }
                }
            }

        accepted = quantized_acceptance(
            benchmark(0.70, 0.72),
            benchmark(0.695, 0.71),
            model_name="selected",
            variant="dense_with_divisions",
        )
        self.assertTrue(accepted["accepted"])
        rejected = quantized_acceptance(
            benchmark(0.70, 0.72),
            benchmark(0.68, 0.71),
            model_name="selected",
            variant="dense_with_divisions",
        )
        self.assertFalse(rejected["accepted"])


if __name__ == "__main__":
    unittest.main()
