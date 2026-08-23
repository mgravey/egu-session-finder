import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from calibrate_prediction_sets import (  # noqa: E402
    conformal_higher_quantile,
    dynamic_set_size,
    evaluate_year,
)


class PredictionSetCalibrationTests(unittest.TestCase):
    def test_finite_sample_higher_quantile(self):
        self.assertEqual(conformal_higher_quantile([0.0, 0.1, 0.2, 0.3], 0.8), 0.3)
        with self.assertRaises(ValueError):
            conformal_higher_quantile([], 0.8)

    def test_dynamic_size_uses_top_gap_and_minimum(self):
        scores = np.asarray([0.9, 0.85, 0.7, 0.1], dtype=np.float32)
        self.assertEqual(dynamic_set_size(scores, 0.1, 1), 2)
        self.assertEqual(dynamic_set_size(scores, 0.1, 3), 3)

    def test_evaluation_reports_uncapped_and_display_capped_coverage(self):
        examples = [
            (np.asarray([1.0, 0.95, 0.7]), [1]),
            (np.asarray([1.0, 0.99, 0.98]), [2]),
        ]
        report = evaluate_year(
            examples,
            max_score_gap=0.1,
            min_results=1,
            max_results=2,
        )
        self.assertEqual(report["empiricalCoverage"], 1.0)
        self.assertEqual(report["displayCappedCoverage"], 0.5)
        self.assertEqual(report["medianSetSize"], 2.5)
        self.assertEqual(report["fractionTruncatedAtDisplayCap"], 0.5)


if __name__ == "__main__":
    unittest.main()
