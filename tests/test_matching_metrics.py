from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from matching_metrics import (  # noqa: E402
    BM25Index,
    confidence_label,
    fit_confidence_bands,
    hybrid_scores,
    retrieval_metrics,
)


class MatchingMetricsTests(unittest.TestCase):
    def test_bm25_prefers_matching_specialist_terms(self):
        index = BM25Index.build(["cloud aerosol", "river flood hydrology"])
        scores = index.scores("hydrology flood")
        self.assertGreater(scores[1], scores[0])

    def test_bm25_postings_match_reference_document_scan(self):
        documents = ["river flood flood model", "cloud aerosol", "river sediment"]
        index = BM25Index.build(documents)
        fast = index.scores("river flood")
        reference = np.zeros(len(documents), dtype=np.float32)
        count = len(documents)
        for term in {"river", "flood"}:
            frequency = index.document_frequency[term]
            inverse = np.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for row, terms in enumerate(index.frequencies):
                occurrences = terms.get(term, 0)
                if occurrences:
                    denominator = occurrences + index.k1 * (
                        1 - index.b + index.b * len(index.documents[row]) / index.average_length
                    )
                    reference[row] += inverse * occurrences * (index.k1 + 1) / denominator
        np.testing.assert_allclose(fast, reference, rtol=1e-6, atol=1e-6)

    def test_hybrid_normalizes_components(self):
        score = hybrid_scores(
            np.array([0.1, 0.2], dtype=np.float32),
            np.array([10.0, 0.0], dtype=np.float32),
            0.75,
        )
        self.assertGreater(score[1], score[0])

    def test_retrieval_metrics_include_missing_labels_in_denominator(self):
        metrics = retrieval_metrics([1, 4, None])
        self.assertEqual(metrics["queries"], 3)
        self.assertEqual(metrics["recall_at_1"], 0.333333)
        self.assertEqual(metrics["recall_at_5"], 0.666667)

    def test_confidence_bands_are_reusable(self):
        bands = fit_confidence_bands(
            [(0.9, 0.2, True), (0.8, 0.1, True), (0.4, 0.01, False)]
        )
        self.assertEqual(confidence_label(0.95, 0.3, bands), "high")
        self.assertEqual(confidence_label(0.1, 0.0, bands), "low")


if __name__ == "__main__":
    unittest.main()
