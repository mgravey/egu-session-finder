from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from benchmark_embeddings import limited_evaluation_sample  # noqa: E402


class BenchmarkSamplingTests(unittest.TestCase):
    def test_limited_sample_is_balanced_across_requested_years(self):
        abstracts = [
            {
                "abstract_key": f"EGU{year}-{index}",
                "event_year": year,
                "has_abstract": True,
                "session_keys": [f"EGU{year}/1"],
            }
            for year in (2024, 2025, 2026)
            for index in range(10)
        ]
        selected = limited_evaluation_sample(abstracts, [2024, 2025, 2026], 7)
        counts = {
            year: sum(item["event_year"] == year for item in selected)
            for year in (2024, 2025, 2026)
        }
        self.assertEqual(counts, {2024: 3, 2025: 2, 2026: 2})

    def test_limited_sample_ignores_unmatched_rows(self):
        values = [
            {"event_year": 2024, "has_abstract": True, "session_keys": []},
            {"event_year": 2024, "has_abstract": False, "session_keys": ["EGU24/1"]},
        ]
        self.assertEqual(limited_evaluation_sample(values, [2024], 10), [])


if __name__ == "__main__":
    unittest.main()
