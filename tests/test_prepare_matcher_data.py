from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from matcher_common import iter_jsonl, session_text  # noqa: E402
from prepare_matcher_data import SPLITS, prepare  # noqa: E402


def write_jsonl(path: Path, values: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value) + "\n")


class PrepareMatcherDataTests(unittest.TestCase):
    def test_sparse_public_years_are_not_supervised_training_data(self):
        self.assertEqual(SPLITS[2020], "auxiliary_unlabeled")
        self.assertEqual(SPLITS[2021], "auxiliary_unlabeled")
        self.assertEqual(SPLITS[2022], "train")
        self.assertEqual(SPLITS[2023], "train")

    def test_joins_multi_session_labels_and_reports_unmatched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sessions = root / "sessions.jsonl.gz"
            crossref = root / "crossref.jsonl.gz"
            output = root / "out"
            write_jsonl(
                sessions,
                [
                    {
                        "event_code": "EGU24",
                        "event_year": 2024,
                        "session_id": "1",
                        "session_code": "HS1",
                        "title": "Water",
                        "description": "Hydrology",
                        "abstract_codes": ["EGU24-10"],
                        "discovery_groups": [
                            "Atmospheric Sciences (AS)",
                            "Hydrological Sciences (HS)",
                            "Nonlinear Processes in Geosciences (NP)",
                        ],
                        "co_organizer_label": "AS1/NP3",
                        "co_organizer_codes": ["AS", "NP"],
                    },
                    {
                        "event_code": "EGU24",
                        "event_year": 2024,
                        "session_id": "2",
                        "session_code": "NH1",
                        "title": "Floods",
                        "description": None,
                        "abstract_codes": ["EGU24-10", "EGU24-999"],
                        "discovery_groups": ["Natural Hazards (NH)"],
                    },
                ],
            )
            write_jsonl(
                crossref,
                [
                    {
                        "doi": "10.5194/egusphere-egu24-10",
                        "event_code": "EGU24",
                        "event_year": 2024,
                        "abstract_id": "10",
                        "title": "River flood",
                        "abstract_text": "Flood model",
                    },
                    {
                        "doi": "10.5194/egusphere-egu24-11",
                        "event_code": "EGU24",
                        "event_year": 2024,
                        "abstract_id": "11",
                        "title": "Unassigned",
                        "abstract_text": "Text",
                    },
                ],
            )
            report = prepare(
                crossref_paths=[crossref],
                sessions_path=sessions,
                output_dir=output,
                review_size=1,
            )
            abstracts = list(iter_jsonl(output / "abstracts.jsonl.gz"))
            assignments = list(iter_jsonl(output / "assignments.jsonl.gz"))
            prepared_sessions = list(iter_jsonl(output / "sessions.jsonl.gz"))
            self.assertEqual(abstracts[0]["session_keys"], ["EGU24/1", "EGU24/2"])
            self.assertEqual(len(assignments), 2)
            self.assertEqual(report["totals"]["unmatched_abstracts"], 1)
            self.assertEqual(report["totals"]["abstracts_with_multiple_sessions"], 1)
            self.assertEqual(report["totals"]["programme_abstract_codes_not_in_crossref"], 1)
            self.assertEqual(prepared_sessions[0]["division_codes"], ["AS", "HS", "NP"])
            self.assertEqual(prepared_sessions[0]["primary_division_code"], "HS")
            self.assertEqual(prepared_sessions[0]["visual_division_codes"], ["HS", "AS", "NP"])
            self.assertEqual(prepared_sessions[0]["co_organizer_label"], "AS1/NP3")
            self.assertTrue(prepared_sessions[1]["description_missing"])

    def test_session_text_never_contains_accepted_abstracts(self):
        value = session_text(
            {
                "session_code": "AS1",
                "title": "Clouds",
                "description": "Cloud processes",
                "discovery_groups": ["Atmospheric Sciences (AS)"],
                "abstract_codes": ["EGU26-123"],
            }
        )
        self.assertNotIn("EGU26-123", value)
        self.assertIn("Cloud processes", value)


if __name__ == "__main__":
    unittest.main()
