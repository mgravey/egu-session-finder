from __future__ import annotations

import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from extract_egu_records import (  # noqa: E402
    clean_abstract,
    normalize_egu_record,
    parse_egu_doi,
)
from fetch_crossref_egu_api import export_year  # noqa: E402


class ExtractEguRecordsTests(unittest.TestCase):
    def write_gzip_jsonl(self, path: Path, records: list[dict]) -> None:
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")

    def test_parses_two_and_four_digit_event_years(self):
        self.assertEqual(
            parse_egu_doi("10.5194/egusphere-egu2020-123"),
            ("EGU2020", 2020, "123"),
        )
        self.assertEqual(
            parse_egu_doi("https://doi.org/10.5194/egusphere-egu25-456"),
            ("EGU25", 2025, "456"),
        )
        self.assertIsNone(parse_egu_doi("10.5194/egusphere-egu25-456-rc1"))

    def test_normalizes_title_and_abstract_but_preserves_raw_record(self):
        original = {
            "DOI": "10.5194/egusphere-egu25-456",
            "type": "posted-content",
            "title": ["Example title"],
            "abstract": "<jats:p>Water &amp; ice</jats:p><jats:p>Second part.</jats:p>",
            "author": [{"given": "Ada", "family": "Lovelace"}],
        }

        normalized = normalize_egu_record(
            original,
            source_file="1.jsonl.gz",
            source_line=7,
            item_index=0,
        )

        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(normalized["title"], "Example title")
        self.assertEqual(normalized["abstract_text"], "Water & ice Second part.")
        self.assertTrue(normalized["has_abstract"])
        self.assertEqual(normalized["crossref_record"], original)

    def test_clean_abstract_handles_missing_values(self):
        self.assertIsNone(clean_abstract(None))
        self.assertIsNone(clean_abstract("  "))

    @unittest.skipUnless(shutil.which("parallel"), "GNU Parallel is not installed")
    def test_parallel_runner_extracts_only_egu_posted_content(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dump_root = root / "dump"
            dump_root.mkdir()
            source = dump_root / "100.jsonl.gz"
            records = [
                {
                    "DOI": "10.5194/egusphere-egu2020-1",
                    "type": "posted-content",
                    "title": ["Has abstract"],
                    "abstract": "<p>Full abstract</p>",
                },
                {
                    "DOI": "10.5194/acp-20-1-2020",
                    "type": "journal-article",
                    "title": ["Journal article"],
                },
                {
                    "DOI": "10.5194/egusphere-egu25-2",
                    "type": "posted-content",
                    "title": ["No abstract"],
                },
            ]
            self.write_gzip_jsonl(source, records)
            preview = root / "preview.jsonl.gz"
            self.write_gzip_jsonl(
                preview,
                [
                    {
                        "source_file": "100.jsonl.gz",
                        "source_line": 1,
                        "item_index": 0,
                        "doi": "10.5194/egusphere-egu2020-1",
                        "type": "posted-content",
                    },
                    {
                        "source_file": "100.jsonl.gz",
                        "source_line": 3,
                        "item_index": 0,
                        "doi": "10.5194/egusphere-egu25-2",
                        "type": "posted-content",
                    },
                ],
            )
            output = root / "egu.jsonl.gz"
            runner = PROJECT_ROOT / "scripts" / "run_extract_egu_records.sh"

            subprocess.run(
                [str(runner), str(dump_root), str(preview), str(output), "2"],
                check=True,
                capture_output=True,
                text=True,
            )

            with gzip.open(output, "rt", encoding="utf-8") as stream:
                extracted = [json.loads(line) for line in stream]
            self.assertEqual(len(extracted), 2)
            self.assertEqual(sum(item["has_abstract"] for item in extracted), 1)
            self.assertEqual(
                {item["event_year"] for item in extracted},
                {2020, 2025},
            )

    def test_exports_cached_api_pages_with_exact_egu_year_filter(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cache = root / "cache"
            cache.mkdir()
            page = {
                "status": "ok",
                "message": {
                    "items": [
                        {
                            "DOI": "10.5194/egusphere-egu26-3285",
                            "type": "posted-content",
                            "title": ["EGU26 abstract"],
                            "abstract": "<p>Abstract text</p>",
                        },
                        {
                            "DOI": "10.5194/egusphere-2026-100",
                            "type": "posted-content",
                            "title": ["Not EGU"],
                        },
                    ]
                },
            }
            with gzip.open(cache / "page_00001.json.gz", "wb") as stream:
                stream.write(json.dumps(page).encode("utf-8"))
            output = root / "egu26.jsonl.gz"

            summary = export_year(cache, output, 2026)

            self.assertEqual(summary["records"], 1)
            self.assertEqual(summary["with_abstract"], 1)
            with gzip.open(output, "rt", encoding="utf-8") as stream:
                record = json.loads(next(stream))
            self.assertEqual(record["doi"], "10.5194/egusphere-egu26-3285")


if __name__ == "__main__":
    unittest.main()
