from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from crossref_prefix_scan import ScanError, scan_file  # noqa: E402


class CrossrefPrefixScanTests(unittest.TestCase):
    def write_gzip(self, root: Path, name: str, lines: list[object | str]) -> Path:
        path = root / name
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            for line in lines:
                if isinstance(line, str):
                    stream.write(line)
                else:
                    stream.write(json.dumps(line))
                stream.write("\n")
        return path

    def scan(self, input_path: Path, root: Path, mode: str = "preview"):
        output = io.BytesIO()
        summary = scan_file(
            input_path,
            dump_root=root,
            doi_prefix="10.5194/",
            mode=mode,
            on_json_error="fail",
            output=output,
        )
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        return summary, records

    def test_matches_normalized_dois_and_tracks_physical_lines(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self.write_gzip(
                root,
                "1.json.gz",
                [
                    {"DOI": "10.1000/not-copernicus", "title": ["No"]},
                    "",
                    {
                        "DOI": "https://doi.org/10.5194/EGUSPHERE-EGU26-3285",
                        "type": "proceedings-article",
                        "publisher": "Copernicus GmbH",
                        "title": ["An EGU abstract"],
                        "container-title": ["EGU General Assembly 2026"],
                    },
                ],
            )

            summary, records = self.scan(source, root)

            self.assertEqual(summary["lines"], 3)
            self.assertEqual(summary["records"], 2)
            self.assertEqual(summary["matches"], 1)
            self.assertEqual(records[0]["source_file"], "1.json.gz")
            self.assertEqual(records[0]["source_line"], 3)
            self.assertEqual(records[0]["item_index"], 0)
            self.assertEqual(
                records[0]["doi"], "10.5194/egusphere-egu26-3285"
            )
            self.assertEqual(records[0]["title"], "An EGU abstract")

    def test_supports_crossref_items_envelopes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self.write_gzip(
                root,
                "2.json.gz",
                [
                    {
                        "message": {
                            "items": [
                                {"DOI": "10.5194/one", "title": ["One"]},
                                {"DOI": "10.2000/two", "title": ["Two"]},
                                {"DOI": "doi:10.5194/three", "title": ["Three"]},
                            ]
                        }
                    }
                ],
            )

            summary, records = self.scan(source, root)

            self.assertEqual(summary["records"], 3)
            self.assertEqual(summary["matches"], 2)
            self.assertEqual([record["item_index"] for record in records], [0, 2])
            self.assertEqual(
                [record["doi"] for record in records],
                ["10.5194/one", "10.5194/three"],
            )

    def test_full_mode_preserves_original_record(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            original = {"DOI": "10.5194/example", "custom": {"value": 7}}
            source = self.write_gzip(root, "3.json.gz", [original])

            _, records = self.scan(source, root, mode="full")

            self.assertEqual(records[0]["record"], original)

    def test_invalid_json_fails_with_source_line(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self.write_gzip(root, "broken.json.gz", ["not-json"])

            with self.assertRaisesRegex(ScanError, r"broken\.json\.gz:1"):
                self.scan(source, root)

    @unittest.skipUnless(shutil.which("parallel"), "GNU Parallel is not installed")
    def test_parallel_runner_combines_workers(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dump_root = root / "dump"
            dump_root.mkdir()
            self.write_gzip(
                dump_root,
                "10.jsonl.gz",
                [{"DOI": "10.5194/first", "title": ["First"]}],
            )
            self.write_gzip(
                dump_root,
                "20.jsonl.gz",
                [
                    {"DOI": "10.1000/ignored", "title": ["Ignored"]},
                    {"DOI": "10.5194/second", "title": ["Second"]},
                ],
            )
            output = root / "matches.jsonl.gz"
            runner = PROJECT_ROOT / "scripts" / "run_crossref_prefix_scan.sh"

            subprocess.run(
                [str(runner), str(dump_root), str(output), "2", "preview"],
                check=True,
                capture_output=True,
                text=True,
            )

            with gzip.open(output, "rt", encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream]
            self.assertEqual(
                {record["doi"] for record in records},
                {"10.5194/first", "10.5194/second"},
            )
            self.assertTrue(Path(f"{output}.joblog.tsv").is_file())
            self.assertTrue(Path(f"{output}.summaries.jsonl").is_file())

    @unittest.skipUnless(shutil.which("parallel"), "GNU Parallel is not installed")
    def test_parallel_runner_rejects_an_empty_dump(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dump_root = root / "empty-dump"
            dump_root.mkdir()
            output = root / "matches.jsonl.gz"
            runner = PROJECT_ROOT / "scripts" / "run_crossref_prefix_scan.sh"

            result = subprocess.run(
                [str(runner), str(dump_root), str(output), "2", "preview"],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("No files matching *.jsonl.gz or *.json.gz", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
