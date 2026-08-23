import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from session_categories import (  # noqa: E402
    is_typical_scientific_session,
    session_code_prefix,
)


class SessionCategoryTests(unittest.TestCase):
    def test_extracts_joint_and_special_prefixes(self):
        self.assertEqual(session_code_prefix("ITS4.18/CL0.1.12"), "ITS")
        self.assertEqual(session_code_prefix("MAL20-HS"), "MAL")
        self.assertEqual(session_code_prefix(" SC 5.1"), "SC")

    def test_keeps_scientific_division_and_interdisciplinary_sessions(self):
        for code in ("HS4.1", "ITS2.3/CL0.1.1", "EOS1.5"):
            self.assertTrue(is_typical_scientific_session({"session_code": code}))

    def test_excludes_nonstandard_programme_categories(self):
        for code in ("MAL20-HS", "SC2.1", "SPM12", "TM3", "GDB2", "VPS4"):
            self.assertFalse(is_typical_scientific_session({"session_code": code}))
