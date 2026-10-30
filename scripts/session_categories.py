"""Explicit EGU programme-category rules used by matcher evaluation."""

from __future__ import annotations

import re
from typing import Any


NONSTANDARD_SESSION_PREFIXES = frozenset(
    {
        "DM",   # Division meetings
        "GDB",  # Great debates
        "GP",   # Geo Panels (EGU27 successor of the great debates)
        "LRS",  # Named and public lectures
        "MAL",  # Medal and award lectures
        "NET",  # Networking and receptions
        "PC",   # Press conferences
        "SC",   # Short courses
        "SPM",  # Splinter meetings
        "TM",   # Town-hall meetings
        "UMI",  # Union administration and internal meetings
        "US",   # Union-wide special symposia
        "VPS",  # Broad virtual-poster programme buckets
    }
)


def session_code_prefix(code: Any) -> str:
    match = re.match(r"[A-Za-z]+", str(code or "").strip())
    return match.group(0).upper() if match else ""


def is_typical_scientific_session(session: dict[str, Any]) -> bool:
    """Return whether a programme entry is an ordinary scientific session."""
    return session_code_prefix(session.get("session_code")) not in NONSTANDARD_SESSION_PREFIXES
