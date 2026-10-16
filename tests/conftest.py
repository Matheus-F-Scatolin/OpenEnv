"""Shared test setup: make the frozen pilot importable for parity tests."""

from __future__ import annotations

import sys
from pathlib import Path

PILOT_DIR = Path(__file__).resolve().parent.parent / "pilot"

if str(PILOT_DIR) not in sys.path:
    sys.path.insert(0, str(PILOT_DIR))
