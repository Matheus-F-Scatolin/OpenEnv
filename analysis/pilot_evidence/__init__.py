# SPDX-License-Identifier: BSD-3-Clause

"""
What the pilot's stored logs show, as one text.

  python -m analysis.pilot_evidence <pilot folder> [--out <file>]

Each section comes from one module of this package and starts with an empty
line and a header such as `===== replay.py =====`.
"""

from __future__ import annotations

from pathlib import Path


def evidence(pilot_dir: Path) -> str:
    """
    Recompute every section of the evidence from the pilot's logs.

    Args:
        pilot_dir (`Path`):
            The pilot's folder, `rfc-013/pilot`.

    Returns:
        `str`: The sections in order: replay, analyze, calib, alloc, rates,
        repro, overlap.
    """
    # Imported here, so that `python -m` on one section finds it not yet imported.
    from analysis.pilot_evidence import (
        alloc,
        analyze,
        calib,
        overlap,
        rates,
        replay,
        repro,
    )

    sections = {
        "replay.py": replay.text(pilot_dir),
        "analyze.py": analyze.text(pilot_dir),
        "calib.py": calib.text(pilot_dir),
        "alloc.py": alloc.text(),
        "rates.py": rates.text(pilot_dir),
        "repro.py": repro.text(pilot_dir),
        "overlap.py": overlap.text(),
    }
    return "".join(f"\n===== {name} =====\n{text}" for name, text in sections.items())


__all__ = ["evidence"]
