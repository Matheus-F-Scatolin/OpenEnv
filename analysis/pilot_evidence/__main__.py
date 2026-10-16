# SPDX-License-Identifier: BSD-3-Clause

"""
Print what the pilot's stored logs show, or write it to a file.

  python -m analysis.pilot_evidence <pilot folder> [--out <file>]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from analysis.pilot_evidence import evidence


def main(argv: list[str] | None = None) -> None:
    """
    Recompute the evidence from the command line.

    Args:
        argv (`list[str]`, *optional*):
            The arguments; `sys.argv[1:]` when left out.
    """
    parser = argparse.ArgumentParser(
        prog="python -m analysis.pilot_evidence",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pilot_dir", type=Path)
    parser.add_argument("--out", type=Path, help="Write here instead of printing.")
    args = parser.parse_args(argv)
    text = evidence(args.pilot_dir)
    if args.out is None:
        sys.stdout.write(text)
        return
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, args.out)


if __name__ == "__main__":
    main()
