# SPDX-License-Identifier: BSD-3-Clause

"""
Did the pilot repeat its smoke run? The first steps of each method, side by side.

  python -m analysis.pilot_evidence.repro <pilot folder>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from analysis.pilot_evidence.pilot_logs import read_pilot_log

CONFIG_KEYS = (
    "steps",
    "max_tokens",
    "micro_batch",
    "eval_per_tier",
    "seed",
    "group_size",
    "prompts_per_step",
    "lr",
    "temperature",
)


def _step(s: dict[str, Any]) -> str:
    return (
        f"gen_tokens={s['gen_tokens']:>5} mean_reward={s['mean_reward']:.4f} "
        f"sig={s['groups_with_signal']} tiers={sorted(s['tier_mean_reward'])}"
    )


def text(pilot_dir: Path) -> str:
    """
    Return the `repro.py` section of the evidence.

    Args:
        pilot_dir (`Path`):
            The pilot's folder, `rfc-013/pilot`.

    Returns:
        `str`: The section's lines, each ending with a newline.
    """
    lines = []
    for arm in ("uniform", "curriculum"):
        smoke = read_pilot_log(pilot_dir, arm, "smoke")
        pilot = read_pilot_log(pilot_dir, arm, "pilot")
        config = {k: v for k, v in smoke.config.items() if k in CONFIG_KEYS}
        lines += [f"== {arm}", f" smoke config: {config}"]
        sms, pis = smoke.steps, pilot.steps
        for a, b in list(zip(sms, pis))[:8]:
            lines.append(f"  step {a['step']:>2}: smoke {_step(a)} | pilot {_step(b)}")
        same = sum(a["gen_tokens"] == b["gen_tokens"] for a, b in zip(sms, pis))
        lines.append(f"  identical gen_tokens in {same}/{len(sms)} first steps")
        for r in smoke.evals:
            lines.append(
                f"  smoke eval {r['step']} {r['success_rate']} "
                f"{r['mean_completion_tokens']} {r['by_tier']}"
            )
        lines.append(f"  smoke peak mem: {[s['peak_mem_gb'] for s in sms]}")
    return "".join(line + "\n" for line in lines)


def main(argv: list[str] | None = None) -> None:
    """
    Print the section from the command line.

    Args:
        argv (`list[str]`, *optional*):
            The arguments; `sys.argv[1:]` when left out.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pilot_dir", type=Path)
    sys.stdout.write(text(parser.parse_args(argv).pilot_dir))


if __name__ == "__main__":
    main()
