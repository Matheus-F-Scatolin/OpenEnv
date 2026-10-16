"""Plot uniform vs curriculum runs against training time.

Usage: uv run python plot.py runs/pilot
Writes <run_root>/comparison.png and prints a summary table.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"uniform": "#6b7280", "curriculum": "#2563eb"}


def load(run_dir: Path):
    rows = [json.loads(line) for line in (run_dir / "log.jsonl").open()]
    return ([r for r in rows if r["event"] == "step"], [r for r in rows if r["event"] == "eval"])


def rolling(values, window=50):
    out, acc = [], []
    for v in values:
        acc.append(v)
        if len(acc) > window:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


def main(root: Path):
    runs = {}
    for d in sorted(p for p in root.iterdir() if (p / "log.jsonl").exists()):
        sampler = "curriculum" if "curriculum" in d.name else "uniform"
        steps, evals = load(d)
        if steps and evals:  # skip runs that have not logged a step and an eval yet
            runs[sampler] = (steps, evals)

    fig, grid = plt.subplots(2, 2, figsize=(13, 9))
    ax_acc, ax_tok, ax_tier, ax_zero = grid.flat
    axes = grid.flat
    for sampler, (steps, evals) in runs.items():
        c = COLORS[sampler]
        mins = [e["train_elapsed_s"] / 60 for e in evals]
        ax_acc.plot(mins, [100 * e["success_rate"] for e in evals], "o-", color=c, label=sampler)
        ax_tok.plot([e["gen_tokens_total"] / 1e6 for e in evals], [100 * e["success_rate"] for e in evals],
                    "o-", color=c, label=sampler)
        for tier, style in (("2", "-"), ("3", "--"), ("4", ":")):
            ax_tier.plot(mins, [100 * e["by_tier"][tier] for e in evals], style, color=c,
                         label=f"{sampler} tier {tier}")
        ax_zero.plot([s["train_elapsed_s"] / 60 for s in steps],
                     rolling([100 * s["zero_signal_fraction_step"] for s in steps]), color=c, label=sampler)

    ax_acc.set(title="Test success (500 tasks)", xlabel="training minutes", ylabel="%")
    ax_tok.set(title="Test success vs generated tokens (hardware-independent)",
               xlabel="million generated tokens", ylabel="%")
    ax_tier.set(title="Test success by tier", xlabel="training minutes", ylabel="%")
    ax_zero.set(title="Groups with no gradient (rolling 50 steps)", xlabel="training minutes", ylabel="%")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = root / "comparison.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")

    print(f"\n{'sampler':<11} {'steps':>6} {'minutes':>8} {'gen tokens':>11} {'zero-signal':>12} "
          f"{'test start':>11} {'test best':>10} {'test end':>9}  by tier (end)")
    for sampler, (steps, evals) in runs.items():
        zero = sum(s["zero_signal_fraction_step"] for s in steps) / max(len(steps), 1)
        print(f"{sampler:<11} {len(steps):>6} {steps[-1]['train_elapsed_s'] / 60:>8.1f} "
              f"{evals[-1]['gen_tokens_total']:>11,} {100 * zero:>11.1f}% "
              f"{100 * evals[0]['success_rate']:>10.1f}% {100 * max(e['success_rate'] for e in evals):>9.1f}% "
              f"{100 * evals[-1]['success_rate']:>8.1f}%  {evals[-1]['by_tier']}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/pilot"))
