"""Two-panel figure for RFC 013: test success per environment episode and per generated token.

Usage: uv run --no-project --with matplotlib python rfc_figure.py runs/pilot ../figures/pilot-episodes-vs-tokens.png
Also prints both arms side by side at matched episode counts, which is the table in the RFC.

An episode is one reset() + step() of the environment, i.e. one completion. Every training step runs
prompts_per_step x group_size episodes, so episodes = step x 64 in the pilot. Neither axis depends on
how fast the machine was, unlike training minutes.
"""

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"uniform": "#6b7280", "curriculum": "#2563eb"}


def load(run_dir: Path):
    rows = [json.loads(line) for line in (run_dir / "log.jsonl").open()]
    config = next(r for r in rows if r["event"] == "config")
    return config, [r for r in rows if r["event"] == "eval"]


def main(root: Path, out: Path):
    runs = {}
    for d in sorted(p for p in root.iterdir() if (p / "log.jsonl").exists()):
        runs["curriculum" if "curriculum" in d.name else "uniform"] = load(d)

    fig, (ax_ep, ax_tok) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for sampler, (config, evals) in runs.items():
        c = COLORS[sampler]
        per_step = config["prompts_per_step"] * config["group_size"]
        n_test = 5 * config["eval_per_tier"]
        acc = [100 * e["success_rate"] for e in evals]
        se = [100 * math.sqrt(e["success_rate"] * (1 - e["success_rate"]) / n_test) for e in evals]
        for ax, x in ((ax_ep, [e["step"] * per_step / 1e3 for e in evals]),
                      (ax_tok, [e["gen_tokens_total"] / 1e6 for e in evals])):
            ax.fill_between(x, [a - s for a, s in zip(acc, se)], [a + s for a, s in zip(acc, se)],
                            color=c, alpha=0.15, linewidth=0)
            ax.plot(x, acc, "o-", color=c, markersize=3, label=sampler)

    ax_ep.set(title="Per environment episode: curriculum ahead", xlabel="thousand training episodes",
              ylabel="test success (%)")
    ax_tok.set(title="Per generated token: uniform ahead", xlabel="million generated tokens")
    for ax in (ax_ep, ax_tok):
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Llama-3.2-1B, GRPO + LoRA, reasoning_gym chain_sum, one seed. "
                 "Bands: ±1 standard error (500 test tasks)", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}\n")

    config = runs["uniform"][0]
    per_step = config["prompts_per_step"] * config["group_size"]
    by_step = {s: {e["step"]: e for e in evals} for s, (_, evals) in runs.items()}
    common = sorted(set(by_step["uniform"]) & set(by_step["curriculum"]))
    print(f"{'step':>5} {'episodes':>9} {'uniform':>8} {'curric.':>8} {'diff':>6}")
    for step in common:
        u, c = by_step["uniform"][step]["success_rate"], by_step["curriculum"][step]["success_rate"]
        print(f"{step:>5} {step * per_step:>9,} {100 * u:>7.1f}% {100 * c:>7.1f}% {100 * (c - u):>+6.1f}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/pilot"),
         Path(sys.argv[2] if len(sys.argv) > 2 else "../figures/pilot-episodes-vs-tokens.png"))
