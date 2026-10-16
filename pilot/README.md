# RFC 013 smoke test: GRPO + curriculum on an Apple Silicon laptop

Trains a small LLM with GRPO (LoRA, in MLX) on OpenEnv's `reasoning_gym_env`, choosing tasks with
the samplers proposed in RFC 013 (`curriculum.py`), and evaluates a fixed test split during training.

- **Environment**: `reasoning_gym_env`, vendored from huggingface/OpenEnv at commit `9e0fd9b` and run
  in-process (no Docker). Each task is `reset(dataset_name="chain_sum", dataset_config=<tier>, seed, size=1)`,
  and the reward is whatever the environment's `step()` returns.
- **Difficulty tiers** (`chain_sum`): 2 one-digit terms, then 3×2 digits, 4×3, 5×4 with negatives, 6×5 with negatives.
- **Splits**: 2,000 train tasks and 40 test tasks per tier, with disjoint seeds.

## Setup

```bash
uv sync
```

The first run downloads the model from Hugging Face (about 2.5 GB for Llama-3.2-1B in bf16).

## Commands

```bash
# Greedy accuracy per tier on the test split, for one or more models
uv run python smoke.py baseline --models llama3.2-1b qwen3-0.6b

# Short GRPO run: 8 prompts x 8 completions per step, eval every 10 steps
uv run python smoke.py train --model llama3.2-1b --sampler curriculum --steps 20 --eval-every 10 --eval-at-start
uv run python smoke.py train --model llama3.2-1b --sampler uniform    --steps 20 --eval-every 10 --eval-at-start
```

Each run writes `runs/<model>-<sampler>-s<seed>/log.jsonl` (one line per step and per eval), the LoRA
adapters, and the sampler state.

For long runs, keep the Mac plugged in and awake:

```bash
caffeinate -i uv run python smoke.py train --model llama3.2-1b --sampler curriculum --steps 1500 --eval-every 100 --eval-at-start --eval-per-tier 100
```

## Smoke results (2026-09-26, MacBook Pro M5 Pro, 24 GB)

**Baseline, 200 test tasks, greedy**

| Model | Overall | Tier 0 | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Eval time |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 51.0% | 100% | 100% | 50% | 5% | 0% | 13 s |
| Qwen3-0.6B (thinking off) | 79.5% | 95% | 80% | 100% | 72.5% | 50% | 16 s |

Llama-3.2-1B was chosen for the experiment. Its accuracy falls tier by tier and the top tiers sit near 0%,
which is the regime where GRPO wastes groups and a curriculum should help. Qwen3-0.6B is already strong on
this task and would need harder tiers.

**20 GRPO steps per sampler (Llama-3.2-1B)**

| | Uniform | Curriculum |
|---|---|---|
| Median seconds per step | 4.9 | 7.8 |
| Groups with no gradient (all rewards equal) | 83% | 70% |
| Groups with gradient per step (of 8) | 1.35 | 2.40 |
| Peak memory | 8.7 GB | 15.2 GB |
| Test success, step 0 → 20 | 51.0% → 53.0% | 51.0% → 51.0% |

- The curriculum does what it is built for: it concentrates on tier 2 (estimated success ≈ 0.44),
  and 78% more groups per step carry a gradient.
- Twenty steps are far too few to move test accuracy; the changes above are within noise (40 tasks per tier).
- Curriculum steps cost more because harder tasks produce longer completions and more groups get trained on.
  Compare arms at equal wall-clock time or equal generated tokens, not at equal steps.
- Strict `<answer>` tag usage (`format_rate`) fell to 0% during training. The answer parser also accepts
  `\boxed{}` and the last number, so the reward does not push the tags. This does not affect the comparison,
  but `format_rate` is not meaningful after training starts.

## Pilot results (2026-09-26): 150 training minutes per sampler, one seed

Plot: `runs/pilot/comparison.png` (regenerate with `uv run python plot.py runs/pilot`).
The two-panel figure in the RFC (test success per episode and per generated token) comes from
`uv run --no-project --with matplotlib python rfc_figure.py runs/pilot ../figures/pilot-episodes-vs-tokens.png`,
which also prints both arms side by side at matched steps.

| | Uniform | Curriculum |
|---|---|---|
| Steps | 2,196 | 1,081 |
| Generated tokens | 6.2 M | 7.3 M |
| Tokens per completion | 44 | 106 |
| Groups with no gradient | 79.5% | 71.5% |
| Groups with gradient per step | 1.64 | 2.28 |
| Groups with gradient per training minute | **24.1** | 16.4 |
| Test success, start → end | 52.0% → 73.2% | 52.0% → 72.4% |
| Mean of the last 3 evals | 73.6% | 70.5% |
| Test success at ~6 M generated tokens | 73.6% | 67.8% |

**The curriculum did not beat uniform sampling.** Both arms reached about the same test accuracy in the same
training time, and uniform was ahead at equal generated tokens. With one seed and 500 test tasks (standard error
about 2 points), the gap is suggestive, not conclusive.

The per-step mechanism worked: the curriculum cut the share of groups with no gradient and produced 39% more
groups with gradient per step. But it chose hard tasks, whose completions are 2.4 times longer, so each step cost
more. Uniform sampling spent many steps on easy tasks that gave no gradient, but those steps were cheap. Per
minute, it got more groups with gradient. Uniform training on tier 2 also transferred to tiers 3 and 4 without
targeting them.

Lessons for RFC 013:

1. **Weight by signal per unit of cost, not by signal alone.** A group's cost depends on completion length,
   which varies by difficulty. A cost-aware weight is `P(signal) / E[tokens]` per key.
2. **Report gradient-bearing groups per generated token** (or per second) as the efficiency metric. The share of
   zero-signal groups on its own is misleading.
3. **Difficulty tiers need finer resolution.** After tier 2 saturated (~95%), no tier was near 50%: tier 3 was at
   about 15 to 40% and tier 4 at 5 to 20%.
4. Caveats: one seed; the arms ran one after the other; the curriculum arm was throttled for its last ~40 minutes
   by an underpowered charger. The per-token comparison is not affected by throttling.

## Notes

- MLX needs the Metal GPU, which is not available inside a sandboxed shell. Run the commands from a normal terminal.
- The GRPO loss is on-policy with a single update per batch (ratio = 1), no KL term, and token-level
  normalization. Groups with no reward variance are skipped for the backward pass.
