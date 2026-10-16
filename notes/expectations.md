# Expectations, written before the runs

Written on 2026-10-12, before any run of the new experiments. The commit date of each prediction is the proof that it came first. A prediction is never edited after its run started; a correction is added below it with its date.

The numbers "from the pilot" were computed from the pilot's logs (seed 0) on 2026-10-12.

## 1. Ten-run check

**Question.** At step 30, on the same 500 test questions, is the mean answer length after `curriculum` training above the length after `uniform` training, in seeds we have not seen?

**From the pilot.** Seed 0 is already known and is not new evidence. On 200 questions: step 10, uniform 43.5 tokens and curriculum 72.2; step 20, 47.3 and 97.2. Untrained model: 73.1 on 500 questions.

**To predict.** In how many of the four new seeds (1 to 4) the curriculum's length at step 30 is above uniform's by at least 20 tokens.

| Who | Prediction | Date |
|---|---|---|
| Tiago | 3 of 4. The mechanism is plausible (harder tasks ask for longer answers; uniform trains mostly on easy ones and shortens them), but 30 steps are few and 20 tokens is a demanding margin. | 2026-10-12 |
| Matheus | _to fill in before the first run_ | |

**How it is read.** 4 of 4 means the split comes from the method (a coin flip per seed gives this 1 time in 16). 0 or 1 means it was mostly seed 0's accident. 2 or 3 means four seeds do not decide it, and we say so.

## 2. The trial

**Question.** Per generated token and per episode, which method has the higher test success on seeds 1 and 2, and by how much?

**From the pilot.** Seed 0, with the definitions below. Per token: uniform 72.2, curriculum 64.5, difference −7.7. Per episode: uniform 61.4, curriculum 63.6, difference +2.3. From one evaluation to the next, test success moves by 2 to 3 points.

**To predict.** The difference curriculum minus uniform, per token and per episode: sign and size in points.

| Who | Per token | Per episode | Date |
|---|---|---|---|
| Tiago | −5 points. The sign should hold: the curriculum spends more tokens per answer, so it gets fewer training steps in 5 million tokens. Smaller than seed 0 because one seed tends to exaggerate. | +1 point, inside the noise: no clear difference. | 2026-10-12 |
| Matheus | _to fill in_ | _to fill in_ | |

## 3. Cost probes

**Question.** What share of the episodes does each cost-aware formula send to the easiest level, which is solved from the start?

**From the pilot.** The pilot's sampler gave that level a probability of 8.6% on average over steps 51 to 100, and 5.7% in the last quarter of the run. Computed for the three formulas of `interface.md`: 23%, 18% and 5% at the pilot's last-quarter numbers; 20%, 17% and 5% at its first-quarter numbers. These are arithmetic, not training results. One probe counts 400 groups, which moves the share by about 2 points either way.

**To predict.** The share for each formula over steps 51 to 100.

| Who | `curriculum-tokens` | `-mix` | `-mix-decay` | Date |
|---|---|---|---|---|
| Tiago | 20% | 17% | 7% (above the computed 5%: the prior takes some steps to fade, and 400 groups move the share by about 2 points) | 2026-10-12 |
| Matheus | _to fill in_ | _to fill in_ | _to fill in_ | |

## 4. Cost-aware runs

**Question.** Per generated token, is the cost-aware curriculum at least as good as uniform?

**From the pilot.** Never run.

**To predict.** The difference cost-aware minus uniform per token on seeds 1 and 2.

| Who | Prediction | Date |
|---|---|---|
| Tiago | −1 point, so "at least as good" by the −2 rule. Dividing by cost moves the sampler to cheap levels, so it should spend tokens much as uniform does. Not better: the cheap levels are the ones that teach least. | 2026-10-12 |
| Matheus | _to fill in_ | |

## How each number is computed

- **Success per token**, for one run: the mean test success of the five evaluations at 4.00, 4.25, 4.50, 4.75 and 5.00 million generated tokens. Where a run has no evaluation at one of those marks (the pilot, and the pinned script), its curve is read there by a straight line between the two nearest evaluations.
- **Success per episode**, for one run: the mean test success of the seven evaluations at steps 100 to 700. If any run of a seed stops before step 700, every run of that seed is read at steps 100 to the last multiple of 100 that all of its runs reached, and the report says so.
- **A difference between two methods** is computed inside each seed, then averaged over the new seeds. Seed 0 is shown beside them and is not in the average, because our questions were written from it.
- **"At least as good"** means the average difference is −2 points or more. **"Better"** means +4 or more and ahead in every new seed. **"Worse"** means −4 or less and behind in every new seed. Anything else is "no clear difference with this many seeds".
- Every success number is also given for levels 2 to 4 alone: levels 0 and 1 are at 100% from the start, and all 100 test questions of level 0 also appear in train.
- No significance test: there are fewer than five seeds.

## Decision rules

- **Which trainer runs the experiments.** The new one, if on the machine that runs them (a) its first 20 steps give the same log as the reference 20 steps, for both methods, (b) a five-minute run on a token budget passes `python -m analysis.validate_run`, and (c) a queue of five tiny runs completes. Otherwise the pinned script. `trial/RUNBOOK.md` calls these gates G4, G5 and G6.
- **Which runs may be compared.** Runs of different methods are compared only if the same trainer made them, or if gate G4 showed that the two trainers train identically.
- **Which cost-aware formulas get full runs.** The RFC's formula, because it is what the RFC proposes, and `curriculum-tokens-mix-decay`. `curriculum-tokens-mix` is probed for the record. A 100-step probe on one seed shows where the episodes go; it cannot rank formulas.
- **When a run counts.** Only if it is complete and `python -m analysis.validate_run` says `OK` or `OK (version 1)`. For each method and seed the first such run counts; a second is reported as a repeat. No run is dropped for its result.
