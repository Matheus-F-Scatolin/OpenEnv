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

## 5. The reference experiment

Written on 2026-11-08, before any run of it. Sections 1 to 4 and the rules above are about the trial; this section has its own rules.

**Question.** On ten `chain_sum` levels (2×1 to 7×5 numbers × digits, without negative numbers), and on five seeds not used so far, is the curriculum's test success at least as high as uniform sampling's, per generated token and per episode?

**Runs.** Two methods: `uniform` (`UniformSampler`) and `curriculum` (`CurriculumSampler` with the RFC's defaults: groups of 8, levels by `difficulty`, decay 0.9, ε 0.05, success at a reward of 1.0, the fixed prior). Seeds 5, 6, 7, 8 and 9; seeds 0 to 4 ran in the pilot, the ten-run check and the trial. Everything else as in the trial: Llama-3.2-1B-Instruct (bf16, snapshot `863c846a`), LoRA rank 16, 8 prompts × 8 answers per step, temperature 1.0, at most 384 tokens per answer, learning rate 2e-5, the prompt date 26 Sep 2026, and a stop at 5 million generated training tokens. `trial/lists/reference.json` runs seed 5 uniform, seed 5 curriculum, then seed 6, and so on, so that a stopped queue leaves whole pairs.

**Tasks.** `--ladder rfc`: the environment's `train` split (2,000 tasks per level) and `test` split (100 per level, 1,000 in all), through its Task API methods, in-process, with question seed 0 in every run, so all runs see the same questions. If a gate fails as the runbook describes, the runs use `--ladder staircase` instead (`trial/lists/reference-staircase.json`): the same ten level settings with the trial's numbering.

**Evaluations.** Greedy, on all 1,000 test questions: at step 0, every 100 steps, each time the generated tokens cross a multiple of 250,000, and at the last step.

**Levels whose test questions are also in train.** A level is left out of the two main measures when any of its test questions also appears among the train questions. `python -m analysis.overlap --ladder rfc` prints, for levels 0 to 9: 100, 47, 0, 0, 0, 0, 0, 0, 0, 0. The main measures therefore use the 800 test questions of levels 2 to 9. With `--ladder staircase` it prints 100, 38, 1, then 0, and the main measures use the 700 questions of levels 3 to 9. The left-out levels are reported apart, and every number is also given on all 1,000 questions and per level.

**The two main measures**, for one run, on the questions above:

- *Success per token*: the mean of the evaluations at 4.00, 4.25, 4.50, 4.75 and 5.00 million generated tokens (the evaluation of the step that crossed each mark; where a run has none, the straight line between the two nearest).
- *Success per episode*: the mean of the evaluations at steps 100, 200, ..., 700. When either run of a seed stops before step 700, both runs of that seed are read at steps 100 to the last multiple of 100 that both reached. A seed whose two runs did not both reach step 100 has no number per episode.

**The comparison.** For each seed, the difference curriculum minus uniform, in points. The report gives the five differences, their mean and their range, and for each evaluation the standard error over its questions.

**The test.** For each main measure, a two-sided paired t-test on the five differences: t = mean / (sd / √5), with the sample standard deviation (dividing by 4). The difference is significant at 5% when its two-sided p-value is 0.05 or less, that is when t, without its sign, is at least 2.776445, the 97.5% point of Student's t with 4 degrees of freedom. The report gives t, the p-value and the interval mean ± 2.776445 × sd / √5. The p-value uses the closed form of Student's t with 4 degrees of freedom, standard library only: with x = |t| / √(4 + t²), p = 1 − 1.5x + 0.5x³. The two tests are not corrected for each other: the chance that at least one is significant by chance alone is up to about 10%. With fewer than five complete pairs there is no test, only the differences.

**How it is read.** *Better*: significant, mean above 0. *Worse*: significant, mean below 0. Otherwise: *no significant difference with five seeds*. The RFC's "What would support the proposal" holds when, for both measures, the curriculum is not *worse* and its mean difference is −2 points or more. The t-test asks whether the two methods differ; it cannot show that the curriculum is at least as good. So for each measure the report also gives the lower end of the interval, mean − 2.776445 × sd / √5, and writes *shown by the interval* when it is −2 points or more, and *on the mean only* when it is below −2 while the mean is −2 or more. Reported beside, without a test: the same on all 1,000 questions and per level; groups with signal per episode and per million generated tokens; the share of episodes per level; per level, the groups with signal that the curriculum's estimates predict against those observed, as `notes/results/calibration.md` does for the trial; the share of training answers cut at 384 tokens; tokens per training answer; training seconds and the number of optimizer updates (steps with at least one group with signal).

**A run that degenerates.** A run degenerates when, in 50 consecutive steps, at least 90% of its training answers were cut at 384 tokens. It stays in every number and in the test, and it is not repeated: the same seed gives the same log. The report names it and adds a second reading of the comparison without every seed that has a degenerate run, labelled as such, with no test (fewer than five seeds).

**When a run counts.** For each method and seed the first complete run counts (its log ends with the `done` event); a later one is reported as a repeat and never replaces it. If `python -m analysis.validate_run` does not say `OK` for the run that counts, the report gives the checker's line next to that run's numbers. A run not complete when the results are written up is reported as not run.

**Checks the analysis makes before reading.** The step-0 evaluation is the same untrained model on the same questions in all ten runs, so it must give the same numbers in each. All runs must log the same `model_snapshot`, `versions` and `prompt_date`. A difference is reported, not corrected.

**To predict.** For each measure: the mean difference in points, and in how many of the five seeds the curriculum is ahead. For reference, the trial (the pilot's five levels, 500 questions, seeds 1 to 3) gave per token +5.5, +18.3, +14.4, and per episode +4.3, +9.7, +6.5 (RFC, "Three more seeds").

| Who | Per token: mean, seeds ahead | Per episode: mean, seeds ahead | Date |
|---|---|---|---|
| Tiago | +5 points; ahead in 4 of 5 seeds | +4 points; ahead in 4 of 5 seeds | 2026-11-08 |
| Matheus | _optional, before the first run_ | | |

Tiago adopted this prediction on 2026-11-08. It is deliberately smaller than the trial's mean advantage: the finer levels change the questions and difficulty distribution, and one of the five seeds may favour uniform sampling.
