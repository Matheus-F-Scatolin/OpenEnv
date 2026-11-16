# Reading the reference experiment

The training code is frozen at tag `reference-freeze` (`9862aa7`). The analysis is added after that tag in `analysis/reference.py`; it implements section 5 of `expectations.md`. It does not change training, dependencies, the tag or any run folder. `analysis.report` continues to describe the October trial.

## One command

From `rfc-013/`, after the real logs have arrived:

```bash
uv run --no-sync python -m analysis.reference --runs trial/runs/reference --out notes/reference-results
```

Only when the gates chose the pre-registered fallback:

```bash
uv run --no-sync python -m analysis.reference --runs trial/runs/reference-staircase --ladder staircase --out notes/reference-staircase-results
```

The command writes `reference.md`, `reference.json`, `success-per-token.png` and `success-per-episode.png`, each through a temporary file and rename, and `falls.md` and `falls.json`, a reading of the falls between consecutive evaluations added after the results (`analysis/falls.py`; not part of the pre-registered reading). Repeating the command on unchanged logs gives identical outputs. JSON retains full precision and records the input roots as given; the tables of `reference.md` show four significant digits, which keeps small p-values and question errors readable, and `falls.md` rounds half up to one decimal; the figures show each seed separately, with the main levels' success and standard error over questions against generated training tokens and training episodes. Both arms use the episode window of their pair for the scalar per-episode measure. `--runs` accepts multiple roots if completed repeats are stored elsewhere. The output must be outside every input root and run folder.

A successful command means the report was generated, not that the inputs passed validation. Read the inventory, warnings and each run's validator status before interpreting it. Empty or partial inputs yield an explicit list of missing runs and no five-seed test; they are never a reference-experiment result.

## Rules implemented

- Only `uniform` and `curriculum`, seeds 5-9, with a single selected numbering. Different numbering is listed but not pooled.
- The first complete run of each method/seed counts, even when the validator rejects it. Completed repeats are listed, never substituted. The log records a start UTC, not a completion UTC: start order equals completion order because the runs ran one after another on one machine. Missing or tied start times produce a warning, with path order as the disclosed tie-break. This cannot verify ordering of externally concurrent runs.
- Main questions are levels 2-9 for `rfc`, or 3-9 for `staircase`. All levels, excluded levels and every individual level are reported separately. Tests recalculate the actual question overlap to guard these sets.
- A token mark uses the evaluation at the training step that crossed it; only when that step has no evaluation is a straight line between the surrounding evaluations used. Nothing is extrapolated. A recorded evaluation with missing question records is unavailable, not repaired by interpolation.
- The token measure averages the five marks from 4.00 to 5.00 million. The episode measure averages exact evaluations at steps 100, 200, ..., up to 700 or the last multiple of 100 both completed arms reached. Neither reaching step 100 means no per-episode value. No complete counterpart means no paired episode window.
- Success is classified from the environment's recorded rewards at the sampler's default threshold, without modifying rewards. A missing, non-finite or unscored answer makes that evaluation's affected scope unavailable. The standard error is `sqrt(p * (1-p) / n)` over the recorded test tasks; it is not uncertainty across training seeds.
- Each method's seed values, means and ranges, and paired curriculum-minus-uniform differences in percentage points, are reported. Only the two main measures receive a two-sided paired t-test on all five seeds, df=4, with the pre-registered closed-form p-value and critical value. Descriptive scopes and sensitivity readings have no significance test. There is no multiple-test correction.
- The -2-point margin is read separately from significance, using the mean and the lower end of the 95% interval as pre-registered. With fewer than five pairs the report makes no inferential support claim. For zero sample variance, t/p use their limiting values: t=0 and p=1 if every difference is zero; signed infinity and p=0 otherwise. Infinity is encoded as a string in JSON, and the interval collapses to the common difference. This arithmetic boundary case should not be read as population certainty.
- A degenerate run has a 50-step window in which at least 90% of answers were cut. It remains in all primary numbers; a labelled sensitivity reading removes its entire seed, with no test. Training tokens, episodes, seconds, updates, signal per episode/token, cut-answer share, answer length, per-level allocation and signal calibration are reported. Calibration reuses `analysis.calibration`, including estimates from the previous step, before the current batch was sampled.
- Step-0 question records and `model_snapshot`, `versions` and `prompt_date` are compared across selected runs. Differences are disclosed, not corrected. Validator failures and unavailable measurements stay next to the selected run; an inability to compute a value is not a zero.

## Verification and publication

Verification on 2026-11-08: the full suite ended with `439 passed in 37.89s`, with no skips (27 new reference-analysis cases). Ruff lint, formatting and usort checks passed. Tests use synthetic records and the real loop's schema, without a model or GPU. No full reference results were read to develop the analysis. The real-model tiny checks before freezing are recorded in the authors' private notes; they are not reference results.

What happened: the reading rules (section 5 of `expectations.md`, commit `cb7c2a7`) were committed on 2026-11-08 at 19:42 UTC and pushed to our private repository at 21:26 UTC, before the first run (2026-11-09 14:01 UTC); every run logs commit `9ac3c5d`, whose tree contains them. The analysis and this note (`0fdac9d`) were committed on 2026-11-08 at 21:48 UTC, before the runs, but pushed only on 2026-11-15 at 21:43 UTC, after the ten runs had been pushed (2026-11-10). Neither the rules nor the analysis was public before the results.

Three changes on 2026-11-15, after the reference results had been read, change no computed value of the pre-registered reading: the Markdown tables show four significant digits (the JSON keeps full precision); `roots` in the JSON keeps the paths as given instead of absolute paths, which named a local folder; and the falls reading above was added, with its 5- and 10-point thresholds chosen after the results. On the real logs, `reference.json` is identical apart from `roots`, and both figures are byte-identical. The suite ended with `487 passed`.
