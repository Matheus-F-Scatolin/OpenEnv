# Task sampling and training evaluations

## Evaluate during TRL training

`trl_eval_callback.py` defines `EvalScheduleCallback`, a Transformers `TrainerCallback` that runs an `openenv.core.evals.EvalSchedule` while a `Trainer` or one of TRL's trainers trains. It evaluates the starting weights (step 0), the weights after every `every_n_steps` optimizer steps (`state.global_step`, counted after gradient accumulation) and the final weights, and passes each step and its `EvalResult` to the schedule's `on_result`. Copy the file into your project; it needs `transformers` and `openenv`, installed from source until a release includes `openenv.core.evals.EvalSchedule`.

### With TRL's vLLM server mode

The evaluation must see the weights being trained. With `GRPOTrainer` in vLLM server mode (TRL 1.11 or later), the vLLM server that generates the training rollouts also serves the evaluation through its OpenAI-compatible API. `trl vllm-serve --model Qwen/Qwen3-0.6B` still starts it, but is deprecated since TRL 1.11: it runs vLLM's own server and prints the command to use instead. With TRL 1.18 that command is the following, plus `--enable-scale-out` on vLLM 0.30.0 or later, and plus the last two flags, which TRL does not print:

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B --weight-transfer-config '{"backend": "nccl"}' --logprobs-mode processed_logprobs --max-logprobs -1 --enable-auto-tool-choice --tool-call-parser hermes
```

The policy below answers through tool calls, and vLLM returns them only with a tool-call parser: without one, every evaluation episode ends without an answer and counts in `error_rate`. `hermes` is the parser that vLLM's documentation gives for Qwen2.5 and that Qwen's documentation gives for Qwen3; use your model's. These flags come from that documentation; the recipe was not run.

Then give the harness a policy that calls this server under the served model name, and attach the callback to the trainer:

```python
import json

from openenv.core import OpenAIClient
from openenv.core.evals import EvalSchedule
from openenv.core.harness.collect import build_model_step
from trl import GRPOConfig, GRPOTrainer
from trl_eval_callback import EvalScheduleCallback

policy = build_model_step(OpenAIClient("http://localhost", 8000, model="Qwen/Qwen3-0.6B"))
# harness: any EvalHarness that answers its tasks with `policy`.


def log_scores(step, result):
    with open("evals.jsonl", "a") as f:
        f.write(json.dumps({"step": step, **result.scores}) + "\n")


schedule = EvalSchedule(harness, config, every_n_steps=50, on_result=log_scores)

trainer = GRPOTrainer(..., args=GRPOConfig(..., use_vllm=True, vllm_mode="server"))
trainer.add_callback(
    EvalScheduleCallback(schedule, sync=trainer.vllm_generation.sync_weights)
)
trainer.train()
```

**Why `sync`.** `GRPOTrainer` sends new weights to the server only right before its next generation, so without `sync` an evaluation at the end of step N would score older weights. `sync_weights` gathers the parameters across processes, so the callback calls it on every process before rank 0 evaluates. Each evaluation costs one extra weight transfer, because the trainer sends the same weights again before its next generation. `trainer.vllm_generation.sync_weights` is an internal method of TRL, read in TRL 1.11, 1.15 and 1.18: check it when you upgrade TRL. When the endpoint already serves the current weights, pass `sync=lambda: None`.

**Logging.** `on_result` receives `(step, result)`. Log the whole `result.scores` to your own logger, so that `success_rate` stays next to `error_rate`, `unscored_rate` and `infra_error_rate`. Do not call `trainer.log` from it: `GRPOTrainer.log` averages and then clears the training metrics it has collected since its last log.

### Limits

- With several processes, the other ranks wait in their next collective while rank 0 evaluates. An evaluation longer than `ddp_timeout` (1800 s by default) stops the run.
- A run resumed from a checkpoint skips the step-0 evaluation and evaluates again at its next marked step.
- The final weights are evaluated in `on_train_end`, unless the last step was a marked one and was evaluated already.
- The training prompts come from the trainer's dataset, not from a task sampler.
