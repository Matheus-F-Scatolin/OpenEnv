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

## Train with a task sampler

`training_loop.py` is a GRPO loop that needs no training framework. It draws each step's tasks from a task sampler (`UniformSampler`, `SequentialSampler`, or `CurriculumSampler` from `curriculum_sampler.py`), generates a group of answers per task with a policy, scores each answer with the environment, and takes one optimizer step. The policy and the environment are two small protocols in the file: a policy encodes prompts, generates in batches and takes a step on `(prompt ids, answer ids, advantage)` samples; an environment returns a task's question and a reply's reward.

```python
from curriculum_sampler import CurriculumSampler
from openenv.core.evals import EvalSchedule
from training_loop import train

sampler = CurriculumSampler(train_tasks, group_size=8, rng_seed=0)
schedule = EvalSchedule(harness, config, every_n_steps=100, on_result=log_eval)
train(
    policy,
    env,
    sampler,
    steps=700,
    prompts_per_step=8,
    group_size=8,
    max_tokens=384,
    temperature=1.0,
    log=log_step,
    schedule=schedule,
)
```

The loop keeps the rules of RFC 013:

- every group updates the sampler, with its rewards and the number of tokens of each answer as its costs, before any group is dropped;
- a group whose scored rewards are all equal, or that has fewer than two, gives no gradient; the others give each scored answer the advantage `(reward - mean) / (std + 1e-4)`;
- a reward of None (the environment could not score the answer) reaches the sampler, which leaves it out of its estimates, and gives no gradient: it is never replaced by zero;
- with a schedule, the policy is evaluated before the first step, at the schedule's interval and after the last step;
- each step's record carries the sampler's `metrics()`, and `on_result` can log them next to each evaluation, so learning curves can be drawn against episodes and against generated tokens.

`tests/scripts/test_task_sampling_training_loop.py` runs the loop with a fake policy and a fake environment.

### On Apple Silicon with MLX

`train_mlx.py` trains a model with LoRA adapters on MLX on the tasks of a `reasoning_gym_env` server, and evaluates it on the server's `test` split with `EnvEvalHarness` while it trains. The environment must implement the Task API with difficulty levels, and its session factory must pass each task's `split` and `index`, with the seed, to `reset()`. The script needs `mlx-lm` (checked with `mlx` 0.32.2 and `mlx-lm` 0.31.3); training and evaluation read each plain-text reply with `reasoning_gym_tasks.answer_from_reply` (the last `<answer>` block, else the last `\boxed{}`, else the last number), and evaluation submits it as the `answer` tool call.

```bash
# Terminal 1, from envs/: the environment server
uvicorn reasoning_gym_env.server.app:app --port 8001

# Terminal 2, from examples/task_sampling/
python train_mlx.py --base-url http://localhost:8001 --sampler curriculum --seed 0 \
    --out runs/curriculum-seed0
```

The folder given to `--out` must not exist. It gets `config.json` (every setting, the prompt date and the library versions), `train.jsonl` (one record per step), `evals.jsonl` (one record per evaluation, with the sampler's counters), `evals/run-NNNN/` (the episodes of each evaluation, as `openenv collect` writes them) and, at the end, `adapters.safetensors`. A fixed date goes into every prompt, because Llama 3.2's chat template otherwise writes the current date into it. Evaluate on fewer test tasks with `--eval-limit`: the server interleaves the levels in the `test` split, so the first 200 tasks hold 20 of each of its 10 levels.

The defaults are the settings of RFC 013's reference experiment: Llama-3.2-1B-Instruct, LoRA rank 16 on the attention and MLP projections, 8 prompts of 8 answers per step, temperature 1.0, at most 384 tokens, Adam with a learning rate of 2e-5. Over five seeds, the curriculum sampler beat uniform sampling on the test questions of levels 2 to 9 by 8.2 points at the same number of generated tokens and by 7.2 points at the same number of episodes ([RFC 013](https://github.com/openenv/OpenEnv/pull/1256)). That experiment used its own trainer, which stops on a token budget and evaluates in batches; this script stops after `--steps` steps and evaluates one question at a time, so its numbers are not expected to match exactly. Both read replies by the same lenient rule. Llama-3.2-1B-Instruct rarely writes the `<answer>` tags before training: on the first 10 test questions it wrote them in 2 replies of 10, and a rule that reads only the tags (`reasoning_gym_env.harness.text_answer_model_step`, which serves every reasoning_gym dataset) scored 0 of 10 where this one scored 5, as the reference's untrained model did.
