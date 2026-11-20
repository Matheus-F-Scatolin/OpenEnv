# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the `openenv eval` CLI command."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from click.testing import Result
from openenv.cli.__main__ import app
from openenv.cli.commands.collect import _SYSTEM_PROMPT
from openenv.core.env_server.mcp_types import Tool
from openenv.core.evals import EvalResult
from openenv.core.harness import (
    ResourceSession,
    ResourceSessionFactory,
    ToolResult,
    VerifyResult,
)
from typer.testing import CliRunner

runner = CliRunner()

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

ANSWER_TOOL = Tool(
    name="answer",
    description="Submit the final answer.",
    input_schema={
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    },
)

SPLITS = [{"name": "train", "type": "train"}, {"name": "test", "type": "test"}]


class FakeSession(ResourceSession):
    """
    One question whose reward is computed here, as an environment would: 1.0 for
    the task's `answer`, else 0.0, and None for a task marked `unscored`.
    """

    def __init__(self, task: dict[str, Any], seed: int | None) -> None:
        self.task = task
        self.seed = seed
        self.reward: float | None = None

    def initial_messages(self) -> list[dict[str, Any]]:
        return [
            {"role": "user", "content": f"question {self.task['id']} seed {self.seed}"}
        ]

    def list_tools(self) -> list[Tool]:
        return [ANSWER_TOOL]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if not self.task.get("unscored"):
            self.reward = float(arguments["answer"] == self.task["answer"])
        return ToolResult(
            data={"reward": self.reward}, done=True, metadata={"reward": self.reward}
        )

    def verify(
        self, transcript: list[dict[str, Any]], final_state: Any | None = None
    ) -> VerifyResult:
        return VerifyResult(env_reward=self.reward, done=True)

    def close(self) -> None:
        pass


class FakeFactory(ResourceSessionFactory):
    def __init__(self) -> None:
        self.spec: tuple[str, str] | None = None
        self.calls: list[tuple[str, int | None, str | None]] = []

    def create(
        self, task: Any, seed: int | None = None, episode_id: str | None = None
    ) -> FakeSession:
        self.calls.append((task["id"], seed, episode_id))
        return FakeSession(task, seed)

    @property
    def episodes(self) -> list[tuple[str, int | None]]:
        return [(task_id, seed) for task_id, seed, episode in self.calls if episode]


def tasks(count: int, split: str = "test") -> list[dict[str, Any]]:
    return [
        {"id": f"t{index}", "split": split, "index": index, "answer": "4"}
        for index in range(count)
    ]


@pytest.fixture
def task_api(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """
    Serve the Task API routes of `reasoning_gym_env` to every `httpx.Client`.

    Set `status` to answer every request with that code, or `cap` to return at most
    that many tasks per range, as an environment may.
    """
    state = SimpleNamespace(
        tasks={"train": [], "test": tasks(2) + [{"id": "t2", "unscored": True}]},
        status=None,
        cap=None,
        requests=[],
    )

    def handle(request: httpx.Request) -> httpx.Response:
        state.requests.append(request)
        env_name, route = request.url.path.strip("/").split("/")
        if env_name != "reasoning_gym_env":
            return httpx.Response(404, json={"detail": f"unknown env {env_name}"})
        if state.status is not None:
            return httpx.Response(state.status, text="refused")
        if route == "splits":
            return httpx.Response(200, json=SPLITS)
        body = json.loads(request.content)
        split_tasks = state.tasks[body["split"]]
        if route == "num_tasks":
            return httpx.Response(200, json={"num_tasks": len(split_tasks)})
        stop = body["stop"]
        if state.cap is not None:
            stop = min(stop, body["start"] + state.cap)
        return httpx.Response(200, json={"tasks": split_tasks[body["start"] : stop]})

    client_cls = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **options: client_cls(transport=httpx.MockTransport(handle), **options),
    )
    return state


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """
    Stand in for `openai.AsyncOpenAI`: each request gets one `answer` tool call
    with `answers[task id]`, or "0" for a task with no entry.
    """
    state = SimpleNamespace(answers={}, options=[], requests=[])

    class FakeAsyncOpenAI:
        def __init__(self, **options: Any) -> None:
            state.options.append(options)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def create(self, **request: Any) -> SimpleNamespace:
            state.requests.append(request)
            task_id = request["messages"][-1]["content"].split()[1]
            arguments = json.dumps({"answer": state.answers.get(task_id, "0")})
            call = SimpleNamespace(
                id="call-0",
                function=SimpleNamespace(name="answer", arguments=arguments),
            )
            message = SimpleNamespace(content="", tool_calls=[call])
            usage = SimpleNamespace(prompt_tokens=12, completion_tokens=3)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)], usage=usage
            )

    monkeypatch.setattr("openenv.core.llm_client.AsyncOpenAI", FakeAsyncOpenAI)
    return state


@pytest.fixture
def factory(monkeypatch: pytest.MonkeyPatch) -> FakeFactory:
    fake = FakeFactory()

    def build(env: str, base_url: str) -> FakeFactory:
        fake.spec = (env, base_url)
        return fake

    monkeypatch.setattr("openenv.cli.commands.evaluate._build_session_factory", build)
    return fake


def invoke(
    out: Path, *extra: str, spec: str = "reasoning_gym:chain_sum", **options: Any
) -> tuple[Result, str]:
    """
    Run `openenv eval` with the usual options, changed or dropped (None) by
    `options`, and return the result with its output flattened to one line.
    """
    values = {
        "base_url": "http://env",
        "split": "test",
        "llm_endpoint": "http://llm/v1",
        "model": "m",
        "output_dir": str(out),
        **options,
    }
    args = ["eval", spec]
    for name, value in values.items():
        if value is not None:
            args += ["--" + name.replace("_", "-"), value]
    result = runner.invoke(
        app, [*args, *extra], env={"COLUMNS": "200", "OPENAI_API_KEY": "sk-test"}
    )
    # Typer wraps usage errors in a box, coloured when GITHUB_ACTIONS or FORCE_COLOR
    # is set; strip the colour codes and flatten before matching.
    plain = _ANSI_ESCAPE.sub("", result.output)
    return result, " ".join(plain.replace("│", " ").split())


def read_result(out: Path) -> EvalResult:
    return EvalResult.model_validate_json((out / "result.json").read_text())


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class TestEvalCommand:
    def test_writes_result_and_episodes(
        self, tmp_path: Path, task_api, llm, factory: FakeFactory
    ) -> None:
        llm.answers.update({"t0": "4", "t1": "5", "t2": "4"})
        out = tmp_path / "out"

        result, output = invoke(out)

        assert result.exit_code == 0, result.output
        evaluation = read_result(out)
        assert evaluation.config.harness_name == "EnvEvalHarness"
        assert evaluation.config.dataset == "reasoning_gym:chain_sum"
        assert evaluation.config.eval_parameters == {
            "split": "test",
            "limit": None,
            "seeds": [0],
            "samples_per_task": 1,
            "temperature": 0.0,
            "success_threshold": 1.0,
            "max_tokens": 200,
            "max_turns": 9,
            "model": "m",
            "llm_endpoint": "http://llm/v1",
            "system_prompt": _SYSTEM_PROMPT,
            "base_url": "http://env",
        }
        scores = evaluation.scores
        assert scores["per_task"] == {
            "t0": "solved",
            "t1": "not_solved",
            "t2": "left_out",
        }
        assert scores["success_rate"] == 0.5
        assert scores["unscored_rate"] == pytest.approx(1 / 3)
        assert len(lines(out / "run-0000" / "results.jsonl")) == 2
        assert len(lines(out / "run-0000" / "errors.jsonl")) == 1
        assert "success_rate=0.5" in output
        assert str(out / "result.json") in output
        assert factory.spec == ("reasoning_gym:chain_sum", "http://env")
        assert {request.url.host for request in task_api.requests} == {"env"}
        assert {request.url.path for request in task_api.requests} == {
            "/reasoning_gym_env/splits",
            "/reasoning_gym_env/num_tasks",
            "/reasoning_gym_env/task_range",
        }
        assert llm.options[0]["base_url"] == "http://llm/v1"
        assert llm.requests[0]["messages"][0] == {
            "role": "system",
            "content": _SYSTEM_PROMPT,
        }

    def test_api_key_is_written_nowhere(
        self, tmp_path: Path, task_api, llm, factory: FakeFactory
    ) -> None:
        out = tmp_path / "out"

        result, _ = invoke(out)

        assert result.exit_code == 0, result.output
        assert llm.options[0]["api_key"] == "sk-test"
        files = [path for path in out.rglob("*") if path.is_file()]
        assert files
        assert all("sk-test" not in path.read_text() for path in files)

    def test_flags_reach_the_harness_and_the_model(
        self, tmp_path: Path, task_api, llm, factory: FakeFactory
    ) -> None:
        task_api.tasks["test"] = tasks(4)
        out = tmp_path / "out"

        result, _ = invoke(
            out,
            *("--limit", "2", "--seed", "0", "--seed", "3"),
            *("--samples-per-task", "2", "--temperature", "0.7"),
            *("--success-threshold", "0.5", "--max-tokens", "64"),
            *("--max-turns", "3", "--system-prompt", "Use the answer tool."),
        )

        assert result.exit_code == 0, result.output
        evaluation = read_result(out)
        assert evaluation.config.eval_parameters == {
            "split": "test",
            "limit": 2,
            "seeds": [0, 3],
            "samples_per_task": 2,
            "temperature": 0.7,
            "success_threshold": 0.5,
            "max_tokens": 64,
            "max_turns": 3,
            "model": "m",
            "llm_endpoint": "http://llm/v1",
            "system_prompt": "Use the answer tool.",
            "base_url": "http://env",
        }
        assert evaluation.scores["num_tasks"] == 2
        assert evaluation.scores["num_episodes"] == 8
        assert sorted(set(factory.episodes)) == [
            ("t0", 0),
            ("t0", 3),
            ("t1", 0),
            ("t1", 3),
        ]
        assert len(llm.requests) == 8
        for request in llm.requests:
            assert request["model"] == "m"
            assert request["temperature"] == 0.7
            assert request["max_tokens"] == 64
            assert request["messages"][0]["content"] == "Use the answer tool."

    def test_capped_task_ranges_are_read_up_to_the_limit(
        self, tmp_path: Path, task_api, llm, factory: FakeFactory
    ) -> None:
        task_api.tasks["test"] = tasks(5)
        task_api.cap = 2
        out = tmp_path / "out"

        result, _ = invoke(out, "--limit", "3")

        assert result.exit_code == 0, result.output
        assert list(read_result(out).scores["per_task"]) == ["t0", "t1", "t2"]

    def test_empty_output_dir_is_accepted(
        self, tmp_path: Path, task_api, llm, factory: FakeFactory
    ) -> None:
        out = tmp_path / "out"
        out.mkdir()

        result, _ = invoke(out)

        assert result.exit_code == 0, result.output
        assert (out / "result.json").exists()


class TestUsageErrors:
    def test_unknown_split(
        self, tmp_path: Path, task_api, llm, factory: FakeFactory
    ) -> None:
        out = tmp_path / "out"

        result, output = invoke(out, split="dev")

        assert result.exit_code == 2, result.output
        assert "--split" in output
        assert "available: train, test" in output
        assert not out.exists()
        assert factory.calls == []
        assert llm.requests == []

    def test_split_without_tasks(
        self, tmp_path: Path, task_api, llm, factory: FakeFactory
    ) -> None:
        out = tmp_path / "out"

        result, output = invoke(out, split="train")

        assert result.exit_code == 2, result.output
        assert "--split" in output
        assert "has no tasks" in output
        assert not out.exists()

    def test_environment_without_task_api(
        self, tmp_path: Path, task_api, llm, factory: FakeFactory
    ) -> None:
        task_api.status = 501
        out = tmp_path / "out"

        result, output = invoke(out)

        assert result.exit_code == 2, result.output
        assert "does not implement the Task API" in output
        assert not out.exists()
        assert factory.calls == []

    def test_non_empty_output_dir_is_refused(
        self, tmp_path: Path, task_api, llm, factory: FakeFactory
    ) -> None:
        out = tmp_path / "out"
        out.mkdir()
        (out / "result.json").write_text("earlier")

        result, output = invoke(out)

        assert result.exit_code == 2, result.output
        assert "--output-dir" in output
        assert (out / "result.json").read_text() == "earlier"
        assert task_api.requests == []
        assert factory.calls == []

    def test_unknown_env_spec(self, tmp_path: Path, task_api, llm) -> None:
        result, output = invoke(tmp_path / "out", spec="unknown:foo")

        assert result.exit_code == 2, result.output
        assert "Unknown env" in output
        assert task_api.requests == []

    def test_bad_llm_endpoint_is_refused_before_any_request(
        self, tmp_path: Path, task_api, llm, factory: FakeFactory
    ) -> None:
        out = tmp_path / "out"

        result, output = invoke(out, llm_endpoint="ftp://llm")

        assert result.exit_code == 2, result.output
        assert "--llm-endpoint" in output
        assert task_api.requests == []
        assert not out.exists()

    @pytest.mark.parametrize(
        "missing", ["base_url", "split", "llm_endpoint", "model", "output_dir"]
    )
    def test_required_options(
        self, tmp_path: Path, task_api, llm, factory: FakeFactory, missing: str
    ) -> None:
        result, output = invoke(tmp_path / "out", **{missing: None})

        assert result.exit_code == 2, result.output
        assert "Missing option" in output
        assert task_api.requests == []

    @pytest.mark.parametrize(
        "option",
        ["--limit", "--samples-per-task", "--max-tokens", "--max-turns"],
    )
    def test_counts_below_one_are_refused(
        self, tmp_path: Path, task_api, llm, factory: FakeFactory, option: str
    ) -> None:
        result, output = invoke(tmp_path / "out", option, "0")

        assert result.exit_code == 2, result.output
        assert option in output
        assert task_api.requests == []

    def test_invalid_settings_write_nothing(
        self, tmp_path: Path, task_api, llm, factory: FakeFactory
    ) -> None:
        out = tmp_path / "out"

        result, _ = invoke(out, "--samples-per-task", "2")

        assert result.exit_code == 1
        assert isinstance(result.exception, ValueError)
        assert "temperature must be > 0" in str(result.exception)
        assert not out.exists()
        assert llm.requests == []


class TestRegistration:
    def test_eval_is_listed(self) -> None:
        result = runner.invoke(app, ["--help"], env={"COLUMNS": "200"})

        assert result.exit_code == 0
        assert "Evaluate a model on a split of an environment" in result.output

    def test_eval_help_shows_its_options(self) -> None:
        result = runner.invoke(app, ["eval", "--help"], env={"COLUMNS": "200"})

        assert result.exit_code == 0
        plain = _ANSI_ESCAPE.sub("", result.output)
        for option in ("--split", "--llm-endpoint", "--seed", "--success-threshold"):
            assert option in plain
