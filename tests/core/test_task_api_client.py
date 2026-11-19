# SPDX-License-Identifier: BSD-3-Clause

"""Tests for TaskAPIClient against the Task API routes of a stub server."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from openenv.core.generic_client import GenericEnvClient
from openenv.core.task_api_client import TaskAPIClient

SPLIT_SIZES = {"train": 7, "test": 3, "empty": 0}
TOKEN_HEADERS = {"Authorization": "Bearer hf_test_token"}


class TaskAction(Action):
    value: str = ""


class TaskObservation(Observation):
    message: str = ""


class TaskEnvironment(Environment):
    """Publishes three splits; `list_tasks` returns a bounded preview of two."""

    def reset(self, **kwargs: Any) -> TaskObservation:
        return TaskObservation()

    def step(self, action: TaskAction, **kwargs: Any) -> TaskObservation:
        return TaskObservation()

    @property
    def state(self) -> State:
        return State()

    def list_splits(self) -> list[str]:
        return list(SPLIT_SIZES)

    def num_tasks(self, split: str) -> int:
        return SPLIT_SIZES[split]

    def get_task(self, split: str, index: int) -> dict[str, Any]:
        if not 0 <= index < self.num_tasks(split):
            raise IndexError(f"index {index} out of range for split {split}")
        return {"id": f"{split}-{index}", "split": split, "index": index}

    def get_task_range(
        self, split: str, start: int | None = None, stop: int | None = None
    ) -> list[dict[str, Any]]:
        indices = range(self.num_tasks(split))[start:stop]
        return [self.get_task(split, index) for index in indices]

    def list_tasks(self, split: str) -> list[dict[str, Any]]:
        return self.get_task_range(split, 0, 2)


class CappedEnvironment(TaskEnvironment):
    """Returns at most `cap` task specs per range, as the guide allows."""

    cap = 2

    def get_task_range(
        self, split: str, start: int | None = None, stop: int | None = None
    ) -> list[dict[str, Any]]:
        return super().get_task_range(split, start, stop)[: self.cap]


class EmptyRangeEnvironment(CappedEnvironment):
    """Counts tasks it never returns."""

    cap = 0


class PlainEnvironment(Environment):
    """Implements no Task API method."""

    def reset(self, **kwargs: Any) -> TaskObservation:
        return TaskObservation()

    def step(self, action: TaskAction, **kwargs: Any) -> TaskObservation:
        return TaskObservation()

    @property
    def state(self) -> State:
        return State()


def _app(env_cls: type[Environment]) -> FastAPI:
    return create_app(env_cls, TaskAction, TaskObservation, env_name="task_env")


def _route(
    monkeypatch: pytest.MonkeyPatch,
    handle: Callable[[httpx.Request], httpx.Response],
) -> list[httpx.Request]:
    """Send every request of an `httpx.Client` to `handle`, and record it."""
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handle(request)

    client_cls = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **options: client_cls(transport=httpx.MockTransport(record), **options),
    )
    return seen


def _serve(monkeypatch: pytest.MonkeyPatch, app: FastAPI) -> list[httpx.Request]:
    """Serve every request of an `httpx.Client` from `app`, in process."""
    server = TestClient(app, raise_server_exceptions=False)

    def handle(request: httpx.Request) -> httpx.Response:
        response = server.request(
            request.method,
            request.url.path,
            content=request.content,
            headers=dict(request.headers),
        )
        return httpx.Response(
            response.status_code,
            content=response.content,
            headers={"content-type": response.headers["content-type"]},
        )

    return _route(monkeypatch, handle)


def _body(request: httpx.Request) -> Any:
    return json.loads(request.content) if request.content else None


class TestReads:
    def test_returns_what_the_routes_serve(self, monkeypatch) -> None:
        _serve(monkeypatch, _app(TaskEnvironment))
        client = TaskAPIClient("http://env.test:8000", env_name="task_env")

        assert client.list_splits() == [
            {"name": "train", "type": "train"},
            {"name": "test", "type": "test"},
            {"name": "empty", "type": "validation"},
        ]
        assert client.num_tasks("train") == 7
        assert client.get_task("test", 2) == {
            "id": "test-2",
            "split": "test",
            "index": 2,
        }
        assert [task["id"] for task in client.list_tasks("train")] == [
            "train-0",
            "train-1",
        ]

    @pytest.mark.parametrize(
        ("start", "stop", "expected"),
        [
            (2, 5, [2, 3, 4]),
            (None, None, [0, 1, 2, 3, 4, 5, 6]),
            (5, None, [5, 6]),
            (None, 2, [0, 1]),
        ],
    )
    def test_get_task_range_follows_slice_bounds(
        self, monkeypatch, start, stop, expected
    ) -> None:
        seen = _serve(monkeypatch, _app(TaskEnvironment))
        client = TaskAPIClient("http://env.test:8000", env_name="task_env")

        tasks = client.get_task_range("train", start, stop)

        assert [task["index"] for task in tasks] == expected
        assert _body(seen[0]) == {"split": "train", "start": start, "stop": stop}

    def test_iter_tasks_pages_up_to_num_tasks(self, monkeypatch) -> None:
        seen = _serve(monkeypatch, _app(TaskEnvironment))
        client = TaskAPIClient("http://env.test:8000", env_name="task_env")

        tasks = list(client.iter_tasks("train", page_size=3))

        assert [task["index"] for task in tasks] == list(range(7))
        assert [request.url.path for request in seen] == [
            "/task_env/num_tasks",
            "/task_env/task_range",
            "/task_env/task_range",
            "/task_env/task_range",
        ]
        assert [_body(request) for request in seen[1:]] == [
            {"split": "train", "start": 0, "stop": 3},
            {"split": "train", "start": 3, "stop": 6},
            {"split": "train", "start": 6, "stop": 7},
        ]

    def test_iter_tasks_reads_no_range_for_an_empty_split(self, monkeypatch) -> None:
        seen = _serve(monkeypatch, _app(TaskEnvironment))
        client = TaskAPIClient("http://env.test:8000", env_name="task_env")

        assert list(client.iter_tasks("empty")) == []
        assert [request.url.path for request in seen] == ["/task_env/num_tasks"]

    def test_iter_tasks_reads_on_after_a_short_page(self, monkeypatch) -> None:
        seen = _serve(monkeypatch, _app(CappedEnvironment))
        client = TaskAPIClient("http://env.test:8000", env_name="task_env")

        tasks = list(client.iter_tasks("train", page_size=5))

        assert [task["index"] for task in tasks] == list(range(7))
        assert [_body(request) for request in seen[1:]] == [
            {"split": "train", "start": 0, "stop": 5},
            {"split": "train", "start": 2, "stop": 7},
            {"split": "train", "start": 4, "stop": 7},
            {"split": "train", "start": 6, "stop": 7},
        ]

    def test_iter_tasks_refuses_an_empty_page_before_the_end(self, monkeypatch) -> None:
        _serve(monkeypatch, _app(EmptyRangeEnvironment))
        client = TaskAPIClient("http://env.test:8000", env_name="task_env")

        with pytest.raises(RuntimeError, match="7 tasks"):
            list(client.iter_tasks("train"))

    @pytest.mark.parametrize("page_size", [0, -1])
    def test_iter_tasks_refuses_a_page_size_below_one(
        self, monkeypatch, page_size
    ) -> None:
        seen = _serve(monkeypatch, _app(TaskEnvironment))
        client = TaskAPIClient("http://env.test:8000", env_name="task_env")

        with pytest.raises(ValueError, match="page_size"):
            client.iter_tasks("train", page_size=page_size)
        assert seen == []


TASK_API_CALLS = {
    "list_splits": lambda client: client.list_splits(),
    "list_tasks": lambda client: client.list_tasks("train"),
    "num_tasks": lambda client: client.num_tasks("train"),
    "get_task": lambda client: client.get_task("train", 0),
    "get_task_range": lambda client: client.get_task_range("train"),
    "iter_tasks": lambda client: list(client.iter_tasks("train")),
}


class TestErrors:
    def test_out_of_range_index_raises_index_error(self, monkeypatch) -> None:
        _serve(monkeypatch, _app(TaskEnvironment))
        client = TaskAPIClient("http://env.test:8000", env_name="task_env")

        with pytest.raises(IndexError):
            client.get_task("test", 3)

    @pytest.mark.parametrize("call", TASK_API_CALLS.values(), ids=TASK_API_CALLS)
    def test_environment_without_task_api_raises_not_implemented(
        self, monkeypatch, call
    ) -> None:
        _serve(monkeypatch, _app(PlainEnvironment))
        client = TaskAPIClient("http://env.test:8000", env_name="task_env")

        with pytest.raises(NotImplementedError, match="not supported"):
            call(client)

    def test_unknown_environment_name_raises_http_status_error(
        self, monkeypatch
    ) -> None:
        _serve(monkeypatch, _app(TaskEnvironment))
        client = TaskAPIClient("http://env.test:8000", env_name="other_env")

        with pytest.raises(httpx.HTTPStatusError) as raised:
            client.list_splits()
        assert raised.value.response.status_code == 404

    def test_server_error_raises_http_status_error(self, monkeypatch) -> None:
        _serve(monkeypatch, _app(TaskEnvironment))
        client = TaskAPIClient("http://env.test:8000", env_name="task_env")

        with pytest.raises(httpx.HTTPStatusError) as raised:
            client.num_tasks("no_such_split")
        assert raised.value.response.status_code == 500


class TestConnection:
    def test_sends_headers_with_every_request(self, monkeypatch) -> None:
        seen = _serve(monkeypatch, _app(TaskEnvironment))
        client = TaskAPIClient(
            "http://env.test:8000", env_name="task_env", headers=TOKEN_HEADERS
        )

        for call in TASK_API_CALLS.values():
            call(client)

        assert len(seen) == 7
        assert all(
            request.headers["authorization"] == TOKEN_HEADERS["Authorization"]
            for request in seen
        )

    @pytest.mark.parametrize(
        ("base_url", "http_url"),
        [
            ("ws://env.test:8000", "http://env.test:8000"),
            ("wss://env.test", "https://env.test"),
            ("http://env.test:8000/", "http://env.test:8000"),
            ("https://env.test/prefix", "https://env.test/prefix"),
            ("env.test:8000", "http://env.test:8000"),
        ],
    )
    def test_calls_the_http_url_of_the_server(
        self, monkeypatch, base_url, http_url
    ) -> None:
        seen = _route(monkeypatch, lambda request: httpx.Response(200, json=[]))
        client = TaskAPIClient(base_url, env_name="task_env")

        client.list_splits()

        assert client.base_url == http_url
        assert str(seen[0].url) == f"{http_url}/task_env/splits"


class TestFromEnvClient:
    def test_takes_url_headers_and_name_from_the_server(self, monkeypatch) -> None:
        seen = _serve(monkeypatch, _app(TaskEnvironment))
        env = GenericEnvClient(base_url="ws://env.test:8000", headers=TOKEN_HEADERS)

        client = TaskAPIClient.from_env_client(env)

        assert client.base_url == "http://env.test:8000"
        assert client.env_name == "task_env"
        assert client.num_tasks("test") == 3
        assert [request.url.path for request in seen] == [
            "/list_environments",
            "/task_env/num_tasks",
        ]
        assert all(
            request.headers["authorization"] == TOKEN_HEADERS["Authorization"]
            for request in seen
        )

    def test_accepts_a_sync_client(self, monkeypatch) -> None:
        _serve(monkeypatch, _app(TaskEnvironment))
        env = GenericEnvClient(base_url="ws://env.test:8000").sync()

        client = TaskAPIClient.from_env_client(env, env_name="task_env")

        assert client.base_url == "http://env.test:8000"
        assert client.num_tasks("train") == 7

    def test_empty_headers_send_none_of_the_env_client(self, monkeypatch) -> None:
        seen = _serve(monkeypatch, _app(TaskEnvironment))
        env = GenericEnvClient(base_url="ws://env.test:8000", headers=TOKEN_HEADERS)

        client = TaskAPIClient.from_env_client(env, env_name="task_env", headers={})
        client.list_splits()

        assert "authorization" not in seen[0].headers

    def test_given_name_and_headers_win(self, monkeypatch) -> None:
        seen = _serve(monkeypatch, _app(TaskEnvironment))
        env = GenericEnvClient(base_url="ws://env.test:8000", headers=TOKEN_HEADERS)

        client = TaskAPIClient.from_env_client(
            env, env_name="task_env", headers={"Authorization": "Bearer other"}
        )
        client.list_splits()

        assert [request.url.path for request in seen] == ["/task_env/splits"]
        assert seen[0].headers["authorization"] == "Bearer other"

    def test_name_lookup_needs_exactly_one_environment(self, monkeypatch) -> None:
        _route(
            monkeypatch, lambda request: httpx.Response(200, json=["a_env", "b_env"])
        )
        env = GenericEnvClient(base_url="ws://env.test:8000")

        with pytest.raises(ValueError, match="a_env"):
            TaskAPIClient.from_env_client(env)

    def test_client_without_url_is_refused(self, monkeypatch) -> None:
        seen = _route(monkeypatch, lambda request: httpx.Response(200, json=[]))
        env = GenericEnvClient(provider=object())

        with pytest.raises(ValueError, match="base URL"):
            TaskAPIClient.from_env_client(env, env_name="task_env")
        assert seen == []


def test_is_exported_from_core() -> None:
    from openenv.core import TaskAPIClient as exported

    assert exported is TaskAPIClient
