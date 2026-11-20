# SPDX-License-Identifier: BSD-3-Clause

"""
Read the tasks any environment server publishes through the HTTP Task API.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .env_client import EnvClient
    from .sync_client import SyncEnvClient

# The default message timeout of `EnvClient`.
_TIMEOUT_S = 60.0


def _http_url(url: str) -> str:
    """Return the HTTP form of a server URL, the reverse of `convert_to_ws_url`."""
    url = url.rstrip("/")
    if url.startswith("ws://"):
        return "http://" + url[len("ws://") :]
    if url.startswith("wss://"):
        return "https://" + url[len("wss://") :]
    if url.startswith(("http://", "https://")):
        return url
    return "http://" + url


def _send(
    method: str,
    url: str,
    headers: dict[str, str] | None,
    body: dict[str, Any] | None = None,
) -> Any:
    """Call one route and turn the server's error statuses back into exceptions."""
    with httpx.Client(headers=headers, timeout=_TIMEOUT_S) as http:
        response = http.request(method, url, json=body)
    if response.status_code == 501:
        raise NotImplementedError(f"{url}: {response.text}")
    if response.status_code == 400:
        raise IndexError(f"{url}: {response.text}")
    response.raise_for_status()
    return response.json()


class TaskAPIClient:
    """
    Read the task specs of any environment server through its Task API routes.

    The routes serve metadata only: a task is still chosen by passing its split and
    index to `reset()`. Task specs come back as untyped JSON, exactly as the server
    serves them, so the client works for every environment that implements the Task
    API. The server's error statuses become exceptions again: `501` raises
    `NotImplementedError` and `400` raises `IndexError`. The core environment
    clients keep no Task API methods of their own.

    Args:
        base_url (`str`):
            URL of the environment server: `http(s)://`, or the `ws(s)://` URL an
            [`~openenv.core.env_client.EnvClient`] uses. A URL without a scheme is
            read as `http://`.
        env_name (`str`):
            The `env_name` the server was created with, which every Task API route
            starts with.
        headers (`dict[str, str]`, *optional*):
            Extra HTTP headers sent with every request, e.g.
            `{"Authorization": f"Bearer {hf_token}"}` for a private Hugging Face Space.

    Attributes:
        base_url (`str`):
            HTTP URL of the server, without a trailing slash.
        env_name (`str`):
            Name the routes start with.

    Examples:

    ```python
    from openenv.core import TaskAPIClient

    client = TaskAPIClient("http://localhost:8000", env_name="latex_ocr_env")
    client.list_splits()  # [{"name": "train", "type": "train"}, ...]
    for task in client.iter_tasks("test"):
        print(task["id"])
    ```
    """

    def __init__(
        self,
        base_url: str,
        *,
        env_name: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = _http_url(base_url)
        self.env_name = env_name
        self._headers = headers

    @classmethod
    def from_env_client(
        cls,
        env: EnvClient | SyncEnvClient,
        *,
        env_name: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> TaskAPIClient:
        """
        Build a client for the server an environment client talks to.

        Args:
            env (`EnvClient` or `SyncEnvClient`):
                An [`~openenv.core.env_client.EnvClient`] or
                [`~openenv.core.sync_client.SyncEnvClient`] that has a base URL: one
                created with `base_url`, or one that has started its server through
                a provider.
            env_name (`str`, *optional*):
                Name the routes start with. Defaults to the one name the server
                lists at `/list_environments`.
            headers (`dict[str, str]`, *optional*):
                Extra HTTP headers. Defaults to the headers `env` was created with.

        Returns:
            `TaskAPIClient`: A client for the same server.

        Raises:
            `ValueError`: If `env` has no base URL yet, or if `env_name` is not
                given and the server does not list exactly one environment.
        """
        if env.base_url is None:
            raise ValueError(
                "The environment client has no base URL yet: connect it first, "
                "or create the TaskAPIClient with the server's URL."
            )
        if headers is None:
            headers = env._headers
        base_url = _http_url(env.base_url)
        if env_name is None:
            names = _send("GET", f"{base_url}/list_environments", headers)
            if len(names) != 1:
                raise ValueError(
                    f"The server lists the environments {names!r}; pass env_name."
                )
            env_name = names[0]
        return cls(base_url, env_name=env_name, headers=headers)

    def _call(self, method: str, route: str, body: dict[str, Any] | None = None) -> Any:
        return _send(
            method, f"{self.base_url}/{self.env_name}/{route}", self._headers, body
        )

    def list_splits(self) -> list[dict[str, Any]]:
        """
        Return the splits the environment publishes.

        Returns:
            `list[dict[str, Any]]`: One descriptor per split. A split the environment
                names by a string comes as `{"name": ..., "type": ...}`; a dict or a
                model comes as the environment wrote it.
        """
        return self._call("GET", "splits")

    def list_tasks(self, split: str) -> list[Any]:
        """
        Return the task specs of a split, or a bounded preview of them.

        An environment with a large split may return only its first tasks here; use
        `iter_tasks` to read all of them.

        Args:
            split (`str`):
                Split name.

        Returns:
            `list[Any]`: Task specs, as the server serves them.
        """
        return self._call("POST", "tasks", {"split": split})["tasks"]

    def num_tasks(self, split: str) -> int:
        """
        Return the number of tasks in a split.

        Args:
            split (`str`):
                Split name.

        Returns:
            `int`: The true number of tasks, even when `list_tasks` returns fewer.
        """
        return self._call("POST", "num_tasks", {"split": split})["num_tasks"]

    def get_task(self, split: str, index: int) -> Any:
        """
        Return one task spec.

        Args:
            split (`str`):
                Split name.
            index (`int`):
                Position of the task in the split.

        Returns:
            `Any`: The task spec, as the server serves it.

        Raises:
            `IndexError`: If the server reports `index` out of range (`400`).
        """
        return self._call("POST", "task", {"split": split, "index": index})["task"]

    def get_task_range(
        self, split: str, start: int | None = None, stop: int | None = None
    ) -> list[Any]:
        """
        Return the task specs between two positions, with Python slice bounds.

        Args:
            split (`str`):
                Split name.
            start (`int`, *optional*):
                First position, included. Defaults to the start of the split.
            stop (`int`, *optional*):
                Last position, excluded. Defaults to the end of the split.

        Returns:
            `list[Any]`: Task specs in `[start, stop)`, as the server serves them.
        """
        body = {"split": split, "start": start, "stop": stop}
        return self._call("POST", "task_range", body)["tasks"]

    def iter_tasks(self, split: str, page_size: int = 256) -> Iterator[Any]:
        """
        Yield every task spec of a split, in order, reading it in pages.

        The total comes from `num_tasks`, so this reads the whole split even when
        `list_tasks` returns only a preview. Each page starts where the last one
        ended, so an environment that returns fewer task specs than asked is still
        read to the end.

        Args:
            split (`str`):
                Split name.
            page_size (`int`, *optional*, defaults to `256`):
                Number of task specs asked for per request.

        Returns:
            `Iterator[Any]`: The task specs, as the server serves them.

        Raises:
            `ValueError`: If `page_size` is below 1, when `iter_tasks` is called.
            `RuntimeError`: If the server returns no task specs before `num_tasks`
                is reached, while iterating.
        """
        if page_size < 1:
            raise ValueError(f"page_size must be at least 1, got {page_size}")
        return self._pages(split, page_size)

    def _pages(self, split: str, page_size: int) -> Iterator[Any]:
        # Apart from `iter_tasks`, so that `page_size` is checked at the call.
        total = self.num_tasks(split)
        start = 0
        while start < total:
            page = self.get_task_range(split, start, min(start + page_size, total))
            if not page:
                raise RuntimeError(
                    f"The server counts {total} tasks in split {split!r} but "
                    f"returned none from position {start}."
                )
            yield from page
            start += len(page)


__all__ = ["TaskAPIClient"]
