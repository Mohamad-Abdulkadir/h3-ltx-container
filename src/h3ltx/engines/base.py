from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol

from h3ltx.params import Common
from h3ltx.workflows.validate import Graph

Progress = Callable[[dict[str, Any]], Awaitable[None]]


class Engine(Protocol):
    async def info(self) -> dict[str, Any]: ...
    async def execute(
        self, graph: Graph, params: Common, output: Path, progress: Progress
    ) -> dict[str, Any]: ...
    async def interrupt(self) -> None: ...
    async def close(self) -> None: ...
