import asyncio
import json
import os
from pathlib import Path
from typing import Any

from h3ltx.config import Settings
from h3ltx.engines.base import Progress
from h3ltx.errors import RuntimeFault
from h3ltx.params import Common
from h3ltx.workflows.validate import Graph


class NativeEngine:
    """Persistent JSON-lines subprocess. One command at a time per GPU."""

    def __init__(self, settings: Settings, index: int):
        self.settings, self.index = settings, index
        self.process: asyncio.subprocess.Process | None = None
        self.log_task: asyncio.Task[None] | None = None
        self.log_path: Path | None = None

    async def info(self) -> dict[str, Any]:
        return {"available": self.settings.ltx_python.is_file(), "engine": "ltx-native"}

    async def start(self) -> None:
        if self.process and self.process.returncode is None:
            return
        self.process = await asyncio.create_subprocess_exec(
            str(self.settings.ltx_python),
            "-m",
            "h3ltx.native_runner",
            env={
                k: v
                for k, v in {**os.environ, "CUDA_VISIBLE_DEVICES": str(self.index)}.items()
                if k not in ("API_TOKEN", "HF_TOKEN", "RUNPOD_API_KEY")
            },
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.log_task = asyncio.create_task(self.logs())

    async def logs(self) -> None:
        assert self.process and self.process.stderr
        while line := await self.process.stderr.readline():
            if self.log_path:
                with self.log_path.open("ab") as output:
                    output.write(line)

    async def execute(
        self, graph: Graph, params: Common, output: Path, progress: Progress
    ) -> dict[str, Any]:
        self.log_path = output.parent / "logs/engine.log"
        await self.start()
        assert self.process and self.process.stdin and self.process.stdout
        command = {
            "params": params.model_dump(),
            "output": str(output / "native.mp4"),
            "models": str(self.settings.volume_root / "models"),
            "inputs": graph.get("native", {}),
        }
        self.process.stdin.write(json.dumps(command).encode() + b"\n")
        await self.process.stdin.drain()
        while line := await self.process.stdout.readline():
            event = json.loads(line)
            if event["type"] == "progress":
                await progress(event["data"])
            elif event["type"] == "result":
                return dict(event["data"])
            elif event["type"] == "error":
                raise RuntimeFault("native_execution", event["message"], 500)
        raise RuntimeFault("native_exit", "ltx-native process exited without a result", 500)

    async def interrupt(self) -> None:
        await self.close()

    async def close(self) -> None:
        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 15)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        if self.log_task:
            await self.log_task
        self.process = None
