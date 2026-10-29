import asyncio
import json
import os
import shutil
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import httpx
from websockets.asyncio.client import connect

from h3ltx.config import Settings
from h3ltx.engines.base import Progress
from h3ltx.errors import RuntimeFault
from h3ltx.params import Common
from h3ltx.workflows.validate import Graph


def stage_for(cls: str) -> str:
    if "Sampler" in cls and cls != "KSamplerSelect":
        return "sampling"
    if "TextEncode" in cls or "MiniMaxH3ImageToVideo" in cls or "ReferenceToVideo" in cls:
        return "text_encode"
    if "Decode" in cls:
        return "vae_decode"
    if "Upscal" in cls or "Upsampl" in cls:
        return "upscale"
    if cls in ("CreateVideo", "SaveVideo"):
        return "video_encode"
    if "Loader" in cls:
        return "model_load"
    return "conditioning"


class EventTimeline:
    """Intervals between server execution events; unobservable stages remain absent."""

    def __init__(self, graph: Graph):
        self.graph = graph
        self.nodes: list[dict[str, Any]] = []
        self.steps: list[dict[str, Any]] = []
        self.cached: list[str] = []
        self.current: tuple[str, float] | None = None
        self.last_step: float | None = None
        self.epoch_offset = time.time() - time.monotonic()

    def consume(self, event: dict[str, Any]) -> dict[str, Any]:
        kind, data = event["type"], event.get("data", {})
        # Normalize server milliseconds to the local monotonic clock.
        now = (
            float(data["timestamp"]) / 1000 - self.epoch_offset
            if "timestamp" in data
            else time.monotonic()
        )
        result: dict[str, Any] = {}
        if kind == "execution_cached":
            self.cached.extend(str(n) for n in data.get("nodes", []))
        if kind == "executing":
            if self.current:
                node, start = self.current
                cls = self.graph[node]["class_type"]
                self.nodes.append(
                    {
                        "node": node,
                        "class_type": cls,
                        "stage": stage_for(cls),
                        "seconds": max(0, now - start),
                    }
                )
            node = data.get("node")
            self.current = (str(node), now) if node else None
            self.last_step = None
            if node and str(node) in self.graph:
                result["stage"] = stage_for(self.graph[str(node)]["class_type"])
        if kind == "progress":
            result.update(
                progress=100 * data.get("value", 0) / max(data.get("max", 1), 1), stage="sampling"
            )
            self.steps.append(
                {
                    "node": data.get("node"),
                    "step": data.get("value"),
                    "seconds": now - self.last_step if self.last_step is not None else None,
                }
            )
            self.last_step = now
        return result

    def result(self) -> dict[str, Any]:
        stages: dict[str, float] = {}
        for node in self.nodes:
            stages[node["stage"]] = stages.get(node["stage"], 0) + node["seconds"]
        return {
            "nodes": self.nodes,
            "sampling_steps": self.steps,
            "stages": stages,
            "cached_nodes": self.cached,
            "timing_source": "ComfyUI websocket execution events; receipt timestamps where server "
            "timestamps absent",
        }


class ComfyClient:
    def __init__(self, url: str, output_root: Path):
        self.url, self.output_root = url, output_root
        self.http = httpx.AsyncClient(base_url=url, timeout=60, trust_env=False)
        self.active_prompt: str | None = None
        self.last_timeline: EventTimeline | None = None

    async def info(self) -> dict[str, Any]:
        response = await self.http.get("/object_info")
        response.raise_for_status()
        return dict(response.json())

    async def execute(
        self, graph: Graph, params: Common, output: Path, progress: Progress
    ) -> dict[str, Any]:
        identifier = uuid.uuid4().hex
        timeline = EventTimeline(graph)
        self.last_timeline = timeline
        async with connect(
            self.url.replace("http", "ws", 1) + f"/ws?clientId={identifier}", max_size=16 * 1024**2
        ) as websocket:
            response = await self.http.post(
                "/prompt", json={"prompt": graph, "client_id": identifier}
            )
            if response.status_code != 200:
                raise RuntimeFault(
                    "engine_validation", "ComfyUI rejected graph", 422, response=response.json()
                )
            prompt = response.json()["prompt_id"]
            self.active_prompt = prompt
            try:
                async for raw in websocket:
                    if not isinstance(raw, str):
                        continue
                    event = json.loads(raw)
                    data = event.get("data", {})
                    if data.get("prompt_id") not in (None, prompt):
                        continue
                    update = timeline.consume(event)
                    if update:
                        await progress(update)
                    if event["type"] in ("execution_error", "execution_interrupted"):
                        raise RuntimeFault(
                            "engine_execution", "ComfyUI execution failed", 500, event=data
                        )
                    if event["type"] == "executing" and data.get("node") is None:
                        break
                history_response = await self.http.get(f"/history/{prompt}")
                history_response.raise_for_status()
                history = history_response.json().get(prompt)
                if not history or history.get("status", {}).get("status_str") != "success":
                    raise RuntimeFault(
                        "engine_history", "Prompt did not complete successfully", 500
                    )
                copied = []
                for node_outputs in history.get("outputs", {}).values():
                    for values in node_outputs.values():
                        if not isinstance(values, list):
                            continue
                        for value in values:
                            if not isinstance(value, dict) or "filename" not in value:
                                continue
                            if value.get("type", "output") != "output":
                                continue
                            source = (
                                self.output_root / value.get("subfolder", "") / value["filename"]
                            ).resolve()
                            if (
                                not source.is_relative_to(self.output_root.resolve())
                                or not source.is_file()
                            ):
                                raise RuntimeFault(
                                    "unsafe_output", "Engine output path is invalid", 500
                                )
                            target = output / source.name
                            await asyncio.to_thread(shutil.move, source, target)
                            copied.append(target.name)
                if not copied:
                    raise RuntimeFault("engine_no_output", "Engine returned no output files", 500)
                return {"timeline": timeline.result(), "history": history, "outputs": copied}
            finally:
                self.active_prompt = None

    async def interrupt(self) -> None:
        # One runtime slot owns this process. Remove only its active prompt if queued.
        if self.active_prompt:
            await self.http.post("/queue", json={"delete": [self.active_prompt]})
        await self.http.post("/interrupt")

    async def diagnostics(self) -> dict[str, Any]:
        response = await self.http.get("/queue")
        return dict(response.json())

    async def close(self) -> None:
        await self.http.aclose()


class ComfySupervisor:
    def __init__(self, settings: Settings, index: int):
        self.settings, self.index = settings, index
        self.root = settings.volume_root / "engines" / f"gpu-{index}"
        self.output_root = self.root / "output"
        self.process: asyncio.subprocess.Process | None = None
        self.log_task: asyncio.Task[None] | None = None
        self.lines: deque[str] = deque(maxlen=200)
        self.job_log: Path | None = None
        self.client = ComfyClient(
            f"http://127.0.0.1:{settings.comfy_port + index}", self.output_root
        )
        self.policy: tuple[str, bool, str] | None = None

    async def start(self, params: Common | None = None) -> None:
        policy = (
            (params.attention, params.reuse_conditioning, params.text_encoder_offload)
            if params
            else ("sdpa", False, "auto")
        )
        if self.process and self.process.returncode is None and policy == self.policy:
            return
        await self.stop()
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in ("input", "output", "temp", "user"):
            (self.root / directory).mkdir(exist_ok=True)
        custom = self.root / "custom_nodes"
        if not custom.exists():
            custom.symlink_to(self.settings.comfy_root / "custom_nodes", target_is_directory=True)
        models = self.root / "models"
        if not models.exists():
            models.symlink_to(self.settings.volume_root / "models", target_is_directory=True)
        flags = (
            ["--use-pytorch-cross-attention"] if policy[0] == "sdpa" else ["--use-sage-attention"]
        )
        if not policy[1]:
            flags.append("--cache-none")
        if policy[2] != "off":
            flags.append("--disable-smart-memory")
        self.process = await asyncio.create_subprocess_exec(
            str(self.settings.comfy_python),
            str(Path(__file__).parents[1] / "engine_boot.py"),
            str(os.getpid()),
            str(self.settings.comfy_root / "main.py"),
            "--listen",
            "127.0.0.1",
            "--port",
            str(self.settings.comfy_port + self.index),
            "--base-directory",
            str(self.root),
            "--disable-partner-nodes",
            *flags,
            cwd=self.settings.comfy_root,
            env={
                k: v
                for k, v in {**os.environ, "CUDA_VISIBLE_DEVICES": str(self.index)}.items()
                if k not in ("API_TOKEN", "HF_TOKEN", "RUNPOD_API_KEY")
            },
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self.policy = policy
        self.log_task = asyncio.create_task(self.read_logs())
        for _ in range(240):
            if self.process.returncode is not None:
                raise RuntimeFault(
                    "engine_start", "ComfyUI exited during startup", 503, log=list(self.lines)[-30:]
                )
            try:
                await self.client.info()
                return
            except httpx.HTTPError:
                await asyncio.sleep(0.5)
        raise RuntimeFault("engine_start", "ComfyUI readiness timed out", 503)

    async def read_logs(self) -> None:
        assert self.process and self.process.stdout
        while raw := await self.process.stdout.readline():
            line = raw.decode(errors="replace").rstrip()
            self.lines.append(line)
            if self.job_log:
                with self.job_log.open("a") as output:
                    output.write(line + "\n")

    async def stop(self) -> None:
        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 20)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        if self.log_task:
            await self.log_task
        self.process = None

    async def close(self) -> None:
        await self.stop()
        await self.client.close()
