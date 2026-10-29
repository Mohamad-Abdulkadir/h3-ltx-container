"""Durable, measured model operations serialized with each GPU's generation worker."""

import asyncio
import contextlib
import time
import uuid
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from h3ltx.engines.base import Progress
from h3ltx.errors import RuntimeFault
from h3ltx.metrics import Sampler
from h3ltx.params import H3, LTX, Common, Face, ICLora, SeedVR, StrictModel
from h3ltx.records import RunRecord
from h3ltx.store import TERMINAL
from h3ltx.workflows import build
from h3ltx.workflows.validate import MODEL_FIELDS, Graph, validate_graph

if TYPE_CHECKING:
    from h3ltx.runtime import Runtime


def pipelines() -> dict[str, tuple[str, Common]]:
    table: dict[str, tuple[str, Common]] = {
        "h3-fl2v": ("h3.t2v", H3(prompt="Preload")),
        "h3-ref2v": ("h3.r2v", H3(prompt="Preload")),
        "h3-upscale": ("h3.t2v", Common()),
        "ltx-distilled": ("ltx.t2v", LTX(prompt="Preload")),
        "ltx-dev": ("ltx.t2v", LTX(prompt="Preload", pipeline="dev")),
        "ltx-dfr": ("ltx.dfr", LTX(prompt="Preload", pipeline="dfr", sampler="euler")),
        "ltx-hdr": ("ltx.dfr", LTX(prompt="Preload", pipeline="dfr", hdr=True)),
        "h3-face": ("refine.h3_face", Face(input_video="0" * 32)),
    }
    for prefix, kind in (("h3-fl2v", "h3.t2v"), ("h3-ref2v", "h3.r2v")):
        table[prefix + "-full"] = (kind, H3(prompt="Preload", weights="full-bf16"))
        table[prefix + "-upscaled"] = (kind, H3(prompt="Preload", latent_upscale="2x"))
        for steps in (4, 8):
            table[prefix + f"-turbo{steps}"] = (
                kind,
                H3(prompt="Preload", turbo="4-step" if steps == 4 else "8-step"),
            )
    for model in ("7b", "7b-sharp", "3b"):
        table["seedvr2-" + model] = ("upscale.seedvr2", SeedVR(input_video="0" * 32, model=model))
    for mode in ("pixel-spatial", "refine-details"):
        table["ltx-iclora-" + mode] = (
            "upscale.ltx_iclora",
            ICLora(input_video="0" * 32, mode=mode),
        )
    return table


def pipeline_for(kind: str, p: Common) -> str:
    if isinstance(p, H3):
        key = "h3-ref2v" if kind == "h3.r2v" else "h3-fl2v"
        if p.weights == "full-bf16":
            key += "-full"
        elif p.latent_upscale == "2x":
            key += "-upscaled"
        elif p.turbo != "none":
            key += "-turbo" + p.turbo[0]
        return key
    if isinstance(p, LTX):
        return "ltx-hdr" if p.hdr else "ltx-" + p.pipeline
    if isinstance(p, SeedVR):
        return "seedvr2-" + p.model
    if isinstance(p, ICLora):
        return "ltx-iclora-" + p.mode
    return "h3-face"


class LoadRequest(StrictModel):
    model: str
    params: dict[str, Any] = Field(default_factory=dict)
    gpus: list[int] | Literal["all"] = "all"
    profile: str | None = None
    engine: Literal["comfy", "ltx-native"] = "comfy"


class UnloadRequest(StrictModel):
    model: str = "all"
    gpus: list[int] | Literal["all"] = "all"
    engine: Literal["comfy", "ltx-native"] = "comfy"


def load_graph(key: str, kind: str, params: Common, action: str = "load") -> Graph:
    if action == "unload":
        graph: Graph = {}
    elif key == "h3-upscale":
        graph = {
            "1": {
                "class_type": "LatentUpscaleModelLoader",
                "inputs": {"model_name": "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"},
            }
        }
    else:
        complete = build(kind, params)
        keep = {
            node
            for node, value in complete.items()
            if value["class_type"]
            in (
                "UNETLoader",
                "CLIPLoader",
                "VAELoader",
                "LoraLoaderModelOnly",
                "LatentUpscaleModelLoader",
                "MiniMaxH3SigmaShift",
            )
        }
        graph = {node: value for node, value in complete.items() if node in keep}
    components = []
    links: dict[str, Any] = {}
    # Load each final patched model and all other loader components; identical underlying
    # tensors are shared by ComfyUI's ModelPatchers, rather than duplicated in the image.
    for i, (identifier, node) in enumerate(graph.items()):
        label = next(
            (value for field, value in node["inputs"].items() if field in MODEL_FIELDS),
            node["class_type"],
        )
        components.append(label)
        links[f"component_{i}"] = [identifier, 0]
    import json

    graph["h3ltx_load"] = {
        "class_type": "H3LTXPreload",
        "inputs": {
            "action": action,
            "pipeline": key,
            "nonce": uuid.uuid4().hex,
            "component_names": json.dumps(components),
            "detector": params.detector if isinstance(params, Face) else "",
            **links,
        },
    }
    return graph


class ModelLoading:
    def __init__(self, runtime: "Runtime"):
        self.runtime = runtime
        self.resident: dict[int, dict[str, Any]] = {i: {} for i in runtime.profiles}
        self.tasks: set[asyncio.Task[None]] = set()
        self.active: set[int] = set()

    def targets(self, selected: list[int] | str, profile: str | None = None) -> list[int]:
        targets = sorted(self.runtime.profiles) if isinstance(selected, str) else list(selected)
        if (
            not targets
            or len(targets) != len(set(targets))
            or any(i not in self.runtime.profiles for i in targets)
        ):
            raise RuntimeFault("gpu_unavailable", "Select eligible GPU indices or all")
        if profile and any(self.runtime.profiles[i]["id"] != profile for i in targets):
            raise RuntimeFault(
                "profile_mismatch", "Requested profile does not match selected GPU hardware"
            )
        return targets

    def submit(self, request: LoadRequest | UnloadRequest, action: str) -> dict[str, Any]:
        if action == "load" and request.model not in pipelines():
            raise RuntimeFault(
                "pipeline_unknown", "Unknown loadable pipeline", models=sorted(pipelines())
            )
        targets = self.targets(request.gpus, getattr(request, "profile", None))
        if (
            request.engine == "ltx-native"
            and not request.model.startswith("ltx-")
            and request.model != "all"
        ):
            raise RuntimeFault("engine_incompatible", "Native engine loads LTX pipelines only")
        job, _ = self.runtime.store.submit(
            {"type": "system." + action, "params": request.model_dump(), "tags": []},
            {
                "params": request.model_dump(),
                "eligible_gpus": [],
                "model": request.model,
                "engine": request.engine,
                "target_gpus": targets,
            },
        )
        record = RunRecord(self.runtime.settings.volume_root, job)
        record.write("workflow", {})
        record.write("metrics", {})
        record.write("timeline", {})
        task = asyncio.create_task(self.execute(job, targets, action))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        self.runtime.idle.last_activity = time.monotonic()
        return {"load_id": job["id"], "state": job["state"], "gpus": targets}

    async def operate(
        self,
        index: int,
        key: str,
        kind: str,
        params: Common,
        record: RunRecord,
        progress: Progress,
        action: str = "load",
    ) -> dict[str, Any]:
        runtime = self.runtime
        engine: Any = runtime.engines[index]
        if not runtime.injected:
            if params.engine == "ltx-native":
                await runtime.supervisors[index].stop()
                engine = runtime.native[index]
            else:
                await runtime.native[index].close()
                supervisor = runtime.supervisors[index]
                previous = supervisor.process
                await supervisor.start(params)
                if previous is not supervisor.process:
                    self.resident[index].clear()
                runtime.schemas[index] = await engine.info()
        graph = load_graph(key, kind, params, action)
        exact_models = (
            runtime.graph_models(graph) if action == "load" and not runtime.injected else []
        )
        record.write(
            f"resolved.gpu-{index}",
            {
                "params": params.model_dump(),
                "models": exact_models,
                "software": runtime.versions,
                "gpu": next(g.as_dict() for g in runtime.gpus if g.index == index),
            },
        )
        if params.engine == "comfy":
            validate_graph(graph, runtime.schemas[index], runtime.manifest)
        record.write(f"workflow.gpu-{index}", graph)
        sampler = Sampler(runtime.backend, index, runtime.settings.metric_interval)
        sampling = asyncio.create_task(sampler.run())
        started = time.monotonic()
        try:
            if params.engine == "ltx-native":
                method = engine.prepare
                result = await method(params, action, record.path / "outputs", progress)
            else:
                method = engine.load
                result = await method(graph, params, record.path / "outputs", progress)
            metadata = {
                "model": key,
                "engine": params.engine,
                "params": params.model_dump(),
                "profile": runtime.profiles[index]["id"],
                "loaded_at": time.time(),
                "resident": True,
                "components": result.get("components", []),
            }
            if action == "load":
                self.resident[index][key] = metadata
            elif key == "all":
                self.resident[index].clear()
            else:
                self.resident[index].pop(key, None)
            reply = {
                "gpu": index,
                "profile": runtime.profiles[index]["id"],
                "model": key,
                "wall_seconds": time.monotonic() - started,
                **result,
            }
            return reply
        finally:
            sampler.stop.set()
            await sampling
            metrics = sampler.summary()
            record.write(f"metrics.gpu-{index}", metrics)
            if "reply" in locals():
                reply["metrics"] = metrics

    async def execute(self, job: dict[str, Any], targets: list[int], action: str) -> None:
        runtime, key = self.runtime, job["resolved"]["model"]
        if job["resolved"]["engine"] == "ltx-native" and key not in (
            "all",
            "ltx-distilled",
            "ltx-dev",
            "ltx-dfr",
            "ltx-hdr",
        ):
            runtime.store.transition(job["id"], "preparing")
            runtime.store.transition(
                job["id"],
                "failed",
                error={
                    "code": "engine_incompatible",
                    "message": "Native IC-LoRA standalone loading is unavailable",
                },
            )
            return
        record = RunRecord(runtime.settings.volume_root, job)
        runtime.store.transition(job["id"], "preparing", run_dir=str(record.path))
        results = []
        state = "succeeded"
        error: dict[str, Any] | None = None
        try:
            runtime.store.transition(job["id"], "running")
            for index in targets:
                async with runtime.gpu_locks[index]:
                    self.active.add(index)
                    kind, params = pipelines().get(key, ("h3.t2v", Common()))
                    params = params.model_copy(deep=True)
                    params = type(params).model_validate(
                        {
                            **params.model_dump(),
                            **job["resolved"]["params"].get("params", {}),
                            "engine": job["resolved"]["engine"],
                        }
                    )

                    async def progress(data: dict[str, Any], gpu: int = index) -> None:
                        runtime.store.progress(job["id"], {**data, "gpu": gpu})

                    try:
                        async with asyncio.timeout(runtime.settings.job_timeout_seconds):
                            results.append(
                                await self.operate(
                                    index, key, kind, params, record, progress, action
                                )
                            )
                    finally:
                        self.active.discard(index)
            runtime.store.transition(job["id"], "finalizing")
        except asyncio.CancelledError:
            state = "cancelled"
            error = {"code": "load_cancelled", "message": "Load operation interrupted"}
        except Exception as exc:
            state = "failed"
            error = {
                "code": "load_failed",
                "message": str(exc),
                "details": exc.details if isinstance(exc, RuntimeFault) else {},
            }
        finally:
            result = {
                "state": state,
                "model": key,
                "operation": action,
                "gpus": results,
                "error": error,
                "cost_estimate_usd": sum(g["wall_seconds"] for g in results)
                / 3600
                * runtime.settings.gpu_hourly_usd,
            }
            record.write("result", result)
            record.write("timeline", {"gpus": results, "operation": action})
            record.write("metrics", {"per_gpu": [f"metrics.gpu-{i}.json" for i in targets]})
            if runtime.store.get(job["id"])["state"] not in TERMINAL:
                runtime.store.transition(job["id"], state, error=error, result=result)
            runtime.store.measure(
                job["id"],
                {
                    "type": "system." + action,
                    "model": key,
                    "profile": ",".join(runtime.profiles[i]["id"] for i in targets),
                    "state": state,
                    "wall_seconds": sum(g["wall_seconds"] for g in results),
                    "warm": True,
                    "stages": {},
                    "operation": action,
                    "cost_estimate_usd": result["cost_estimate_usd"],
                },
            )
            runtime.idle.last_activity = time.monotonic()

    async def refresh(self, index: int, engine: str = "comfy") -> None:
        if engine != "comfy":
            return
        client = self.runtime.engines[index]
        if hasattr(client, "resident"):
            with contextlib.suppress(Exception):
                report = await client.resident()
                for key, metadata in self.resident[index].items():
                    metadata.update(report.get("pipelines", {}).get(key, {"resident": False}))

    def warm(self, index: int, key: str, params: Common) -> bool:
        resident = self.resident[index].get(key)
        if (
            not resident
            or resident["engine"] != params.engine
            or not resident.get("resident", True)
        ):
            return False
        previous = resident["params"]
        return all(
            previous.get(name) == getattr(params, name, None)
            for name in (
                "weights",
                "turbo",
                "turbo_version",
                "latent_upscale",
                "attention",
                "text_encoder_offload",
                "reuse_conditioning",
                "pipeline",
                "hdr",
            )
        )

    async def close(self) -> None:
        for task in list(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
