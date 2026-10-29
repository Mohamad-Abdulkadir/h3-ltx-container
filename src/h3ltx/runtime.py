import asyncio
import contextlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from filelock import FileLock
from pydantic import ValidationError

from h3ltx import __version__
from h3ltx.config import Settings
from h3ltx.doctor import report, software
from h3ltx.engines.base import Engine
from h3ltx.engines.comfy import ComfySupervisor
from h3ltx.engines.native import NativeEngine
from h3ltx.errors import RuntimeFault
from h3ltx.gpu import FakeBackend, GPUBackend, NVMLBackend, admit, profile_for
from h3ltx.idle import IdleStop
from h3ltx.media import encode_output, ffmpeg
from h3ltx.metrics import Sampler
from h3ltx.models import Manifest, status
from h3ltx.params import H3, LTX, REGISTRY, Common, ICLora, JobRequest, VideoInput
from h3ltx.records import RunRecord, output_metadata
from h3ltx.store import TERMINAL, Store
from h3ltx.uploads import Uploads
from h3ltx.workflows import build
from h3ltx.workflows.validate import MODEL_FIELDS, Graph, validate_graph

log = logging.getLogger(__name__)


class Runtime:
    def __init__(
        self,
        settings: Settings,
        backend: GPUBackend | None = None,
        engines: dict[int, Engine] | None = None,
    ):
        self.settings = settings
        settings.validate_startup()
        self.manifest = Manifest.load(settings.manifest_path)
        self.store = Store(settings.volume_root / "state/runtime.db")
        self.uploads = Uploads(settings.volume_root / "uploads", settings.upload_max_bytes)
        if backend is None:
            try:
                backend = NVMLBackend()
            except Exception:
                backend = FakeBackend()
        self.backend = backend
        self.gpus = backend.detect()
        self.profiles = {g.index: p for g in self.gpus if (p := profile_for(g)) is not None}
        self.supervisors = {index: ComfySupervisor(settings, index) for index in self.profiles}
        self.native = {index: NativeEngine(settings, index) for index in self.profiles}
        self.engines: dict[int, Engine] = engines or {
            i: s.client for i, s in self.supervisors.items()
        }
        self.injected = engines is not None
        self.schemas: dict[int, dict[str, Any]] = {}
        self.start_errors: dict[int, str] = {}
        self.active: dict[str, tuple[int, asyncio.Task[None], Engine]] = {}
        self.workers: list[asyncio.Task[None]] = []
        self.shutdown = asyncio.Event()
        self.owner = FileLock(str(settings.volume_root / "state/runtime.owner.lock"))
        self.idle = IdleStop(settings)
        self.versions: dict[str, Any] = {}

    async def start(self) -> None:
        self.owner.acquire(timeout=0)
        for job in self.store.recover():
            record = RunRecord(self.settings.volume_root, job)
            record.write(
                "result",
                {"state": "failed", "error": self.store.get(job["id"])["error"], "outputs": []},
            )
        self.versions = (
            await software(self.settings) if not self.injected else {"test_backend": True}
        )
        for index in self.profiles:
            try:
                if not self.injected:
                    await self.supervisors[index].start()
                self.schemas[index] = await self.engines[index].info()
            except Exception as exc:
                self.start_errors[index] = str(exc)
                log.error("GPU %d engine unavailable: %s", index, exc)
        log.info(
            "Startup doctor: %d GPUs, %d profiles, %d ComfyUI schemas; model status: %s",
            len(self.gpus),
            len(self.profiles),
            len(self.schemas),
            {
                s: sum(
                    m["status"] == s
                    for m in status(self.manifest.files, self.settings.volume_root / "models")
                )
                for s in ("ready", "missing", "size_mismatch")
            },
        )
        if not self.injected:
            diagnostics = await report(self.settings, self.gpus, self.manifest, self.schemas)
            for warning in diagnostics["warnings"]:
                log.warning("%s", warning)
        self.workers = [asyncio.create_task(self.worker(index)) for index in self.profiles]
        self.workers.append(asyncio.create_task(self.idle_loop()))

    async def close(self) -> None:
        self.shutdown.set()
        for identifier in list(self.active):
            await self.cancel(identifier)
        for task in self.workers:
            task.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        if self.injected:
            for engine in self.engines.values():
                await engine.close()
        else:
            for index in self.profiles:
                await self.supervisors[index].close()
                await self.native[index].close()
        self.store.close()
        self.owner.release()

    def input_files(self, p: Common, verify: bool = True) -> dict[str, str]:
        identifiers: list[str] = []
        if isinstance(p, H3):
            identifiers += [x for x in (p.first_frame, p.last_frame) if x]
            identifiers += [r.upload_id for r in [*p.references, *p.guides]]
        if isinstance(p, LTX):
            identifiers += [r.upload_id for r in p.keyframes]
        if isinstance(p, VideoInput):
            identifiers.append(p.input_video)
        return {
            identifier: identifier + "_" + self.uploads.get(identifier)["filename"]
            if verify
            else identifier + ".png"
            for identifier in identifiers
        }

    def inspect_inputs(self, p: Common) -> None:
        if isinstance(p, H3):
            for identifier in (p.first_frame, p.last_frame):
                if identifier and self.uploads.get(identifier)["media_type"] != "image":
                    raise RuntimeFault("media_type", "First and last frames must be images")
            for ref in [*p.references, *p.guides]:
                meta = self.uploads.get(ref.upload_id)
                if meta["media_type"] != ref.kind:
                    raise RuntimeFault("media_type", "Reference media type mismatch")
                if ref.kind != "image":
                    ref.duration_seconds = meta["duration_seconds"]
            # Re-check reference limits against measured durations, not caller claims.
            H3.model_validate(p.model_dump())
        if isinstance(p, LTX) and any(
            self.uploads.get(r.upload_id)["media_type"] != "image" for r in p.keyframes
        ):
            raise RuntimeFault("media_type", "LTX keyframes must be images")
        if isinstance(p, VideoInput):
            meta = self.uploads.get(p.input_video)
            if meta["media_type"] != "video":
                raise RuntimeFault("media_type", "This job requires a video upload")
            stream = next(s for s in meta["ffprobe"]["streams"] if s["codec_type"] == "video")
            p.width, p.height = stream["width"], stream["height"]
            a, b = map(float, stream.get("avg_frame_rate", "24/1").split("/"))
            p.fps = max(1, round(a / b)) if b else 24
            p.frames = max(1, round(meta["duration_seconds"] * p.fps))

    def graph_models(self, graph: Graph) -> list[dict[str, Any]]:
        files: dict[str, dict[str, Any]] = {}
        for node in graph.values():
            for key, value in node.get("inputs", {}).items():
                if key in MODEL_FIELDS and isinstance(value, str):
                    model = self.manifest.by_name(Path(value).name)
                    state = status([model], self.settings.volume_root / "models")[0]
                    if state["status"] != "ready":
                        raise RuntimeFault(
                            "model_missing",
                            f"Model unavailable: {model.key}",
                            model_status=state["status"],
                        )
                    files[model.key] = model.model_dump()
        return list(files.values())

    def preflight(self, kind: str, p: Common, files: dict[str, str], index: int) -> dict[str, Any]:
        gpu = next(g for g in self.gpus if g.index == index)
        arch = f"sm_{gpu.capability[0]}{gpu.capability[1]}"
        if self.versions.get("arches") and arch not in self.versions["arches"]:
            raise RuntimeFault(
                "gpu_incompatible", f"Torch build lacks {arch}; use a compatible CUDA wheel"
            )
        admission = admit(kind, p, self.profiles[index])
        if p.engine == "ltx-native" and not self.injected:
            if not self.settings.ltx_python.is_file():
                raise RuntimeFault("engine_unavailable", "ltx-native Python is unavailable", 503)
            if isinstance(p, LTX) and (
                p.sampler not in ("euler", "euler_ancestral")
                or (p.pipeline != "dev" and (p.cfg != 1 or p.negative_prompt))
            ):
                raise RuntimeFault(
                    "native_parameter",
                    "Native distilled/DFR uses cfg=1 and empty negative prompt; native "
                    "samplers are euler/euler_ancestral",
                )
            if p.attention == "sage":
                raise RuntimeFault(
                    "attention_unavailable",
                    "Pinned ltx-native supports SDPA; Sage is a ComfyUI comparison",
                )
        graph = build(kind, p, files)
        if p.engine == "comfy":
            if index not in self.schemas:
                raise RuntimeFault(
                    "engine_unavailable", self.start_errors.get(index, "ComfyUI not ready"), 503
                )
            validate_graph(graph, self.schemas[index], self.manifest, set(files.values()))
        exact_models = self.graph_models(graph)
        if isinstance(p, LTX) and p.engine == "ltx-native" and p.hdr:
            embedding = self.manifest.by_name(
                "ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors"
            )
            if status([embedding], self.settings.volume_root / "models")[0]["status"] != "ready":
                raise RuntimeFault("model_missing", "Native HDR scene embeddings unavailable")
            exact_models.append(embedding.model_dump())
        if kind == "refine.h3_face":
            detector = self.manifest.by_name(getattr(p, "detector", "face_yolov8m.pt"))
            if status([detector], self.settings.volume_root / "models")[0]["status"] != "ready":
                raise RuntimeFault("model_missing", "Face detector unavailable")
            exact_models.append(detector.model_dump())
        return {"admission": admission, "models": exact_models}

    def submit(self, request: JobRequest, parent: str | None = None) -> dict[str, Any]:
        raw = request.model_dump()
        if request.client_request_id:
            # Resolve retries before randomness, model availability, or new admission decisions.
            with self.store.lock:
                old = self.store.db.execute(
                    "SELECT * FROM jobs WHERE client_key=?", (request.client_request_id,)
                ).fetchone()
            if old:
                existing, _ = self.store.submit(raw, {}, parent)
                return self.job(existing["id"])
        try:
            p = request.resolve()
            self.inspect_inputs(p)
        except (ValidationError, ValueError) as exc:
            raise RuntimeFault("parameters_invalid", str(exc)) from exc
        files = self.input_files(p)
        eligible: list[int] = []
        selected: dict[str, Any] = {}
        reasons = []
        for index in self.profiles:
            try:
                selected = self.preflight(request.type, p, files, index)
                eligible.append(index)
            except RuntimeFault as exc:
                reasons.append(exc.payload()["error"])
        if not eligible:
            code = reasons[0]["code"] if reasons else "gpu_unavailable"
            raise RuntimeFault(code, "No eligible GPU for this request", reasons=reasons)
        # Validate post-chain parameters before consuming GPU time. Input is supplied by the parent.
        for post in request.post:
            post_params = {**post.params, "input_video": "0" * 32}
            try:
                REGISTRY[post.type].model_validate(post_params)
            except ValueError as exc:
                raise RuntimeFault("post_invalid", str(exc)) from exc
        resolved = {
            "params": p.model_dump(),
            "eligible_gpus": eligible,
            **selected,
            "software": {
                **self.versions,
                "h3ltx": __version__,
                "image_version": self.settings.image_version,
                "git_sha": self.settings.git_sha,
            },
            "environment": {
                "SLOTS_PER_GPU": self.settings.slots_per_gpu,
                **{
                    key: os.environ[key]
                    for key in (
                        "PYTORCH_CUDA_ALLOC_CONF",
                        "NVIDIA_TF32_OVERRIDE",
                        "CUBLAS_WORKSPACE_CONFIG",
                    )
                    if key in os.environ
                },
            },
            "profile": self.profiles[eligible[0]]["id"],
            "memory_policy": "aggressive offload"
            if p.text_encoder_offload != "off"
            else "smart residency",
            "cache_policy": "enabled" if p.reuse_conditioning else "cache-none",
        }
        job, created = self.store.submit(raw, resolved, parent)
        if created:
            record = RunRecord(self.settings.volume_root, job)
            record.write("workflow", {})
            record.write("timeline", {"stages": {}, "cached_nodes": []})
            record.write("metrics", {"sample_count": 0})
        self.idle.last_activity = time.monotonic()
        return self.job(job["id"])

    def job(self, identifier: str) -> dict[str, Any]:
        job = self.store.get(identifier)
        queue = self.store.list(states=["queued"], limit=1000000)
        job["queue_position"] = next(
            (i + 1 for i, row in enumerate(queue) if row["id"] == identifier), None
        )
        durations: dict[str, float] = {}
        stage, anchor = "queued", job["created"]
        for event in self.store.events(identifier):
            marker = event["data"].get("stage", event["data"].get("state"))
            if marker and marker != stage:
                if stage not in TERMINAL:
                    key = "queue_wait" if stage == "queued" else stage
                    durations[key] = durations.get(key, 0) + max(0, event["timestamp"] - anchor)
                stage, anchor = marker, event["timestamp"]
        end = job["updated"] if job["state"] in TERMINAL else time.time()
        if stage not in TERMINAL:
            key = "queue_wait" if stage == "queued" else stage
            durations[key] = durations.get(key, 0) + max(0, end - anchor)
        job["timings"] = {
            "elapsed_seconds": max(0, end - job["created"]),
            "stage_seconds": durations,
            "source": "durable progress events",
        }
        return job

    def capacity(self) -> dict[str, Any]:
        busy = {entry[0] for entry in self.active.values()}
        total = len(self.profiles)
        return {
            "slots_total": total,
            "slots_busy": len(busy),
            "slots_free": total - len(busy),
            "queue_length": len(self.store.list(states=["queued"], limit=1000000)),
            "gpus": [
                {"index": i, "state": "busy" if i in busy else "idle", "profile": p["id"]}
                for i, p in self.profiles.items()
            ],
        }

    def capabilities(self) -> dict[str, Any]:
        capabilities = []
        for kind, cls in REGISTRY.items():
            params: dict[str, Any] = (
                {"prompt": "Capability probe"}
                if kind.startswith(("h3.", "ltx."))
                else {"input_video": "0" * 32}
            )
            if kind == "h3.fl2v":
                params["first_frame"] = "0" * 32
            if kind == "h3.r2v":
                params["references"] = [{"upload_id": "0" * 32}]
            if kind in ("ltx.i2v", "ltx.flf2v"):
                params["keyframes"] = [
                    {"upload_id": "0" * 32, "frame": 0},
                    {"upload_id": "1" * 32, "frame": -1},
                ]
            reasons = []
            available = []
            p = JobRequest(type=kind, params=params).resolve()
            for index in self.profiles:
                try:
                    self.preflight(kind, p, self.input_files(p, verify=False), index)
                    available.append(index)
                except RuntimeFault as exc:
                    reasons.append({"gpu": index, **exc.payload()["error"]})
            capabilities.append(
                {
                    "type": kind,
                    "available": bool(available),
                    "eligible_gpus": available,
                    "reasons": reasons or ([] if available else [{"code": "gpu_unavailable"}]),
                    "parameters": cls.model_json_schema(),
                }
            )
        return {"job_types": capabilities, "profiles": self.profiles, "limits_provisional": True}

    async def worker(self, index: int) -> None:
        while not self.shutdown.is_set():
            eligible = [
                j["id"]
                for j in self.store.list(states=["queued"], limit=1000000)
                if index in j["resolved"].get("eligible_gpus", [])
            ]
            job = self.store.claim(index, eligible)
            if not job:
                await asyncio.sleep(self.settings.poll_seconds)
                continue
            params = REGISTRY[job["type"]].model_validate(job["resolved"]["params"])
            engine = (
                self.engines[index]
                if params.engine == "comfy" or self.injected
                else self.native[index]
            )
            task = asyncio.create_task(self.execute(job, params, index, engine))
            self.active[job["id"]] = (index, task, engine)
            try:
                await task
            except asyncio.CancelledError:
                if self.shutdown.is_set():
                    raise
            finally:
                self.active.pop(job["id"], None)
                self.idle.last_activity = time.monotonic()

    async def execute(self, job: dict[str, Any], p: Common, index: int, engine: Engine) -> None:
        record = RunRecord(self.settings.volume_root, job)
        sampler = Sampler(self.backend, index, self.settings.metric_interval)
        sampling = asyncio.create_task(sampler.run())
        started = time.monotonic()
        timeline: dict[str, Any] = {
            "stages": {"queue_wait": time.time() - job["created"]},
            "cached_nodes": [],
        }
        outputs: list[dict[str, Any]] = []
        state = "failed"
        files: dict[str, str] = {}
        engine_started = False
        error: dict[str, Any] | None = None
        try:
            preparation = time.monotonic()
            files = self.input_files(p)
            input_root = self.supervisors[index].root / "input"
            input_root.mkdir(parents=True, exist_ok=True)
            for identifier, filename in files.items():
                meta = self.uploads.get(identifier)
                target = input_root / filename
                await asyncio.to_thread(
                    shutil.copyfile, self.uploads.root / identifier / meta["filename"], target
                )
            graph = build(job["type"], p, files, prefix=job["id"])
            self.preflight(job["type"], p, files, index)
            resolved = {
                **job["resolved"],
                "gpu": next(g.as_dict() for g in self.gpus if g.index == index),
                "profile": self.profiles[index]["id"],
            }
            resolved["environment"]["CUDA_VISIBLE_DEVICES"] = str(index)
            record.write("resolved", resolved)
            if p.engine == "ltx-native":
                graph = {"native": {key: str(input_root / value) for key, value in files.items()}}
            workflow: dict[str, Any] = graph
            if p.engine == "ltx-native":
                workflow = {
                    "command": [str(self.settings.ltx_python), "-m", "h3ltx.native_runner"],
                    "protocol": {
                        "params": p.model_dump(),
                        "inputs": graph["native"],
                        "models": str(self.settings.volume_root / "models"),
                        "output": str(record.path / "outputs/native.mp4"),
                    },
                }
            record.write("workflow", workflow)
            timeline["stages"]["input_staging"] = time.monotonic() - preparation
            if not self.injected:
                if p.engine == "comfy":
                    await self.native[index].close()
                    self.supervisors[index].job_log = record.path / "logs/engine.log"
                    await self.supervisors[index].start(p)
                    self.schemas[index] = await engine.info()
                    validate_graph(graph, self.schemas[index], self.manifest, set(files.values()))
                else:
                    await self.supervisors[index].stop()
            self.store.transition(job["id"], "running", run_dir=str(record.path), resolved=resolved)

            async def progress(data: dict[str, Any]) -> None:
                self.store.progress(job["id"], data)

            engine_started = True
            async with asyncio.timeout(self.settings.job_timeout_seconds):
                if (
                    isinstance(p, ICLora)
                    and self.uploads.get(p.input_video)["duration_seconds"] > p.clip_seconds
                ):
                    result = await self.execute_clips(
                        job, p, index, engine, record, files, progress
                    )
                else:
                    result = await engine.execute(graph, p, record.path / "outputs", progress)
            if "software" in result:
                resolved["native_software"] = result["software"]
                record.write("resolved", resolved)
            observed = result.get("timeline", {})
            timeline["stages"].update(observed.pop("stages", {}))
            timeline.update(observed)
            self.store.transition(job["id"], "finalizing")
            finalizing = time.monotonic()
            for name in result.get("outputs", []):
                output = (record.path / "outputs" / name).resolve()
                if not output.is_relative_to((record.path / "outputs").resolve()):
                    raise RuntimeFault("unsafe_output", "Engine output escapes run directory", 500)
                if not self.injected:
                    gpu = next(g for g in self.gpus if g.index == index)
                    encoder = await encode_output(
                        output, gpu.nvenc, index, getattr(p, "hdr", False)
                    )
                    resolved["video_encoder"] = encoder
                    record.write("resolved", resolved)
                outputs.append(await asyncio.to_thread(output_metadata, output))
            timeline["stages"]["finalization"] = time.monotonic() - finalizing
            state = "succeeded"
        except asyncio.CancelledError:
            state = "cancelled"
            with contextlib.suppress(Exception):
                await engine.interrupt()
                if not self.injected and p.engine == "comfy":
                    await self.supervisors[index].stop()
        except TimeoutError:
            error = {"code": "job_timeout", "message": "Job exceeded configured timeout"}
            diagnostics: dict[str, Any] = {"gpu": self.backend.sample(index)}
            supervisor = self.supervisors[index]
            diagnostics["engine_log_tail"] = list(supervisor.lines)[-30:]
            if p.engine == "comfy" and not self.injected:
                with contextlib.suppress(Exception):
                    diagnostics["queue"] = await supervisor.client.diagnostics()
            with (record.path / "logs/engine.log").open("a") as stream:
                stream.write("TIMEOUT gpu=" + json.dumps(diagnostics["gpu"])[:500] + "\n")
                stream.write(
                    "TIMEOUT queue=" + json.dumps(diagnostics.get("queue", {}))[:500] + "\n"
                )
                for line in diagnostics["engine_log_tail"]:
                    stream.write("TIMEOUT log=" + line[:500] + "\n")
            record.write("hang_diagnostics", diagnostics)
            with contextlib.suppress(Exception):
                await engine.interrupt()
                if not self.injected and p.engine == "comfy":
                    await self.supervisors[index].stop()
        except Exception as exc:
            error = {
                "code": exc.code if isinstance(exc, RuntimeFault) else "job_failed",
                "message": str(exc),
                "details": exc.details if isinstance(exc, RuntimeFault) else {},
            }
            log.exception("Job %s failed", job["id"])
        finally:
            sampler.stop.set()
            await sampling
            if (
                state != "succeeded"
                and engine_started
                and (partial := getattr(engine, "last_timeline", None))
            ):
                observed = partial.result()
                timeline["stages"].update(observed.pop("stages", {}))
                timeline.update(observed)
                timeline["incomplete"] = True
            metrics = sampler.summary()
            if error and any(
                x in json.dumps(error).lower() for x in ("outofmemory", "out of memory", "cuda oom")
            ):
                metrics["oom_detected"] = True
            wall = time.monotonic() - started
            cost = wall / 3600 * self.settings.gpu_hourly_usd
            result_record = {
                "state": state,
                "error": error,
                "outputs": outputs,
                "wall_seconds": wall,
                "cost_estimate_usd": cost,
            }
            record.write("timeline", timeline)
            record.write("metrics", metrics)
            record.write("result", result_record)
            current = self.store.get(job["id"])["state"]
            if current not in TERMINAL:
                self.store.transition(
                    job["id"], state, error=error, result=result_record, run_dir=str(record.path)
                )
            self.store.measure(
                job["id"],
                {
                    "id": job["id"],
                    "type": job["type"],
                    "state": state,
                    "profile": self.profiles[index]["id"],
                    "engine": p.engine,
                    "cached": bool(timeline.get("cached_nodes")),
                    "resolution": (
                        f"{outputs[0].get('width', p.width)}x{outputs[0].get('height', p.height)}"
                        if outputs
                        else f"{p.width}x{p.height}"
                    ),
                    "steps": getattr(p, "steps", None),
                    "wall_seconds": wall,
                    "stages": timeline["stages"],
                    "cost_estimate_usd": cost,
                    "peak_vram_bytes": metrics["vram_bytes_peak"],
                    "peak_ram_bytes": metrics["ram_bytes_peak"],
                    "output_seconds": sum(o["duration_seconds"] for o in outputs),
                },
            )
            if not self.injected:
                self.supervisors[index].job_log = None
            for filename in files.values():
                (self.supervisors[index].root / "input" / filename).unlink(missing_ok=True)
            shutil.rmtree(
                self.supervisors[index].root / "input" / (job["id"] + "-clips"), ignore_errors=True
            )
        if state == "succeeded" and job["request"].get("post") and outputs:
            await self.enqueue_post(job, record, outputs[0])

    async def execute_clips(
        self,
        job: dict[str, Any],
        p: ICLora,
        index: int,
        engine: Engine,
        record: RunRecord,
        files: dict[str, str],
        progress: Any,
    ) -> dict[str, Any]:
        input_root = self.supervisors[index].root / "input"
        chunks = input_root / (job["id"] + "-clips")
        chunks.mkdir(exist_ok=True)
        source = input_root / files[p.input_video]
        await ffmpeg(
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-force_key_frames",
            f"expr:gte(t,n_forced*{p.clip_seconds})",
            "-f",
            "segment",
            "-segment_time",
            str(p.clip_seconds),
            "-reset_timestamps",
            "1",
            str(chunks / "%05d.mp4"),
        )
        graphs: list[Graph] = []
        outputs: list[Path] = []
        timelines: list[dict[str, Any]] = []
        for n, chunk in enumerate(sorted(chunks.glob("*.mp4"))):
            relative = chunk.relative_to(input_root).as_posix()
            mapping = {**files, p.input_video: relative}
            graph = build(job["type"], p, mapping, prefix=f"{job['id']}-clip-{n}")
            validate_graph(graph, self.schemas[index], self.manifest, set(mapping.values()))
            graphs.append(graph)
            record.write("workflow", {"chunks": graphs, "clip_seconds": p.clip_seconds})
            folder = record.path / "outputs" / "clips" / str(n)
            folder.mkdir(parents=True, exist_ok=True)
            result = await engine.execute(graph, p, folder, progress)
            timelines.append(result["timeline"])
            outputs.extend(folder / name for name in result["outputs"])
        if not outputs:
            raise RuntimeFault("media_processing", "No video clips produced", 500)
        listing = record.path / "outputs" / "clips.concat.txt"
        listing.write_text(
            "".join(f"file '{path.relative_to(listing.parent).as_posix()}'\n" for path in outputs)
        )
        target = record.path / "outputs" / "upscaled.mp4"
        await ffmpeg("-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(target))
        stages: dict[str, float] = {}
        for timeline in timelines:
            for stage, seconds in timeline.get("stages", {}).items():
                stages[stage] = stages.get(stage, 0) + seconds
        return {
            "outputs": [target.name],
            "timeline": {
                "chunks": timelines,
                "stages": stages,
                "cached_nodes": [node for t in timelines for node in t.get("cached_nodes", [])],
            },
        }

    async def enqueue_post(
        self, job: dict[str, Any], record: RunRecord, output: dict[str, Any]
    ) -> None:
        first, *rest = job["request"]["post"]
        identifier, path = self.uploads.begin()
        await asyncio.to_thread(shutil.copyfile, record.path / "outputs" / output["name"], path)
        await self.uploads.finish(identifier, path)
        request = JobRequest(
            type=first["type"],
            params={**first["params"], "input_video": identifier},
            client_request_id=f"post:{job['id']}",
            tags=job["tags"],
            post=rest,
        )
        try:
            child = self.submit(request, parent=job["id"])
            self.store.event(job["id"], {"child_run": child["id"]})
        except RuntimeFault as exc:
            # Preserve the failure as a durable linked run, even when unavailable at dispatch.
            child, _ = self.store.submit(request.model_dump(), {"eligible_gpus": []}, job["id"])
            child_record = RunRecord(self.settings.volume_root, child)
            self.store.transition(child["id"], "preparing")
            result = {"state": "failed", "error": exc.payload()["error"], "outputs": []}
            child_record.write("result", result)
            self.store.transition(
                child["id"],
                "failed",
                error=result["error"],
                result=result,
                run_dir=str(child_record.path),
            )

    async def cancel(self, identifier: str) -> dict[str, Any]:
        job = self.store.get(identifier)
        if job["state"] in TERMINAL:
            return self.job(identifier)
        if identifier in self.active:
            _, task, engine = self.active[identifier]
            await engine.interrupt()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self.store.get(identifier)["state"] not in TERMINAL:
            record = RunRecord(self.settings.volume_root, job)
            result: dict[str, Any] = {"state": "cancelled", "error": None, "outputs": []}
            record.write("result", result)
            self.store.transition(identifier, "cancelled", result=result, run_dir=str(record.path))
        return self.job(identifier)

    async def system(self) -> dict[str, Any]:
        self.gpus = self.backend.detect()
        return await report(self.settings, self.gpus, self.manifest, self.schemas)

    async def idle_loop(self) -> None:
        while not self.shutdown.is_set():
            try:
                await self.idle.check(
                    lambda: bool(self.active) or bool(self.store.list(states=["queued"], limit=1))
                )
            except Exception:
                log.exception("Idle stop failed; will retry")
            await asyncio.sleep(30)
