"""Original ComfyUI extension: persistent loader objects and measured GPU residency.

Copied into custom_nodes/h3ltx_memory at image build. Imports only the standard
library until ComfyUI loads the extension. No runtime/API secrets are required.
"""

import functools
import inspect
import json
import os
import sys
import time
from typing import Any, cast

_CACHE: dict[str, dict[str, Any]] = {}
_PIPELINES: dict[str, list[Any]] = {}
_INSTALLED = False
_READS = 0
_READ_SECONDS = 0.0


def _key(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_key(v) for v in value]
    if isinstance(value, dict):
        return {k: _key(v) for k, v in value.items()}
    return {"object": id(value)}


def _cached(name: str, function: Any, arguments: dict[str, Any]) -> Any:
    global _READS, _READ_SECONDS
    key = json.dumps([name, _key(arguments)], sort_keys=True)
    if key in _CACHE:
        _CACHE[key]["last_read_s"] = 0.0
        return _CACHE[key]["value"]
    tick = time.perf_counter()
    value = function()
    elapsed = time.perf_counter() - tick
    if name != "MiniMaxH3SigmaShift":
        _READS += 1
        _READ_SECONDS += elapsed
    _CACHE[key] = {"value": value, "last_read_s": time.perf_counter() - tick, "class": name}
    return value


def _outputs(value: Any) -> Any:
    result = getattr(value, "result", value)
    return result if isinstance(result, (tuple, list)) else (result,)


def _patcher(value: Any) -> Any:
    if hasattr(value, "load_device") and hasattr(value, "model_size"):
        return value
    return getattr(value, "patcher", None)


def residency() -> dict[str, Any]:
    result = {}
    for key, values in list(_PIPELINES.items()):
        components = []
        for value in values:
            patcher = _patcher(value)
            size = int(patcher.model_size()) if patcher else 0
            used = int(patcher.loaded_size()) if patcher else 0
            if patcher is None and hasattr(value, "model"):
                parameters = list(value.model.parameters())
                size = sum(p.nelement() * p.element_size() for p in parameters)
                used = sum(p.nelement() * p.element_size() for p in parameters if p.is_cuda)
            components.append(
                {
                    "bytes": size,
                    "gpu_bytes": used,
                    "resident": bool(used >= size and used > 0),
                }
            )
        result[key] = {"resident": all(c["resident"] for c in components), "components": components}
    return result


class H3LTXPreload:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "action": (["load", "unload"],),
                "pipeline": ("STRING",),
                "nonce": ("STRING",),
                "component_names": ("STRING",),
            },
            "optional": {
                "detector": ("STRING", {"default": ""}),
                **{f"component_{i}": ("*",) for i in range(32)},
            },
        }

    RETURN_TYPES: tuple[str, ...] = ()
    FUNCTION = "operate"
    OUTPUT_NODE = True
    CATEGORY = "h3ltx/internal"

    def operate(
        self, action: str, pipeline: str, nonce: str, component_names: str, **components: Any
    ) -> dict[str, Any]:
        import comfy.model_management as memory
        import torch

        _ = nonce
        tick = time.perf_counter()
        reports = []
        detector = components.pop("detector", "")
        if action == "unload":
            selected = list(_PIPELINES) if pipeline == "all" else [pipeline]
            targets = [v for name in selected for v in _PIPELINES.pop(name, [])]
            retained = [v for values in _PIPELINES.values() for v in values]
            release = {id(_patcher(v)) for v in targets if all(v is not r for r in retained)}
            keep = [m for m in memory.current_loaded_models if id(m.model) not in release]
            memory.free_memory(1e32, memory.get_torch_device(), keep_loaded=keep)
            if pipeline == "all":
                memory.unload_all_models()
                _CACHE.clear()
            else:
                for key, entry in list(_CACHE.items()):
                    values = _outputs(entry["value"])
                    if any(any(v is target for target in targets) for v in values) and not any(
                        any(v is r for r in retained) for v in values
                    ):
                        _CACHE.pop(key)
            memory.soft_empty_cache()
        else:
            names = json.loads(component_names)
            values = [components[f"component_{i}"] for i in range(len(names))]
            # Keep all supplied components together; fail if the selected GPU cannot fit.
            patchers = [_patcher(v) for v in values]
            for i, (name, value, patcher) in enumerate(zip(names, values, patchers, strict=True)):
                read_s = next(
                    (
                        e["last_read_s"]
                        for e in _CACHE.values()
                        if any(v is value for v in _outputs(e["value"]))
                    ),
                    None,
                )
                transfer = time.perf_counter()
                if patcher is not None:
                    memory.load_models_gpu(
                        [p for p in patchers[: i + 1] if p is not None], force_full_load=True
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                reports.append(
                    {
                        "component": name,
                        "read_from_disk_s": read_s,
                        "transfer_to_gpu_s": time.perf_counter() - transfer,
                        "compile_s": None,
                        "compile_status": "not_triggered",
                        "timing_note": (
                            "loader wall time; mmap page faults may occur during transfer"
                        ),
                    }
                )
            if detector:
                import importlib

                import nodes

                module = importlib.import_module(
                    nodes.NODE_CLASS_MAPPINGS["H3FaceTrackCrop"].__module__
                )
                read = time.perf_counter()
                face = _cached(
                    "detector", lambda: module._load_detector(detector), {"name": detector}
                )
                read_s = time.perf_counter() - read
                transfer = time.perf_counter()
                face.to(memory.get_torch_device())
                torch.cuda.synchronize()
                values.append(face)
                reports.append(
                    {
                        "component": detector,
                        "read_from_disk_s": read_s,
                        "transfer_to_gpu_s": time.perf_counter() - transfer,
                        "compile_s": None,
                    }
                )
            _PIPELINES[pipeline] = values
            if not residency()[pipeline]["resident"]:
                raise RuntimeError(
                    "Pipeline did not become fully GPU resident; reduce the selected load set"
                )
        report = {
            "pipeline": pipeline,
            "action": action,
            "seconds": time.perf_counter() - tick,
            "components": reports,
            "resident": residency(),
        }
        return {"ui": {"h3ltx_load": [report]}, "result": ()}


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    import nodes
    from comfy_api.latest import io

    for name in (
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "LoraLoaderModelOnly",
        "LatentUpscaleModelLoader",
        "MiniMaxH3SigmaShift",
    ):
        cls = nodes.NODE_CLASS_MAPPINGS[name]
        method = getattr(cls, "FUNCTION", "execute")
        original = getattr(cls, method)
        if issubclass(cls, io.ComfyNode):

            def factory(function: Any, label: str) -> Any:
                @functools.wraps(function)
                def wrapper(subclass: Any, **kwargs: Any) -> Any:
                    return _cached(label, lambda: function(**kwargs), kwargs)

                return classmethod(cast(Any, wrapper))

            setattr(cls, method, factory(original, name))
        else:

            def legacy_factory(function: Any, label: str) -> Any:
                @functools.wraps(function)
                def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                    bound = inspect.signature(function).bind(self, *args, **kwargs)
                    bound.apply_defaults()
                    parameters = {k: v for k, v in bound.arguments.items() if k != "self"}
                    return _cached(label, lambda: function(self, *args, **kwargs), parameters)

                return wrapper

            setattr(cls, method, legacy_factory(original, name))
    # Offload text encoders immediately before sampling, without clearing persistent loader objects.
    sampler = nodes.NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]
    original_sample = sampler.execute

    @functools.wraps(original_sample)
    def sample(subclass: Any, **kwargs: Any) -> Any:
        if os.getenv("H3LTX_TEXT_OFFLOAD", "auto") != "off":
            import comfy.model_management as memory

            text = {
                id(_patcher(v))
                for e in _CACHE.values()
                if e["class"] == "CLIPLoader"
                for v in _outputs(e["value"])
            }
            keep = [m for m in memory.current_loaded_models if id(m.model) not in text]
            memory.free_memory(1e32, memory.get_torch_device(), keep_loaded=keep)
        return original_sample(**kwargs)

    sampler.execute = classmethod(cast(Any, sample))
    sampler.sample = sampler.execute
    from aiohttp import web
    from server import PromptServer

    async def status(request: Any) -> Any:
        return web.json_response(
            {
                "pipelines": residency(),
                "cached_objects": len(_CACHE),
                "loader_reads": _READS,
                "loader_read_s": _READ_SECONDS,
            }
        )

    PromptServer.instance.routes.get("/h3ltx/resident")(status)
    _INSTALLED = True


NODE_CLASS_MAPPINGS = {"H3LTXPreload": H3LTXPreload}
if "nodes" in sys.modules:
    install()
