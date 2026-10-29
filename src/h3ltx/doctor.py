import asyncio
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any

import httpx
import psutil

from h3ltx.config import Settings
from h3ltx.gpu import GPU, encode_probe
from h3ltx.models import Manifest, status


async def software(settings: Settings) -> dict[str, Any]:
    code = (
        "import json,torch,importlib.metadata as m\n"
        "d={'torch':str(torch.__version__),'cuda':torch.version.cuda,"
        "'arches':torch.cuda.get_arch_list(),"
        "'bf16':torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}\n"
        "for p in ('transformers','comfy-kitchen','comfy-aimdo','sageattention','av'):\n"
        " try: d[p]=m.version(p)\n"
        " except m.PackageNotFoundError: d[p]=None\n"
        "print(json.dumps(d))"
    )
    try:
        process = await asyncio.create_subprocess_exec(
            str(settings.comfy_python),
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={
                k: v
                for k, v in os.environ.items()
                if k not in ("API_TOKEN", "HF_TOKEN", "RUNPOD_API_KEY")
            },
        )
        out, _ = await asyncio.wait_for(process.communicate(), 30)
        data: dict[str, Any] = (
            json.loads(out) if process.returncode == 0 else {"error": "torch probe failed"}
        )
    except (OSError, ValueError, TimeoutError):
        data = {"error": "ComfyUI Python unavailable"}
    versions = Path("/opt/h3ltx-versions.json")
    data.update(
        json.loads(versions.read_text())
        if versions.exists()
        else {"comfy_commit": "e6354106fa0761ea1f40773b9971985dbe51421a", "custom_nodes": {}}
    )
    return data


async def report(
    settings: Settings,
    gpus: list[GPU],
    manifest: Manifest,
    engine_info: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    versions = await software(settings)
    for gpu in gpus:
        gpu.nvenc = await asyncio.to_thread(encode_probe, gpu.index)
    files = status(
        manifest.select(settings.model_groups.split(",")), settings.volume_root / "models"
    )
    warnings = []
    dc = os.getenv("RUNPOD_DC_ID", "unknown")
    country = os.getenv("RUNPOD_COUNTRY", "unknown").upper()
    if country == "UNKNOWN":
        if dc.startswith("EUR-IS-"):
            country = "IS"
        elif dc.startswith("EU-"):
            country = "EU"
        elif dc.startswith("US-"):
            country = "US"
    excluded = {
        "EU",
        "US",
        "GB",
        "UK",
        "KR",
        "AT",
        "BE",
        "BG",
        "HR",
        "CY",
        "CZ",
        "DK",
        "EE",
        "FI",
        "FR",
        "DE",
        "GR",
        "HU",
        "IE",
        "IT",
        "LV",
        "LT",
        "LU",
        "MT",
        "NL",
        "PL",
        "PT",
        "RO",
        "SK",
        "SI",
        "ES",
        "SE",
    }
    if country == "UNKNOWN" or country in excluded:
        warnings.append(
            "MiniMax H3 licence: deployment territory is unknown or excluded "
            "(US/EU/UK/South Korea). Verify location and terms; not legal advice. "
            "Warning only."
        )
    if (
        any(m.gated for m in manifest.select(settings.model_groups.split(",")))
        and not settings.hf_token.get_secret_value()
    ):
        warnings.append("HF_TOKEN absent; gated model sync requires a token and accepted licences")
    for gpu in gpus:
        arch = f"sm_{gpu.capability[0]}{gpu.capability[1]}"
        if arch not in versions.get("arches", []):
            warnings.append(
                f"GPU {gpu.index}: torch build does not list {arch}; verify CUDA compatibility"
            )
    summaries = {}
    required = {
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "SamplerCustomAdvanced",
        "SaveVideo",
        "MiniMaxH3ImageToVideo",
        "MiniMaxH3ReferenceToVideo",
        "MiniMaxH3LatentUpscaleWithModel",
        "LTXVAddGeneratedKeyframes",
        "LTXVLatentUpsampler",
        "SeedVR2Conditioning",
        "SeedVR2TemporalChunk",
        "SeedVR2TemporalMerge",
        "H3FaceTrackCrop",
        "H3FaceStitch",
    }
    async with httpx.AsyncClient(timeout=2) as client:
        for gpu in gpus:
            schema = (engine_info or {}).get(gpu.index, {})
            if engine_info is None:
                try:
                    response = await client.get(
                        f"http://127.0.0.1:{settings.comfy_port + gpu.index}/object_info"
                    )
                    response.raise_for_status()
                    schema = response.json()
                except (httpx.HTTPError, ValueError):
                    schema = {}
            summaries[gpu.index] = {
                "reachable": bool(schema),
                "node_count": len(schema),
                "missing_required_nodes": sorted(required - schema.keys()),
            }
    return {
        "gpus": [g.as_dict() for g in gpus],
        "software": versions,
        "image_version": settings.image_version,
        "git_sha": settings.git_sha,
        "cpu_cores": os.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "disk_free_volume": shutil.disk_usage(settings.volume_root).free,
        "disk_free_container": shutil.disk_usage("/").free,
        "volume": {
            "path": str(settings.volume_root),
            "mounted": os.path.ismount(settings.volume_root),
            "writable": os.access(settings.volume_root, os.W_OK),
        },
        "platform": platform.platform(),
        "runpod": {"pod_id": os.getenv("RUNPOD_POD_ID"), "dc": dc, "country": country},
        "models": files,
        "engines": summaries,
        "warnings": warnings,
    }
