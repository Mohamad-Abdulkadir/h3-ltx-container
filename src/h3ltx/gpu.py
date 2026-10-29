import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from h3ltx.errors import RuntimeFault
from h3ltx.params import H3, LTX, Common, SeedVR, h3_frames


@dataclass
class GPU:
    index: int
    name: str
    uuid: str
    capability: tuple[int, int]
    total: int
    used: int = 0
    temperature: int = 0
    power_watts: float = 0
    utilization: float = 0
    driver: str = "unknown"
    nvenc: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class GPUBackend(Protocol):
    def detect(self) -> list[GPU]: ...
    def sample(self, index: int) -> dict[str, float]: ...


class NVMLBackend:
    def __init__(self) -> None:
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()

    def detect(self) -> list[GPU]:
        n = self.nvml
        devices = []
        for i in range(n.nvmlDeviceGetCount()):
            h = n.nvmlDeviceGetHandleByIndex(i)
            mem = n.nvmlDeviceGetMemoryInfo(h)
            devices.append(
                GPU(
                    i,
                    str(n.nvmlDeviceGetName(h)),
                    str(n.nvmlDeviceGetUUID(h)),
                    tuple(n.nvmlDeviceGetCudaComputeCapability(h)),
                    mem.total,
                    driver=str(n.nvmlSystemGetDriverVersion()),
                )
            )
            sample = self.sample(i)
            devices[-1].used = int(sample["vram_bytes"])
            devices[-1].temperature = int(sample["temperature_c"])
            devices[-1].power_watts = sample["power_watts"]
            devices[-1].utilization = sample["utilization"]
        return devices

    def sample(self, index: int) -> dict[str, float]:
        n = self.nvml
        h = n.nvmlDeviceGetHandleByIndex(index)

        def safe(call: Any, fallback: float = 0) -> float:
            try:
                return float(call())
            except n.NVMLError:
                return fallback

        return {
            "vram_bytes": float(n.nvmlDeviceGetMemoryInfo(h).used),
            "utilization": safe(lambda: n.nvmlDeviceGetUtilizationRates(h).gpu),
            "power_watts": safe(lambda: n.nvmlDeviceGetPowerUsage(h)) / 1000,
            "temperature_c": safe(lambda: n.nvmlDeviceGetTemperature(h, 0)),
        }


class FakeBackend:
    def __init__(self, devices: list[GPU] | None = None):
        self.devices = devices or []

    def detect(self) -> list[GPU]:
        return self.devices

    def sample(self, index: int) -> dict[str, float]:
        return {"vram_bytes": 1024, "utilization": 50, "power_watts": 100, "temperature_c": 40}


def encode_probe(index: int) -> bool:
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=size=128x128:rate=24",
                "-t",
                "1",
                "-c:v",
                "h264_nvenc",
                "-gpu",
                str(index),
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=20,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def profiles() -> list[dict[str, Any]]:
    return list(
        yaml.safe_load((Path(__file__).parent / "data/profiles.yaml").read_text())["profiles"]
    )


def profile_for(gpu: GPU) -> dict[str, Any] | None:
    for profile in profiles():
        if list(gpu.capability) == profile["capability"] and gpu.total / 1e9 >= profile["min_gb"]:
            return profile
    return None


def admit(kind: str, params: Common, profile: dict[str, Any]) -> dict[str, Any]:
    weights = getattr(params, "weights", "bf16")
    if profile["precision"] == "quantized" and "bf16" in weights:
        raise RuntimeFault(
            "precision_incompatible", "ada-48 requires quantized weights; BF16 refused"
        )
    if weights == "nvfp4" and profile["capability"][0] < 12:
        raise RuntimeFault("precision_incompatible", "NVFP4 requires Blackwell")
    if isinstance(params, H3):
        frames = h3_frames(params.duration_seconds)
        factor = 4 if params.latent_upscale == "2x" else 1
    elif isinstance(params, LTX):
        frames = (params.frames - 1) * 2**params.temporal_upscale + 1
        factor = 1
        if params.width >= 3840 and not params.tiled:
            raise RuntimeFault("tiling_required", "Experimental LTX 4K requires tiling")
    else:
        frames, factor = getattr(params, "frames", 121), 1
    width, height = params.width, params.height
    if isinstance(params, SeedVR):
        scale = params.target_short_side / min(width, height)
        width, height = round(width * scale), round(height * scale)
    pixels_frames = width * height * frames * factor
    budget = profile["budgets"][kind.split(".")[0]]
    if pixels_frames > budget and not params.force:
        raise RuntimeFault(
            "admission_budget",
            "Provisional GPU budget exceeded; reduce size/duration "
            "or set force=true after reviewing memory requirements",
            requested=pixels_frames,
            maximum=budget,
            profile=profile["id"],
        )
    return {
        "pixels_x_frames": pixels_frames,
        "budget": budget,
        "provisional": True,
        "experimental": width >= 3840,
        "forced": params.force,
    }
