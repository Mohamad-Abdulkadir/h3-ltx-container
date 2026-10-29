import os
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")
    api_token: SecretStr = SecretStr("")
    hf_token: SecretStr = SecretStr("")
    runpod_api_key: SecretStr = SecretStr("")
    volume_root: Path = Field(
        default_factory=lambda: Path(
            "/runpod-volume" if Path("/runpod-volume").exists() else "/workspace"
        )
    )
    mode: Literal["pod", "serverless"] = "pod"
    allow_ephemeral: bool = False
    slots_per_gpu: int = Field(default=1, ge=1)
    comfy_root: Path = Path("/opt/ComfyUI")
    comfy_python: Path = Path("/opt/venvs/comfy/bin/python")
    ltx_python: Path = Path("/opt/venvs/ltx/bin/python")
    comfy_port: int = 8188
    manifest_path: Path = Path(__file__).parent / "data/manifest.yaml"
    model_groups: str = "h3-core,h3-upscale,seedvr2,ltx-core,ltx-iclora"
    upload_max_bytes: int = 2 * 1024**3
    job_timeout_seconds: int = Field(default=7200, ge=1)
    idle_stop_minutes: float = Field(default=0, ge=0)
    gpu_hourly_usd: float = Field(default=0, ge=0)
    image_version: str = "development"
    git_sha: str = "unknown"
    poll_seconds: float = 0.25
    metric_interval: float = 0.5

    def validate_startup(self) -> None:
        if len(self.api_token.get_secret_value()) < 32:
            raise ValueError("API_TOKEN must contain at least 32 characters")
        if self.slots_per_gpu != 1:
            raise ValueError("BF16 profiles require SLOTS_PER_GPU=1")
        root = self.volume_root
        if not self.allow_ephemeral and self.mode == "pod":
            # Mountinfo handles a bind-mounted child as well as the mount itself.
            mounts = Path("/proc/self/mountinfo")
            mounted = os.path.ismount(root)
            if mounts.exists():
                mounted |= any(
                    line.split()[4] == str(root) for line in mounts.read_text().splitlines()
                )
            if not mounted:
                raise ValueError("Writable mounted volume required; ALLOW_EPHEMERAL=true overrides")
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".h3ltx-write-probe"
        probe.write_text("ok")
        probe.unlink()
        for name in ("state", "models", "runs", "uploads", "engines"):
            (root / name).mkdir(exist_ok=True)
