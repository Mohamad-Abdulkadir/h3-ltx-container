import secrets
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

H3_FRAME_EXPRESSION = "max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17"


def h3_frames(seconds: float) -> int:
    base = max(5, round(seconds * 24))
    return base + (5 - base % 17) % 17


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Common(StrictModel):
    seed: int = Field(default_factory=lambda: secrets.randbits(63), ge=0, le=2**64 - 1)
    force: bool = False
    engine: Literal["comfy", "ltx-native"] = "comfy"
    attention: Literal["sdpa", "sage"] = "sdpa"
    text_encoder_offload: Literal["auto", "on", "off"] = "auto"
    reuse_conditioning: bool = False
    width: int = Field(default=1152, ge=64, le=8192, multiple_of=32)
    height: int = Field(default=640, ge=64, le=8192, multiple_of=32)
    aspect_ratio: Literal["16:9", "9:16", "1:1"] | None = None
    resolution: int | None = Field(default=None, ge=64, multiple_of=32)

    @model_validator(mode="after")
    def resolve_size(self) -> Self:
        if self.aspect_ratio is not None and self.resolution is not None:
            a, b = map(int, self.aspect_ratio.split(":"))
            if a >= b:
                self.height, self.width = self.resolution, round(self.resolution * a / b / 32) * 32
            else:
                self.width, self.height = self.resolution, round(self.resolution * b / a / 32) * 32
        return self


class Reference(StrictModel):
    upload_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    kind: Literal["image", "video", "audio"] = "image"
    duration_seconds: float = Field(default=0, ge=0, le=15)
    frame: int = Field(default=0, ge=-1)
    strength: float = Field(default=1, ge=0, le=1)


class H3(Common):
    prompt: str = Field(min_length=1, max_length=50000)
    duration_seconds: float = Field(default=5, ge=4, le=15)
    steps: int = Field(default=20, ge=1, le=200)
    sampler: str = "res_multistep"
    scheduler: str = "simple"
    shift: float = Field(default=12, gt=0, le=100)
    turbo: Literal["none", "4-step", "8-step"] = "none"
    turbo_version: Literal["default", "latest"] = "default"
    weights: Literal["pruned-bf16", "full-bf16", "int8", "nvfp4"] = "pruned-bf16"
    latent_upscale: Literal["off", "2x"] = "off"
    refine_steps: int = Field(default=6, ge=1, le=100)
    first_frame: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    last_frame: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    references: list[Reference] = Field(default_factory=list, max_length=12)
    guides: list[Reference] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def limits(self) -> Self:
        if self.engine != "comfy":
            raise ValueError("H3 supports ComfyUI only")
        for kind, maximum in (("image", 9), ("video", 3), ("audio", 3)):
            if sum(r.kind == kind for r in self.references) > maximum:
                raise ValueError(f"At most {maximum} {kind} references")
        clips = [r for r in self.references if r.kind != "image"]
        if any(not 2 <= r.duration_seconds <= 15 for r in clips):
            raise ValueError("Reference clips must be 2–15 seconds")
        if sum(r.duration_seconds for r in clips) > 15:
            raise ValueError("Reference clips must total at most 15 seconds")
        if self.turbo != "none":
            self.steps = 4 if self.turbo == "4-step" else 8
        return self


class LTX(Common):
    width: int = Field(default=1280, ge=64, le=8192, multiple_of=64)
    height: int = Field(default=704, ge=64, le=8192, multiple_of=64)
    prompt: str = Field(min_length=1, max_length=50000)
    negative_prompt: str = ""
    frames: int = Field(default=121, ge=9, le=993)
    fps: int = Field(default=24, ge=1, le=120)
    steps: int = Field(default=8, ge=1, le=100)
    cfg: float = Field(default=1, ge=0, le=30)
    sampler: str = "euler_ancestral"
    pipeline: Literal["distilled", "dev", "dfr"] = "distilled"
    temporal_upscale: Literal[0, 1, 2] = 0
    hdr: bool = False
    tiled: bool = True
    keyframes: list[Reference] = Field(default_factory=list, max_length=12)
    weights: Literal["bf16", "int8", "nvfp4"] = "bf16"

    @model_validator(mode="after")
    def grid(self) -> Self:
        if self.width % 64 or self.height % 64:
            raise ValueError("Resolved LTX dimensions must be multiples of 64")
        if self.frames % 8 != 1:
            raise ValueError("LTX frames must be 8k+1")
        if self.engine == "ltx-native" and self.weights != "bf16":
            raise ValueError("ltx-native requires BF16 checkpoints")
        if any(r.kind != "image" or r.frame >= self.frames for r in self.keyframes):
            raise ValueError("Keyframes must be images at valid frame positions")
        if self.temporal_upscale and self.pipeline != "dfr":
            raise ValueError("Temporal upscaling requires pipeline=dfr")
        if self.temporal_upscale and self.fps != 24:
            raise ValueError("Temporal DFR upscale uses a 24 fps source")
        return self


class VideoInput(Common):
    input_video: str = Field(pattern=r"^[a-f0-9]{32}$")
    frames: int = Field(default=121, ge=1)
    fps: int = Field(default=24, ge=1, le=120)


class SeedVR(VideoInput):
    model: Literal["7b", "7b-sharp", "3b"] = "7b"
    target_short_side: int = Field(default=1080, ge=64, le=4320)
    tiled: bool = True
    tile_size: int = Field(default=512, ge=128, le=2048, multiple_of=64)
    frames_per_chunk: int = Field(default=21, ge=5, le=257)


class ICLora(VideoInput):
    mode: Literal["pixel-spatial", "refine-details"] = "pixel-spatial"
    prompt: str = Field(default="High fidelity video.", min_length=1)
    clip_seconds: int = Field(default=5, ge=1, le=15)
    tiled: bool = True


class Face(VideoInput):
    prompt: str = "Natural detailed face, preserve identity."
    steps: int = Field(default=6, ge=1, le=100)
    denoise: float = Field(default=0.25, gt=0, le=1)
    detector: str = "face_yolov8m.pt"


REGISTRY: dict[str, type[Common]] = {
    "h3.t2v": H3,
    "h3.fl2v": H3,
    "h3.r2v": H3,
    "ltx.t2v": LTX,
    "ltx.i2v": LTX,
    "ltx.flf2v": LTX,
    "ltx.dfr": LTX,
    "upscale.seedvr2": SeedVR,
    "upscale.ltx_iclora": ICLora,
    "refine.h3_face": Face,
}


class Post(StrictModel):
    type: Literal["upscale.seedvr2", "upscale.ltx_iclora", "refine.h3_face"]
    params: dict[str, Any] = Field(default_factory=dict)


class JobRequest(StrictModel):
    type: str
    params: dict[str, Any]
    client_request_id: str | None = Field(default=None, min_length=1, max_length=128)
    priority: int = Field(default=0, ge=-100, le=100)
    tags: list[str] = Field(default_factory=list, max_length=32)
    post: list[Post] = Field(default_factory=list, max_length=8)

    def resolve(self) -> Common:
        if self.type not in REGISTRY:
            raise ValueError(f"Unknown job type: {self.type}")
        raw = (
            {"sampler": "euler", **self.params, "pipeline": "dfr"}
            if self.type == "ltx.dfr"
            else self.params
        )
        p = REGISTRY[self.type].model_validate(raw)
        if isinstance(p, H3):
            if self.type == "h3.fl2v" and not (p.first_frame or p.last_frame or p.guides):
                raise ValueError("h3.fl2v requires a first frame, last frame or guides")
            if self.type == "h3.r2v" and not p.references:
                raise ValueError("h3.r2v requires references")
        if isinstance(p, LTX):
            if self.type in ("ltx.i2v", "ltx.flf2v") and not p.keyframes:
                raise ValueError("Image-conditioned LTX requires keyframes")
            if self.type == "ltx.flf2v" and not {0, p.frames - 1} <= {
                p.frames - 1 if r.frame == -1 else r.frame for r in p.keyframes
            }:
                raise ValueError("ltx.flf2v requires first and last keyframes")
            if self.type == "ltx.dfr":
                p.pipeline = "dfr"
        return p
