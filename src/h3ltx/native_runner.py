"""ltx-pipelines adapter; executed only inside /opt/venvs/ltx.

Stdout is a machine protocol, stderr is engine logging. A single pipeline remains
resident until its model/policy configuration changes or the subprocess is stopped.
"""

import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any, TextIO


def run(protocol: TextIO) -> None:
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_pipelines.dfr_pipeline import DFRPipeline
    from ltx_pipelines.distilled import DistilledPipeline
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
    from ltx_pipelines.utils.helpers import get_device
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.types import ImageConditioningInput, OffloadMode

    pipeline: Any = None
    configuration: Any = None

    def emit(kind: str, data: Any) -> None:
        protocol.write(json.dumps({"type": kind, "data": data}) + "\n")
        protocol.flush()

    from collections.abc import Iterator
    from functools import wraps
    from inspect import isgeneratorfunction, signature

    from ltx_pipelines.utils.blocks import (
        AudioDecoder,
        DiffusionStage,
        PromptEncoder,
        VideoDecoder,
        VideoUpsampler,
    )

    stages: dict[str, float] = {}
    node_timings: list[dict[str, Any]] = []
    sampling_steps: list[dict[str, Any]] = []

    def instrument(cls: Any, stage: str) -> None:
        original = cls.__call__

        def finish(start: float) -> None:
            elapsed = time.monotonic() - start
            stages[stage] = stages.get(stage, 0) + elapsed
            node_timings.append({"class": cls.__name__, "stage": stage, "seconds": elapsed})

        if isgeneratorfunction(original):

            @wraps(original)
            def generator(self: Any, *args: Any, **kwargs: Any) -> Any:
                emit("progress", {"stage": stage, "progress": 0})
                started = time.monotonic()
                try:
                    yield from original(self, *args, **kwargs)
                finally:
                    finish(started)

            cls.__call__ = generator
        else:

            @wraps(original)
            def measured(self: Any, *args: Any, **kwargs: Any) -> Any:
                emit("progress", {"stage": stage, "progress": 0})
                started = time.monotonic()
                if cls is DiffusionStage:
                    bound = signature(original).bind(self, *args, **kwargs)
                    denoiser = bound.arguments["denoiser"]
                    total = len(bound.arguments["sigmas"]) - 1

                    def timed_denoiser(*values: Any, **options: Any) -> Any:
                        tick = time.monotonic()
                        result = denoiser(*values, **options)
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        step = int(values[4] if len(values) > 4 else options["step_index"])
                        sampling_steps.append({"step": step, "seconds": time.monotonic() - tick})
                        emit(
                            "progress",
                            {"stage": "sampling", "progress": 100 * (step + 1) / max(total, 1)},
                        )
                        return result

                    bound.arguments["denoiser"] = timed_denoiser
                    result = original(*bound.args, **bound.kwargs)
                else:
                    result = original(self, *args, **kwargs)
                if isinstance(result, Iterator):
                    creation_seconds = time.monotonic() - started

                    def timed_iterator() -> Any:
                        elapsed = creation_seconds
                        try:
                            while True:
                                tick = time.monotonic()
                                try:
                                    item = next(result)
                                except StopIteration:
                                    break
                                elapsed += time.monotonic() - tick
                                yield item
                        finally:
                            stages[stage] = stages.get(stage, 0) + elapsed
                            node_timings.append(
                                {"class": cls.__name__, "stage": stage, "seconds": elapsed}
                            )

                    return timed_iterator()
                finish(started)
                return result

            cls.__call__ = measured

    for cls, stage in (
        (PromptEncoder, "text_encode"),
        (DiffusionStage, "sampling"),
        (VideoDecoder, "vae_decode"),
        (AudioDecoder, "audio_decode"),
        (VideoUpsampler, "upscale"),
    ):
        instrument(cls, stage)

    with contextlib.redirect_stdout(sys.stderr):
        for line in sys.stdin:
            stages = {}
            node_timings = []
            sampling_steps = []
            try:
                command = json.loads(line)
                p, root = command["params"], Path(command["models"])
                if p["weights"] != "bf16":
                    raise ValueError("Native runner accepts BF16 only")
                config = (str(root), p["pipeline"], p["text_encoder_offload"], p["attention"])
                if config != configuration:
                    pipeline = None
                    import gc

                    import torch

                    gc.collect()
                    torch.cuda.empty_cache()
                    start = time.monotonic()
                    emit("progress", {"stage": "model_load", "progress": 0})
                    kind = "dev" if p["pipeline"] == "dev" else "distilled"
                    paths = ModelPaths.from_split(
                        transformer_path=str(
                            root
                            / f"diffusion_models/ltx-2.5-22b-{kind}-transformer-bf16.safetensors"
                        ),
                        text_encoder_path=str(
                            root / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
                        ),
                        video_vae_path=str(root / "vae/ltx-2.5-video-vae-bf16.safetensors"),
                        audio_vae_path=str(root / "vae/ltx-2.5-audio-vae-bf16.safetensors"),
                    )
                    kwargs: dict[str, Any] = dict(
                        model_paths=paths,
                        spatial_upsampler_path=str(
                            root
                            / (
                                "latent_upscale_models/"
                                "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
                            )
                        ),
                        loras=[],
                        device=get_device(),
                    )

                    kwargs["offload_mode"] = (
                        OffloadMode.NONE if p["text_encoder_offload"] == "off" else OffloadMode.CPU
                    )

                    def lora(filename: str, model_root: Path = root) -> Any:
                        return LoraPathStrengthAndSDOps(
                            str(model_root / "loras" / filename), 1.0, LTXV_LORA_COMFY_RENAMING_MAP
                        )

                    if p["pipeline"] == "dfr":
                        kwargs.update(
                            detailing_lora=[
                                lora(
                                    "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
                                )
                            ],
                            temporal_upsampler_path=str(
                                root
                                / (
                                    "latent_upscale_models/"
                                    "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"
                                )
                            ),
                        )
                        pipeline = DFRPipeline(**kwargs)
                    elif p["pipeline"] == "dev":
                        kwargs["distilled_lora"] = [
                            lora("ltx-2.5-22b-distilled-lora-450-bf16.safetensors")
                        ]
                        pipeline = TI2VidTwoStagesPipeline(**kwargs)
                    else:
                        pipeline = DistilledPipeline(**kwargs)
                    stages["model_load"] = time.monotonic() - start
                    configuration = config
                images = [
                    ImageConditioningInput.at(
                        command["inputs"][r["upload_id"]],
                        p["frames"] - 1 if r["frame"] == -1 else r["frame"],
                        r["strength"],
                    )
                    for r in p["keyframes"]
                ]
                kwargs = dict(
                    prompt=p["prompt"],
                    seed=p["seed"],
                    height=p["height"],
                    width=p["width"],
                    num_frames=p["frames"],
                    frame_rate=p["fps"],
                    images=images,
                )
                from ltx_core.model.video_vae import AUTO_TILING

                kwargs["tiling_config"] = AUTO_TILING if p["tiled"] else None
                if p["pipeline"] != "dev" and p["steps"] != 8:
                    import torch

                    from h3ltx.schedules import distilled_sigmas

                    kwargs["stage_1_sigmas"] = torch.tensor(distilled_sigmas(p["steps"]))
                if hasattr(pipeline, "use_ancestral_sampler"):
                    pipeline.use_ancestral_sampler = p["sampler"] == "euler_ancestral"
                if p["pipeline"] == "dfr":
                    kwargs["temporal_upscalings"] = p["temporal_upscale"]
                if p["pipeline"] == "dev":
                    from ltx_core.components.guiders import MultiModalGuiderParams

                    kwargs.update(
                        negative_prompt=p["negative_prompt"],
                        num_inference_steps=p["steps"],
                        video_guider_params=MultiModalGuiderParams(cfg_scale=p["cfg"]),
                        audio_guider_params=MultiModalGuiderParams(cfg_scale=p["cfg"]),
                    )
                emit("progress", {"stage": "native_pipeline", "progress": 5})
                started = time.monotonic()
                result = pipeline(**kwargs)
                stages["native_pipeline"] = time.monotonic() - started
                emit("progress", {"stage": "video_encode", "progress": 95})
                started = time.monotonic()
                from ltx_core.model.video_vae import get_video_chunks_number

                encode_video(
                    video=result.video,
                    audio=result.audio,
                    fps=p["fps"] * 2 ** p["temporal_upscale"],
                    output_path=command["output"],
                    video_chunks_number=get_video_chunks_number(
                        result.num_frames, result.tiling_config
                    ),
                )
                stages["video_encode"] = time.monotonic() - started
                if p["hdr"]:
                    import subprocess

                    from ltx_pipelines.hdr_ic_lora import HDRICLoraPipeline
                    from ltx_pipelines.utils.media_io import EXRColorSpace
                    from ltx_pipelines.utils.media_io.inputs import VideoInput

                    # Release the generation transformer before the finishing transformer.
                    pipeline = None
                    configuration = None
                    result = None
                    import gc

                    import torch

                    gc.collect()
                    torch.cuda.empty_cache()
                    emit("progress", {"stage": "hdr_refine", "progress": 97})
                    hdr_started = time.monotonic()
                    hdr_pipeline = HDRICLoraPipeline(
                        model_paths=paths,
                        hdr_lora=root / "loras/ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors",
                        text_embeddings_path=root
                        / "loras/ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors",
                        quantization=None,
                        offload_mode=OffloadMode.CPU
                        if p["text_encoder_offload"] != "off"
                        else OffloadMode.NONE,
                    )
                    target = Path(command["output"])
                    hdr, hdr_fps = hdr_pipeline(video=VideoInput(target, True), seed=p["seed"])
                    hdr_target = target.with_name("hdr-silent.mp4")
                    encode_video(
                        video=hdr,
                        audio=None,
                        fps=round(hdr_fps),
                        output_path=str(hdr_target),
                        video_chunks_number=1,
                        color_space=EXRColorSpace.ACESCCT,
                    )
                    muxed = target.with_name("hdr-muxed.mp4")
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-nostdin",
                            "-v",
                            "error",
                            "-y",
                            "-i",
                            str(hdr_target),
                            "-i",
                            str(target),
                            "-map",
                            "0:v:0",
                            "-map",
                            "1:a?",
                            "-c",
                            "copy",
                            str(muxed),
                        ],
                        check=True,
                        timeout=600,
                    )
                    muxed.replace(target)
                    hdr_target.unlink()
                    stages["hdr_refine"] = time.monotonic() - hdr_started
                    del hdr_pipeline, hdr
                    torch.cuda.empty_cache()
                emit(
                    "result",
                    {
                        "outputs": [Path(command["output"]).name],
                        "software": {
                            "torch": str(__import__("torch").__version__),
                            "cuda": __import__("torch").version.cuda,
                            "ltx_pipelines": __import__(
                                "importlib.metadata", fromlist=["version"]
                            ).version("ltx-pipelines"),
                        },
                        "timeline": {
                            "stages": stages,
                            "cached_nodes": [],
                            "nodes": node_timings,
                            "sampling_steps": sampling_steps,
                            "timing_source": (
                                "native adapter stage events and synchronized denoiser call timings"
                            ),
                        },
                    },
                )
            except Exception as exc:
                import traceback

                traceback.print_exc(file=sys.stderr)
                protocol.write(json.dumps({"type": "error", "message": str(exc)}) + "\n")
                protocol.flush()


if __name__ == "__main__":
    protocol_output = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        run(protocol_output)
