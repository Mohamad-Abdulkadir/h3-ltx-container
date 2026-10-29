# GPU acceptance and comparison matrix

CPU tests use fake NVML/engines and a recorded real ComfyUI schema. No model weights,
CUDA inference or Docker image build were run on the development Mac. Treat every
GPU/profile budget as provisional until these experiments have run.

For each run retain the full bundle. Compare the same prompt, seed, references,
resolution, duration, sampler, scheduler and software/model pins. Separate cold
loads, warm model residency and conditioning-cache hits. A warm native pipeline is
not evidence of a cold GPU. Never infer speedups from cached ComfyUI nodes.

| Experiment | Controls and evidence |
| --- | --- |
| H3 pruned/full BF16 | `weights=pruned-bf16/full-bf16`; compare R2V and optionally FL2V with identical inputs |
| H3 turbo | none/4-step/8-step, then each with `latent_upscale=2x`; inspect quality and target-resolution VRAM |
| H3 engine | ComfyUI only in v0.1; diffusers/SGLang need separate model/operator parity work and are reserved |
| H3 finishing | Latent+refine, SeedVR2 7B and sharp, LTX pixel spatial, each with/without face refinement |
| LTX engines | `engine=comfy/ltx-native`; distilled, DFR and dev; identify pipeline sampler/stage differences |
| LTX quality ladder | Distilled, DFR 1080p/1440p/4K; tile 4K; temporal 0/1/2; HDR off/on |
| Memory/attention | `text_encoder_offload=on/off`, ComfyUI `attention=sdpa/sage`; native SDPA only |
| Limits | Increase pixels×frames in controlled increments, use force for exploratory runs, inspect OOM logs |

Run `uv run python scripts/matrix.py examples/matrix.yaml --run-id <unique-comparison-id>`.
Use the same run ID to resume idempotently; use a new ID for a new measurement.
The script submits Cartesian variants, waits, reads result/metrics/timeline artifacts,
prints a comparison table and fetches `/v1/stats`. Its statistics query covers all
runs since the matrix started; the displayed run IDs identify the exact cohort.
Use `--dry-run` before GPU execution. The script loads local `.env` only when invoked;
no secret values are printed or embedded in submitted requests.

Before accepting a release, verify:

- Build linux/amd64, inspect installed source commits/lock versions, measure image size.
- On sm_120, confirm torch compiled arches, BF16, native model loads and a real NVENC encode.
- Test every job example with real media; check audio sample rate/channels, frame count,
  aspect ratio, output duration and ffprobe metadata.
- Inspect DFR generated keyframes, stage-2 guide crops, temporal 48/96 fps, first/last anchors,
  HDR transfer/primaries/10-bit output and scene preservation. HDR is not established by tags alone.
- Compare ltx-native's BF16 split-component loader and persistent second-run behaviour.
- Cancel sampling and then run a fresh job; kill the runtime during preparing/running/finalizing,
  restart, and verify only previously queued jobs run.
- Run one job per GPU on a multi-GPU machine; confirm CUDA bindings and no shared interruption.
- Interrupt a model sync, resume it, deep-verify, and test disk exhaustion without damaging installed weights.
- Download Range slices, all terminal bundles and multiple pull pages, including out-of-order completion.
- Measure cgroup/VRAM peaks, long IC-LoRA clip joins, face alignment and HDR audio preservation.
- Exercise the Serverless SDK handler and graceful pod shutdown/optional idle stop on a disposable deployment.

Known verification limits: native callable/denoiser timing hooks require GPU verification; exact model-level output parity and all performance claims await GPUs.
Document observed failures as failures, not as successful schema validation.
