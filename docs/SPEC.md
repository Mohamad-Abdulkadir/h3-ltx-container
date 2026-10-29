# h3-ltx-container: design specification

Status: v0.1 design, agreed 2026-10-29. This document is the source of truth for the
first implementation. Where it says "test both", the implementation must support both
options behind a parameter so they can be compared on real GPUs.

## 1. Purpose

A self-hosted, **API-only** GPU runtime for open video models, deployed on Runpod.
It is the cheap tier of a larger production pipeline (ArcReel / GNI Studio): about 80% of
shots run here, and complex shots go to closed models through a separate API (KIE).
GNI Studio will be the UI later. This repository has no UI.

In scope for v0.1:

- **MiniMax H3** (open H3-Base, 768p-class, 4–15 s, 24 fps, 32 kHz stereo audio):
  text-to-video, first/last-frame-to-video, reference-to-video.
- **LTX 2.5** (22B, audio + video): text-to-video, image-to-video, first/last frame,
  and the production-quality DFR pipeline.
- **Upscaling and finishing** for H3 output: H3 latent upscale + refine pass,
  SeedVR2, LTX-2.5 IC-LoRA upscaling, H3 face refinement.
- GPU detection, compatibility checks, admission control, job queue, complete run
  records with timings and resource metrics, statistics, and a download API so every run
  (outputs + full configuration) can be pulled to the operator's machine.
- Runpod Pod deployment (primary) and a Runpod Serverless handler (secondary,
  same image).

Out of scope for v0.1 (keep extension points, do not implement):

- Image generation models (planned later; job types must be pluggable).
- NVIDIA Open-H3-Regenerate-2K (needs ~150 GB VRAM / B200; rejected on cost).
- Any UI. ComfyUI's own UI is not exposed by default.

## 2. Hardware

Default and reference GPU: **RTX PRO 6000 Blackwell, 96 GB** (compute capability
`sm_120`, has NVENC). One full-precision generation per GPU at a time.

The runtime must work on any single- or multi-GPU pod and choose behaviour from what it
detects. GPU profiles (data, not code branches) define defaults and limits:

| Profile id | Match | Notes |
|---|---|---|
| `blackwell-96` | sm_120, ≥ 90 GB | Default. BF16 everything, text encoder unloaded before the DiT. |
| `hopper-80` | sm_90, ≥ 79 GB | H100. No NVENC: CPU x264 encode. |
| `hopper-141` | sm_90, ≥ 135 GB | H200. No NVENC. |
| `ampere-80` | sm_80, ≥ 79 GB | A100 80 GB fallback. No FP8. |
| `ada-48` | sm_89, ≥ 44 GB | L40S class. Requires quantized weights; not a v0.1 default, include as a profile that refuses BF16 jobs with a clear error. |

Each profile has a per-job-type admission budget `max_pixels_x_frames` (width × height ×
frames). Values are **provisional** until measured, and the stats system must make
measuring them easy. Starting values: H3 on `blackwell-96` with pruned BF16 = 1152×640×361;
LTX DFR 1080p allowed, 4K marked experimental (tiling required). Reject over-budget jobs
with HTTP 422 and an explanation, unless the request sets `"force": true`.

Multi-GPU: one engine instance per eligible GPU (separate ComfyUI process bound to
`127.0.0.1:<8188+i>` with `CUDA_VISIBLE_DEVICES=i`). Scheduler slots = number of eligible
GPUs. `SLOTS_PER_GPU` env (default 1) exists for future quantized profiles; values > 1
must be refused for BF16 profiles.

## 3. Models

Testing mode (current phase): no network volume. `VOLUME_ROOT` points at the pod's own
volume disk (`/workspace`), and `h3ltx models sync --groups <only what is being tested>`
downloads from Hugging Face when the pod starts. The pod startup logs the time and GB that
download took. Region and licence-territory checks are warnings only and must never block
anything. A network volume stays supported as an optional later step.

Weights never go in the image. They live on a Runpod network volume under
`$VOLUME_ROOT/models` in ComfyUI folder layout. A manifest (`models/manifest.yaml`)
declares every file: key, group, Hugging Face repo, path in repo, destination
folder, size in bytes, sha256 (the LFS oid), pinned revision, licence id, gated flag.
Generate size/sha256/revision from the Hugging Face API; do not hand-type them.

Groups (default install = `h3-core,h3-upscale,seedvr2,ltx-core,ltx-iclora`, ~455 GB):

- `h3-core` (Comfy-Org/MiniMax-H3): `diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors`,
  `diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors`,
  `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors`,
  `vae/minimax_h3_video_vae_fp16.safetensors`, `vae/minimax_h3_audio_vae_fp32.safetensors`,
  and the three Comfy-format turbo LoRAs in `loras/` (fl2v 4-step 768p v1.0, fl2v 8-step v1.0,
  ref2v 4-step v0.1). Check lightx2v/Minimax-h3-Turbo for newer ComfyUI-format versions
  (fl2v 4-step v1.3 768p, ref2v 8-step v1.1 768p) and include them as optional `h3-turbo-extra`.
- `h3-ab-full`: `diffusion_models/minimax_h3_ref2va_bf16.safetensors` (66 GB). Temporary,
  used once to confirm pruned BF16 matches full BF16; not in the default install.
- `h3-upscale`: LBH-123-AI/Minimax_h3_latent_Upscaler 3D conv v1 **safetensors** (bf16 and
  fp16). Do not use the fp32 `.pth` (pickle) by default.
- `seedvr2` (Comfy-Org/SeedVR2): 7B fp16, 7B sharp fp16, 3B fp16 and the VAE the native
  ComfyUI SeedVR2 workflow expects (check the official template).
- `ltx-core` (Lightricks/LTX-2.5, gated): distilled BF16 and dev BF16 transformers, Gemma-4
  12B BF16 text encoder, video VAE, audio VAE, spatial x2 and temporal x2 latent upscalers,
  distilled LoRA.
- `ltx-iclora` (separate gated repos): Pixel-Spatial-Upscaler, Refine-Details, SDR-To-HDR,
  Deblur, Decompression.
- `face-refine`: whatever detection models ComfyUI-H3-FaceRefine requires.
- `quantized-extra` (optional, speed comparisons only): H3 pruned INT8 ConvRot / pruned
  NVFP4, LTX INT8 / NVFP4 distilled.

Tooling: `h3ltx models sync --groups ...` (resumable, parallel, `hf_transfer`/Xet,
verifies size and sha256, atomic rename into place, disk-space precheck, writes
`models.lock.json`), `h3ltx models verify [--deep]` (size check by default, sha256 with
`--deep`), `h3ltx models status` (JSON). `HF_TOKEN` is required for gated LTX repos and is
never logged or written to disk. Downloads should normally run on a cheap CPU pod
attached to the same network volume; document this.

## 4. Engines

- **ComfyUI** (primary): pinned commit (start from `e6354106fa0761ea1f40773b9971985dbe51421a`,
  v0.43.0) plus pinned custom nodes: ComfyUI-VideoHelperSuite, ComfyUI-KJNodes (only if a
  workflow needs it), LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler, Lightricks
  ComfyUI-LTXVideo (only if native ComfyUI LTX nodes are not enough),
  Carasibana/ComfyUI-H3-FaceRefine, and SeedVR2 (prefer ComfyUI's native nodes; fall back to
  numz/ComfyUI-SeedVR2_VideoUpscaler). Always listen on loopback. Started and supervised
  by the runtime. ComfyUI's output cache must not falsify measurements (see §7).
- **ltx-native** (engine B for LTX, "test both"): Lightricks `ltx-pipelines` in its own
  virtualenv, driven by a small runner that keeps the pipeline loaded between jobs.
  Supports distilled, DFR, and the two-stage dev pipeline. The ComfyUI `*-comfy-int8-convrot`
  files do not load here; BF16 files do.

Graphs are built in Python from the **official Comfy-Org workflow templates**
(`video_minimax_h3_t2v.json`, `_i2v.json`, `_r2v.json`, `_multiframe_reference.json`,
`video_ltx2_5_t2v.json`, `_i2v.json`, `_flf2v.json`, `_dfr_i2v.json`,
`utility_seedvr2_video_upscale.json`), replicating their topology (including subgraph
contents) as API-format graphs. Every builder must be validated against a recorded
`/object_info` snapshot in tests, and against the live server at startup (missing node
classes or model files make that job type unavailable rather than crash the server).

## 5. Job types and parameters

Registry of job types, each with a pydantic parameter model and a builder:

- `h3.t2v`, `h3.fl2v` (first and/or last frame), `h3.r2v` (≤ 9 images, ≤ 3 videos, ≤ 3 audio,
  ≤ 12 files total, clip lengths 2–15 s, total ≤ 15 s; validate these limits)
- `ltx.t2v`, `ltx.i2v` (start keyframe + optional middle/end keyframes with strength),
  `ltx.flf2v`, `ltx.dfr` (production pipeline; optional temporal upscale 0/1/2 → 24/48/96 fps;
  optional HDR IC-LoRA)
- `upscale.seedvr2` (input video, model 7b / 7b-sharp / 3b, target short side, tiling)
- `upscale.ltx_iclora` (input video, pixel-spatial or refine-details, scene prompt, clip splitting)
- `refine.h3_face` (input video)

Common H3 parameters: prompt, seed (random by default, always recorded), aspect ratio +
resolution or explicit width/height (multiples of 32), duration seconds (H3 frame-count rule
from the template: frames = max(5, round(s×24)) adjusted to 17k+5... copy the template's
expression exactly), steps, sampler, scheduler, shift, turbo LoRA (none / 4-step / 8-step),
`latent_upscale` (off / 2x with refine steps), `text_encoder_offload` (auto/on/off),
`attention` (sdpa / sage), `reuse_conditioning` (bool). LTX parameters follow the LTX
templates and `ltx-pipelines` CLI (frames 8k+1, sizes multiple of 32/64). Any job can carry
a `post` chain: ordered list of `{type, params}` that runs follow-up jobs on its output
(e.g. h3.t2v → upscale.seedvr2), recorded as linked child runs.

Default H3 configuration on `blackwell-96`: pruned BF16 DiT, BF16 text encoder, text encoder
unloaded before DiT, fp16 video VAE, no turbo, template default steps/sampler.

## 6. HTTP API

FastAPI on port `8000`, every route except `GET /health` requires
`Authorization: Bearer $API_TOKEN` (constant-time compare; refuse to start without a token
of ≥ 32 chars). Modelled on hosted video APIs (submit / poll / fetch), so a KIE-style
provider in ArcReel can swap in.

- `GET /health` → `{status}` only.
- `GET /v1/system` → GPUs (name, index, UUID, capability, VRAM total/used, temperature,
  power, driver, NVENC available by actual encode test), CUDA/torch/ComfyUI/node versions,
  image version/git sha, CPU cores, RAM, disk free on volume and container, Runpod pod/DC env.
- `GET /v1/capabilities` → job types available on this machine (with reasons for
  unavailable ones), selected profile per GPU, limits, parameter JSON schemas.
- `GET /v1/capacity` → slots total/busy/free, queue length, per-GPU state.
- `GET /v1/models` → manifest status per file/group.
- `POST /v1/uploads` (multipart) → `{upload_id, sha256, bytes, media_type}`; also accept
  `{"url": ...}` JSON to fetch. Size limit configurable. Validate media with ffprobe.
- `POST /v1/jobs` → `{type, params, client_request_id?, priority?, tags?, post?}`;
  `client_request_id` is an idempotency key. Returns 202 with job id and queue position.
- `GET /v1/jobs?state=&type=&tag=&since=&limit=` , `GET /v1/jobs/{id}` (state, progress
  %, current stage, timings so far, queue position, error).
- `GET /v1/jobs/{id}/events` → Server-Sent Events for progress.
- `POST /v1/jobs/{id}/cancel` → removes queued work or interrupts the running prompt.
- `GET /v1/jobs/{id}/artifacts` and `/artifacts/{name}` (supports HTTP Range).
- `GET /v1/jobs/{id}/bundle` → zip (or tar) of the full run record directory.
- `GET /v1/runs/export?since_cursor=` → list of finished runs with bundle URLs and a cursor;
  used by the client to download everything new.
- `GET /v1/stats?group_by=type,profile,resolution,steps&since=` → count, success rate,
  p50/p90/max wall time, per-stage medians, peak VRAM, peak RAM, cost estimate
  (`GPU_HOURLY_USD` env), seconds of output per GPU-minute.
- `POST /v1/admin/doctor` → runs the doctor checks and returns the report.

### Explicit model loading (required)
Loading is a separate, measured operation, so generation timings never include it.

- `POST /v1/load` `{model, gpus?: [index...] | "all", profile?}`. `model` is a loadable
  pipeline key such as `h3-fl2v`, `h3-ref2v`, `h3-upscale`, `ltx-distilled`, `ltx-dev`,
  `seedvr2-7b` or `seedvr2-3b`; list the full set in `/v1/capabilities`. It loads every
  component of that pipeline into VRAM on each selected engine (text encoder, DiT, VAEs,
  LoRAs), using the same ComfyUI loaders and the same cached objects that generation uses.
  So a following job reuses them, and must not reload them. It returns 202 with a `load_id`.
- `GET /v1/load/{load_id}` and SSE `/v1/load/{load_id}/events`: state, per-component and
  per-GPU timings (read from disk, transfer to GPU, compile/TensorRT build if any), and peak
  VRAM/RAM. Every load writes a record under `$VOLUME_ROOT/runs/<date>/load-<id>/`, and
  `/v1/stats` reports load times separately (`group_by=model,profile`).
- `POST /v1/unload` `{model?|"all", gpus?}` frees VRAM, and records the time it takes.
- `GET /v1/capacity` and `/v1/system` show what is resident per GPU.
- Jobs: each job's `timeline.json` has `model_load_s`, which is 0 when the pipeline was
  already resident. Each record states `warm: true|false`. `/v1/stats` excludes cold jobs
  from generation-time percentiles by default (`include_cold=true` to include them).
  A job whose pipeline isn't loaded still works: it loads lazily and is marked cold.
- Scheduler: if a load and a job target the same engine, they run one after the other,
  never at the same time. A load never evicts a model a running job is using.
- Client/CLI: `h3ltx load <model> [--gpus all]`, `h3ltx unload`, and a `--preload` flag on
  the benchmark matrix runner. That flag loads, waits, then runs the jobs, so every
  measured job is warm.
- Tests: with a fake engine, a load followed by a job gives the job `model_load_s == 0` and
  `warm: true`; a job without a load is cold and is excluded from the default stats.

States: `queued → preparing → running → finalizing → succeeded | failed | cancelled`.
Jobs survive a restart: on boot, `running` jobs become `failed` with reason
`interrupted_by_restart` (never silently re-run a paid job), `queued` stay queued.

Errors are JSON `{error: {code, message, details}}` with stable codes.

## 7. Run records, metrics, statistics

Every job writes `$VOLUME_ROOT/runs/<YYYY-MM-DD>/<job_id>/`:

- `request.json`: as submitted.
- `resolved.json`: all parameters after defaults, GPU profile, the exact model files used
  (manifest key, path, size, sha256, revision), software versions (image tag + git sha,
  ComfyUI commit, custom node commits, torch, CUDA, driver), GPU identity, env flags that
  affect output.
- `workflow.json`: the exact API graph submitted to ComfyUI (or the native runner args).
- `timeline.json`: server-side stage timings (queue wait, input staging, model load, text
  encode, sampling with per-step times, VAE decode, upscale passes, video encode,
  finalization) taken from ComfyUI websocket events / history, not a stopwatch around
  the HTTP call. Flag `cached_nodes` if ComfyUI reports `execution_cached`; seeds default to
  random so benchmarks stay honest.
- `metrics.json`: sampled every 0.5 s via NVML: VRAM used peak/mean, GPU util mean, power
  mean/peak, temperature peak; process/cgroup RAM peak; OOM detection.
- `logs/engine.log`: the engine log lines for this job.
- `outputs/`: final media plus `thumbnail.jpg` and a contact sheet; `ffprobe.json` per output.
- `result.json`: state, error, output list with sha256, durations, cost estimate.

A SQLite database `$VOLUME_ROOT/state/runtime.db` (WAL mode) indexes jobs, runs, stage
timings and metrics for `/v1/stats`. Records are append-only; deleting runs is a separate
explicit admin command with a dry-run.

**Client**: `h3ltx-client` (pure-Python CLI in the same repo, installable on macOS):
`submit`, `status`, `wait`, `pull` (downloads every new run bundle into a local directory,
unpacks it, keeps a local cursor file; idempotent), `stats`, `system`. Base URL and token
from env or `~/.config/h3ltx/config.toml`.

## 8. Diagnostics

`h3ltx doctor` (CLI, also `/v1/admin/doctor`, also run at startup in summary form):
driver and CUDA versions; torch build CUDA and compiled arch list vs. the GPU's compute
capability; VRAM; bf16 support; NVENC via a real 1-second ffmpeg encode; disk free on
volume and container disk; RAM; volume mounted and writable; each manifest file present
with correct size; HF token present when gated groups are requested; ComfyUI reachable and
required node classes present; custom node versions; a licence notice when the Runpod data
centre cannot be determined or is in an excluded territory for MiniMax H3 (US, EU, UK,
South Korea). The H3 licence check is a **warning**, never a hard block, and documented as
not legal advice. Output: human table + JSON.

Hang diagnostics: when a job exceeds its timeout, log GPU state, ComfyUI queue state and
the last 30 engine log lines as short lines, then interrupt.

## 9. Container

- Base: pinned `nvidia/cuda:13.0.x-cudnn-runtime-ubuntu24.04` (or `-devel` only if a
  dependency needs compilation; prefer prebuilt wheels) by digest. linux/amd64 only.
- Python 3.12 via `uv`. Separate virtualenvs: `/opt/venvs/comfy` (torch cu130 build
  supporting sm_120), `/opt/venvs/ltx` (ltx-pipelines), `/opt/venvs/runtime` (API) may share
  with comfy if dependencies allow; document the choice. Lock files committed.
- ffmpeg with NVENC support where possible (NVENC used when the actual encode test passes,
  libx264 otherwise).
- Volume root detection: `$VOLUME_ROOT`, else `/runpod-volume` (serverless), else `/workspace`
  (pod). Refuse to start in pod mode without a writable volume unless
  `ALLOW_EPHEMERAL=true` (for the first temporary-disk experiments).
- `MODE=pod|serverless`. Pod mode: API on 8000; optional SSH if `PUBLIC_KEY` set; tini as PID 1;
  graceful shutdown. Serverless mode: Runpod queue handler mapping `input` to the same job
  system (synchronous per worker), outputs stay on the network volume and the response
  returns paths and metadata, never base64 video.
- Optional idle auto-stop: `IDLE_STOP_MINUTES` + `RUNPOD_API_KEY` (pod-scoped key) stop the
  pod through the Runpod REST API when no job has run for that long. Off by default.
- Non-secret configuration via env with documented defaults; secrets only via env:
  `API_TOKEN`, `HF_TOKEN`, `RUNPOD_API_KEY`.
- Image size target < 25 GB. No weights, no secrets, no HF cache in layers.

## 10. Repository, quality, CI

- Python package `h3ltx` under `src/`, `pyproject.toml`, ruff (lint+format), mypy on the
  package, pytest. Tests run without a GPU (fake engine, recorded `/object_info`, fake NVML).
- GitHub Actions: `ci.yml` (lint, type check, tests on PRs/pushes); `image.yml` (build and
  push `linux/amd64` to GHCR on tags `v*`, free runner disk first, cache layers, image
  labels with git sha, SBOM/provenance where available).
- `runpod/`: template definition (image, ports `8000/http,22/tcp`, container disk, volume
  mount `/workspace`, env vars with descriptions), plus `docs/runpod.md` with exact steps:
  create a 550 GB network volume in a suitable data centre, populate it from a CPU pod,
  deploy the GPU pod, verify with `h3ltx-client system` and a smoke job.
- Docs: `README.md` (what it is, quick start, API overview, hardware, licences), `docs/api.md`,
  `docs/architecture.md`, `docs/models.md`, `docs/runpod.md`, `docs/testing-matrix.md`,
  `docs/licensing.md`, `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`, `LICENSE`
  (Apache-2.0 for this code; model weights keep their own licences, listed in
  `docs/licensing.md`: MiniMax H3 Community License with territory exclusions and the
  "MiniMax H3" attribution requirement; LTX-2.x Community License with the $10M revenue
  threshold; SeedVR2 and custom node licences as found).
- No vendored third-party code without checking its licence. Pin every external repo by
  commit.

## 11. Test matrix (docs/testing-matrix.md)

Experiments the runtime must be able to run and record side by side:

1. H3 pruned BF16 vs full BF16 (same prompt, seed, references).
2. H3 full steps vs turbo 4-step vs turbo 8-step; turbo + latent upscale.
3. H3 engine: ComfyUI only in v0.1 (diffusers/SGLang reserved for later; document why).
4. H3 upscale chains to 1080p/1440p: latent upscale + refine; SeedVR2 7B; SeedVR2 7B sharp;
   LTX IC-LoRA pixel upscaler; each optionally followed by face refine.
5. LTX engine: ComfyUI vs ltx-native, same settings.
6. LTX quality ladder: distilled → DFR 1080p → DFR 1440p → DFR 4K (experimental, tiled);
   temporal upscale 0/1/2; HDR on/off; dev two-stage pipeline.
7. Memory policy: text encoder offload on/off; attention sdpa vs sage.
8. Limits: find the max pixels × frames per job type on `blackwell-96` (raise the budgets
   from measurements).

Provide `scripts/matrix.py` that submits a matrix from a YAML file through the API and
prints a comparison table from `/v1/stats` and the run records.

## 12. Research notes to honour

- ComfyUI caches node outputs; a repeated seed measures only the uncached part (one
  community benchmark reported 130 s instead of 198 s). Default random seeds, flag cache hits.
- API-format graphs flatten `DynamicCombo` inputs (`mode.width`, not nested dicts); a nested
  dict passes validation and fails at execution.
- `ComfyMathExpression` returns `(FLOAT, INT, BOOL)`; frame counts use slot 1.
- H100/H200/B200 have no NVENC; detect by an actual encode, not the codec list.
- Turbo LoRAs apply to pruned/quantized checkpoints without key mismatches.
- Latent upscale saves time, not VRAM; refinement runs at the target resolution.
- SeedVR2 7B FP8 has reported quality issues; use FP16.
- A community repo `irregular-dressing1531/Comfyui-MMH3-UltimateUpscale` is a malware lure;
  never reference or install it.
