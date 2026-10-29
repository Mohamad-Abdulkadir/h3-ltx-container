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

## Unattended one-pod acceptance run

Use the current stock-image/pod-disk path in `docs/runpod.md`, with the API running
on a dedicated RTX PRO 6000 pod. Do not submit other jobs during a benchmark matrix.
All load/unload and generation operations are serialized by the same per-GPU lock.
The matrix unloads, loads and waits before **every** measured request, then asserts
that the resulting job is warm. It records failures instead of silently reporting cold work as warm.
The acceptance shell uses --continue-on-error to finish the remaining cases, renders
the report, and exits nonzero if any case failed. Full residency can fail if a pipeline cannot fit; retain that load
record and assess a smaller/quantized selection or a larger GPU explicitly.

Prerequisites, before starting unattended execution:

1. Supply your own licensed `benchmark-assets/first.png`, `last.png`, `clip.mp4`
   and `face-clip.mp4` (a short clip with a clearly detectable real face). Keep the
   assets private and gitignored. Use 24 fps and 4–8 seconds for the input videos.
2. Accept gated HF licences and supply HF_TOKEN/API_TOKEN via the pod environment.
   Configure GPU_HOURLY_USD from the actual pod rate. Confirm enough disk space;
   the serial optional-group sync fits the 550 GB plan before large run bundles accumulate.
3. Run `h3ltx doctor --json`, verify the image/torch/driver and actual encode test,
   and inspect `/v1/capabilities` for expected nodes/model availability.
4. Set a unique `BENCHMARK_RUN_ID`, set `H3LTX_URL=http://127.0.0.1:8000`, and run
   `bash scripts/run-acceptance.sh`. It does not provision, stop or publish anything.

`examples/acceptance-matrix.yaml` covers every ComfyUI loadable pipeline, including
standalone H3 latent-upscaler loading, H3 pruned/full and turbo variants, both upscale
paths, SeedVR2 variants, IC-LoRAs, face refinement, LTX distilled/dev/keyframes/DFR,
temporal 24/48/96 fps and HDR. `examples/native-matrix.yaml` follows with matched
ComfyUI/native settings. The native adapter's full-residency/build-cache behaviour
must be accepted on the actual GPU; failures are retained while the remaining cases run.

The first pass is a bounded correctness/residency baseline. Follow it with controlled
repeats (at least three distinct run IDs) and the experimental tiled 1440p/4K ladder.
Run admission-limit exploration separately in small steps; do not put an unbounded
OOM search into the baseline. Face alignment, HDR colour/bit depth/audio and clip joins
still require human output review after unattended execution.

The runner stores asset hashes/upload IDs in `.matrix-state` so resuming with the
same run ID does not create new paid requests because of changed upload IDs. It
pulls successful and failed bundles and exports statistics. `scripts/report.py`
regenerates `docs/BENCHMARKS.md` from those artifacts, keeping load timings separate,
excluding cold generation percentiles and retaining GPU/software/cost provenance.
Review the report and samples, then commit it after the real GPU run. The committed
initial report contains no GPU measurements.


## Open GPU-only questions for the Runpod pass

- Confirm that a full pipeline's components fit simultaneously in 96 GB, including
  BF16 text encoder, DiT, VAEs, LoRA patch state and allocator overhead. Full H3 BF16
  A/B weights plus the BF16 text encoder may exceed that capacity. A failed full
  load is evidence of a limit, not a successful warm benchmark; use an explicitly
  selected larger GPU or a separately labelled offloaded/cold experiment.
- Verify the real ComfyUI extension against the pinned V3 loader/sampler lifecycle:
  object identity, loader-read counters, model-management residency and no new disk
  reads between preload and generation, even with --cache-none. Local tests use
  fake ComfyUI/model-management objects, not real model files.
- Verify native LTX builder/component reuse and memory behaviour. Its upstream
  builders combine disk reads and assignment; the native load report currently
  preserves a combined build/transfer duration and null separate values. GPU
  instrumentation must establish those individual timings before claiming parity.
- Validate the native HDR path's additional model residency, DFR patched weights,
  optional offload policies and all reported warm flags on real hardware.
- Confirm HDR transfer/primaries/bit depth, 48/96 fps, face alignment, actual NVENC,
  Sage kernels, multi-GPU isolation, and measured admission budgets.
- Exercise the stock-image setup script and container build on Linux. Neither was
  executed on the development Mac. No GPU wheels, images or weights may be downloaded
  to the Mac for this follow-up.

The standalone matrix stops on failures by default. The acceptance shell deliberately
continues remaining cases, retains failed load/job records, and returns a nonzero
status after writing the report. It never labels failed or cold work as warm.

## Required before the next GPU run (owner notes, 2026-10-29)

1. **Fix preload residency.** Preload currently requires every component to be fully GPU-resident. H3 BF16 (DiT, Qwen3-VL 32B and the VAEs) needs about 98 GB, more than a 96 GB card, so every preload and every cold job failed until the check was weakened on the pod. Preload must support a declared partial-residency policy, such as the text encoder held in CPU RAM and moved in for encoding. It must report what is resident where, so that `model_load_s` and `warm` are valid measurements. Add tests for this.
2. **Clip length is 10–15 seconds.** Every generated test clip is 10–15 s long; 5 s is too short to judge motion and coherence. Update every example and matrix to match. For H3 use `duration_seconds` 10–15. For LTX at 24 fps use `frames` that are 8k+1, for example 241 (10 s), 289 (12 s) or 361 (15 s).
