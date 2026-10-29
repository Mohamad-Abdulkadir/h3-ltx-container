# Runpod testing and deployment

## Current testing path: stock image, pod disk, no template

The current phase uses the GPU pod's own disk at `/workspace`. A network volume and
a custom image/template are optional later steps. The operator creates the pod;
repository tooling does not provision resources or spend money.

1. Choose a Secure Cloud **RTX PRO 6000 Blackwell 96 GB** pod. Select a stock Runpod
   PyTorch/CUDA image with a CUDA-13-compatible host, or the stock
   `nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04` image. The setup script installs the
   locked cu130 Python environments rather than relying on the stock Python packages.
2. Allocate a **550 GB pod volume disk**, mounted at `/workspace`, and a 50 GB
   container disk. Use at least 128 GB host RAM for BF16 offload where available
   (provisional, not measured). Expose HTTP port **8000**; keep ComfyUI ports private.
3. Supply `API_TOKEN` (32+ characters) and, for gated LTX groups, `HF_TOKEN` through
   Runpod environment secrets. Accept the required HF licences first. Do not copy a
   local `.env` into source control or place tokens in shell history or request files.
4. In the pod terminal, after the owner has published the repository:

   ```sh
   cd /workspace
   git clone https://github.com/<owner>/<repository>.git h3ltx
   cd h3ltx
   bash scripts/setup-pod.sh
   export MODEL_GROUPS=h3-core
   bash scripts/start-pod.sh
   ```

   `setup-pod.sh` is guarded against execution on a Mac. It installs small system
   tools, Python 3.12.15, pinned sources and isolated runtime/ComfyUI/LTX environments
   on the GPU pod. It never runs upstream install scripts. The first startup syncs
   **only MODEL_GROUPS** from the manifest. Choose the groups for each test phase:
   `h3-core`, then `h3-upscale,seedvr2`, then `ltx-core,ltx-iclora`, with optional groups
   only when testing them. Existing verified files are reused. No weights enter an image.
5. Startup logs download GB and elapsed seconds (including integrity verification),
   and saves `/workspace/state/startup-download.json`. Model download/verification
   finishes before serving jobs; it is never included in generation timing.

Keep the same `MODEL_GROUPS` environment on restarts. Set `MODEL_SYNC_ON_START=false`
only when deliberately using a prepopulated, verified disk. `start-pod.sh` permits a
writable pod disk with `ALLOW_EPHEMERAL=true`; no network mount is required in testing
mode. Stopping/replacing/deleting a pod affects its local disk according to Runpod's
storage lifecycle. Pull all run bundles before deleting the pod.

## Location and licence warnings

The owner's **2026-10-29** stock check found low-stock RTX PRO 6000 96 GB pods in
EU-CZ-1, EU-NL-1, EU-RO-1, EUR-IS-1, EUR-IS-2, US-MD-1, US-MO-2, US-NC-2 and US-PA-1.
Recheck stock before allocating. Prefer **EUR-IS-1 or EUR-IS-2**: Iceland is not an EU
member state and is the only listed region outside MiniMax H3's excluded territories
(US, EU, UK, South Korea). This is not legal advice; verify the licence and physical
location yourself. Country/region checks are **warnings only** and never block work.
Set `RUNPOD_COUNTRY=IS` only after confirming the location.

## Load, generate and archive

On your Mac install the pure Python client with `uv tool install .`. Set
`H3LTX_URL=https://<pod-id>-8000.proxy.runpod.net` and supply the API token through
an exported environment variable or your own local gitignored configuration.

```sh
h3ltx-client system
h3ltx load h3-fl2v --gpus all
h3ltx-client submit examples/h3.t2v.json
h3ltx-client wait <job-id>
h3ltx-client pull ./runs
h3ltx unload
```

`h3ltx load` waits by default and returns per-GPU/component load measurements.
`/v1/capabilities` lists all loadable pipeline keys. An explicit full-residency load
can fail if all components cannot fit together; the error is retained as a load run.
Never interpret a partial GPU load as a warm pipeline. H3 text-encoder offload during
sampling remains the default policy; the same cached objects remain reusable.

A job without a preload loads lazily, records `warm=false` and `model_load_s`, and
is excluded from default generation percentiles. Inspect `timeline.json`, the
`cache_proof` before/after loader counters, and `/v1/capacity` residency. Kernel JIT
warmth, disk-cache warmth, model residency and cached sampling are separate concepts.

For unattended testing, supply the media assets described in
[testing-matrix.md](testing-matrix.md), then run:

```sh
uv run python scripts/matrix.py examples/acceptance-matrix.yaml \
  --run-id <unique-test-id> --preload --pull ./runs --stats-output stats.json
uv run python scripts/report.py --runs ./runs --stats stats.json --output docs/BENCHMARKS.md
```

Review outputs and the generated report, then commit the report yourself after the
Runpod run. No benchmark results are fabricated in this repository.

## Optional later: network volume and custom image

The 550 GB network-volume design remains supported. Create a volume in the same data
centre as the GPU, populate the required groups on a cheap CPU pod, and attach it at
`/workspace` (pods) or `/runpod-volume` (Serverless). Use the same sync/verify commands:

```sh
h3ltx models sync --groups h3-core,h3-upscale,seedvr2,ltx-core,ltx-iclora
h3ltx models verify --deep
```

Default files currently total **312,331,947,147 bytes**, rather than the earlier
approximate 455 GB estimate. A 550 GB disk leaves staging and run-record headroom;
optional full BF16/quantized comparisons can need more space or phased cleanup.

`docker/Dockerfile` pins CUDA and uv images by digest, engine dependencies by hashes,
and upstream repositories by commit. Build it on an appropriate Linux builder only,
never on this development Mac. The release workflow pushes GHCR images only on `v*`
tags. Review `runpod/template.json` when choosing this optional route; replace image
placeholders and inject secrets through the console, never a committed populated file.
The <25 GB image-size goal is unmeasured until a Linux build has completed.

Serverless uses the same image with `MODE=serverless`, API_TOKEN set, and a prefilled
network volume. Queue responses contain run paths/metadata, never base64 video.
Use one runtime owner per SQLite database. Pod mode provides the HTTP download API.

Idle stop is disabled by default. Opt in with `IDLE_STOP_MINUTES` and a pod-scoped
`RUNPOD_API_KEY`; it calls the documented [stop operation](https://docs.runpod.io/api-reference/pods/POST/pods/podId/stop)
only after engines, model operations and queue are idle. Storage can continue billing
after stop. Manage resources explicitly in the console; no deployment is performed here.
