# Runpod deployment

These are operator steps. No resource is created or paid for by repository tooling.
The image has not been built or GPU-tested on the development Mac.

## Choose the data centre first

The owner's Runpod catalog check on **2026-10-29** found low-stock RTX PRO 6000 96 GB
pods in EU-CZ-1, EU-NL-1, EU-RO-1, EUR-IS-1, EUR-IS-2, US-MD-1, US-MO-2,
US-NC-2, and US-PA-1. Recheck current stock before allocating anything.

Recommend **EUR-IS-1 or EUR-IS-2**, and place both network volume and GPU pod in
the same selected data centre. Iceland is not an EU member state and is the only
listed location outside MiniMax H3's excluded territories (US, EU, UK, South Korea).
This is not legal advice: verify the model licence and actual deployment location
yourself. The runtime emits a warning for excluded or undetermined territory; it
never turns this into a hardware admission block.

## Build an image when ready

The owner publishes the repository and builds the image separately. The tag
workflow builds amd64 and pushes to GHCR only on `v*` tags. For a manual build on
a Linux Docker host with enough disk:

```sh
docker buildx build --platform linux/amd64 -f docker/Dockerfile \
  --build-arg VERSION=0.1.0 --build-arg GIT_SHA="$(git rev-parse HEAD)" \
  -t ghcr.io/<owner>/<repository>:0.1.0 --load .
```

Test the image before publishing it. Replace all namespace/image placeholders in
`runpod/template.json`. The CUDA 13.0.2 runtime and uv images are digest-pinned;
source repositories and Python packages are pinned in `docker/` and `uv.lock`.
The target image size is below 25 GB, but actual size must be measured after the
first Linux build. CUDA wheels are hardlinked across isolated engine environments
where identical to reduce duplication.

## Populate a 550 GB network volume on a CPU pod

1. In Runpod's console, create a **550 GB network volume** in EUR-IS-1 or EUR-IS-2.
   A pod's ordinary volume disk is not a replacement for a network volume.
2. Create a temporary **CPU pod in that same data centre**, attach the network
   volume at `/workspace`, use the built image, and set its command to `sleep infinity`.
   Set `VOLUME_ROOT=/workspace`; inject `HF_TOKEN` as a secret. No API token is needed
   by the population command. Do not start the GPU runtime.
3. Accept the gated Lightricks model agreements in the Hugging Face account associated
   with the token. In the pod terminal:

   ```sh
   h3ltx models sync --groups h3-core,h3-upscale,seedvr2,ltx-core,ltx-iclora
   h3ltx models verify --deep
   h3ltx models status
   ```

   Downloads are resumable through Hugging Face Hub/Xet. The runtime checks free
   space, size, SHA256, and pinned revisions before atomically installing each file.
   Default weights currently total **312,331,947,147 bytes** (about 312.3 GB), rather
   than the design's approximate 455 GB. The 550 GB recommendation retains room
   for download staging, optional comparison groups, uploads and run records.
   Large optional groups may require removing temporary A/B files or expanding storage.
4. Optionally populate `face-refine`, `h3-turbo-extra`, `h3-ab-full`, or
   `quantized-extra`, using `--groups` with the desired groups. Optional detector
   weights are pickle-format `.pt`; review their source/licence before opting in.
5. Verify completion and **terminate only the temporary CPU pod** in the console.
   Keep the network volume. Terminating a pod is distinct from deleting the volume.

## Create the GPU pod

In Runpod Secure Cloud select:

| Setting | Value |
| --- | --- |
| GPU | RTX PRO 6000 Blackwell, 96 GB, one GPU initially |
| Data centre | Same EUR-IS-1 or EUR-IS-2 as the network volume |
| Minimum host CUDA compatibility | 13.0 |
| System RAM | Prefer at least 128 GB for BF16 offload; provisional until measured |
| Image | Your tested GHCR image, ideally by release digest |
| Container disk | 50 GB |
| Network volume | Existing 550 GB volume |
| Mount | `/workspace` |
| HTTP port | `8000` |
| TCP port | `22` only if SSH is wanted |
| Start command | Image default |

Set `MODE=pod`, `VOLUME_ROOT=/workspace`, `API_TOKEN=<your-api-token>` (32+ characters),
`SLOTS_PER_GPU=1`, `RUNPOD_COUNTRY=IS` after confirming location. Set `PUBLIC_KEY`
only for SSH; the image disables password authentication. The full environment
reference is `.env.example`. Populate secrets in Runpod, never in the template file.
`HF_TOKEN` is unnecessary for offline inference once every selected weight is installed.

Models, uploads, SQLite state and run records stay on the network volume. ComfyUI
listens only on `127.0.0.1:8188+i`; expose neither ComfyUI nor its custom-node APIs.
The API requires a writable mounted volume in pod mode. Only temporary experiments
should set `ALLOW_EPHEMERAL=true`.

## Smoke test from the Mac

```sh
uv tool install .
export H3LTX_URL=https://<pod-id>-8000.proxy.runpod.net
# Supply API_TOKEN through your shell or a local gitignored .env; do not paste it into logs.
h3ltx-client system
h3ltx-client submit examples/h3.t2v.json
h3ltx-client wait <job-id>
h3ltx-client pull ./runs
```

Check the bundle's video/audio, resolved model hashes, zero cache hits, stage
measurements and resource peaks. Run `h3ltx doctor --json` in the pod. Repeat with
an LTX request and with `engine=ltx-native`. The detector and optional LoRAs must
be installed before their capabilities become available.

For multi-GPU pods, one loopback ComfyUI process and one runtime slot are created
per eligible GPU. Keep `SLOTS_PER_GPU=1`. A native LTX job stops that GPU's ComfyUI
process before starting its isolated native runner; switching back releases native
weights. Unavailable GPUs have diagnostic reasons rather than enabling extra slots.

## Serverless and idle stop

For Runpod queue workers use the same image with `MODE=serverless`, a prepopulated
network volume at `/runpod-volume`, and `VOLUME_ROOT=/runpod-volume`. Set one SDK
job at a time per worker; the local scheduler still owns GPU slots. Submit the same
job object under `input`. Responses contain run paths/metadata, never base64 video.
The queue mode does not expose port 8000; use pod mode for the HTTP download client,
or retrieve bundles through a separately started API against the same idle volume.
Do not run two runtime owners against one SQLite database simultaneously.

Idle stop is disabled by default. Opt in with `IDLE_STOP_MINUTES` and a pod-scoped
`RUNPOD_API_KEY`; the runtime calls the documented
[stop API](https://docs.runpod.io/api-reference/pods/POST/pods/podId/stop) only after
both queue and engines are idle. Stopped pods and network volumes can continue to
incur storage charges; use the console to manage them explicitly.

On stop/restart, queued jobs remain queued and interrupted jobs fail visibly.
Before deleting anything, pull the run bundles. `h3ltx prune --before <unix-seconds>`
is dry-run; add `--apply` only after inspecting the printed run directories.
