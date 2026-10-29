# h3ltx

An API-only, self-hosted Runpod container for MiniMax H3 and LTX 2.5 video generation,
upscaling, and face refinement. It includes a durable GPU queue, per-run records,
resource metrics, statistics, and a Mac-compatible download client.

The reference GPU is an **RTX PRO 6000 Blackwell 96 GB**. H3 defaults to pruned BF16
weights and a BF16 text encoder, with text-encoder offload enabled. ComfyUI is the
primary engine; `ltx-native` provides a separate LTX comparison engine.

## Quick start

1. Follow [deployment](docs/runpod.md): create a 550 GB network volume, populate
   weights on a CPU pod, and attach the same volume to the GPU pod.
2. Supply `API_TOKEN` (at least 32 characters) and `HF_TOKEN` as environment secrets.
   Set `VOLUME_ROOT=/workspace`, `MODE=pod`. Only port 8000 exposes the runtime;
   ComfyUI stays on loopback.
3. Install the client on your Mac with `uv tool install .` or `pip install .`.
   Set `H3LTX_URL=https://<pod-id>-8000.proxy.runpod.net` and `API_TOKEN`.
4. Submit and archive a run:

   ```sh
   h3ltx-client submit examples/h3.t2v.json
   h3ltx-client wait <job-id>
   h3ltx-client pull ./runs
   ```

Local CLIs also support a gitignored `.env` without overriding exported variables.
The service uses environment variables only. Never put credentials in request JSON.

## API overview

`GET /health` is public. All other routes require `Authorization: Bearer <your-api-token>`.
Submit to `POST /v1/jobs`, poll `/v1/jobs/{id}`, follow `/events`, or download `/bundle`.
`/v1/system`, `/v1/capabilities`, `/v1/capacity`, `/v1/models`, and `/v1/stats` expose
hardware, availability, queue state, model inventory, and measurements.

Generation: `h3.t2v`, `h3.fl2v`, `h3.r2v`, `ltx.t2v`, `ltx.i2v`, `ltx.flf2v`, `ltx.dfr`.
Finishing: `upscale.seedvr2`, `upscale.ltx_iclora`, `refine.h3_face`.
Ordered `post` chains create linked child runs. See [API details](docs/api.md) and [examples](examples/).

## Hardware and storage

One BF16 job runs per eligible GPU. Profiles include RTX PRO 6000 96 GB, H100 80 GB,
H200 141 GB, A100 80 GB, and a quantized-only Ada 48 GB profile. Budgets are provisional;
4K LTX is experimental and requires tiling. Unsupported models or nodes disable a job
capability with a reason. Weights live on the volume, never in the image.

This repository was developed on Apple Silicon without Docker or a GPU. CPU schema
and fake-engine tests do not establish inference quality, VRAM fit, image size, or
Runpod compatibility. Complete the [GPU acceptance matrix](docs/testing-matrix.md)
before production use. See [architecture and decisions](docs/architecture.md).

## Licences

Runtime code: [Apache-2.0](LICENSE). Model weights retain their own licences.
MiniMax H3 has territory exclusions and an attribution requirement. LTX uses the
LTX-2.x Community License, including a revenue threshold. ComfyUI and some bundled
dependencies have separate copyleft terms. Read [licensing](docs/licensing.md) before
publishing the image or using the models commercially.
