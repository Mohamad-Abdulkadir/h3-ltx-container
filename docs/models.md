# Models

`models/manifest.yaml` is the reviewed source manifest. A packaged identical copy
lives in `src/h3ltx/data/manifest.yaml`. Every selected file has its actual Hugging
Face size, LFS SHA256 and immutable repository revision. `scripts/generate_manifest.py`
refreshes metadata only; review diffs rather than refreshing during inference.

| Group | Purpose | Default |
| --- | --- | --- |
| h3-core | Pruned BF16 FL2V/R2V, BF16 Qwen, video/audio VAEs, three turbo LoRAs | Yes |
| h3-upscale | 3D-conv-v1 BF16 and FP16 safetensors | Yes |
| seedvr2 | Native ComfyUI 7B, 7B sharp, 3B FP16 and native-layout VAE | Yes |
| ltx-core | Split BF16 dev/distilled transformers, Gemma 4 12B with projection, VAEs, spatial/temporal upscalers, distilled LoRA | Yes |
| ltx-iclora | Pixel-Spatial, Refine-Details, SDR-To-HDR plus scene embeddings, Deblur, Decompression | Yes |
| h3-ab-full | Full BF16 FL2V and R2V for controlled A/B tests | No |
| h3-turbo-extra | Newer ComfyUI-format turbo adapters | No |
| face-refine | Bingsu face_yolov8m detector | No |
| quantized-extra | Available H3 pruned INT8 and LTX INT8/NVFP4 transformers | No |

Default files total 312,331,947,147 bytes at the initial pins. The original 455 GB
estimate is retained in SPEC.md as historical design context; no fabricated files
were added to match it. H3 pruned NVFP4 is not published in the inspected
Comfy-Org repository. Requesting an absent checkpoint fails availability checks;
add it only when an actual supported artifact can be pinned.

Use `h3ltx models sync --groups <comma-separated-groups>` on a CPU pod with the
same volume. Downloads use Hub/Xet resumability, a per-volume process lock,
parallel transfers, a conservative free-space check, full SHA256 verification,
fsynced partial files and atomic replacement. Completed source blobs are removed
to avoid storing a second copy. `models/models.lock.json` records installed pins.
No token is written to the manifest, lock or cache configuration by this code.

`h3ltx models verify` checks sizes; `--deep` verifies hashes. `models status` prints
JSON. Inference does not download weights. An upstream Hub failure can produce a
normal CLI error; never publish unredacted shell environments or HTTP debug traces.

The official SeedVR2 template currently uses external upscaler wrapper classes.
This runtime implements the equivalent native preprocess → VAE encode → temporal
chunk → SeedVR conditioning → sample → merge → decode → colour correction chain,
using the matching Comfy-Org layout. It avoids that wrapper's different model names.

ComfyUI's native loader at the pinned commit recognizes the LBH 3D convolution
upscaler safetensors. The LBH custom-node source is also pinned in the image, but
builders use the native implementation. No FP32 pickle upscaler is installed.
