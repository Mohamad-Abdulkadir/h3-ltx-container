# Smoke test, 29 October 2026

First run on real hardware: MiniMax H3 and LTX 2.5 each generated 5-second clips with audio on one GPU. This is an intermediate result, not the full benchmark. The full matrix in `docs/testing-matrix.md` is still to run.

**Setup**
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB, Runpod community cloud at $1.69/h, driver 580.82.
- Models: H3 pruned BF16 with the 4-step turbo LoRA, and LTX 2.5 22B distilled BF16.
- Plan: `examples/smoke-matrix.yaml`.
- Inputs: two frames generated for this test (`media/input-first.png`, `media/input-last.png`).

Open `index.html` locally for the visual report with the clips playing inline. GitHub doesn't render HTML pages.

## Results

All 6 clips succeeded. "Total" is the time on the GPU for one clip, without queue wait.

| Clip | Steps | Resolution | Text encode (s) | Sampling (s) | Decode (s) | Video encode (s) | Total (s) | Peak GPU memory | GPU cost |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [H3 FL2V turbo-4, first run](media/h3-turbo4-seed11.mp4) | 4 | 1152×640 | 5.8 | 36.6 | 5.6 | 9.4 | **59.3** | 51 GB | $0.028 |
| [H3 FL2V turbo-4](media/h3-turbo4-seed12.mp4) | 4 | 1152×640 | 4.5 | 30.1 | 5.2 | 10.2 | **52.9** | 93 GB | $0.025 |
| [H3 FL2V base](media/h3-base20-seed11.mp4) | 20 | 1152×640 | 5.0 | 151.1 | 5.4 | 9.8 | **173.9** | 93 GB | $0.082 |
| [LTX T2V distilled, first run](media/ltx-t2v-seed11.mp4) | 8 | 1280×704 | 8.3 | 21.6 | 10.9 | 10.7 | **54.7** | 67 GB | $0.026 |
| [LTX T2V distilled](media/ltx-t2v-seed12.mp4) | 8 | 1280×704 | 7.9 | 8.6 | 10.7 | 10.2 | **40.2** | 90 GB | $0.019 |
| [LTX I2V distilled](media/ltx-i2v-seed11.mp4) | 8 | 1280×704 | 4.1 | 9.2 | 10.2 | 9.0 | **35.8** | 90 GB | $0.017 |

<p>
<img src="media/h3-base20-seed11-contact-sheet.jpg" width="49%" alt="H3 first/last-frame contact sheet">
<img src="media/ltx-i2v-seed11-contact-sheet.jpg" width="49%" alt="LTX image-to-video contact sheet">
</p>

## Findings

- **LTX 2.5 distilled** takes about 40 s and costs about $0.02 for a warm 5-second clip at 1280×704. Sampling is only about 9 s of that.
- **H3 turbo-4** takes about 53 s for a warm clip at 1152×640, and it ends on the requested last frame. **H3 base** at 20 steps took 174 s.
- **The first run of each model is slower:** about 6 s more for H3 and 14 s more for LTX. Loading happened inside that first job, because preload didn't work (next point).
- **Bug: preload requires full GPU residency.** H3 BF16 (DiT, Qwen3-VL 32B and the VAEs) needs about 98 GB, more than the 96 GB card. Every preload, and every cold job, failed until the residency check was turned into a warning on the pod. That fix is **not yet in the repo**. As a result, `model_load_s` and the warm/cold flags in these records aren't valid load measurements.
- **Video encoding takes about 10 s per clip,** a fifth of an LTX clip's time. Test hardware (NVENC) encoding next.
- **Setup dominated the cost.** 170.6 GB of downloads took 21 minutes on this community host, then installs, restarts and debugging. The pod ran 76 minutes (about $2.15), and the six clips used about $0.20 of that.
- **Fixed during the run, in 717a40c:** the LTX manifest checksums were placeholders, and the stock image's `uv` couldn't install Python 3.12.15.

## Files

- `media/`: the six clips, their thumbnails and contact sheets, and the two input frames.
- `records/<clip>/`: request, resolved settings, timeline, result and GPU metric samples.
- `records/stats*.json`: the API's `/v1/stats` exports. Their percentiles are empty because every job was flagged cold.
- `records/system.json`: GPU and software versions.
- `records/prefetch.json`: per-file model download times.
