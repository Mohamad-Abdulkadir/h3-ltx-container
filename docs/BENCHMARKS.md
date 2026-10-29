# Benchmarks

> **Latest intermediate result:** [smoke test, 2026-10-29](benchmarks/2026-10-29-smoke/README.md). On an RTX PRO 6000 a warm 5-second clip takes about 40 s with LTX 2.5 distilled, 53 s with H3 turbo-4 and 174 s with H3 base. The full benchmark below has not run yet.

Generated: 2026-10-29

These figures come from the supplied run records. No GPU measurements are invented.
Cold generations are excluded by default. Model loading is reported separately.

## Generation

| Model | Profile | Resolution | Duration (s) | Steps | Engine | Runs | Wall p50/p90 (s) | Peak VRAM/RAM (GiB) | Output s/GPU-min | Estimated cost (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

No measured warm generation runs have been supplied.

### Per-stage medians

| Model/profile/resolution | Stage | Median (s) |
| --- | --- | --- |

## Load and unload operations

| Model | Profile | Operation | Runs | Wall p50/p90 (s) |
| --- | --- | --- | --- | --- |

### Component loading

| Model | Component | Disk read (s) | GPU transfer (s) | Compile (s) |
| --- | --- | --- | --- | --- |

## Hardware and software

No target GPU/driver/software records have been supplied.

## Statistics export

No `/v1/stats` export supplied.

## Sample outputs

No sample output artifacts supplied.

A zero cost estimate means no hourly rate was configured unless the run says otherwise.
Null timing fields are unmeasured, never assumed to be zero. Disk-cache warmth and
compiled-kernel warmth are separate from model residency. Quality needs visual/audio review.
