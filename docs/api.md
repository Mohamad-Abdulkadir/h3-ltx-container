# HTTP API

The API listens on port 8000. `GET /health` returns only `{"status":"ok"}` and requires
no token. Every `/v1` route requires `Authorization: Bearer <your-api-token>`.
Errors use `{"error":{"code":"stable_code","message":"...","details":{}}}`.
FastAPI's interactive documentation routes are disabled.

| Method | Route | Result |
| --- | --- | --- |
| GET | `/v1/system` | GPU identity/telemetry, actual NVENC test, software, disk, RAM and location |
| GET | `/v1/capabilities` | Availability, reasons, profiles, budgets, parameter JSON schemas |
| GET | `/v1/capacity` | Total/busy/free slots, queue length, per-GPU state |
| GET | `/v1/models` | Manifest metadata and file status |
| POST | `/v1/uploads` | Multipart `file` or JSON `{"url":"https://..."}`; returns upload ID and digest |
| POST | `/v1/jobs` | 202; accepted job, queue position and resolved configuration |
| GET | `/v1/jobs` | Filters: `state`, `type`, `tag`, `since` (Unix seconds), `limit` (1–1000) |
| GET | `/v1/jobs/{id}` | State, progress, current stage, timings/result and queue position |
| GET | `/v1/jobs/{id}/events` | Durable SSE progress; reconnect with integer `Last-Event-ID` |
| POST | `/v1/jobs/{id}/cancel` | Cancel queued work or interrupt the owning GPU engine |
| GET | `/v1/jobs/{id}/artifacts` | Record files, bytes and SHA256 |
| GET | `/v1/jobs/{id}/artifacts/{name}` | File download with byte Range support |
| GET | `/v1/jobs/{id}/bundle` | Finished run as a ZIP64 archive; 409 while active |
| GET | `/v1/runs/export?since_cursor=0` | Finished-run page and monotonic completion cursor |
| GET | `/v1/stats` | Aggregates; `group_by=type,profile,resolution,steps,engine,cached`, `since` |
| POST | `/v1/admin/doctor` | Full diagnostic report |

## Requests

```json
{
  "type": "h3.t2v",
  "params": {"prompt": "A quiet coastal sunrise.", "seed": 1234},
  "client_request_id": "shot-001-attempt-1",
  "priority": 0,
  "tags": ["project-a"],
  "post": [{"type": "upscale.seedvr2", "params": {"model": "7b", "target_short_side": 1080}}]
}
```

Omit the seed for a random recorded seed. Reusing a client request ID with the same
request returns the original job; a different request returns 409
`idempotency_conflict`. Priority is -100 to 100, with higher values dispatched first;
within a priority, submission order is preserved. There is no preemption.

Upload first; use the returned 32-character upload ID in `first_frame`, `last_frame`,
`references`, `keyframes`, or `input_video`. Example files contain zero/one placeholder
IDs and require replacement. URLs permit public HTTPS only; each redirect and DNS
answer is checked, and connections are pinned to validated addresses. File names
are server-generated and media is inspected with ffprobe. Upload metadata, rather
than claimed dimensions or clip durations, determines admission for input-video jobs.

H3 widths/heights are multiples of 32; duration is 4–15 seconds at 24 fps. The exact
frame expression is retained in `params.py`; 15 seconds resolves to 362 frames and
can exceed the provisional 361-frame reference budget. `force=true` bypasses the
pixel-frame budget, never media, precision, schema, model, or tiling checks.
`h3.fl2v` also accepts `guides` for multiframe references and continuation.
References permit nine images, three videos, three audio files, twelve total files,
2–15 seconds per clip, and fifteen seconds of combined clip duration.

LTX dimensions are multiples of 64, frames are `8k+1` (up to 993 in the current
native-node schema). Use 1920×1088 for a grid-aligned 1080p-class DFR output.
`ltx.dfr` selects DFR; other LTX jobs accept `pipeline=distilled|dev|dfr`.
Temporal upscale 0/1/2 is a DFR option at a 24 fps source and yields 24/48/96 fps.
Keyframe positions are pixel frame indices; -1 means last frame. Strength is 0–1.

Post stages execute in order as linked child runs; a parent's success means its
own output succeeded. Check child states for the complete chain. A finishing
stage that becomes unavailable is retained as a failed child with an error.

## States and downloads

`queued → preparing → running → finalizing → succeeded|failed|cancelled`.
Queued jobs survive restarts. Preparing/running/finalizing jobs fail with
`interrupted_by_restart` and are never automatically re-executed. Cancellation
is idempotent for terminal jobs. Worker shutdown cancels active jobs gracefully;
an abrupt process/pod loss is identified during the next startup.

A completion cursor is assigned when a job finishes, independent of submission
order. `h3ltx-client pull` downloads pages and advances its local cursor only after
each archive is safely extracted and atomically placed. Use one local destination
per server. The cursor is scoped to the server URL. Keep a separate destination
after replacing the volume/database, because completion cursors can start again.

Artifact names are relative to the run root, such as `outputs/video.mp4`.
File downloads support `Range: bytes=0-1048575`; invalid ranges return 416.
Bundles use stored ZIP entries to avoid expensive recompression of video.
