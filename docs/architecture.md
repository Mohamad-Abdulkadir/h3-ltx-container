# Architecture

The runtime is the sole owner of the SQLite queue and GPU engine processes. FastAPI
and the Runpod handler call the same submission, validation, scheduling and record
code. Engine processes are isolated from API/HF/Runpod secrets. Weight population is
an explicit CPU-pod operation; startup never downloads large files.

A worker claims one eligible queued job in an immediate SQLite transaction. GPU
profiles are YAML data. Preflight checks schema, model sizes, precision, input media
and provisional pixel-frame budgets. A job can run only on GPUs that passed its
admission checks. A file lock prevents a second runtime owner from recovering or
claiming the same database. Run the API with one worker process.

ComfyUI is a pinned, separately supervised process per eligible GPU, listening on
loopback. Its input/output/user directories live below `engines/gpu-N`; models and
custom nodes are explicitly linked from their managed locations. Process policy
changes (attention, cache, offload) trigger a restart. `--cache-none` is the default;
`execution_cached` is always retained. Native LTX runs in a separate virtualenv and
persistent JSON-lines process. Switching engines releases the previous engine's
GPU weights. The native pipeline object and upstream weight caches are reused across compatible requests; upstream offload policy may still evict individual modules.

Run records are initialized at submission, enriched during execution and finalized
before the terminal transition. Finalized files are not rewritten by download
routes. SQLite uses WAL and FULL synchronous mode. Active states fail during
restart recovery; queued jobs remain queued. Completion cursors use a separate
monotonic completion table, so out-of-order completion cannot skip a run.

ComfyUI timing comes from execution/progress websocket events and history. Stages
that cannot be distinguished are not assigned invented timings. A node producing
both text conditioning and latents is recorded under text encoding. Per-step deltas
begin at the second progress event; the first step has no measured duration.
Native timing comes from adapter stage events. NVML samples every 0.5 seconds;
Linux cgroup RAM includes the whole pod and is labelled as such, so concurrent GPU
runs can observe the same RAM pressure. Peaks are sampled observations, not exact
allocation maxima. OOM kills are detected from cgroup counters and engine errors.

## Decisions

Details left open by SPEC.md were resolved as follows:

1. Use Python 3.12.15, Pydantic 2, synchronous SQLite connections guarded by a
   reentrant lock, and asyncio engine workers. No external broker is required.
2. Use one process owner per network-volume database, enforced by a file lock;
   Uvicorn runs one worker. Multi-GPU parallelism belongs to the scheduler.
3. Treat preparing and finalizing as interrupted work during crash recovery, along
   with running. No active paid job is replayed automatically.
4. Use descending integer priority (-100 to 100), then submission sequence. There
   is no preemption. Match jobs to any GPU that passed preflight.
5. Use UUID4 hex IDs; idempotency compares canonical request JSON, before generating
   a new random seed. A conflicting reuse returns HTTP 409.
6. Represent upload references by opaque IDs, never filesystem paths. Public HTTPS
   URL ingestion rejects private DNS answers and non-443 ports, pins the connected
   IP, preserves TLS SNI, and revalidates every redirect (maximum six hops).
7. Use ffprobe's measured media type, duration, dimensions and frame rate for
   input validation and finishing admission; do not trust declared clip lengths.
8. Use 2 GiB upload and 7,200-second job limits, configurable by environment. Limit
   multipart fields to one file. No automatic run deletion or retention expiry.
9. Use 64-pixel LTX size increments for both engines, `8k+1` frames, capped at 993
   because the pinned native audio node limits frame counts to 1,000. A 1080p-class
   grid-aligned request is 1920×1088. H3 uses the exact template expression and
   integer output slot 1; no replacement rounding rule is introduced.
10. Use explicit H3 `weights` and LTX `pipeline`/`weights` parameters for comparisons.
    Optional full BF16 includes FL2V as well as the specified R2V comparison weight.
    Newer turbo files are optional, immutable pins. No absent NVFP4 artifact is invented.
11. Use native ComfyUI SeedVR2 primitives instead of the official template's external
    wrapper, because this commit has the complete native model implementation. Preserve
    its preprocess/conditioning/sampling/merge/decode/postprocess topology. Use native
    H3 latent upscale as supported by the pinned native loader.
12. Omit KJNodes, ComfyUI-LTXVideo and the external SeedVR2 wrapper because the authored
    graphs need no classes from them. Pin VHS, LBH and FaceRefine sources. FaceRefine's
    minimal detector path avoids InsightFace/SAM and their extra model licences.
13. Derive four FaceRefine schemas statically from its pinned source, without executing
    third-party schema/install scripts. Record this separately from the 686 live
    native schemas. Overlay model combos from the manifest only in offline tests;
    runtime checks the actual files and validates against the live schema.
14. Default caching is disabled. `reuse_conditioning=true` enables ComfyUI's cache
    policy, and cache hits are reported rather than counted as uncached benchmarks.
    Native pipeline residency is not a conditioning cache.
15. Map text-encoder offload `auto/on` to aggressive ComfyUI offloading and native CPU
    weight streaming; `off` enables normal ComfyUI smart residency/native GPU residency.
    This is a policy request, not a promise that no model will ever be evicted.
16. Provide SDPA by default and a locked pure-Python SageAttention/Triton backend for
    ComfyUI comparisons. Native LTX exposes SDPA; Sage requests to native are refused.
17. Give each engine its own virtualenv and use PyTorch 2.11.0 cu130 for the CUDA image,
    with the matching torchvision/torchaudio and transformers 5.14.1. This is the
    mutually resolvable cu130 set; no cu132-only upstream indexes are inherited.
    Linux locks have hashes. Identical installed wheel files are hardlinked.
18. Use Ubuntu's ffmpeg, a real NVENC probe, and a final hardware encode when supported;
    CPU libx264 is the fallback. Probe success and the selected encoder are recorded.
    Preserve audio and colour metadata. Verify HDR bit depth/transfer tags on GPU.
19. Split long IC-LoRA inputs into configurable, keyframe-aligned clips with ffmpeg,
    execute a graph per clip, record every graph/timeline, and concatenate outputs.
    Preserve clip intermediates in the run bundle for inspection.
20. Post stages are durable, linked child runs, created one stage at a time. Parent
    success describes its own output; an unavailable next stage becomes a failed child.
21. Use ZIP64 with stored entries for bundles, generated on volume disk and deleted
    after transmission. Artifact downloads use Starlette's byte-range implementation.
    Finished-run export includes success, failure and cancellation.
22. The Mac client extracts into a temporary directory, rejects archive traversal and
    symlinks, atomically renames, and advances a server-URL-scoped cursor after each
    complete run. An incomplete pre-existing destination requires operator inspection.
23. Use linear-interpolated p50/p90 and per-stage medians. Include failures and flag
    cached runs; permit grouping by engine/cache. Cost estimates use occupied worker
    time excluding queue wait and the configured per-GPU hourly rate. A zero configured
    rate is an unspecified estimate, not a claim of free compute.
24. Store raw metric samples plus summaries. RAM uses cgroup v2, cgroup v1 fallback,
    then runtime-process RSS on non-Linux systems. Sampled GPU values are device-wide.
25. Keep extra per-profile budgets conservative and provisional in YAML. Force only
    bypasses budget checks; incompatible precision and untiled 4K remain errors.
26. Keep API auto-documentation disabled and use constant-time token comparison.
    Secrets stay in environment variables; local CLIs/matrix support gitignored `.env`
    without overriding exported values. The service does not load env files.
27. Require a mounted writable volume in pod mode. Explicit `ALLOW_EPHEMERAL=true`
    permits experiments. CLI model population needs neither API auth nor a GPU.
28. Use an optional SSH service only when a public key is supplied; disable password
    login. Use tini for process-group signal forwarding. Serverless returns volume
    paths and metadata rather than hosting a second download API.
29. Use the documented Runpod REST stop operation for opt-in idle stop. It is disabled
    by default; no provisioning or deployment is performed by application code.
30. Prefer EUR-IS-1/EUR-IS-2 using the owner's dated stock/location information. Warn
    about unknown/excluded H3 territory without blocking execution; no legal conclusion
    is encoded in admission.
31. Keep runtime code Apache-2.0 and upstream source/licence files in the image. A
    combined image is not labelled as wholly Apache-licensed. Maintainers must retain
    and satisfy each upstream licence before distributing it.

32. Interpolate the official distilled sigma schedule for non-default LTX step counts;
    these variants are experimental. Native supports Euler/Euler-ancestral. Native
    distilled/DFR requires cfg=1 and an empty negative prompt; dev exposes both.
33. Instrument native callable stages and synchronized denoiser calls without copying
    upstream implementation code. Iterator timings include only iterator execution,
    and unavailable crash timings remain null rather than invented.
34. Count every terminal job in statistics, including queued cancellations and restart
    failures. Exclude unknown crash wall times from duration percentiles and expose
    an incomplete-timings count.

35. Use the parameter models as the complete bound registry: 64-bit unsigned seeds,
    prompts up to 50,000 characters, 8,192-pixel dimensions, H3 up to 200 steps, other
    generation/refine paths up to 100, 32 tags, eight post stages and 128-character
    idempotency keys. Finishing defaults are 1080 short-side SeedVR2 with 21-frame
    chunks, five-second IC-LoRA clips, and face refinement at six steps/0.25 denoise
    on a 768-square crop. These bounds/defaults are exposed by JSON schemas.
36. Move verified ComfyUI outputs into the run record and remove engine-owned input
    staging after each attempt, retaining original uploads and IC-LoRA clip outputs.
    Preserve 10-bit formats for HDR encoding and record hardware-encoder fallback.
37. Prefer 128 GB host RAM for BF16 offload as an unmeasured deployment recommendation;
    do not present it as a measured minimum.

38. Fsync an append-only metric sample journal so crash recovery can reconstruct
    observed peaks/means even when final metrics.json was never written. Ignore a
    truncated final journal line and mark recovered metrics as interrupted.

39. On Linux, start engines through a small exec wrapper that requests SIGTERM on
    parent death and checks for a parent-exit race. This prevents an orphan engine
    from continuing GPU work after the runtime crashes. Normal shutdown still uses
    explicit interrupts, bounded waits and process termination.

## Verification boundaries

The recorded native schemas came from CPU ComfyUI, not an invented fixture. The
fixture does not establish successful weight loading, H3 joint AV decoding, DFR
keyframe semantics, HDR quality, native engine parity or peak VRAM. The native
adapter's internal denoiser stages require GPU acceptance. See the testing matrix.
