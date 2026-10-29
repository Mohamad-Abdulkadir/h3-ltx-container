`template.json` is a Runpod template request body for operator review. Replace image
placeholders and inject secrets through the console's secret mechanism. It does not
create a network volume or select a GPU. Never commit a populated copy.

| Setting | Meaning |
| --- | --- |
| `containerDiskInGb=50` | Temporary container disk; no model weights here |
| `volumeMountPath=/workspace` | Attach an existing 550 GB network volume here |
| `ports` | API on 8000; optional public-key SSH on 22 |
| `API_TOKEN` | Required bearer secret, minimum 32 characters |
| `HF_TOKEN` | Model population secret, only needed for gated downloads |
| `MODE` | `pod` or `serverless` |
| `SLOTS_PER_GPU` | One BF16 execution per GPU |
| `IDLE_STOP_MINUTES` | Zero disables self-stop; requires a pod-scoped key when enabled |
| `RUNPOD_COUNTRY` | Set only after confirming the physical deployment country |

GPU selection: RTX PRO 6000 96 GB, Secure Cloud, CUDA >=13.0. Network volume and GPU
must be in the same data centre. As of 2026-10-29 the owner's catalog check reported
low stock in EU-CZ-1, EU-NL-1, EU-RO-1, EUR-IS-1, EUR-IS-2, US-MD-1, US-MO-2,
US-NC-2, US-PA-1. Prefer EUR-IS-1 or EUR-IS-2: Iceland is outside the EU and the
listed MiniMax H3 excluded territories. This is not legal advice; verify the licence
and location yourself. Capacity must be rechecked before provisioning.

See [exact deployment steps](../docs/runpod.md). These files have not been submitted
to Runpod; no deployment is performed by this repository.
