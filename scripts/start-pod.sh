#!/usr/bin/env bash
set -euo pipefail
if [[ "$(uname -s)" != Linux ]] || ! command -v nvidia-smi >/dev/null; then
  printf 'This entrypoint is for the GPU pod only.\n' >&2
  exit 2
fi
: "${MODEL_GROUPS:?Choose only the model groups under test before starting}"
: "${API_TOKEN:?Supply API_TOKEN via the pod environment}"
export VOLUME_ROOT="${VOLUME_ROOT:-/workspace}"
export MODEL_SYNC_ON_START="${MODEL_SYNC_ON_START:-true}"
# Testing mode permits the pod's own disk; no network volume is required.
export ALLOW_EPHEMERAL="${ALLOW_EPHEMERAL:-true}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility,video}"
exec /opt/venvs/runtime/bin/h3ltx serve
