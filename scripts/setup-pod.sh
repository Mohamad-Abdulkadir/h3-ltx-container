#!/usr/bin/env bash
# Operator-invoked bootstrap for a stock Runpod PyTorch/CUDA pod. Never run on a Mac.
set -euo pipefail
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]] || ! command -v nvidia-smi >/dev/null; then
  printf 'Run this script only on the target linux/amd64 NVIDIA GPU pod.\n' >&2
  exit 2
fi
H3LTX_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DEBIAN_FRONTEND=noninteractive
export UV_CACHE_DIR=/opt/h3ltx-install-cache UV_LINK_MODE=hardlink
apt-get update
apt-get install -y --no-install-recommends ca-certificates git ffmpeg python3 python3-pip libglib2.0-0 libgomp1 libgl1
# Stock images may ship an older uv that cannot install the pinned Python; always install the pinned one.
python3 -m pip install --break-system-packages --upgrade "uv>=0.13.3"
hash -r
uv python install 3.12.15
for h3ltx_environment in runtime comfy ltx; do uv venv --python 3.12.15 "/opt/venvs/$h3ltx_environment"; done
cp "$H3LTX_REPO_ROOT/docker/sources.lock.json" /opt/source-pins.json
/opt/venvs/runtime/bin/python "$H3LTX_REPO_ROOT/docker/fetch_sources.py"
for h3ltx_environment in comfy ltx; do
  uv pip sync --python "/opt/venvs/$h3ltx_environment/bin/python" \
    --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match \
    --require-hashes "$H3LTX_REPO_ROOT/docker/locks/$h3ltx_environment.txt"
done
uv pip install --python /opt/venvs/ltx/bin/python --no-deps /opt/LTX-2/packages/ltx-core /opt/LTX-2/packages/ltx-pipelines
cd "$H3LTX_REPO_ROOT"
uv export --frozen --no-dev --extra serverless --no-emit-project -o /tmp/h3ltx-runtime-requirements.txt
uv pip sync --python /opt/venvs/runtime/bin/python /tmp/h3ltx-runtime-requirements.txt
for h3ltx_environment in runtime ltx; do uv pip install --python "/opt/venvs/$h3ltx_environment/bin/python" --no-deps .; done
rm /tmp/h3ltx-runtime-requirements.txt
install -d /opt/ComfyUI/custom_nodes/h3ltx_memory
cp src/h3ltx/comfy_extension.py /opt/ComfyUI/custom_nodes/h3ltx_memory/__init__.py
rm -rf /opt/h3ltx-install-cache
printf 'Setup complete. Supply secrets via Runpod environment, select MODEL_GROUPS, then run scripts/start-pod.sh.\n'
