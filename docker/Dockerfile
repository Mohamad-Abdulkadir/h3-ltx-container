# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.13.3@sha256:bee0a2a458fd956c5df7b8b4c5fed2257b0af34866e3da7b51db0be3acb375d1 AS uv
# CUDA 13.0.2 runtime, amd64 manifest resolved from NVIDIA's registry, 2026-10-29.
FROM --platform=linux/amd64 nvidia/cuda:13.4.3-cudnn-runtime-ubuntu24.04@sha256:534ea2e94fd17f053b4422d094f50a43a7e75a03240c4b89cbe9ad42085ec1c3
COPY --from=uv /uv /uvx /usr/local/bin/
ENV DEBIAN_FRONTEND=noninteractive UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy UV_NO_CACHE=1 PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    PATH=/opt/venvs/runtime/bin:/usr/local/bin:/usr/bin:/bin \
    MODE=pod HF_HOME=/workspace/.cache/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git ffmpeg tini openssh-server libglib2.0-0 libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && uv python install 3.12.15 \
    && uv venv --python 3.12.15 /opt/venvs/runtime \
    && uv venv --python 3.12.15 /opt/venvs/comfy \
    && uv venv --python 3.12.15 /opt/venvs/ltx
COPY docker/sources.lock.json /opt/source-pins.json
COPY docker/fetch_sources.py /opt/fetch_sources.py
RUN /opt/venvs/runtime/bin/python /opt/fetch_sources.py
COPY docker/locks/ /opt/locks/
# Hardlinks across isolated venvs deduplicate matching CUDA wheels on this filesystem.
RUN UV_LINK_MODE=hardlink UV_NO_CACHE=0 uv pip sync --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match --require-hashes --python /opt/venvs/comfy/bin/python /opt/locks/comfy.txt \
    && UV_LINK_MODE=hardlink UV_NO_CACHE=0 uv pip sync --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match --require-hashes --python /opt/venvs/ltx/bin/python /opt/locks/ltx.txt \
    && rm -rf /root/.cache/uv
# The local packages use reviewed PEP 517 metadata, not repository setup scripts.
RUN uv pip install --python /opt/venvs/ltx/bin/python --no-deps \
      /opt/LTX-2/packages/ltx-core /opt/LTX-2/packages/ltx-pipelines
WORKDIR /opt/h3ltx
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ ./src/
RUN mkdir -p /opt/ComfyUI/custom_nodes/h3ltx_memory \
    && cp src/h3ltx/comfy_extension.py /opt/ComfyUI/custom_nodes/h3ltx_memory/__init__.py
RUN uv export --frozen --no-dev --extra serverless --no-emit-project -o /tmp/runtime.txt \
    && uv pip sync --python /opt/venvs/runtime/bin/python /tmp/runtime.txt \
    && uv pip install --python /opt/venvs/runtime/bin/python --no-deps . \
    && uv pip install --python /opt/venvs/ltx/bin/python --no-deps . \
    && rm /tmp/runtime.txt
COPY docker/entrypoint.sh /usr/local/bin/h3ltx-entrypoint
RUN chmod 0755 /usr/local/bin/h3ltx-entrypoint
ARG VERSION=development
ARG GIT_SHA=unknown
ENV IMAGE_VERSION=$VERSION GIT_SHA=$GIT_SHA
LABEL org.opencontainers.image.title="h3ltx" \
      org.opencontainers.image.description="API-only MiniMax H3 / LTX 2.5 runtime" \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.revision=$GIT_SHA
EXPOSE 8000 22
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s CMD /opt/venvs/runtime/bin/python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3) if os.getenv('MODE') == 'pod' else None"
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/usr/local/bin/h3ltx-entrypoint"]
