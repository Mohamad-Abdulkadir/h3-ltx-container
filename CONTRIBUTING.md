# Contributing

Use Python 3.12 and uv. Keep weights and credentials outside version control.

```sh
uv sync --frozen --dev --extra serverless --python 3.12
uv run ruff check .
uv run ruff format --check .
uv run mypy src/h3ltx
uv run pytest
uv build
uv run python scripts/verify_local.py
```

Tests run without a GPU. They validate API graphs against the recorded fixture and
exercise queue recovery, auth, uploads, model integrity, metrics and downloads.
The provenance sidecar distinguishes live native schemas from source-derived ones.
Never claim GPU performance from mocks. See `docs/testing-matrix.md`.

When changing a builder, inspect the official pinned workflow including subgraphs,
update graph tests and validate against a live target server. Flatten DynamicCombo
and Autogrow fields; never use FLOAT slot 0 for integer frame counts.
When changing model pins, regenerate metadata from Hugging Face rather than typing
sizes/hashes by hand. Review licences before adding third-party source or models.

Container dependency updates must regenerate both Linux engine locks and the runtime
lock. Keep CUDA/torch/compiled architecture compatibility explicit. Do not run upstream
install hooks without reviewing them. Do not publish images from untrusted pull requests.
