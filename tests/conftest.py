import gzip
import json
from pathlib import Path

import pytest

from h3ltx.config import Settings
from h3ltx.gpu import GPU, FakeBackend
from h3ltx.models import Manifest
from h3ltx.records import write_json
from h3ltx.runtime import Runtime

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def object_info():
    return json.load(gzip.open(ROOT / "tests/fixtures/object_info.json.gz"))


@pytest.fixture(scope="session")
def manifest():
    return Manifest.load(ROOT / "models/manifest.yaml")


class FakeEngine:
    def __init__(self, info):
        self.schema = info
        self.calls = []
        self.delay = 0
        self.error = None
        self.interrupts = 0

    async def info(self):
        return self.schema

    async def execute(self, graph, params, output, progress):
        import asyncio

        self.calls.append(graph)
        await progress({"progress": 50, "stage": "sampling"})
        await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        (output / "video.mp4").write_bytes(b"fake-media")
        return {
            "outputs": ["video.mp4"],
            "timeline": {"stages": {"sampling": 0.01}, "cached_nodes": []},
        }

    async def interrupt(self):
        self.interrupts += 1

    async def close(self):
        pass


@pytest.fixture
def runtime(tmp_path, object_info, manifest, monkeypatch):
    import h3ltx.runtime as module

    # Test fixtures use one-byte model files; real metadata lives in the separate manifest fixture.
    small = manifest.model_copy(deep=True)
    for model in small.files:
        model.size_bytes = 1
        target = model.target(tmp_path / "models")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
    manifest_path = tmp_path / "manifest.yaml"
    import yaml

    manifest_path.write_text(yaml.safe_dump(small.model_dump()))
    settings = Settings(
        api_token="t" * 32,
        volume_root=tmp_path,
        allow_ephemeral=True,
        manifest_path=manifest_path,
        poll_seconds=0.005,
        metric_interval=0.005,
    )
    engine = FakeEngine(object_info)
    backend = FakeBackend([GPU(0, "RTX PRO 6000", "GPU-test", (12, 0), 96 * 10**9)])
    service = Runtime(settings, backend=backend, engines={0: engine})
    monkeypatch.setattr(
        module,
        "output_metadata",
        lambda path: {
            "name": path.name,
            "sha256": "fake",
            "bytes": path.stat().st_size,
            "duration_seconds": 5,
        },
    )
    service.test_engine = engine
    return service


@pytest.fixture
def upload_id(runtime):
    identifier, path = runtime.uploads.begin()
    path.rename(path.with_name("media.png")) if path.exists() else None
    (path.parent / "media.png").write_bytes(b"x")
    write_json(
        path.parent / "metadata.json",
        {
            "upload_id": identifier,
            "filename": "media.png",
            "media_type": "image",
            "duration_seconds": 0,
        },
    )
    return identifier
