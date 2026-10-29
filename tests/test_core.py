import asyncio
import hashlib

import pytest
from pydantic import ValidationError

from h3ltx.config import Settings
from h3ltx.engines.comfy import EventTimeline
from h3ltx.errors import RuntimeFault
from h3ltx.gpu import GPU, FakeBackend, admit, profile_for, profiles
from h3ltx.metrics import Sampler, cgroup_oom, cgroup_ram
from h3ltx.models import Manifest, ModelFile, sha256, status, sync
from h3ltx.params import H3, LTX, JobRequest, h3_frames
from h3ltx.stats import percentile, summarize
from h3ltx.store import Store
from h3ltx.uploads import public_url


@pytest.mark.parametrize("seconds", [4, 4.1, 5, 6, 7.25, 10, 15])
def test_h3_grid(seconds):
    assert h3_frames(seconds) % 17 == 5
    assert h3_frames(seconds) >= round(seconds * 24)


@pytest.mark.parametrize(
    "data",
    [
        {"width": 641},
        {"duration_seconds": 16},
        {"unknown": 2},
        {"engine": "ltx-native"},
        {"references": [{"upload_id": "0" * 32}] * 10},
        {"references": [{"upload_id": "0" * 32, "kind": "audio", "duration_seconds": 1}]},
        {"references": [{"upload_id": "0" * 32, "kind": "video", "duration_seconds": 10}] * 2},
    ],
)
def test_bad_h3(data):
    with pytest.raises(ValidationError):
        H3(prompt="test", **data)


@pytest.mark.parametrize(
    "data",
    [
        {"frames": 120},
        {"width": 1056},
        {"engine": "ltx-native", "weights": "int8"},
        {"temporal_upscale": 1},
        {"keyframes": [{"upload_id": "0" * 32, "kind": "audio"}]},
    ],
)
def test_bad_ltx(data):
    with pytest.raises(ValidationError):
        LTX(prompt="test", **data)


@pytest.mark.parametrize("kind", ["unknown", "h3.fl2v", "h3.r2v", "ltx.i2v", "ltx.flf2v"])
def test_job_requirements(kind):
    with pytest.raises(ValueError):
        JobRequest(type=kind, params={"prompt": "test"}).resolve()


def test_random_seed_size():
    assert H3(prompt="x").seed != H3(prompt="x").seed
    p = H3(prompt="x", aspect_ratio="9:16", resolution=640)
    assert p.width == 640 and p.height == 1152


@pytest.mark.parametrize("profile", profiles())
def test_profiles(profile):
    gpu = GPU(0, "test", "uuid", tuple(profile["capability"]), int(profile["min_gb"] * 1e9))
    assert profile_for(gpu)["id"] == profile["id"]


def test_admission():
    p = profiles()[0]
    assert admit("h3.t2v", H3(prompt="x"), p)["provisional"]
    with pytest.raises(RuntimeFault, match="budget"):
        admit("h3.t2v", H3(prompt="x", width=4096, height=4096), p)
    assert admit("h3.t2v", H3(prompt="x", width=4096, height=4096, force=True), p)["forced"]
    with pytest.raises(RuntimeFault, match="quantized"):
        admit("h3.t2v", H3(prompt="x", force=True), profiles()[-1])
    with pytest.raises(RuntimeFault, match="tiling"):
        admit("ltx.dfr", LTX(prompt="x", width=3840, height=2176, tiled=False), p)
    assert profile_for(GPU(0, "other", "u", (7, 0), 16 * 10**9)) is None


def test_startup_validation(tmp_path):
    with pytest.raises(ValueError, match="API_TOKEN"):
        Settings(volume_root=tmp_path).validate_startup()
    with pytest.raises(ValueError, match="SLOTS"):
        Settings(api_token="x" * 32, slots_per_gpu=2).validate_startup()
    with pytest.raises(ValueError, match="volume"):
        Settings(api_token="x" * 32, volume_root=tmp_path).validate_startup()


def test_store_durability(tmp_path):
    path = tmp_path / "runtime.db"
    store = Store(path)
    assert store.db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    request = JobRequest(
        type="h3.t2v", params={"prompt": "x"}, client_request_id="one"
    ).model_dump()
    a, created = store.submit(request, {})
    assert created
    assert not store.submit(request, {})[1]
    with pytest.raises(RuntimeFault, match="different"):
        store.submit({**request, "priority": 9}, {})
    b, _ = store.submit(
        {**request, "client_request_id": "two", "priority": 10, "tags": ["test"]}, {}
    )
    assert store.claim(1)["id"] == b["id"]
    store.transition(b["id"], "running")
    store.progress(b["id"], {"progress": 25, "stage": "sampling"})
    store.close()
    store = Store(path)
    interrupted = store.recover()
    assert interrupted[0]["id"] == b["id"]
    assert store.get(b["id"])["error"]["code"] == "interrupted_by_restart"
    assert store.get(a["id"])["state"] == "queued"
    assert len(store.list(tag="test", kind="h3.t2v", since=0)) == 1
    assert len(store.events(b["id"])) >= 4
    cursor = store.export(0)
    assert cursor["runs"][0]["id"] == b["id"]
    assert not store.export(cursor["cursor"])["runs"]
    with pytest.raises(RuntimeFault):
        store.transition(b["id"], "running")
    with pytest.raises(RuntimeFault):
        store.get("absent")
    store.measure(b["id"], {"x": 1})
    assert store.measurements() == [{"x": 1}]
    store.close()


def test_model_integrity(tmp_path):
    data = b"weights"
    model = ModelFile(
        key="test",
        group="test",
        repo="org/model",
        path="a",
        destination="a.safetensors",
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        revision="a" * 40,
        license="Apache-2.0",
        gated=False,
    )
    assert status([model], tmp_path)[0]["status"] == "missing"
    target = model.target(tmp_path)
    target.write_bytes(data)
    assert sha256(target) == model.sha256
    assert status([model], tmp_path, True)[0]["status"] == "ready"
    target.write_bytes(b"bad")
    assert status([model], tmp_path)[0]["status"] == "size_mismatch"
    target.write_bytes(b"changed")
    assert status([model], tmp_path, True)[0]["status"] == "hash_mismatch"
    with pytest.raises(RuntimeFault):
        model.model_copy(update={"destination": "../escape"}).target(tmp_path)
    with pytest.raises(RuntimeFault):
        Manifest(files=[model]).select(["unknown"])
    with pytest.raises(RuntimeFault):
        Manifest(files=[model]).by_name("absent")


def test_model_sync(tmp_path, monkeypatch):
    import h3ltx.models as module

    source = tmp_path / "source"
    source.write_bytes(b"safe")
    model = ModelFile(
        key="test",
        group="test",
        repo="org/model",
        path="file",
        destination="vae/file",
        size_bytes=4,
        sha256=hashlib.sha256(b"safe").hexdigest(),
        revision="a" * 40,
        license="Apache-2.0",
        gated=True,
    )
    with pytest.raises(RuntimeFault, match="HF_TOKEN"):
        sync([model], tmp_path / "models", None)
    monkeypatch.setattr(module, "hf_hub_download", lambda **kwargs: str(source))
    sync([model], tmp_path / "models", "placeholder", parallel=1)
    assert (tmp_path / "models/vae/file").read_bytes() == b"safe"
    assert (tmp_path / "models/models.lock.json").is_file()


async def test_sampler(tmp_path):
    (tmp_path / "memory.current").write_text("123")
    (tmp_path / "memory.events").write_text("oom_kill 2\n")
    assert cgroup_ram(tmp_path) == 123
    assert cgroup_oom(tmp_path) == 2
    sampler = Sampler(FakeBackend(), 0, 0.001)
    task = asyncio.create_task(sampler.run())
    await asyncio.sleep(0.01)
    sampler.stop.set()
    await task
    assert sampler.summary()["sample_count"] >= 1
    assert sampler.summary()["vram_bytes_peak"] == 1024


def test_timeline():
    timeline = EventTimeline({"1": {"class_type": "SamplerCustomAdvanced"}})
    timeline.consume({"type": "executing", "data": {"node": "1", "timestamp": 1000}})
    timeline.consume({"type": "progress", "data": {"value": 1, "max": 2, "timestamp": 1200}})
    timeline.consume({"type": "progress", "data": {"value": 2, "max": 2, "timestamp": 1400}})
    timeline.consume({"type": "execution_cached", "data": {"nodes": ["2"]}})
    timeline.consume({"type": "executing", "data": {"node": None, "timestamp": 2000}})
    assert timeline.result()["stages"]["sampling"] == 1
    assert timeline.result()["cached_nodes"] == ["2"]
    assert timeline.result()["sampling_steps"][1]["seconds"] == pytest.approx(0.2)


def test_stats():
    rows = [
        {
            "type": "h3.t2v",
            "state": state,
            "wall_seconds": wall,
            "stages": {"sampling": wall},
            "output_seconds": 5,
            "cached": state == "succeeded",
        }
        for state, wall in [("succeeded", 60), ("failed", 120)]
    ]
    group = summarize(rows, "type")[0]
    assert group["success_rate"] == 0.5
    assert group["wall_seconds"]["p50"] == 90
    assert group["wall_seconds"]["p90"] == 114
    assert percentile([], 0.5) is None
    with pytest.raises(RuntimeFault):
        summarize(rows, "injection")


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/a",
        "https://user:pass@example.com/a",
        "https://127.0.0.1/x",
        "https://169.254.169.254/x",
        "file:///etc/passwd",
        "https://example.com:8000/x",
    ],
)
def test_ssrf(url):
    with pytest.raises(RuntimeFault):
        public_url(url)


def test_manifest_pins_are_complete(manifest):
    assert len({m.key for m in manifest.files}) == len(manifest.files)
    assert len({m.destination for m in manifest.files}) == len(manifest.files)
    for model in manifest.files:
        assert len(model.revision) == 40
        assert model.sha256 and len(model.sha256) == 64
        assert model.size_bytes > 0
        assert ".pth" not in model.destination
    assert sum(m.size_bytes for m in manifest.select()) == 312331947147
