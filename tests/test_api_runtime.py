import io
import json
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from h3ltx.api import create_app
from h3ltx.errors import RuntimeFault
from h3ltx.params import H3, JobRequest
from h3ltx.records import RunRecord, write_json

TOKEN = {"Authorization": "Bearer " + "t" * 32}


def wait_finished(client, identifier):
    for _ in range(300):
        response = client.get(f"/v1/jobs/{identifier}", headers=TOKEN)
        job = response.json()
        if job["state"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.005)
    pytest.fail("fake job did not complete")


def test_all_api_routes(runtime, monkeypatch):
    from unittest.mock import AsyncMock

    monkeypatch.setattr(runtime, "system", AsyncMock(return_value={"gpus": []}))
    with TestClient(create_app(runtime.settings, runtime)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        for url in (
            "/v1/system",
            "/v1/capabilities",
            "/v1/capacity",
            "/v1/models",
            "/v1/jobs",
            "/v1/stats",
            "/v1/runs/export",
        ):
            assert client.get(url).status_code == 401
            assert client.get(url, headers=TOKEN).status_code == 200
        assert client.get("/docs").status_code == 404
        assert client.post("/v1/admin/doctor", headers=TOKEN).status_code == 200
        assert (
            client.post(
                "/v1/jobs", headers=TOKEN, json={"type": "unknown", "params": {}}
            ).status_code
            == 422
        )
        assert (
            client.post("/v1/jobs", headers=TOKEN, json={}).json()["error"]["code"]
            == "request_invalid"
        )
        request = {
            "type": "h3.t2v",
            "params": {"prompt": "test"},
            "client_request_id": "stable",
            "tags": ["api"],
        }
        submitted = client.post("/v1/jobs", headers=TOKEN, json=request)
        assert submitted.status_code == 202, submitted.text
        identifier = submitted.json()["id"]
        assert client.post("/v1/jobs", headers=TOKEN, json=request).json()["id"] == identifier
        assert (
            client.post("/v1/jobs", headers=TOKEN, json={**request, "priority": 1}).status_code
            == 409
        )
        finished = wait_finished(client, identifier)
        assert finished["state"] == "succeeded", finished
        assert len(client.get("/v1/jobs?tag=api&type=h3.t2v", headers=TOKEN).json()["jobs"]) == 1
        events = client.get(f"/v1/jobs/{identifier}/events", headers=TOKEN)
        assert "event: progress" in events.text
        assert '"state": "succeeded"' in events.text
        assert (
            client.get(
                f"/v1/jobs/{identifier}/events", headers={**TOKEN, "Last-Event-ID": "bad"}
            ).status_code
            == 422
        )
        artifacts = client.get(f"/v1/jobs/{identifier}/artifacts", headers=TOKEN).json()[
            "artifacts"
        ]
        assert {
            "request.json",
            "resolved.json",
            "workflow.json",
            "timeline.json",
            "metrics.json",
            "result.json",
        } <= {a["name"] for a in artifacts}
        response = client.get(
            f"/v1/jobs/{identifier}/artifacts/outputs/video.mp4",
            headers={**TOKEN, "Range": "bytes=0-3"},
        )
        assert response.status_code == 206
        assert response.content == b"fake"
        assert (
            client.get(
                f"/v1/jobs/{identifier}/artifacts/outputs/video.mp4",
                headers={**TOKEN, "Range": "bytes=100-200"},
            ).status_code
            == 416
        )
        bundle = client.get(f"/v1/jobs/{identifier}/bundle", headers=TOKEN)
        assert bundle.status_code == 200
        archive = zipfile.ZipFile(io.BytesIO(bundle.content))
        assert json.loads(archive.read("result.json"))["state"] == "succeeded"
        assert client.get("/v1/runs/export", headers=TOKEN).json()["runs"][0]["id"] == identifier
        stats = client.get("/v1/stats", headers=TOKEN).json()["groups"][0]
        assert stats["success_rate"] == 1
        assert client.get("/v1/stats?group_by=invalid", headers=TOKEN).status_code == 422
        assert client.get("/v1/jobs/missing", headers=TOKEN).status_code == 404
        assert (
            client.post(f"/v1/jobs/{identifier}/cancel", headers=TOKEN).json()["state"]
            == "succeeded"
        )


def test_cancel_running_and_queued(runtime):
    runtime.test_engine.delay = 5
    with TestClient(create_app(runtime.settings, runtime)) as client:
        ids = [
            client.post(
                "/v1/jobs", headers=TOKEN, json={"type": "h3.t2v", "params": {"prompt": str(i)}}
            ).json()["id"]
            for i in range(2)
        ]
        for _ in range(100):
            if client.get(f"/v1/jobs/{ids[0]}", headers=TOKEN).json()["state"] == "running":
                break
            time.sleep(0.005)
        assert client.get(f"/v1/jobs/{ids[0]}/bundle", headers=TOKEN).status_code == 409
        assert (
            client.post(f"/v1/jobs/{ids[1]}/cancel", headers=TOKEN).json()["state"] == "cancelled"
        )
        assert (
            client.post(f"/v1/jobs/{ids[0]}/cancel", headers=TOKEN).json()["state"] == "cancelled"
        )
        assert runtime.test_engine.interrupts >= 1
        assert len(client.get("/v1/runs/export", headers=TOKEN).json()["runs"]) == 2


@pytest.mark.parametrize(
    "error,code",
    [
        (RuntimeFault("specific", "Failure", 500), "specific"),
        (ValueError("broken"), "job_failed"),
        (TimeoutError(), "job_timeout"),
    ],
)
def test_worker_failures(runtime, error, code):
    runtime.test_engine.error = error
    with TestClient(create_app(runtime.settings, runtime)) as client:
        job = client.post(
            "/v1/jobs", headers=TOKEN, json={"type": "h3.t2v", "params": {"prompt": "x"}}
        ).json()
        job = wait_finished(client, job["id"])
        assert job["state"] == "failed"
        assert job["error"]["code"] == code


def test_model_and_capability_failures(runtime):
    runtime.schemas[0] = runtime.test_engine.schema
    target = runtime.manifest.by_name("minimax_h3_fl2va_pruned_bf16.safetensors").target(
        runtime.settings.volume_root / "models"
    )
    target.unlink()
    capabilities = runtime.capabilities()
    h3 = next(c for c in capabilities["job_types"] if c["type"] == "h3.t2v")
    assert not h3["available"] and h3["reasons"][0]["code"] == "model_missing"
    with pytest.raises(RuntimeFault):
        runtime.submit(JobRequest(type="h3.t2v", params={"prompt": "x"}))
    runtime.schemas[0] = {}
    with pytest.raises(RuntimeFault):
        runtime.submit(JobRequest(type="ltx.t2v", params={"prompt": "x"}))
    runtime.store.close()


def test_upload_and_measured_reference_limits(runtime, monkeypatch):
    import h3ltx.uploads as module

    monkeypatch.setattr(
        module,
        "probe",
        lambda path: {
            "streams": [{"codec_type": "video", "codec_name": "png", "width": 128, "height": 128}],
            "format": {"duration": "0"},
        },
    )
    with TestClient(create_app(runtime.settings, runtime)) as client:
        response = client.post(
            "/v1/uploads", headers=TOKEN, files={"file": ("../../image.png", b"image", "image/png")}
        )
        assert response.status_code == 201, response.text
        identifier = response.json()["upload_id"]
        assert response.json()["media_type"] == "image"
        assert (
            client.post(
                "/v1/jobs",
                headers=TOKEN,
                json={"type": "h3.fl2v", "params": {"prompt": "x", "first_frame": identifier}},
            ).status_code
            == 202
        )
        assert (
            client.post(
                "/v1/uploads", headers=TOKEN, json={"url": "http://127.0.0.1/a"}
            ).status_code
            == 422
        )
        assert client.post("/v1/uploads", headers=TOKEN, json={"extra": 1}).status_code == 422


def test_upload_limit(runtime):
    runtime.settings.upload_max_bytes = 4
    with TestClient(create_app(runtime.settings, runtime)) as client:
        response = client.post("/v1/uploads", headers=TOKEN, files={"file": ("file.mp4", b"12345")})
        assert response.status_code == 413


async def test_post_chain_linked_failure(runtime, monkeypatch):

    await runtime.start()
    try:
        job = runtime.submit(
            JobRequest(
                type="h3.t2v",
                params={"prompt": "x"},
                post=[{"type": "upscale.seedvr2", "params": {}}],
            )
        )

        async def fake_finish(identifier, path):
            target = path.with_name("media.mp4")
            path.rename(target)
            data = {
                "filename": target.name,
                "media_type": "video",
                "duration_seconds": 5,
                "ffprobe": {
                    "streams": [
                        {
                            "codec_type": "video",
                            "width": 1152,
                            "height": 640,
                            "avg_frame_rate": "24/1",
                        }
                    ]
                },
            }
            write_json(path.parent / "metadata.json", data)
            return data

        monkeypatch.setattr(runtime.uploads, "finish", fake_finish)
        import asyncio

        for _ in range(300):
            children = runtime.store.list()
            if len(children) >= 2 and all(j["state"] in ("succeeded", "failed") for j in children):
                break
            await asyncio.sleep(0.005)
        assert len(children) == 2
        assert children[1]["parent"] == job["id"]
    finally:
        await runtime.close()


async def test_long_iclora_clip_splitting(runtime, monkeypatch):
    import asyncio
    import shutil

    from h3ltx.media import ffmpeg
    from h3ltx.records import probe

    identifier, partial = runtime.uploads.begin()
    source = partial.with_name("media.mp4")
    await ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=size=128x128:rate=24",
        "-t",
        "2.5",
        "-c:v",
        "libx264",
        str(source),
    )
    write_json(
        source.parent / "metadata.json",
        {
            "filename": source.name,
            "media_type": "video",
            "duration_seconds": 2.5,
            "ffprobe": probe(source),
        },
    )
    calls = []

    async def execute(graph, params, output, progress):
        calls.append(graph)
        load = next(n for n in graph.values() if n["class_type"] == "LoadVideo")
        chunk = runtime.supervisors[0].root / "input" / load["inputs"]["file"]
        shutil.copyfile(chunk, output / "clip.mp4")
        return {
            "outputs": ["clip.mp4"],
            "timeline": {"stages": {"sampling": 0.01}, "cached_nodes": []},
        }

    monkeypatch.setattr(runtime.test_engine, "execute", execute)
    await runtime.start()
    try:
        job = runtime.submit(
            JobRequest(
                type="upscale.ltx_iclora", params={"input_video": identifier, "clip_seconds": 1}
            )
        )
        for _ in range(1000):
            finished = runtime.job(job["id"])
            if finished["state"] in ("succeeded", "failed"):
                break
            await asyncio.sleep(0.005)
        assert finished["state"] == "succeeded", finished
        assert len(calls) == 3
        record = RunRecord(runtime.settings.volume_root, finished, initialize=False)
        graphs = json.loads((record.path / "workflow.json").read_text())["chunks"]
        assert len(graphs) == 3
        assert float(probe(record.path / "outputs/upscaled.mp4")["format"]["duration"]) >= 2.5
    finally:
        await runtime.close()


async def test_restart_recovers_metric_journal(runtime):
    params = H3(prompt="recover")
    request = JobRequest(type="h3.t2v", params={"prompt": "recover"})
    job, _ = runtime.store.submit(
        request.model_dump(), {"params": params.model_dump(), "eligible_gpus": [0]}
    )
    record = RunRecord(runtime.settings.volume_root, job)
    (record.path / "metrics.samples.jsonl").write_text(
        '{"vram_bytes":42,"ram_bytes":43}\npartial-write'
    )
    runtime.store.claim(0)
    await runtime.start()
    try:
        assert runtime.job(job["id"])["state"] == "failed"
        metrics = json.loads((record.path / "metrics.json").read_text())
        assert metrics["vram_bytes_peak"] == 42
        assert metrics["sample_count"] == 1 and metrics["interrupted"]
        assert runtime.store.measurements()[0]["timings_incomplete"]
    finally:
        await runtime.close()
