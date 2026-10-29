import asyncio

import pytest
from fastapi.testclient import TestClient

from h3ltx.api import create_app
from h3ltx.loading import LoadRequest, load_graph, pipelines
from h3ltx.params import JobRequest
from h3ltx.workflows.validate import validate_graph

TOKEN = {"Authorization": "Bearer " + "t" * 32}


def terminal(client, identifier):
    import time

    for _ in range(500):
        data = client.get(f"/v1/load/{identifier}", headers=TOKEN).json()
        if data["state"] in ("succeeded", "failed", "cancelled"):
            return data
        time.sleep(0.005)
    pytest.fail("load operation timed out")


@pytest.mark.parametrize("key", sorted(pipelines()))
def test_load_graph_schemas(key, object_info, manifest):
    kind, params = pipelines()[key]
    graph = load_graph(key, kind, params)
    validate_graph(graph, object_info, manifest)
    assert all(
        n["class_type"] not in ("SamplerCustomAdvanced", "SaveVideo") for n in graph.values()
    )


def test_explicit_load_then_warm_job_and_unload(runtime):
    with TestClient(create_app(runtime.settings, runtime)) as client:
        assert client.post("/v1/load", json={"model": "h3-fl2v"}).status_code == 401
        assert client.post("/v1/load", headers=TOKEN, json={"model": "missing"}).status_code == 422
        assert (
            client.post(
                "/v1/load", headers=TOKEN, json={"model": "h3-fl2v", "gpus": [99]}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/v1/load", headers=TOKEN, json={"model": "h3-fl2v", "profile": "ada-48"}
            ).status_code
            == 422
        )
        load = client.post("/v1/load", headers=TOKEN, json={"model": "h3-fl2v"})
        assert load.status_code == 202, load.text
        loaded = terminal(client, load.json()["load_id"])
        assert loaded["state"] == "succeeded", loaded
        assert "load-" in loaded["run_dir"]
        assert loaded["result"]["gpus"][0]["components"][0]["read_from_disk_s"] > 0
        assert loaded["result"]["gpus"][0]["metrics"]["vram_bytes_peak"] == 1024
        events = client.get(f"/v1/load/{loaded['id']}/events", headers=TOKEN)
        assert "succeeded" in events.text
        job = client.post(
            "/v1/jobs", headers=TOKEN, json={"type": "h3.t2v", "params": {"prompt": "warm"}}
        ).json()
        import time

        for _ in range(200):
            finished = client.get(f"/v1/jobs/{job['id']}", headers=TOKEN).json()
            if finished["state"] in ("succeeded", "failed"):
                break
            time.sleep(0.005)
        assert finished["state"] == "succeeded", finished
        assert finished["result"]["warm"] is True
        timeline = client.get(f"/v1/jobs/{job['id']}/artifacts/timeline.json", headers=TOKEN).json()
        assert timeline["model_load_s"] == 0
        assert len(runtime.test_engine.loads) == 1
        assert "h3-fl2v" in client.get("/v1/capacity", headers=TOKEN).json()["resident"]["0"]
        assert client.get("/v1/stats?group_by=model,profile", headers=TOKEN).status_code == 200
        unloaded = client.post("/v1/unload", headers=TOKEN, json={"model": "h3-fl2v"}).json()
        assert terminal(client, unloaded["load_id"])["state"] == "succeeded"
        assert not runtime.loading.resident[0]


def test_lazy_job_is_cold_and_percentiles_exclude_it(runtime):
    import time

    with TestClient(create_app(runtime.settings, runtime)) as client:
        job = client.post(
            "/v1/jobs", headers=TOKEN, json={"type": "h3.t2v", "params": {"prompt": "cold"}}
        ).json()
        for _ in range(200):
            finished = client.get(f"/v1/jobs/{job['id']}", headers=TOKEN).json()
            if finished["state"] == "succeeded":
                break
            time.sleep(0.005)
        assert finished["result"]["warm"] is False
        assert finished["result"]["model_load_s"] > 0
        group = client.get("/v1/stats", headers=TOKEN).json()["groups"][0]
        assert group["cold_runs_excluded"] == 1 and group["wall_seconds"]["p50"] is None
        assert (
            client.get("/v1/stats?include_cold=true", headers=TOKEN).json()["groups"][0][
                "wall_seconds"
            ]["p50"]
            > 0
        )


async def test_load_waits_for_running_job_on_same_gpu(runtime):
    runtime.test_engine.delay = 0.1
    await runtime.start()
    try:
        job = runtime.submit(JobRequest(type="h3.t2v", params={"prompt": "busy"}))
        for _ in range(100):
            if job["id"] in runtime.active:
                break
            await asyncio.sleep(0.002)
        request = runtime.loading.submit(LoadRequest(model="ltx-distilled"), "load")
        await asyncio.sleep(0.01)
        assert not any(key == "ltx-distilled" for _, key in runtime.test_engine.loads)
        for _ in range(200):
            if runtime.store.get(request["load_id"])["state"] in ("succeeded", "failed"):
                break
            await asyncio.sleep(0.005)
        assert runtime.store.get(request["load_id"])["state"] == "succeeded"
        assert runtime.store.get(job["id"])["state"] == "succeeded"
    finally:
        await runtime.close()
