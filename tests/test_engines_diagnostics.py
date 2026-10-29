import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from h3ltx.config import Settings
from h3ltx.doctor import report, software
from h3ltx.engines.comfy import ComfyClient, ComfySupervisor
from h3ltx.engines.native import NativeEngine
from h3ltx.errors import RuntimeFault
from h3ltx.gpu import GPU, NVMLBackend, encode_probe
from h3ltx.idle import IdleStop
from h3ltx.media import encode_output, ffmpeg
from h3ltx.params import H3, LTX
from h3ltx.schedules import distilled_sigmas


async def test_comfy_transport(tmp_path, monkeypatch, object_info):
    import h3ltx.engines.comfy as module

    root = tmp_path / "engine"
    root.mkdir()
    (root / "video.mp4").write_bytes(b"media")
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path == "/object_info":
            return httpx.Response(200, json=object_info)
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "test"})
        if request.url.path.startswith("/history"):
            return httpx.Response(
                200,
                json={
                    "test": {
                        "status": {"status_str": "success"},
                        "outputs": {
                            "1": {
                                "images": [
                                    {"filename": "video.mp4", "subfolder": "", "type": "output"}
                                ]
                            }
                        },
                    }
                },
            )
        return httpx.Response(200, json={})

    class Socket:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def __aiter__(self):
            async def events():
                yield b"preview"
                for event in [
                    {"type": "executing", "data": {"prompt_id": "other", "node": "1"}},
                    {"type": "executing", "data": {"prompt_id": "test", "node": "1"}},
                    {"type": "progress", "data": {"prompt_id": "test", "value": 1, "max": 1}},
                    {"type": "execution_cached", "data": {"prompt_id": "test", "nodes": ["2"]}},
                    {"type": "executing", "data": {"prompt_id": "test", "node": None}},
                ]:
                    yield json.dumps(event)

            return events()

    monkeypatch.setattr(module, "connect", lambda *a, **kw: Socket())
    client = ComfyClient("http://local", root)
    await client.http.aclose()
    client.http = httpx.AsyncClient(base_url="http://local", transport=httpx.MockTransport(handler))
    assert "SaveVideo" in await client.info()
    output = tmp_path / "outputs"
    output.mkdir()
    progress = AsyncMock()
    result = await client.execute(
        {"1": {"class_type": "SaveVideo", "inputs": {}}}, H3(prompt="x"), output, progress
    )
    assert result["outputs"] == ["video.mp4"]
    assert result["timeline"]["cached_nodes"] == ["2"]
    assert (output / "video.mp4").read_bytes() == b"media"
    client.active_prompt = "test"
    await client.interrupt()
    await client.diagnostics()
    assert {r.url.path for r in seen} >= {"/prompt", "/interrupt", "/queue"}
    await client.close()


async def test_native_jsonl_protocol(tmp_path, monkeypatch):
    import h3ltx.engines.native as module

    script = tmp_path / "worker.py"
    script.write_text("""import sys,json
for line in sys.stdin:
 c=json.loads(line)
 print(json.dumps({"type":"progress","data":{"stage":"sampling","progress":50}}),flush=True)
 print("engine log",file=sys.stderr,flush=True)
 print(json.dumps({"type":"result","data":{"outputs":["native.mp4"],"timeline":{"cached_nodes":[]}}}),flush=True)
""")
    original = asyncio.create_subprocess_exec

    async def spawn(*args, **kwargs):
        assert "API_TOKEN" not in kwargs["env"]
        return await original(sys.executable, str(script), **kwargs)

    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", spawn)
    settings = Settings(volume_root=tmp_path, ltx_python=Path(sys.executable))
    engine = NativeEngine(settings, 0)
    (tmp_path / "logs").mkdir()
    output = tmp_path / "outputs"
    output.mkdir()
    assert (await engine.info())["available"]
    progress = AsyncMock()
    result = await engine.execute(
        {"native": {}}, LTX(prompt="x", engine="ltx-native"), output, progress
    )
    assert result["outputs"] == ["native.mp4"]
    first_process = engine.process
    await engine.execute({"native": {}}, LTX(prompt="x", engine="ltx-native"), output, progress)
    assert engine.process is first_process
    await engine.interrupt()
    assert engine.process is None
    assert "engine log" in (tmp_path / "logs/engine.log").read_text()


async def test_supervisor_process_contract(tmp_path, monkeypatch, object_info):
    import h3ltx.engines.comfy as module

    class Process:
        returncode = None
        stdout = asyncio.StreamReader()

        def terminate(self):
            self.returncode = 0

        def kill(self):
            self.returncode = -9

        async def wait(self):
            return self.returncode

    process = Process()
    process.stdout.feed_data(b"ready\n")
    process.stdout.feed_eof()
    calls = []

    async def spawn(*args, **kwargs):
        calls.append((args, kwargs))
        return process

    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", spawn)
    settings = Settings(volume_root=tmp_path, comfy_root=tmp_path / "comfy")
    (settings.comfy_root / "custom_nodes").mkdir(parents=True)
    (tmp_path / "models").mkdir()
    supervisor = ComfySupervisor(settings, 2)
    monkeypatch.setattr(supervisor.client, "info", AsyncMock(return_value=object_info))
    await supervisor.start(H3(prompt="x"))
    assert "--cache-none" in calls[0][0]
    assert "--listen" in calls[0][0] and "127.0.0.1" in calls[0][0]
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert "API_TOKEN" not in calls[0][1]["env"]
    assert (supervisor.root / "custom_nodes").is_symlink()
    assert (supervisor.root / "models").is_symlink()
    await supervisor.start(H3(prompt="other"))
    assert len(calls) == 1
    await supervisor.close()
    assert list(supervisor.lines) == ["ready"]


async def test_doctor_and_idle(tmp_path, monkeypatch, manifest):
    import h3ltx.doctor as module
    import h3ltx.idle as idle_module

    monkeypatch.setattr(
        module, "software", AsyncMock(return_value={"arches": ["sm_120"], "bf16": True})
    )
    monkeypatch.setattr(module, "encode_probe", lambda index: True)
    monkeypatch.setenv("RUNPOD_DC_ID", "EUR-IS-1")
    monkeypatch.delenv("RUNPOD_COUNTRY", raising=False)
    settings = Settings(volume_root=tmp_path, idle_stop_minutes=1, runpod_api_key="placeholder")
    gpu = GPU(0, "test", "uuid", (12, 0), 96 * 10**9)
    data = await report(settings, [gpu], manifest)
    assert data["runpod"]["country"] == "IS"
    assert gpu.nvenc
    assert not any("excluded" in w for w in data["warnings"])
    monkeypatch.setenv("RUNPOD_DC_ID", "EU-CZ-1")
    assert any("excluded" in w for w in (await report(settings, [gpu], manifest))["warnings"])
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-id")
    calls = []
    original = httpx.AsyncClient

    def handler(request):
        calls.append(request)
        return httpx.Response(204)

    monkeypatch.setattr(
        idle_module.httpx,
        "AsyncClient",
        lambda **kw: original(transport=httpx.MockTransport(handler)),
    )
    stop = IdleStop(settings)
    assert not await stop.check(lambda: True)
    stop.last_activity = time.monotonic() - 61
    assert await stop.check(lambda: False)
    assert calls[0].url.path == "/v1/pods/pod-id/stop"
    assert not await stop.check(lambda: False)


async def test_missing_software_python(tmp_path):
    result = await software(Settings(comfy_python=tmp_path / "absent"))
    assert "error" in result


def test_fake_nvml(monkeypatch):
    import types

    module = types.ModuleType("pynvml")

    class NVMLError(Exception):
        pass

    module.NVMLError = NVMLError
    module.nvmlInit = lambda: None
    module.nvmlDeviceGetCount = lambda: 1
    module.nvmlDeviceGetHandleByIndex = lambda i: i
    module.nvmlDeviceGetName = lambda h: "GPU"
    module.nvmlDeviceGetUUID = lambda h: "uuid"
    module.nvmlDeviceGetCudaComputeCapability = lambda h: (12, 0)
    module.nvmlDeviceGetMemoryInfo = lambda h: SimpleNamespace(total=96 * 10**9, used=1024)
    module.nvmlSystemGetDriverVersion = lambda: "driver"
    module.nvmlDeviceGetUtilizationRates = lambda h: SimpleNamespace(gpu=25)
    module.nvmlDeviceGetPowerUsage = lambda h: 200000
    module.nvmlDeviceGetTemperature = lambda h, kind: 45
    monkeypatch.setitem(sys.modules, "pynvml", module)
    backend = NVMLBackend()
    assert backend.detect()[0].power_watts == 200
    assert backend.sample(0)["utilization"] == 25


def test_encode_probe_failure(monkeypatch):
    import h3ltx.gpu as module

    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1))
    assert not encode_probe(0)

    def fail(*args, **kwargs):
        raise OSError("missing")

    monkeypatch.setattr(module.subprocess, "run", fail)
    assert not encode_probe(0)


async def test_media_encoder(tmp_path):

    path = tmp_path / "video.mp4"
    await ffmpeg(
        "-f", "lavfi", "-i", "color=size=128x128:rate=24", "-t", "1", "-c:v", "libx264", str(path)
    )
    assert await encode_output(path, False, 0) == "libx264"
    with pytest.raises(RuntimeFault):
        await ffmpeg("-i", str(tmp_path / "missing"), str(tmp_path / "out.mp4"))


@pytest.mark.parametrize("steps", [1, 4, 8, 20])
def test_sigma_interpolation(steps):
    values = distilled_sigmas(steps)
    assert len(values) == steps + 1
    assert values[0] == 1 and values[-1] == 0
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))


def test_engine_parent_death_contract(monkeypatch):
    import h3ltx.engine_boot as module

    calls = []
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(
        module.ctypes, "CDLL", lambda *a, **kw: SimpleNamespace(prctl=lambda *args: 0)
    )
    monkeypatch.setattr(module.os, "getppid", lambda: 123)
    monkeypatch.setattr(module.os, "execv", lambda executable, args: calls.append(args))
    module.launch(123, ["main.py", "--cpu"])
    assert calls[0][-2:] == ["main.py", "--cpu"]
    with pytest.raises(SystemExit):
        module.launch(456, ["main.py"])
