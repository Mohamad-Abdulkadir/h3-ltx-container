from types import SimpleNamespace
from unittest.mock import AsyncMock

from typer.testing import CliRunner

from h3ltx import cli, client

runner = CliRunner()


def test_cli_model_commands(runtime, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "Settings", lambda: runtime.settings)
    assert runner.invoke(cli.app, ["models", "status"]).exit_code == 0
    assert runner.invoke(cli.app, ["models", "verify"]).exit_code == 0
    # One-byte test fixtures deliberately do not claim real model checksums.
    assert runner.invoke(cli.app, ["models", "verify", "--deep"]).exit_code == 1
    sync_calls = []
    monkeypatch.setattr(cli, "sync", lambda *a: sync_calls.append(a))
    assert (
        runner.invoke(
            cli.app, ["models", "sync", "--groups", "h3-core", "--parallel", "1"]
        ).exit_code
        == 0
    )
    assert sync_calls
    runtime.store.close()


def test_cli_doctor_and_serve(runtime, tmp_path, monkeypatch):
    import uvicorn

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "Settings", lambda: runtime.settings)
    monkeypatch.setattr(cli, "NVMLBackend", lambda: SimpleNamespace(detect=lambda: []))
    data = {
        "software": {"torch": "test", "cuda": "none"},
        "models": [],
        "disk_free_volume": 42,
        "ram_bytes": 8,
        "warnings": ["test warning"],
    }
    monkeypatch.setattr(cli, "report", AsyncMock(return_value=data))
    assert runner.invoke(cli.app, ["doctor", "--json"]).exit_code == 0
    assert "test warning" in runner.invoke(cli.app, ["doctor"]).output
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.append(kw))
    assert runner.invoke(cli.app, ["serve", "--port", "8001"]).exit_code == 0
    assert calls[0]["workers"] == 1
    runtime.store.close()


def test_cli_prune_dry_run(runtime, tmp_path, monkeypatch):
    import time

    from h3ltx.params import JobRequest
    from h3ltx.records import RunRecord

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "Settings", lambda: runtime.settings)
    request = JobRequest(type="h3.t2v", params={"prompt": "x"})
    job, _ = runtime.store.submit(request.model_dump(), {})
    record = RunRecord(tmp_path, job)
    runtime.store.transition(job["id"], "cancelled", run_dir=str(record.path))
    args = ["prune", "--before", str(time.time() + 1)]
    assert runner.invoke(cli.app, args).exit_code == 0
    assert record.path.exists()
    assert runner.invoke(cli.app, [*args, "--apply"]).exit_code == 0
    assert not record.path.exists()
    runtime.store.close()


def test_client_commands(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        return {"state": "succeeded"}

    monkeypatch.setattr(client, "request", fake_request)
    source = tmp_path / "request.json"
    source.write_text('{"type":"h3.t2v","params":{"prompt":"x"}}')
    for args in (
        ["submit", str(source)],
        ["status", "id"],
        ["wait", "id"],
        ["stats"],
        ["system"],
        ["upload", str(source)],
    ):
        result = runner.invoke(client.app, args)
        assert result.exit_code == 0, result.output
    assert len(calls) == 6
    monkeypatch.setattr(client, "request", lambda *a, **kw: {"state": "failed"})
    assert runner.invoke(client.app, ["wait", "id"]).exit_code == 1
    monkeypatch.setattr(client, "request", lambda *a, **kw: {"state": "running"})
    assert runner.invoke(client.app, ["wait", "id", "--timeout", "0"]).exit_code != 0


async def test_serverless_uses_shared_queue(runtime, monkeypatch):
    import importlib

    import h3ltx.serverless as module

    real_import = importlib.import_module
    observed = []

    def fake_start(options):
        response = options["handler"](
            {"id": "sdk-job", "input": {"type": "h3.t2v", "params": {"prompt": "x"}}}
        )
        observed.append(response)

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(start=fake_start) if name == "runpod.serverless" else real_import(name)
        ),
    )
    monkeypatch.setattr(module, "Settings", lambda: runtime.settings)
    monkeypatch.setattr(module, "Runtime", lambda settings: runtime)
    await module.serve()
    assert observed[0]["state"] == "succeeded"
    assert observed[0]["run_dir"]
    assert "base64" not in observed[0]
