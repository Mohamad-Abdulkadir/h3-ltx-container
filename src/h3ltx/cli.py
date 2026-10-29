import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from h3ltx.config import Settings
from h3ltx.doctor import report
from h3ltx.gpu import NVMLBackend
from h3ltx.models import Manifest, status, sync

app = typer.Typer(no_args_is_help=True)
models_app = typer.Typer(help="Populate and verify persistent model storage.")
app.add_typer(models_app, name="models")


@app.callback()
def local_environment() -> None:
    load_dotenv(Path.cwd() / ".env", override=False)


def selection(groups: str | None) -> tuple[Settings, Manifest, list[str]]:
    settings = Settings()
    manifest = Manifest.load(settings.manifest_path)
    return settings, manifest, (groups or settings.model_groups).split(",")


@models_app.command("status")
def models_status(groups: str | None = None) -> None:
    s, manifest, selected = selection(groups)
    typer.echo(
        json.dumps({"files": status(manifest.select(selected), s.volume_root / "models")}, indent=2)
    )


@models_app.command("verify")
def models_verify(groups: str | None = None, deep: bool = False) -> None:
    s, manifest, selected = selection(groups)
    files = status(manifest.select(selected), s.volume_root / "models", deep)
    typer.echo(json.dumps({"files": files}, indent=2))
    if any(f["status"] != "ready" for f in files):
        raise typer.Exit(1)


@models_app.command("sync")
def models_sync(
    groups: str | None = None, parallel: Annotated[int, typer.Option(min=1, max=8)] = 2
) -> None:
    s, manifest, selected = selection(groups)
    sync(
        manifest.select(selected),
        s.volume_root / "models",
        s.hf_token.get_secret_value() or None,
        parallel,
    )
    typer.echo("Selected model files verified and installed")


@app.command()
def doctor(json_output: Annotated[bool, typer.Option("--json")] = False) -> None:
    s = Settings()
    try:
        gpus = NVMLBackend().detect()
    except Exception:
        gpus = []
    s.volume_root.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(report(s, gpus, Manifest.load(s.manifest_path)))
    if json_output:
        typer.echo(json.dumps(result, indent=2))
        return
    table = Table("Check", "Result")
    table.add_row("GPUs", str(len(gpus)))
    table.add_row(
        "Torch / CUDA", f"{result['software'].get('torch')} / {result['software'].get('cuda')}"
    )
    table.add_row("Models ready", str(sum(f["status"] == "ready" for f in result["models"])))
    table.add_row("Volume free bytes", str(result["disk_free_volume"]))
    table.add_row("RAM bytes", str(result["ram_bytes"]))
    for warning in result["warnings"]:
        table.add_row("Warning", warning)
    Console().print(table)


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    from h3ltx.api import create_app

    uvicorn.run(create_app(), host=host, port=port, workers=1)


@app.command("prune")
def prune(before: Annotated[float, typer.Option()], apply: bool = False) -> None:
    """Explicit run-directory cleanup; dry run by default. Database index is retained."""
    import shutil

    from h3ltx.store import TERMINAL, Store

    settings = Settings()
    store = Store(settings.volume_root / "state/runtime.db")
    try:
        for job in store.list(states=sorted(TERMINAL), limit=1000000):
            if job["updated"] >= before:
                continue
            path = Path(job["run_dir"]) if job["run_dir"] else None
            if path and path.resolve().is_relative_to((settings.volume_root / "runs").resolve()):
                typer.echo(str(path))
                if apply and path.exists():
                    shutil.rmtree(path)
    finally:
        store.close()
