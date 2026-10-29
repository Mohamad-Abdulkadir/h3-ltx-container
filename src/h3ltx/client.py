import json
import os
import shutil
import stat
import tempfile
import time
import tomllib
import zipfile
from pathlib import Path
from typing import Any

import httpx
import typer
from dotenv import load_dotenv

from h3ltx.records import write_json

app = typer.Typer(help="Submit video jobs and archive complete run records.", no_args_is_help=True)


def connection() -> httpx.Client:
    load_dotenv(Path.cwd() / ".env", override=False)
    path = Path.home() / ".config/h3ltx/config.toml"
    config = tomllib.loads(path.read_text()) if path.exists() else {}
    base_url = os.getenv("H3LTX_URL", config.get("base_url", "http://127.0.0.1:8000"))
    token = os.getenv("API_TOKEN", os.getenv("H3LTX_TOKEN", config.get("token", "")))
    if not token:
        raise typer.BadParameter(
            "Set API_TOKEN or H3LTX_TOKEN, or token in ~/.config/h3ltx/config.toml"
        )
    return httpx.Client(
        base_url=base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,
        follow_redirects=False,
    )


def request(method: str, path: str, **kwargs: Any) -> Any:
    with connection() as client:
        response = client.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()


def print_json(data: Any) -> None:
    typer.echo(json.dumps(data, indent=2))


@app.command()
def submit(file: Path) -> None:
    """Submit a request JSON file."""
    print_json(request("POST", "/v1/jobs", json=json.loads(file.read_text())))


@app.command()
def upload(file: Path) -> None:
    """Stage an image, audio clip or video and print its upload ID."""
    with file.open("rb") as stream:
        print_json(request("POST", "/v1/uploads", files={"file": (file.name, stream)}))


@app.command()
def status(identifier: str) -> None:
    print_json(request("GET", f"/v1/jobs/{identifier}"))


@app.command()
def wait(identifier: str, interval: float = 2, timeout: float = 14400) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = request("GET", f"/v1/jobs/{identifier}")
        if job["state"] in ("succeeded", "failed", "cancelled"):
            print_json(job)
            if job["state"] != "succeeded":
                raise typer.Exit(1)
            return
        time.sleep(interval)
    raise typer.BadParameter("Wait timeout exceeded; remote job continues")


def unpack(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            path = destination / member.filename
            if not path.resolve().is_relative_to(destination.resolve()) or stat.S_ISLNK(
                member.external_attr >> 16
            ):
                raise ValueError("Unsafe path or symlink in run bundle")
        bundle.extractall(destination)


def pull_runs(client: httpx.Client, directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    cursor_path = directory / ".h3ltx-cursor.json"
    identity = str(client.base_url)
    cursor_data = json.loads(cursor_path.read_text()) if cursor_path.exists() else {}
    if cursor_data.get("server", identity) != identity:
        raise ValueError("Cursor belongs to a different server; use a separate destination")
    cursor, downloaded = int(cursor_data.get("cursor", 0)), 0
    while True:
        response = client.get("/v1/runs/export", params={"since_cursor": cursor})
        response.raise_for_status()
        batch = response.json()
        if not batch["runs"]:
            return downloaded
        for run in batch["runs"]:
            identifier = run["id"]
            if len(identifier) != 32 or any(c not in "0123456789abcdef" for c in identifier):
                raise ValueError("Invalid server run identifier")
            target = directory / identifier
            if not (target / ".complete").exists():
                scratch = Path(tempfile.mkdtemp(prefix=".pull-", dir=directory))
                try:
                    archive = scratch / "bundle.zip"
                    # Construct local paths; never send credentials to remote bundle URLs.
                    with client.stream("GET", f"/v1/jobs/{identifier}/bundle") as stream:
                        stream.raise_for_status()
                        with archive.open("wb") as output:
                            for chunk in stream.iter_bytes(1024**2):
                                output.write(chunk)
                    extracted = scratch / "run"
                    unpack(archive, extracted)
                    (extracted / ".complete").write_text("ok\n")
                    if target.exists():
                        raise ValueError(
                            "Incomplete existing run directory; move it aside before retrying"
                        )
                    extracted.rename(target)
                    downloaded += 1
                finally:
                    shutil.rmtree(scratch)
            cursor = run["cursor"]
            write_json(cursor_path, {"server": identity, "cursor": cursor})


@app.command()
def pull(directory: Path = Path("runs")) -> None:
    """Download and unpack every newly finished run; advance cursor after each success."""
    with connection() as client:
        count = pull_runs(client, directory)
    typer.echo(f"Downloaded {count} run bundles")


@app.command()
def stats(group_by: str = "type,profile,resolution,steps") -> None:
    print_json(request("GET", "/v1/stats", params={"group_by": group_by}))


@app.command()
def system() -> None:
    print_json(request("GET", "/v1/system"))
