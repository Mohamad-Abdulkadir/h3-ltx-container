import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from h3ltx.models import sha256


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


class RunRecord:
    def __init__(self, volume: Path, job: dict[str, Any], initialize: bool = True):
        date = datetime.fromtimestamp(job["created"], UTC).strftime("%Y-%m-%d")
        self.path = volume / "runs" / date / job["id"]
        if not initialize:
            if not self.path.is_dir():
                from h3ltx.errors import RuntimeFault

                raise RuntimeFault("run_pruned", "Run record is no longer present", 410)
            return
        self.path.mkdir(parents=True, exist_ok=True)
        for name in ("logs", "outputs"):
            (self.path / name).mkdir(exist_ok=True)
        if not (self.path / "request.json").exists():
            self.write("request", job["request"])
        if not (self.path / "resolved.json").exists():
            self.write("resolved", job["resolved"])
        (self.path / "logs/engine.log").touch(exist_ok=True)

    def write(self, name: str, data: Any) -> None:
        write_json(self.path / (name + ".json"), data)

    def artifacts(self) -> list[dict[str, Any]]:
        return [
            {
                "name": p.relative_to(self.path).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
            }
            for p in sorted(self.path.rglob("*"))
            if p.is_file()
            and not p.is_symlink()
            and p.resolve().is_relative_to(self.path.resolve())
            and not p.name.endswith(".partial")
        ]


def probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        timeout=60,
        check=True,
    )
    return dict(json.loads(result.stdout))


def output_metadata(path: Path) -> dict[str, Any]:
    info = probe(path)
    write_json(path.with_suffix(path.suffix + ".ffprobe.json"), info)
    duration = float(info.get("format", {}).get("duration", 0))
    thumbnail = path.parent / "thumbnail.jpg"
    contact = path.parent / "contact-sheet.jpg"
    for target, filters in (
        (thumbnail, "thumbnail,scale=640:-2"),
        (contact, f"fps=1/{max(duration / 9, 0.1)},scale=320:-2,tile=3x3"),
    ):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(path),
                "-vf",
                filters,
                "-frames:v",
                "1",
                str(target),
            ],
            capture_output=True,
            timeout=120,
            check=True,
        )
    video: dict[str, Any] = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "video"), {}
    )
    return {
        "width": video.get("width"),
        "height": video.get("height"),
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "duration_seconds": duration,
    }
