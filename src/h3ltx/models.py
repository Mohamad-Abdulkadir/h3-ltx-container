import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml
from filelock import FileLock
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field

from h3ltx.errors import RuntimeFault


class ModelFile(BaseModel):
    key: str
    group: str
    repo: str
    path: str
    destination: str
    size_bytes: int
    sha256: str | None
    revision: str
    license: str
    gated: bool

    def target(self, root: Path) -> Path:
        dest = (root / self.destination).resolve()
        if not dest.is_relative_to(root.resolve()):
            raise RuntimeFault("unsafe_path", "Model destination escapes model root")
        return dest


class Manifest(BaseModel):
    schema_version: int = 1
    default_groups: list[str] = Field(default_factory=list)
    files: list[ModelFile]

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        return cls.model_validate(yaml.safe_load(path.read_text()))

    def select(self, groups: list[str] | None = None) -> list[ModelFile]:
        groups = groups or self.default_groups
        unknown = set(groups) - {m.group for m in self.files}
        if unknown:
            raise RuntimeFault("unknown_group", "Unknown model groups", groups=sorted(unknown))
        return [m for m in self.files if m.group in groups]

    def by_name(self, filename: str) -> ModelFile:
        for item in self.files:
            if Path(item.destination).name == filename:
                return item
        raise RuntimeFault("model_unknown", f"Model absent from manifest: {filename}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def status(files: list[ModelFile], root: Path, deep: bool = False) -> list[dict[str, Any]]:
    result = []
    for model in files:
        path = model.target(root)
        state = "missing"
        if path.is_file():
            state = "ready" if path.stat().st_size == model.size_bytes else "size_mismatch"
            if deep and state == "ready":
                state = (
                    "ready" if model.sha256 and sha256(path) == model.sha256 else "hash_mismatch"
                )
        result.append({**model.model_dump(), "status": state})
    return result


def sync(files: list[ModelFile], root: Path, token: str | None, parallel: int = 2) -> None:
    root.mkdir(parents=True, exist_ok=True)
    # Serialize population per volume; individual HF downloads resume via Xet/Hub cache.
    with FileLock(str(root / ".sync.lock")):
        missing = [
            m
            for m, s in zip(files, status(files, root, deep=True), strict=True)
            if s["status"] != "ready"
        ]
        if any(m.gated for m in missing) and not token:
            raise RuntimeFault("hf_token_required", "HF_TOKEN required for selected gated weights")
        if any(m.sha256 is None for m in missing):
            raise RuntimeFault(
                "unverified_manifest", "Missing SHA256; refresh manifest before sync"
            )
        # cache + materialization can coexist; strict, conservative peak-disk estimate.
        required = sum(m.size_bytes for m in missing) + max(
            (m.size_bytes for m in missing), default=0
        )
        if shutil.disk_usage(root).free < required + 1024**3:
            raise RuntimeFault(
                "disk_space", "Insufficient free space for atomic downloads", required=required
            )
        cache = root / ".download-cache"

        def download(model: ModelFile) -> None:
            dest = model.target(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = Path(
                hf_hub_download(
                    repo_id=model.repo,
                    filename=model.path,
                    revision=model.revision,
                    token=token,
                    cache_dir=cache,
                )
            )
            if src.stat().st_size != model.size_bytes or sha256(src) != model.sha256:
                raise RuntimeFault("model_checksum", f"Integrity check failed: {model.key}")
            partial = dest.with_suffix(dest.suffix + ".partial")
            shutil.copyfile(src, partial)
            with partial.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(partial, dest)
            # Each verified source blob is expendable; do not duplicate hundreds of GB.
            src.resolve().unlink(missing_ok=True)

        with ThreadPoolExecutor(max_workers=parallel) as pool:
            list(pool.map(download, missing))
        lock = root / "models.lock.json"
        partial_lock = lock.with_suffix(".partial")
        partial_lock.write_text(json.dumps({"files": [m.model_dump() for m in files]}, indent=2))
        os.replace(partial_lock, lock)
        shutil.rmtree(cache, ignore_errors=True)
