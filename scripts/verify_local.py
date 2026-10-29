"""Verify shipped examples, real-schema fixture, model pins and package contents without a GPU."""

import gzip
import json
import sys
import tarfile
import zipfile
from pathlib import Path

# Also works from a source checkout whose editable .pth files are hidden by macOS.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from h3ltx.models import Manifest  # noqa: E402
from h3ltx.params import JobRequest  # noqa: E402
from h3ltx.workflows import build  # noqa: E402
from h3ltx.workflows.validate import validate_graph  # noqa: E402


def main() -> None:
    manifest = Manifest.load(ROOT / "models/manifest.yaml")
    assert (ROOT / "models/manifest.yaml").read_bytes() == (
        ROOT / "src/h3ltx/data/manifest.yaml"
    ).read_bytes()
    with gzip.open(ROOT / "tests/fixtures/object_info.json.gz") as stream:
        info = json.load(stream)
    examples = sorted((ROOT / "examples").glob("*.json"))
    for path in examples:
        request = JobRequest.model_validate(json.loads(path.read_text()))
        graph = build(request.type, request.resolve())
        validate_graph(graph, info, manifest, {"0" * 32 + ".png", "1" * 32 + ".png"})
    for model in manifest.files:
        assert model.sha256 and len(model.sha256) == 64
        assert len(model.revision) == 40 and model.size_bytes > 0
    packages = []
    for path in (ROOT / "dist").glob("*.whl"):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            assert "h3ltx/data/manifest.yaml" in names
            assert "h3ltx/data/profiles.yaml" in names
            assert not any(Path(n).name == ".env" for n in names)
            packages.append(path.name)
    for path in (ROOT / "dist").glob("*.tar.gz"):
        with tarfile.open(path) as archive:
            assert not any(Path(n).name == ".env" for n in archive.getnames())
            packages.append(path.name)
    print(
        json.dumps(
            {
                "schemas": len(info),
                "examples_valid": len(examples),
                "pinned_models": len(manifest.files),
                "default_model_bytes": sum(m.size_bytes for m in manifest.select()),
                "packages_checked": packages,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
