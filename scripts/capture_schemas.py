"""Capture a running local ComfyUI server; derive lazy FaceRefine schemas from AST.

No third-party module or script is executed by the source-schema reader.
"""

import argparse
import ast
import gzip
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

CONSTANTS = {}


def literal(node):
    if isinstance(node, ast.Name) and node.id in CONSTANTS:
        return CONSTANTS[node.id]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_detector_list"
    ):
        return ["face_yolov8m.pt"]
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return [literal(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {literal(k): literal(v) for k, v in zip(node.keys, node.values, strict=True)}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return literal(node.left) + literal(node.right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -literal(node.operand)
    raise ValueError(ast.dump(node))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:18188")
    p.add_argument("--comfy", type=Path, default=Path("/tmp/src/comfyui"))
    p.add_argument("--face", type=Path, default=Path("/tmp/ctr/Carasibana_ComfyUI-H3-FaceRefine"))
    args = p.parse_args()
    raw = urllib.request.urlopen(args.url + "/object_info").read()
    info = json.loads(raw)
    native_count = len(info)
    derived = []
    tree = ast.parse((args.face / "nodes.py").read_text())
    for item in tree.body:
        if isinstance(item, ast.Assign):
            try:
                val = literal(item.value)
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        CONSTANTS[target.id] = val
            except ValueError:
                pass
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in (
            "H3FaceTrackCrop",
            "H3FaceStitch",
            "H3InjectVideoLatent",
            "H3PerFrameDenoise",
        ):
            continue
        schema = None
        outputs = None
        for member in cls.body:
            if isinstance(member, ast.FunctionDef) and member.name == "INPUT_TYPES":
                ret = next(n for n in ast.walk(member) if isinstance(n, ast.Return))
                schema = literal(ret.value)
            if isinstance(member, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "RETURN_TYPES" for t in member.targets
            ):
                outputs = literal(member.value)
        info[cls.name] = {"input": schema, "output": outputs, "output_node": False}
        derived.append(cls.name)
    root = Path("tests/fixtures")
    with gzip.GzipFile(filename=str(root / "object_info.json.gz"), mode="wb", mtime=0) as output:
        output.write(json.dumps(info, sort_keys=True).encode())

    def commit(path):
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()

    sidecar = dict(
        comfy_commit=commit(args.comfy),
        native_node_count=native_count,
        raw_snapshot_sha256=hashlib.sha256(raw).hexdigest(),
        command="main.py --cpu --listen 127.0.0.1 --port 18188 "
        "--disable-all-custom-nodes --disable-api-nodes --base-directory "
        "/tmp/h3ltx-comfy-state",
        platform="macOS arm64, CPython 3.12.15, CPU torch 2.15.0",
        source_derived={
            "repository": "https://github.com/Carasibana/ComfyUI-H3-FaceRefine",
            "commit": commit(args.face),
            "nodes": derived,
            "method": "AST-only literal INPUT_TYPES and RETURN_TYPES; detector list "
            "substituted with manifest filename. Optional inference dependencies "
            "not installed.",
        },
        model_combos="Empty model lists retained unchanged; validation overlays filenames "
        "from the pinned manifest. Upload combos use staged input basenames.",
    )
    (root / "object_info.provenance.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(native_count, "live native schemas;", len(derived), "source-derived schemas")


if __name__ == "__main__":
    main()
