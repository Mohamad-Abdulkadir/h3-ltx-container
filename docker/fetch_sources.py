"""Fetch exact upstream trees; never execute repository install scripts."""

import json
import subprocess
from pathlib import Path

pins = json.loads(Path("/opt/source-pins.json").read_text())
for name, pin in pins.items():
    if pin.get("reference_only"):
        continue
    if name == "comfyui":
        destination = Path("/opt/ComfyUI")
    elif name == "ltx":
        destination = Path("/opt/LTX-2")
    else:
        destination = Path("/opt/ComfyUI/custom_nodes") / name
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(destination)], check=True)
    subprocess.run(
        ["git", "-C", str(destination), "fetch", "--depth=1", pin["url"], pin["commit"]], check=True
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"], check=True
    )
    actual = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != pin["commit"]:
        raise RuntimeError("Upstream source revision mismatch")
# Keep exact source and licence files in the image for copyleft compliance.
versions = {
    "comfy_commit": pins["comfyui"]["commit"],
    "ltx_commit": pins["ltx"]["commit"],
    "custom_nodes": {
        k: v["commit"] for k, v in pins.items() if k not in ("comfyui", "ltx", "workflow_templates")
    },
}
Path("/opt/h3ltx-versions.json").write_text(json.dumps(versions, indent=2))
