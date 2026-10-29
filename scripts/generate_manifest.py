"""Refresh model pins from HF metadata, without downloading weights.

Run manually and review the diff; normal sync never refreshes revisions.
"""

import argparse
import json
import urllib.request
from pathlib import Path

import yaml

SOURCES = {
    "Bingsu/adetailer": "Apache-2.0",
    "Comfy-Org/MiniMax-H3": "minimax-h3-community",
    "lightx2v/Minimax-h3-Turbo": "minimax-h3-community",
    "LBH-123-AI/Minimax_h3_latent_Upscaler": "Apache-2.0",
    "Comfy-Org/SeedVR2": "Apache-2.0",
    "Lightricks/LTX-2.5": "ltx-2.x-community",
    **{
        f"Lightricks/LTX-2.5-22b-IC-LoRA-{s}": "ltx-2.x-community"
        for s in (
            "Pixel-Spatial-Upscaler",
            "Refine-Details",
            "SDR-To-HDR",
            "Deblur",
            "Decompression",
        )
    },
}


def metadata(repo, cache):
    path = cache / (repo.replace("/", "__") + ".json")
    if path.exists():
        return json.loads(path.read_text())
    meta = json.load(urllib.request.urlopen("https://huggingface.co/api/models/" + repo))
    url = f"https://huggingface.co/api/models/{repo}/tree/{meta['sha']}?recursive=true&limit=1000"
    tree = []
    while url:
        response = urllib.request.urlopen(url)
        tree.extend(json.load(response))
        link = response.headers.get("Link", "")
        url = link.split("<")[1].split(">")[0] if 'rel="next"' in link else None
    return {"meta": meta, "tree": tree}


def select(repo, path):
    name = Path(path).name
    if repo == "Bingsu/adetailer" and name == "face_yolov8m.pt":
        return "face-refine", "ultralytics/bbox/" + name
    if not name.endswith(".safetensors"):
        return None
    if repo == "Comfy-Org/MiniMax-H3":
        if name in (
            "minimax_h3_fl2va_pruned_bf16.safetensors",
            "minimax_h3_ref2va_pruned_bf16.safetensors",
            "qwen3vl_32b_minimax_h3_bf16.safetensors",
            "minimax_h3_video_vae_fp16.safetensors",
            "minimax_h3_audio_vae_fp32.safetensors",
        ) or path.startswith("loras/"):
            return "h3-core", path
        if name in ("minimax_h3_ref2va_bf16.safetensors", "minimax_h3_fl2va_bf16.safetensors"):
            return "h3-ab-full", path
        if "pruned_int8_convrot" in name:
            return "quantized-extra", path
    if (
        repo.startswith("lightx2v")
        and "comfyui_bf16" in name
        and (
            "fl2v_turbo_4step_v1.3_768p" in name
            or "ref2v_turbo_8step_v1.1_768p" in name
            or "fl2v_turbo_8step_v1.1_768p" in name
        )
    ):
        return "h3-turbo-extra", "loras/" + name
    if repo.startswith("LBH") and "3d_conv_v1_" in name:
        return "h3-upscale", "latent_upscale_models/" + name
    if repo == "Comfy-Org/SeedVR2" and (
        path
        in (
            "diffusion_models/seedvr2_7b_fp16.safetensors",
            "diffusion_models/seedvr2_7b_sharp_fp16.safetensors",
            "diffusion_models/seedvr2_3b_fp16.safetensors",
            "vae/seedvr2_ema_vae_fp16.safetensors",
        )
    ):
        return "seedvr2", path
    if repo == "Lightricks/LTX-2.5":
        if "comfy-int8-convrot" in name or "nvfp4" in name:
            return ("quantized-extra", path) if "transformer" in name else None
        if "bf16" in name and "video-vae-conv" not in name and "duration-head" not in name:
            return "ltx-core", path
    if "IC-LoRA" in repo:
        return "ltx-iclora", "loras/" + name
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("/nonexistent"))
    args = parser.parse_args()
    files = []
    for repo, licence in SOURCES.items():
        data = metadata(repo, args.cache)
        for item in data["tree"]:
            choice = select(repo, item["path"])
            if not choice:
                continue
            group, dest = choice
            files.append(
                dict(
                    key=Path(dest).stem,
                    group=group,
                    repo=repo,
                    path=item["path"],
                    destination=dest,
                    size_bytes=item["size"],
                    sha256=item.get("lfs", {}).get("oid"),
                    revision=data["meta"]["sha"],
                    license=licence,
                    gated=bool(data["meta"].get("gated")),
                )
            )
    document = dict(
        schema_version=1,
        default_groups=["h3-core", "h3-upscale", "seedvr2", "ltx-core", "ltx-iclora"],
        files=files,
    )
    output = yaml.safe_dump(document, sort_keys=False)
    for path in (Path("models/manifest.yaml"), Path("src/h3ltx/data/manifest.yaml")):
        path.write_text(output)
    total = sum(f["size_bytes"] for f in files if f["group"] in document["default_groups"])
    print(f"{len(files)} pinned files; default bytes: {total}")


if __name__ == "__main__":
    main()
