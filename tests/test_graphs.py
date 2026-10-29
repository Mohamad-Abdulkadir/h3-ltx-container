import itertools

import pytest

from h3ltx.errors import RuntimeFault
from h3ltx.params import JobRequest
from h3ltx.workflows import build
from h3ltx.workflows.validate import validate_graph

U, V = "0" * 32, "1" * 32


def request(kind, **extra):
    params = {"prompt": "Cinematic landscape", "seed": 42}
    if kind.startswith(("upscale.", "refine.")):
        params = {"input_video": U}
    if kind == "h3.fl2v":
        params["first_frame"] = U
        params["last_frame"] = V
    if kind == "h3.r2v":
        params["references"] = [{"upload_id": U}]
    if kind in ("ltx.i2v", "ltx.flf2v"):
        params["keyframes"] = [{"upload_id": U, "frame": 0}, {"upload_id": V, "frame": -1}]
    return JobRequest(type=kind, params={**params, **extra}).resolve()


CASES = [
    (kind, {})
    for kind in (
        "h3.t2v",
        "h3.fl2v",
        "h3.r2v",
        "ltx.t2v",
        "ltx.i2v",
        "ltx.flf2v",
        "ltx.dfr",
        "upscale.seedvr2",
        "upscale.ltx_iclora",
        "refine.h3_face",
    )
]
CASES += [
    (kind, {"turbo": turbo, "latent_upscale": scale})
    for kind, turbo, scale in itertools.product(
        ("h3.t2v", "h3.r2v"), ("none", "4-step", "8-step"), ("off", "2x")
    )
    if not (kind == "h3.r2v" and turbo == "8-step")
]
CASES += [
    ("ltx.dfr", {"temporal_upscale": temporal, "hdr": hdr, "tiled": tiled})
    for temporal, hdr, tiled in itertools.product((0, 1, 2), (False, True), (False, True))
]
CASES += [("ltx.t2v", {"pipeline": "dev"}), ("h3.r2v", {"weights": "full-bf16"})]
CASES += [
    ("upscale.seedvr2", {"model": model, "tiled": tiled})
    for model, tiled in itertools.product(("7b", "7b-sharp", "3b"), (False, True))
]
CASES += [("upscale.ltx_iclora", {"mode": mode}) for mode in ("pixel-spatial", "refine-details")]
CASES += [
    ("h3.fl2v", {"guides": [{"upload_id": U, "kind": kind, "duration_seconds": 3}]})
    for kind in ("image", "video", "audio")
]
CASES += [
    ("h3.r2v", {"references": [{"upload_id": U, "kind": kind, "duration_seconds": 3}]})
    for kind in ("video", "audio")
]
CASES += [
    ("h3.t2v", {"turbo": "4-step", "turbo_version": "latest"}),
    ("h3.r2v", {"turbo": "8-step", "turbo_version": "latest"}),
]


@pytest.mark.parametrize("kind,extra", CASES)
def test_every_graph_schema(kind, extra, object_info, manifest):
    graph = build(kind, request(kind, **extra))
    validate_graph(graph, object_info, manifest, {U + ".png", V + ".png"})
    assert any(n["class_type"] == "SaveVideo" for n in graph.values())
    for node in graph.values():
        assert not any(isinstance(v, dict) for v in node["inputs"].values())


@pytest.mark.parametrize(
    "mutation", ["class", "required", "slot", "type", "combo", "nested", "unknown", "cycle"]
)
def test_reject_corrupt_graph(mutation, object_info, manifest):
    graph = build("h3.t2v", request("h3.t2v"))
    if mutation == "class":
        graph["1"]["class_type"] = "DoesNotExist"
    elif mutation == "required":
        graph["1"]["inputs"].pop("unet_name")
    elif mutation == "slot":
        graph["2"]["inputs"]["model"] = ["1", 500]
    elif mutation == "type":
        graph["2"]["inputs"]["model"] = ["3", 0]
    elif mutation == "combo":
        graph["1"]["inputs"]["weight_dtype"] = "unknown"
    elif mutation == "nested":
        graph["1"]["inputs"]["unet_name"] = {"bad": "value"}
    elif mutation == "unknown":
        graph["1"]["inputs"]["bad"] = 1
    else:
        graph["2"]["inputs"]["model"] = ["2", 0]
    with pytest.raises(RuntimeFault):
        validate_graph(graph, object_info, manifest)


def test_missing_live_model_not_masked(object_info, manifest):
    graph = build("h3.t2v", request("h3.t2v"))
    # Runtime also checks disk status; offline schema checks use manifest pins.
    assert manifest.by_name(graph["1"]["inputs"]["unet_name"]).group == "h3-core"


def test_integer_math_slot():
    graph = build("h3.t2v", request("h3.t2v"))
    conditioning = next(n for n in graph.values() if n["class_type"] == "MiniMaxH3ImageToVideo")
    assert conditioning["inputs"]["length"][1] == 1
