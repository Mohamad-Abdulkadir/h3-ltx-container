"""Strict API graph schema checks including flattened v3 dynamic inputs."""

from typing import Any

from h3ltx.errors import RuntimeFault
from h3ltx.models import Manifest

Graph = dict[str, dict[str, Any]]
Schema = dict[str, Any]
MODEL_FIELDS = {
    "unet_name": "diffusion_models",
    "clip_name": "text_encoders",
    "vae_name": "vae",
    "lora_name": "loras",
    "model_name": "latent_upscale_models",
}


def expanded(schema: Schema, values: dict[str, Any], prefix: str = "") -> tuple[Schema, set[str]]:
    fields: Schema = {}
    required: set[str] = set()
    for section in ("required", "optional"):
        for key, spec in schema.get(section, {}).items():
            name = prefix + key
            kind, opts = spec[0], spec[1] if len(spec) > 1 else {}
            if kind == "COMFY_AUTOGROW_V3":
                template = opts["template"]["input"]["required"]
                template_spec = next(iter(template.values()))
                members = {n: template_spec for n in values if n.startswith(name + ".")}
                count = len(members)
                if count < opts.get("min", 0) or count > opts.get("max", 100000):
                    raise ValueError(f"{name}: invalid autogrow count")
                for member in members:
                    suffix = member[len(name) + 1 :]
                    if "names" in opts and suffix not in opts["names"]:
                        raise ValueError(f"{member}: invalid autogrow name")
                    if "prefix" in opts and not suffix.startswith(opts["prefix"]):
                        raise ValueError(f"{member}: invalid autogrow prefix")
                fields.update(members)
                required.update(members)
                continue
            if section == "required":
                required.add(name)
            fields[name] = spec
            if kind == "COMFY_DYNAMICCOMBO_V3":
                selected = values.get(name)
                options = opts["options"]
                match = next((o for o in options if o["key"] == selected), None)
                if selected is not None and match is None:
                    raise ValueError(f"{name}: invalid DynamicCombo value {selected!r}")
                if match:
                    children, child_required = expanded(match["inputs"], values, name + ".")
                    fields.update(children)
                    required.update(child_required)
    return fields, required


def validate_graph(
    graph: Graph, info: Schema, manifest: Manifest, staged_files: set[str] | None = None
) -> None:
    staged = staged_files or set()
    for identifier, node in graph.items():
        cls = node["class_type"]
        if cls not in info:
            raise RuntimeFault("node_missing", f"Missing node class: {cls}")
        inputs = node["inputs"]
        try:
            fields, required = expanded(info[cls]["input"], inputs)
            if required - inputs.keys():
                raise ValueError(f"missing inputs: {sorted(required - inputs.keys())}")
            if inputs.keys() - fields.keys():
                raise ValueError(f"unknown inputs: {sorted(inputs.keys() - fields.keys())}")
            for key, value in inputs.items():
                spec = fields[key]
                kind, opts = spec[0], spec[1] if len(spec) > 1 else {}
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    origin, slot = value
                    if origin not in graph or not isinstance(slot, int) or slot < 0:
                        raise ValueError(f"{key}: invalid link")
                    outputs = info[graph[origin]["class_type"]]["output"]
                    if slot >= len(outputs):
                        raise ValueError(f"{key}: output slot {slot} does not exist")
                    actual = outputs[slot]
                    expected = "COMBO" if isinstance(kind, list) else kind
                    if expected != "*" and actual != "*" and actual not in expected.split(","):
                        raise ValueError(f"{key}: link type {actual} does not match {expected}")
                    continue
                if isinstance(value, (dict, list)):
                    raise ValueError(f"{key}: nested values are invalid; flatten dynamic inputs")
                if isinstance(kind, list) or kind == "COMBO":
                    legal = kind if isinstance(kind, list) else opts.get("options", [])
                    if key in MODEL_FIELDS:
                        category = MODEL_FIELDS[key]
                        legal = sorted(
                            set(legal)
                            | {
                                m.destination.removeprefix(category + "/")
                                for m in manifest.files
                                if m.destination.startswith(category + "/")
                            }
                        )
                    elif key in ("image", "file", "audio"):
                        legal = sorted(set(legal) | staged)
                    if value not in legal:
                        raise ValueError(f"{key}: illegal combo {value!r}")
                elif kind in (
                    "INT",
                    "FLOAT",
                    "BOOLEAN",
                    "STRING",
                    "FLOAT,INT,BOOLEAN",
                    "FLOAT,INT",
                ):
                    good = {
                        "INT": isinstance(value, int) and not isinstance(value, bool),
                        "FLOAT": isinstance(value, (float, int)) and not isinstance(value, bool),
                        "BOOLEAN": isinstance(value, bool),
                        "STRING": isinstance(value, str),
                        "FLOAT,INT,BOOLEAN": isinstance(value, (float, int, bool)),
                        "FLOAT,INT": isinstance(value, (float, int))
                        and not isinstance(value, bool),
                    }[kind]
                    if not good:
                        raise ValueError(f"{key}: expected {kind}")
                    if (
                        isinstance(value, (float, int))
                        and not isinstance(value, bool)
                        and (
                            value < opts.get("min", float("-inf"))
                            or value > opts.get("max", float("inf"))
                        )
                    ):
                        raise ValueError(f"{key}: outside schema bounds")
                elif kind != "COMFY_DYNAMICCOMBO_V3":
                    raise ValueError(f"{key}: {kind} must be a node link")
        except (ValueError, KeyError) as exc:
            raise RuntimeFault("graph_invalid", f"Node {identifier} ({cls}): {exc}") from exc
    visited: set[str] = set()
    active: set[str] = set()

    def visit(key: str) -> None:
        if key in active:
            raise RuntimeFault("graph_invalid", "Graph contains a cycle")
        if key in visited:
            return
        active.add(key)
        for value in graph[key]["inputs"].values():
            if isinstance(value, list):
                visit(value[0])
        active.remove(key)
        visited.add(key)

    for key in graph:
        visit(key)
