"""Generate an honest, reviewable benchmark document from stats JSON and complete run bundles."""

import argparse
import json
import os
import statistics
import zipfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    low = int(position)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (position - low)


def load_records(root):
    seen = set()
    records = []
    if not root.exists():
        return records
    for path in sorted(root.rglob("result.json")):
        if path.parent.name in seen:
            continue

        def read(name, parent=path.parent):
            source = parent / (name + ".json")
            return json.loads(source.read_text()) if source.exists() else {}

        records.append(
            {
                "root": path.parent,
                **{
                    key: read(key)
                    for key in ("request", "resolved", "result", "timeline", "metrics")
                },
            }
        )
        seen.add(path.parent.name)
    for path in sorted(root.rglob("*.zip")):
        if path.stem in seen:
            continue
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            records.append(
                {
                    "root": path,
                    **{
                        key: json.loads(archive.read(key + ".json"))
                        if key + ".json" in names
                        else {}
                        for key in ("request", "resolved", "result", "timeline", "metrics")
                    },
                }
            )
        seen.add(path.stem)
    return records


def number(value):
    return "unmeasured" if value is None else f"{value:.3f}"


def render(records, stats, include_cold=False, output=Path("docs/BENCHMARKS.md")):
    generation = defaultdict(list)
    loads = defaultdict(list)
    versions = set()
    samples = []
    for record in records:
        request, resolved, result = record["request"], record["resolved"], record["result"]
        kind = request.get("type", "unknown")
        params = resolved.get("params", request.get("params", {}))
        software = resolved.get("software", {})
        gpu = resolved.get("gpu", {})
        versions.add(
            json.dumps(
                {"gpu": gpu.get("name"), "driver": gpu.get("driver"), "software": software},
                sort_keys=True,
            )
        )
        if kind in ("system.load", "system.unload"):
            for item in result.get("gpus", []):
                loads[
                    (result.get("model"), item.get("profile"), result.get("operation", "load"))
                ].append(item)
            continue
        if not include_cold and not result.get("warm", resolved.get("warm", False)):
            continue
        duration = params.get("duration_seconds", params.get("frames", 0) / params.get("fps", 24))
        key = (
            resolved.get("pipeline", kind),
            resolved.get("profile"),
            f"{params.get('width')}x{params.get('height')}",
            duration,
            params.get("steps"),
            params.get("engine", "comfy"),
        )
        generation[key].append(record)
        for artifact in result.get("outputs", []):
            path = record["root"] / "outputs" / artifact["name"]
            relative = os.path.relpath(path, output.parent)
            samples.append(
                f"- `{record['root'].name}`: [{artifact['name']}]({relative.replace(' ', '%20')})"
            )
    lines = [
        "# Benchmarks",
        "",
        f"Generated: {datetime.now(UTC).date().isoformat()}",
        "",
        "These figures come from the supplied run records. No GPU measurements are invented.",
        "Cold generations are excluded by default. Model loading is reported separately.",
        "",
        "## Generation",
        "",
        (
            "| Model | Profile | Resolution | Duration (s) | Steps | Engine | Runs | "
            "Wall p50/p90 (s) | Peak VRAM/RAM (GiB) | Output s/GPU-min | Estimated cost (USD) |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, group in sorted(generation.items(), key=lambda item: str(item[0])):
        walls = [
            r["result"]["wall_seconds"]
            for r in group
            if r["result"].get("wall_seconds") is not None
        ]
        vram = max((r["metrics"].get("vram_bytes_peak", 0) for r in group), default=0) / 1024**3
        ram = max((r["metrics"].get("ram_bytes_peak", 0) for r in group), default=0) / 1024**3
        seconds = sum(
            sum(o.get("duration_seconds", 0) for o in r["result"].get("outputs", [])) for r in group
        )
        cost = sum(r["result"].get("cost_estimate_usd", 0) for r in group)
        throughput = seconds / (sum(walls) / 60) if sum(walls) else None
        lines.append(
            "| "
            + " | ".join(map(str, key))
            + f" | {len(group)} | {number(percentile(walls, 0.5))} / "
            f"{number(percentile(walls, 0.9))} | {vram:.2f} / {ram:.2f} | "
            f"{number(throughput)} | {cost:.5f} |"
        )
    export_generations = [
        g
        for g in stats.get("groups", [])
        if g.get("operation", "generation") == "generation"
        and g.get("wall_seconds", {}).get("p50") is not None
    ]
    if not generation:
        for group in export_generations:
            wall = group.get("wall_seconds", {})
            fields = [
                group.get(k, "unspecified")
                for k in ("model", "profile", "resolution", "duration", "steps", "engine")
            ]
            peaks = (
                f"{group.get('peak_vram_bytes', 0) / 1024**3:.2f} / "
                f"{group.get('peak_ram_bytes', 0) / 1024**3:.2f}"
            )
            lines.append(
                "| " + " | ".join(map(str, fields)) + f" | {group.get('count', 0)} | "
                f"{number(wall.get('p50'))} / {number(wall.get('p90'))} | {peaks} | "
                f"{number(group.get('output_seconds_per_gpu_minute'))} | "
                f"{group.get('cost_estimate_usd', 0):.5f} |"
            )
        if not export_generations:
            lines += ["", "No measured warm generation runs have been supplied."]
    lines += [
        "",
        "### Per-stage medians",
        "",
        "| Model/profile/resolution | Stage | Median (s) |",
        "| --- | --- | --- |",
    ]
    for key, group in sorted(generation.items(), key=lambda item: str(item[0])):
        stages = defaultdict(list)
        for record in group:
            for stage, seconds in record["timeline"].get("stages", {}).items():
                stages[stage].append(seconds)
        for stage, seconds in sorted(stages.items()):
            lines.append(
                f"| {' / '.join(map(str, key[:3]))} | {stage} | {statistics.median(seconds):.3f} |"
            )
    lines += [
        "",
        "## Load and unload operations",
        "",
        "| Model | Profile | Operation | Runs | Wall p50/p90 (s) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, group in sorted(loads.items(), key=lambda item: str(item[0])):
        walls = [r["wall_seconds"] for r in group]
        lines.append(
            "| "
            + " | ".join(map(str, key))
            + f" | {len(group)} | {number(percentile(walls, 0.5))} / "
            f"{number(percentile(walls, 0.9))} |"
        )
    lines += [
        "",
        "### Component loading",
        "",
        "| Model | Component | Disk read (s) | GPU transfer (s) | Compile (s) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, group in sorted(loads.items(), key=lambda item: str(item[0])):
        for run in group:
            for component in run.get("components", []):
                lines.append(
                    f"| {key[0]} | {component.get('component')} | "
                    f"{number(component.get('read_from_disk_s'))} | "
                    f"{number(component.get('transfer_to_gpu_s'))} | "
                    f"{number(component.get('compile_s'))} |"
                )
    for group in stats.get("groups", []):
        if group.get("operation") in ("load", "unload") and not loads:
            wall = group.get("wall_seconds", {})
            lines.append(
                f"- Exported {group.get('operation')} `{group.get('model')}` / "
                f"`{group.get('profile')}`: p50 {number(wall.get('p50'))} s, "
                f"p90 {number(wall.get('p90'))} s."
            )
    if not versions and (stats.get("hardware") or stats.get("software")):
        versions.add(
            json.dumps(
                {"hardware": stats.get("hardware"), "software": stats.get("software")},
                sort_keys=True,
            )
        )
    lines += ["", "## Hardware and software", ""]
    if versions:
        lines += ["```json", json.dumps([json.loads(v) for v in sorted(versions)], indent=2), "```"]
    else:
        lines.append("No target GPU/driver/software records have been supplied.")
    lines += ["", "## Statistics export", ""]
    if stats:
        lines += ["```json", json.dumps(stats, indent=2), "```"]
    else:
        lines.append("No `/v1/stats` export supplied.")
    lines += [
        "",
        "## Sample outputs",
        "",
        *(samples or ["No sample output artifacts supplied."]),
        "",
        "A zero cost estimate means no hourly rate was configured unless the run says otherwise.",
        "Null timing fields are unmeasured, never assumed to be zero. Disk-cache warmth and",
        "compiled-kernel warmth are separate from model residency. "
        "Quality needs visual/audio review.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/BENCHMARKS.md"))
    parser.add_argument("--include-cold", action="store_true")
    args = parser.parse_args()
    stats = json.loads(args.stats.read_text()) if args.stats else {}
    records = load_records(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(records, stats, args.include_cold, args.output))
    print(f"Wrote {args.output} from {len(records)} run records")


if __name__ == "__main__":
    main()
