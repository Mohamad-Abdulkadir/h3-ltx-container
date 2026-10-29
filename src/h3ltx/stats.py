from collections import defaultdict
from statistics import median
from typing import Any

from h3ltx.errors import RuntimeFault


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * quantile
    lo = int(pos)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (pos - lo)


def summarize(
    rows: list[dict[str, Any]], group_by: str = "type,profile,resolution,steps"
) -> list[dict[str, Any]]:
    keys = group_by.split(",") if group_by else []
    if set(keys) - {"type", "profile", "resolution", "steps", "engine", "cached"}:
        raise RuntimeFault("stats_group", "Unsupported group_by field")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in keys)].append(row)
    output = []
    for group, members in groups.items():
        walls = [float(r["wall_seconds"]) for r in members if r.get("wall_seconds") is not None]
        stages: dict[str, list[float]] = defaultdict(list)
        for row in members:
            for stage, seconds in row.get("stages", {}).items():
                stages[stage].append(seconds)
        output.append(
            {
                **dict(zip(keys, group, strict=True)),
                "count": len(members),
                "success_rate": sum(r["state"] == "succeeded" for r in members) / len(members),
                "wall_seconds": {
                    "p50": percentile(walls, 0.5),
                    "p90": percentile(walls, 0.9),
                    "max": max(walls, default=None),
                },
                "stage_medians": {k: median(v) for k, v in stages.items()},
                "peak_vram_bytes": max(r.get("peak_vram_bytes", 0) for r in members),
                "peak_ram_bytes": max(r.get("peak_ram_bytes", 0) for r in members),
                "cost_estimate_usd": sum(r.get("cost_estimate_usd", 0) for r in members),
                "output_seconds_per_gpu_minute": sum(r.get("output_seconds", 0) for r in members)
                / (sum(walls) / 60)
                if sum(walls)
                else 0,
                "cached_runs": sum(bool(r.get("cached")) for r in members),
                "timings_incomplete_runs": sum(bool(r.get("timings_incomplete")) for r in members),
            }
        )
    return output
