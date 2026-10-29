"""Submit reproducible Cartesian experiments and print run/statistics comparisons."""

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table


def expand(document):
    for experiment in document["experiments"]:
        axes = experiment.get("vary", {})
        for values in itertools.product(*axes.values()):
            params = {**experiment.get("params", {}), **dict(zip(axes, values, strict=True))}
            yield {
                "type": experiment["type"],
                "params": params,
                "post": experiment.get("post", []),
                "tags": [document["name"], experiment["name"]],
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument(
        "--run-id", required=True, help="Stable operator-chosen ID; reuse to resume"
    )
    parser.add_argument("--timeout", type=float, default=86400)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    # Never override exported secrets or print their values.
    load_dotenv(Path.cwd() / ".env", override=False)
    import os

    document = yaml.safe_load(args.matrix.read_text())
    requests = list(expand(document))
    if args.dry_run:
        print(json.dumps(requests, indent=2))
        return
    token = os.getenv("API_TOKEN", os.getenv("H3LTX_TOKEN", ""))
    if not token:
        parser.error("API_TOKEN or H3LTX_TOKEN is required")
    started = time.time()
    with httpx.Client(
        base_url=os.getenv("H3LTX_URL", "http://127.0.0.1:8000"),
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,
    ) as client:
        ids = []
        for request in requests:
            fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[
                :20
            ]
            request["client_request_id"] = f"matrix:{args.run_id}:{fingerprint}"
            response = client.post("/v1/jobs", json=request)
            response.raise_for_status()
            ids.append(response.json()["id"])
        table = Table("Job", "Type", "Engine", "State", "Wall (s)", "Cached", "Peak VRAM (GiB)")
        deadline = time.monotonic() + args.timeout
        for identifier in ids:
            while True:
                response = client.get(f"/v1/jobs/{identifier}")
                response.raise_for_status()
                job = response.json()
                if job["state"] in ("succeeded", "failed", "cancelled"):
                    break
                if time.monotonic() > deadline:
                    parser.error("Matrix wait timed out; submitted jobs continue remotely")
                time.sleep(2)
            artifacts = {}
            for name in ("result", "metrics", "timeline"):
                response = client.get(f"/v1/jobs/{identifier}/artifacts/{name}.json")
                response.raise_for_status()
                artifacts[name] = response.json()
            table.add_row(
                identifier[:10],
                job["type"],
                job["resolved"]["params"]["engine"],
                job["state"],
                str(round(artifacts["result"].get("wall_seconds", 0), 2)),
                str(bool(artifacts["timeline"].get("cached_nodes"))),
                str(round(artifacts["metrics"].get("vram_bytes_peak", 0) / 1024**3, 2)),
            )
        Console().print(table)
        response = client.get(
            "/v1/stats",
            params={"group_by": "type,profile,resolution,steps,engine,cached", "since": started},
        )
        response.raise_for_status()
        print(json.dumps(response.json(), indent=2))
        print(
            "Statistics include all runs since matrix start; use the listed run "
            "IDs for exact comparisons."
        )


if __name__ == "__main__":
    main()
