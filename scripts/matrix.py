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
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Unload, load and wait before each sequential measured job",
    )
    parser.add_argument(
        "--pull", type=Path, help="Archive every finished run bundle after the matrix"
    )
    parser.add_argument("--stats-output", type=Path, help="Write the complete stats export as JSON")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Retain failures and finish remaining cases",
    )
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
        from h3ltx.models import sha256
        from h3ltx.records import write_json

        state_root = Path(".matrix-state")
        state_root.mkdir(exist_ok=True)
        state_key = hashlib.sha256(
            (str(client.base_url) + args.run_id + str(args.matrix.resolve())).encode()
        ).hexdigest()
        state_path = state_root / (state_key + ".json")
        saved = json.loads(state_path.read_text()) if state_path.exists() else {"assets": {}}
        assets = {}
        for name, path in document.get("assets", {}).items():
            source = (args.matrix.parent / Path(path)).resolve()
            digest = sha256(source)
            if saved["assets"].get(name, {}).get("sha256") == digest:
                assets[name] = saved["assets"][name]["upload_id"]
                continue
            with source.open("rb") as stream:
                response = client.post("/v1/uploads", files={"file": (source.name, stream)})
                response.raise_for_status()
                assets[name] = response.json()["upload_id"]
                saved["assets"][name] = {"sha256": digest, "upload_id": assets[name]}
                write_json(state_path, saved)

        def resolve(value):
            if isinstance(value, str) and value.startswith("@asset:"):
                return assets[value[7:]]
            if isinstance(value, list):
                return [resolve(v) for v in value]
            if isinstance(value, dict):
                return {k: resolve(v) for k, v in value.items()}
            return value

        requests = [resolve(request) for request in requests]

        def wait_for(path):
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                response = client.get(path)
                response.raise_for_status()
                result = response.json()
                if result["state"] in ("succeeded", "failed", "cancelled"):
                    if result["state"] != "succeeded":
                        raise RuntimeError(json.dumps(result.get("error")))
                    return result
                time.sleep(1)
            raise TimeoutError(path)

        for model in document.get("load_only", []):
            response = client.post("/v1/load", json={"model": model, "gpus": [0]})
            response.raise_for_status()
            wait_for("/v1/load/" + response.json()["load_id"])
        ids = []
        failures = []
        for request in requests:
            try:
                if args.preload:
                    from h3ltx.loading import pipeline_for
                    from h3ltx.params import JobRequest

                    params = JobRequest.model_validate(request).resolve()
                    key = pipeline_for(request["type"], params)
                    for action, model in (("unload", "all"), ("load", key)):
                        payload = {"model": model, "gpus": [0], "engine": params.engine}
                        if action == "load":
                            payload["params"] = params.model_dump()
                        response = client.post("/v1/" + action, json=payload)
                        response.raise_for_status()
                        wait_for("/v1/load/" + response.json()["load_id"])
                fingerprint = hashlib.sha256(
                    json.dumps(request, sort_keys=True).encode()
                ).hexdigest()[:20]
                request["client_request_id"] = f"matrix:{args.run_id}:{fingerprint}"
                response = client.post("/v1/jobs", json=request)
                response.raise_for_status()
                ids.append(response.json()["id"])
                if args.preload:
                    completed = wait_for("/v1/jobs/" + ids[-1])
                    if not completed["result"].get("warm"):
                        raise RuntimeError(
                            "Preload invariant failed: job was cold; inspect its run record"
                        )
            except (httpx.HTTPError, RuntimeError, TimeoutError) as exc:
                failures.append({"type": request["type"], "message": str(exc)})
                if not args.continue_on_error:
                    raise
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
            params={
                "group_by": "model,profile,resolution,duration,steps,engine,cached",
                "since": started,
            },
        )
        response.raise_for_status()
        if args.stats_output:
            args.stats_output.write_text(json.dumps(response.json(), indent=2) + "\n")
        if args.pull:
            from h3ltx.client import pull_runs

            pull_runs(client, args.pull)
        print(json.dumps(response.json(), indent=2))
        if failures:
            print(json.dumps({"failed_cases": failures}, indent=2))
            raise SystemExit(1)
        print(
            "Statistics include all runs since matrix start; use the listed run "
            "IDs for exact comparisons."
        )


if __name__ == "__main__":
    main()
