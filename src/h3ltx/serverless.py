import asyncio
from typing import Any

from h3ltx.config import Settings
from h3ltx.params import JobRequest
from h3ltx.runtime import Runtime
from h3ltx.store import TERMINAL


async def serve() -> None:
    from importlib import import_module

    serverless = import_module("runpod.serverless")

    runtime = Runtime(Settings())
    await runtime.start()

    async def handler(event: dict[str, Any]) -> dict[str, Any]:
        request = JobRequest.model_validate(event["input"])
        request.client_request_id = request.client_request_id or "runpod:" + event["id"]
        job = runtime.submit(request)
        while True:
            job = runtime.job(job["id"])
            if job["state"] in TERMINAL:
                return {
                    "id": job["id"],
                    "state": job["state"],
                    "run_dir": job["run_dir"],
                    "result": job["result"],
                    "error": job["error"],
                }
            await asyncio.sleep(0.5)

    try:
        # SDK async worker owns its event loop; run in a thread with a bridge to this loop.
        loop = asyncio.get_running_loop()

        def sync_handler(event: dict[str, Any]) -> dict[str, Any]:
            return asyncio.run_coroutine_threadsafe(handler(event), loop).result()

        await asyncio.to_thread(serverless.start, {"handler": sync_handler})
    finally:
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(serve())
