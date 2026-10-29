import asyncio
import hmac
import json
import shutil
import tempfile
import zipfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import httpx
from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from h3ltx.config import Settings
from h3ltx.errors import RuntimeFault
from h3ltx.loading import LoadRequest, UnloadRequest
from h3ltx.models import status
from h3ltx.params import JobRequest
from h3ltx.records import RunRecord
from h3ltx.runtime import Runtime
from h3ltx.stats import summarize
from h3ltx.store import TERMINAL


class APIFileResponse(FileResponse):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        replaced = False

        async def stable_send(message: Message) -> None:
            nonlocal replaced
            if message["type"] == "http.response.start" and message["status"] >= 400:
                replaced = True
                response = JSONResponse(
                    {
                        "error": {
                            "code": "range_invalid",
                            "message": "Requested byte range is not satisfiable",
                            "details": {},
                        }
                    },
                    status_code=message["status"],
                    headers={"Content-Range": f"bytes */{Path(self.path).stat().st_size}"},
                )
                await response(scope, receive, send)
            elif not replaced:
                await send(message)

        await super().__call__(scope, receive, stable_send)


class BodyLimit:
    def __init__(self, app: ASGIApp, maximum: int):
        self.app, self.maximum = app, maximum

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] != "/v1/uploads":
            await self.app(scope, receive, send)
            return
        count = 0

        async def bounded() -> Message:
            nonlocal count
            message = await receive()
            count += len(message.get("body", b""))
            if count > self.maximum + 1024**2:
                raise RuntimeFault("upload_too_large", "Upload exceeds size limit", 413)
            return message

        await self.app(scope, bounded, send)


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    config = settings or Settings()
    service: Runtime | None = runtime

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal service
        service = service or Runtime(config)
        await service.start()
        try:
            yield
        finally:
            await service.close()

    app = FastAPI(
        title="h3ltx",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(BodyLimit, maximum=config.upload_max_bytes)

    def current() -> Runtime:
        assert service is not None
        return service

    def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
        expected = config.api_token.get_secret_value()
        supplied = (
            authorization.removeprefix("Bearer ")
            if authorization and authorization.startswith("Bearer ")
            else ""
        )
        if not expected or not hmac.compare_digest(supplied.encode(), expected.encode()):
            raise RuntimeFault("unauthorized", "Valid Bearer token required", 401)

    auth = [Depends(authorize)]

    @app.exception_handler(RuntimeFault)
    async def fault_handler(request: Request, exc: RuntimeFault) -> JSONResponse:
        return JSONResponse(exc.payload(), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            {
                "error": {
                    "code": "request_invalid",
                    "message": "Request validation failed",
                    "details": json.loads(exc.json())
                    if hasattr(exc, "json")
                    else json.loads(json.dumps(exc.errors(), default=str)),
                }
            },
            status_code=422,
        )

    @app.exception_handler(HTTPException)
    async def http_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            {
                "error": {
                    "code": f"http_{exc.status_code}",
                    "message": str(exc.detail),
                    "details": {},
                }
            },
            status_code=exc.status_code,
        )

    @app.exception_handler(Exception)
    async def internal_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            {
                "error": {
                    "code": "internal_error",
                    "message": "Internal server error",
                    "details": {},
                }
            },
            status_code=500,
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/system", dependencies=auth)
    async def system() -> dict[str, Any]:
        return await current().system()

    @app.get("/v1/capabilities", dependencies=auth)
    async def capabilities() -> dict[str, Any]:
        return current().capabilities()

    @app.get("/v1/capacity", dependencies=auth)
    async def capacity() -> dict[str, Any]:
        for index in current().profiles:
            await current().loading.refresh(index)
        return current().capacity()

    @app.get("/v1/models", dependencies=auth)
    async def models() -> dict[str, Any]:
        s = current()
        return {"files": status(s.manifest.files, s.settings.volume_root / "models")}

    @app.post("/v1/uploads", dependencies=auth, status_code=201)
    async def upload(request: Request) -> dict[str, Any]:
        s = current()
        if request.headers.get("content-type", "").startswith("application/json"):
            body = await request.json()
            if (
                not isinstance(body, dict)
                or set(body) != {"url"}
                or not isinstance(body["url"], str)
            ):
                raise RuntimeFault("upload_invalid", "Expected a JSON object containing only url")
            try:
                return await s.uploads.from_url(body["url"])
            except httpx.HTTPError as exc:
                raise RuntimeFault("url_fetch_failed", "Remote media could not be fetched") from exc
        async with request.form(max_files=1, max_fields=0) as form:
            source = form.get("file")
            if not isinstance(source, UploadFile):
                raise RuntimeFault("upload_invalid", "Multipart field 'file' required")
            identifier, path = s.uploads.begin()
            try:
                total = 0
                with path.open("wb") as output:
                    while block := await source.read(1024**2):
                        total += len(block)
                        if total > config.upload_max_bytes:
                            raise RuntimeFault("upload_too_large", "Upload exceeds size limit", 413)
                        output.write(block)
                return await s.uploads.finish(identifier, path)
            finally:
                path.unlink(missing_ok=True)

    @app.post("/v1/load", dependencies=auth, status_code=202)
    async def load_models(request: LoadRequest) -> dict[str, Any]:
        return current().loading.submit(request, "load")

    @app.post("/v1/unload", dependencies=auth, status_code=202)
    async def unload_models(request: UnloadRequest | None = None) -> dict[str, Any]:
        return current().loading.submit(request or UnloadRequest(), "unload")

    @app.get("/v1/load/{identifier}", dependencies=auth)
    async def load_status(identifier: str) -> dict[str, Any]:
        result = current().job(identifier)
        if result["type"] not in ("system.load", "system.unload"):
            raise RuntimeFault("load_not_found", "Load operation not found", 404)
        result["load_id"] = identifier
        return result

    @app.post("/v1/jobs", dependencies=auth, status_code=202)
    async def submit(request: JobRequest) -> dict[str, Any]:
        return current().submit(request)

    @app.get("/v1/jobs", dependencies=auth)
    async def jobs(
        state: str | None = None,
        type: str | None = None,
        tag: str | None = None,
        since: float | None = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    ) -> dict[str, Any]:
        return {
            "jobs": current().store.list(
                states=state.split(",") if state else None,
                kind=type,
                tag=tag,
                since=since,
                limit=limit,
            )
        }

    @app.get("/v1/jobs/{identifier}", dependencies=auth)
    async def job(identifier: str) -> dict[str, Any]:
        return current().job(identifier)

    @app.get("/v1/load/{identifier}/events", dependencies=auth)
    @app.get("/v1/jobs/{identifier}/events", dependencies=auth)
    async def events(
        identifier: str, request: Request, last_event_id: Annotated[str | None, Header()] = None
    ) -> StreamingResponse:
        s = current()
        s.store.get(identifier)
        try:
            after = int(last_event_id or 0)
        except ValueError as exc:
            raise RuntimeFault("cursor_invalid", "Last-Event-ID must be an integer") from exc

        async def generate() -> AsyncIterator[str]:
            nonlocal after
            while not await request.is_disconnected():
                for event in s.store.events(identifier, after):
                    after = event["sequence"]
                    yield f"id: {after}\nevent: progress\ndata: {json.dumps(event['data'])}\n\n"
                if s.store.get(identifier)["state"] in TERMINAL:
                    return
                yield ": keepalive\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/jobs/{identifier}/cancel", dependencies=auth)
    async def cancel(identifier: str) -> dict[str, Any]:
        return await current().cancel(identifier)

    def record_for(identifier: str, finished: bool = False) -> RunRecord:
        s = current()
        job = s.store.get(identifier)
        if finished and job["state"] not in TERMINAL:
            raise RuntimeFault("run_not_finished", "Run bundle available after completion", 409)
        return RunRecord(s.settings.volume_root, job, initialize=False)

    @app.get("/v1/jobs/{identifier}/artifacts", dependencies=auth)
    async def artifacts(identifier: str) -> dict[str, Any]:
        record = record_for(identifier)
        return {"artifacts": await asyncio.to_thread(record.artifacts)}

    @app.get("/v1/jobs/{identifier}/artifacts/{name:path}", dependencies=auth)
    async def artifact(identifier: str, name: str) -> FileResponse:
        record = record_for(identifier)
        path = (record.path / name).resolve()
        if (
            not path.is_relative_to(record.path.resolve())
            or not path.is_file()
            or path.name.endswith(".partial")
        ):
            raise RuntimeFault("artifact_not_found", "Artifact not found", 404)
        # Starlette FileResponse implements RFC byte ranges, 206 and 416.
        return APIFileResponse(path, filename=path.name)

    @app.get("/v1/jobs/{identifier}/bundle", dependencies=auth)
    async def bundle(identifier: str) -> FileResponse:
        record = record_for(identifier, finished=True)
        scratch = Path(
            tempfile.mkdtemp(prefix="bundle-", dir=current().settings.volume_root / "state")
        )
        target = scratch / (identifier + ".zip")

        def archive() -> None:
            with zipfile.ZipFile(
                target, "w", compression=zipfile.ZIP_STORED, allowZip64=True
            ) as zip_file:
                for path in sorted(record.path.rglob("*")):
                    if (
                        path.is_file()
                        and not path.is_symlink()
                        and not path.name.endswith(".partial")
                    ):
                        zip_file.write(path, path.relative_to(record.path).as_posix())

        try:
            await asyncio.to_thread(archive)
        except BaseException:
            shutil.rmtree(scratch)
            raise
        return APIFileResponse(
            target,
            filename=target.name,
            media_type="application/zip",
            background=BackgroundTask(shutil.rmtree, scratch),
        )

    @app.get("/v1/runs/export", dependencies=auth)
    async def export(since_cursor: Annotated[int, Query(ge=0)] = 0) -> dict[str, Any]:
        return current().store.export(since_cursor)

    @app.get("/v1/stats", dependencies=auth)
    async def stats(
        group_by: str = "type,profile,resolution,steps",
        since: float | None = None,
        include_cold: bool = False,
    ) -> dict[str, Any]:
        return {
            "groups": summarize(current().store.measurements(since), group_by, include_cold),
            "cost_rate_usd_per_gpu_hour": config.gpu_hourly_usd,
            "hardware": [g.as_dict() for g in current().gpus],
            "software": current().versions,
        }

    @app.post("/v1/admin/doctor", dependencies=auth)
    async def doctor() -> dict[str, Any]:
        return await current().system()

    return app
