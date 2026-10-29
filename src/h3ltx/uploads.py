import asyncio
import hashlib
import ipaddress
import json
import socket
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from h3ltx.errors import RuntimeFault
from h3ltx.records import probe, write_json


class Uploads:
    def __init__(self, root: Path, maximum: int):
        self.root, self.maximum = root, maximum
        root.mkdir(parents=True, exist_ok=True)

    def get(self, identifier: str) -> dict[str, Any]:
        if len(identifier) != 32 or any(c not in "0123456789abcdef" for c in identifier):
            raise RuntimeFault("upload_not_found", "Upload not found", 404)
        path = self.root / identifier / "metadata.json"
        if not path.is_file():
            raise RuntimeFault("upload_not_found", "Upload not found", 404)
        return dict(json.loads(path.read_text()))

    def begin(self) -> tuple[str, Path]:
        identifier = uuid.uuid4().hex
        directory = self.root / identifier
        directory.mkdir()
        return identifier, directory / "media.partial"

    async def finish(self, identifier: str, path: Path) -> dict[str, Any]:
        try:
            metadata = await asyncio.to_thread(probe, path)
            streams = metadata.get("streams", [])
            kinds = {s["codec_type"] for s in streams}
            fmt = metadata.get("format", {})
            image_codec = {"mjpeg", "png", "webp", "bmp", "tiff", "gif"}
            if "video" in kinds:
                video = next(s for s in streams if s["codec_type"] == "video")
                still = (
                    video.get("codec_name") in image_codec and float(fmt.get("duration", 0)) <= 0.1
                )
                kind, extension = ("image", ".png") if still else ("video", ".mp4")
            elif "audio" in kinds:
                kind, extension = "audio", ".wav"
            else:
                raise RuntimeFault("media_invalid", "Upload has no supported media stream")
            # Derive the suffix from ffprobe; retain the original container bytes.
            if kind == "video":
                extension = ".webm" if "webm" in fmt.get("format_name", "") else ".mp4"
            elif kind == "image":
                extension = {"mjpeg": ".jpg", "webp": ".webp", "gif": ".gif"}.get(
                    video.get("codec_name"), ".png"
                )
            else:
                extension = ".mp3" if "mp3" in fmt.get("format_name", "") else ".wav"
            target = path.with_name("media" + extension)
            path.rename(target)
            digest = hashlib.sha256()
            with target.open("rb") as stream:
                for block in iter(lambda: stream.read(1024**2), b""):
                    digest.update(block)
            result = {
                "upload_id": identifier,
                "sha256": digest.hexdigest(),
                "bytes": target.stat().st_size,
                "media_type": kind,
                "filename": target.name,
                "ffprobe": metadata,
                "duration_seconds": float(fmt.get("duration", 0)),
            }
            write_json(target.parent / "metadata.json", result)
            return result
        except RuntimeFault:
            raise
        except Exception as exc:
            raise RuntimeFault("media_invalid", "ffprobe rejected the upload") from exc

    async def from_url(self, url: str) -> dict[str, Any]:
        identifier, path = self.begin()
        try:
            async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
                for _ in range(6):
                    host, ip = await asyncio.to_thread(public_url, url)
                    parsed = httpx.URL(url)
                    # Pin connection to the validated IP; preserve TLS SNI and Host.
                    destination = parsed.copy_with(host=ip)
                    async with client.stream(
                        "GET",
                        destination,
                        headers={"Host": host},
                        extensions={"sni_hostname": host},
                    ) as response:
                        if response.is_redirect:
                            url = urljoin(url, response.headers["location"])
                            continue
                        response.raise_for_status()
                        written = 0
                        with path.open("wb") as output:
                            async for chunk in response.aiter_bytes():
                                written += len(chunk)
                                if written > self.maximum:
                                    raise RuntimeFault(
                                        "upload_too_large", "Upload exceeds size limit", 413
                                    )
                                output.write(chunk)
                        return await self.finish(identifier, path)
                raise RuntimeFault("url_redirects", "Too many URL redirects")
        finally:
            path.unlink(missing_ok=True)


def public_url(url: str) -> tuple[str, str]:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        raise RuntimeFault("url_unsafe", "Only public HTTPS URLs on port 443 are allowed")
    addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
    ips = [str(a[4][0]) for a in addresses]
    if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
        raise RuntimeFault("url_unsafe", "URL resolves to a non-public address")
    return parsed.hostname, ips[0]
