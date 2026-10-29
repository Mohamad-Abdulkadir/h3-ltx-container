"""Bounded ffmpeg subprocesses used for staging and clip-safe finishing."""

import asyncio
from pathlib import Path

from h3ltx.errors import RuntimeFault


async def ffmpeg(*args: str, timeout: float = 600) -> None:
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        *args,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, err = await asyncio.wait_for(process.communicate(), timeout)
        if process.returncode:
            raise RuntimeFault(
                "media_processing",
                "ffmpeg failed",
                500,
                stderr=err.decode(errors="replace")[-2000:],
            )
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise


async def encode_output(path: Path, nvenc: bool, gpu: int, hdr: bool = False) -> str:
    """Select the tested hardware encoder or CPU x264; retain audio and colour metadata."""
    temporary = path.with_name(path.stem + ".encoded" + path.suffix)
    try:
        if nvenc:
            codec = "hevc_nvenc" if hdr else "h264_nvenc"
            try:
                await ffmpeg(
                    "-i",
                    str(path),
                    "-map",
                    "0",
                    "-c:v",
                    codec,
                    "-gpu",
                    str(gpu),
                    *(["-pix_fmt", "p010le"] if hdr else []),
                    "-preset",
                    "p5",
                    "-cq",
                    "18",
                    "-c:a",
                    "copy",
                    str(temporary),
                )
                temporary.replace(path)
                return codec
            except RuntimeFault:
                temporary.unlink(missing_ok=True)
        await ffmpeg(
            "-i",
            str(path),
            "-map",
            "0",
            "-c:v",
            "libx264",
            *(["-pix_fmt", "yuv420p10le"] if hdr else []),
            "-preset",
            "medium",
            "-crf",
            "18",
            "-c:a",
            "copy",
            str(temporary),
        )
        temporary.replace(path)
        return "libx264"
    finally:
        temporary.unlink(missing_ok=True)
