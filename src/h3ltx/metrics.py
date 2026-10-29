import asyncio
import contextlib
import json
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any

import psutil

from h3ltx.gpu import GPUBackend


def cgroup_ram(root: Path = Path("/sys/fs/cgroup")) -> int:
    for name in ("memory.current", "memory/memory.usage_in_bytes"):
        try:
            return int((root / name).read_text().strip())
        except (OSError, ValueError):
            continue
    return psutil.Process().memory_info().rss


def cgroup_oom(root: Path = Path("/sys/fs/cgroup")) -> int:
    try:
        values = dict(line.split() for line in (root / "memory.events").read_text().splitlines())
        return int(values.get("oom_kill", 0))
    except (OSError, ValueError):
        return 0


class Sampler:
    def __init__(
        self, backend: GPUBackend, index: int, interval: float = 0.5, journal: Path | None = None
    ):
        self.backend, self.index, self.interval = backend, index, interval
        self.samples: list[dict[str, float]] = []
        self.stop = asyncio.Event()
        self.initial_oom = cgroup_oom()
        self.journal = journal

    async def run(self) -> None:
        while not self.stop.is_set():
            try:
                sample = self.backend.sample(self.index)
                sample.update(timestamp=time.time(), ram_bytes=float(cgroup_ram()))
                self.samples.append(sample)
            except Exception:
                # A disappearing GPU must not mask the engine's failure.
                self.samples.append({"timestamp": time.time(), "ram_bytes": float(cgroup_ram())})
            if self.journal and self.samples:
                with self.journal.open("a") as stream:
                    stream.write(json.dumps(self.samples[-1]) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.stop.wait(), self.interval)

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sample_interval_seconds": self.interval,
            "sample_count": len(self.samples),
            "samples": self.samples,
            "oom_detected": cgroup_oom() > self.initial_oom,
        }
        for key in ("vram_bytes", "ram_bytes", "utilization", "power_watts", "temperature_c"):
            values = [s[key] for s in self.samples if key in s]
            result[key + "_peak"] = max(values, default=0)
            result[key + "_mean"] = mean(values) if values else None
        return result
