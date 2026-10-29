import os
import time
from collections.abc import Callable

import httpx

from h3ltx.config import Settings


class IdleStop:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.last_activity = time.monotonic()
        self.stopped = False

    async def check(self, busy: Callable[[], bool]) -> bool:
        s = self.settings
        if busy():
            self.last_activity = time.monotonic()
            return False
        if self.stopped or not s.idle_stop_minutes:
            return False
        if time.monotonic() - self.last_activity < s.idle_stop_minutes * 60:
            return False
        token, pod = s.runpod_api_key.get_secret_value(), os.getenv("RUNPOD_POD_ID")
        if not token or not pod or s.mode != "pod":
            return False
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"https://rest.runpod.io/v1/pods/{pod}/stop",
                headers={"Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()
        self.stopped = True
        return True
