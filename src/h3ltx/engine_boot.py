"""Exec an engine with Linux parent-death signalling, without importing runtime dependencies."""

import ctypes
import os
import signal
import sys


def launch(expected_parent: int, arguments: list[str]) -> None:
    if sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM) != 0:  # PR_SET_PDEATHSIG
            raise OSError(ctypes.get_errno(), "Cannot set engine parent-death signal")
        if os.getppid() != expected_parent:
            raise SystemExit(143)
    os.execv(sys.executable, [sys.executable, *arguments])


if __name__ == "__main__":
    launch(int(sys.argv[1]), sys.argv[2:])
