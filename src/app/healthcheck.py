"""The Docker health check: the WebSocket port accepts and the agent has descriptors left.

Run as ``python -m app.healthcheck`` inside the container. A port that accepts
is not enough: on 2026-11-23 the agent ran out of file descriptors, new
WebSocket handshakes hung, and the old port check stayed "healthy". This also
fails once the agent process (PID 1) holds more than ``FD_LIMIT_SHARE`` of its
soft ``RLIMIT_NOFILE``. Exit 0 is healthy, 1 unhealthy (Docker's contract).
"""

from __future__ import annotations

import os
import socket
import sys
from typing import Optional, Tuple

PORT = 8765
FD_LIMIT_SHARE = 0.9


def open_fd_count(pid: int = 1) -> Optional[int]:
    """Open descriptors of ``pid``, or None where /proc is not readable."""
    try:
        return len(os.listdir(f"/proc/{pid}/fd"))
    except OSError:
        return None


def soft_fd_limit(pid: int = 1) -> Optional[int]:
    """The soft "Max open files" limit of ``pid`` from /proc, or None."""
    try:
        with open(f"/proc/{pid}/limits") as fh:
            for line in fh:
                if line.startswith("Max open files"):
                    value = line.split()[3]
                    return None if value == "unlimited" else int(value)
    except (OSError, ValueError, IndexError):
        return None
    return None


def fd_verdict(open_fds: Optional[int], limit: Optional[int], share: float = FD_LIMIT_SHARE) -> Tuple[bool, str]:
    """Whether the descriptor count is healthy, and why. Unknown counts pass."""
    if open_fds is None or not limit:
        return True, "descriptor count unknown"
    if open_fds > limit * share:
        return False, f"{open_fds} of {limit} file descriptors open"
    return True, f"{open_fds} of {limit} file descriptors open"


def port_accepts(port: int = PORT, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection(("localhost", port), timeout=timeout):
            return True
    except OSError:
        return False


def main() -> int:
    if not port_accepts():
        print(f"unhealthy: port {PORT} does not accept connections")
        return 1
    ok, detail = fd_verdict(open_fd_count(), soft_fd_limit())
    print(("healthy: " if ok else "unhealthy: ") + detail)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
