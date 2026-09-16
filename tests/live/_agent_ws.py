"""Connect to the deployed agent the way a browser does.

The agent refuses a WebSocket upgrade whose ``Origin`` is not allow-listed,
and the Python ``websockets`` client sends no ``Origin`` at all (browsers
do). Every live probe therefore got HTTP 403 rather than a socket.

That cost real time. The deploy smoke gate swallowed the 403 in a bare
``except Exception`` and reported "agent WebSocket not reachable within boot
window", so the failure looked like a slow boot — and ``BOOT_WAIT`` was duly
raised from 180s to 360s, which fixed nothing because the connection was
being rejected, not delayed. The gate failed on every single agent deploy
(confirmed 2026-09-16: no Origin -> 403, with Origin -> 101).

Use ``connect()`` here instead of calling ``websockets.connect`` directly, so
the Origin is set in exactly one place.
"""
from __future__ import annotations

import os

import websockets

#: Must match an entry in the agent's WebSocket origin allowlist.
AGENT_WS_ORIGIN = os.environ.get(
    "AGENT_WS_ORIGIN", "https://experimental.besser-pearl.org"
)


def connect(url: str, **kwargs):
    """``websockets.connect`` with the browser-equivalent defaults.

    Returns the Connect object, which is both an async context manager and
    awaitable, so it drops into ``async with connect(...) as ws`` and
    ``await connect(...)`` alike.
    """
    kwargs.setdefault("origin", AGENT_WS_ORIGIN)
    kwargs.setdefault("max_size", None)
    kwargs.setdefault("ping_interval", 20)
    return websockets.connect(url, **kwargs)
