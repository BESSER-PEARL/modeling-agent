"""Connect to a running agent the way a browser does.

The agent refuses a WebSocket upgrade whose ``Origin`` is not allow-listed,
and the Python ``websockets`` client sends no ``Origin`` at all (browsers
do), so a bare client gets HTTP 403 rather than a socket.

Use ``connect()`` here instead of calling ``websockets.connect`` directly, so
the target URL and the Origin are handled in exactly one place.
"""
from __future__ import annotations

import os
from urllib.parse import urlsplit

import websockets


def _default_origin(url: str) -> str:
    """``wss://host/...`` -> ``https://host``; anything else is treated as a
    local agent, whose config allow-lists ``http://localhost:8080``."""
    parts = urlsplit(url)
    if parts.scheme == "wss":
        return f"https://{parts.netloc}"
    return "http://localhost:8080"


def connect(url: str, **kwargs):
    """``websockets.connect`` with the browser-equivalent defaults.

    Returns the Connect object, which is both an async context manager and
    awaitable, so it drops into ``async with connect(...) as ws`` and
    ``await connect(...)`` alike. ``AGENT_WS_ORIGIN`` overrides the Origin.
    """
    if not url:
        raise SystemExit(
            "AGENT_WS_URL is not set. Point it at a running agent, e.g. "
            "AGENT_WS_URL=ws://localhost:8765"
        )
    kwargs.setdefault(
        "origin", os.environ.get("AGENT_WS_ORIGIN") or _default_origin(url)
    )
    kwargs.setdefault("max_size", None)
    kwargs.setdefault("ping_interval", 20)
    return websockets.connect(url, **kwargs)
