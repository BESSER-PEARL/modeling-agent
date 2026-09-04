"""Pilot-experiment telemetry — fire-and-forget event posting.

During a pilot session (the frontend tab was opened via a facilitator's
``?pilot=P3`` link) the frontend attaches a ``pilotParticipant`` label to the
workspace context of every message. When that label is present, the agent
posts one ``prompt`` event per handled user message to the BESSER backend's
telemetry collector: what was asked and what the agent did with it.

Contract (frozen, shared with the collector):
    POST {BESSER_BACKEND_URL}/besser_api/telemetry/event
    {"session": str, "participant": str, "kind": "prompt", "payload": {...}}
The collector answers 204 whether or not telemetry is enabled server-side,
so posting is always safe.

Everything here is best-effort by design: the POST runs on a short-timeout
daemon thread, every exception is swallowed and logged at debug level, and a
telemetry hiccup can NEVER delay or break a reply.
"""

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Contract cap for the recorded prompt text (the collector additionally caps
# the whole payload at 8KB serialized).
PROMPT_TEXT_MAX_CHARS = 2000

# Short timeout: the post runs off-thread, but a hung connection should still
# release its thread quickly.
_POST_TIMEOUT_SECONDS = 5


def _telemetry_url() -> str:
    """Collector endpoint, from the same env var the validation bridge uses."""
    base = os.environ.get("BESSER_BACKEND_URL", "http://localhost:3001")
    return f"{base.rstrip('/')}/besser_api/telemetry/event"


def _post_event(url: str, body: dict) -> None:
    """Blocking POST, run on a daemon thread. Swallows everything."""
    try:
        import requests

        requests.post(url, json=body, timeout=_POST_TIMEOUT_SECONDS)
    except Exception as exc:
        logger.debug(f"[Telemetry] post failed (best-effort): {exc}")


def emit_prompt_event(
    session_id: Optional[str],
    participant: Optional[str],
    text: str,
    action_taken: str,
    diagram_type: Optional[str] = None,
) -> None:
    """Fire-and-forget ``prompt`` telemetry event.

    Skipped silently unless BOTH a session id and a participant label are
    present (i.e. outside pilot sessions this is a no-op). Never raises,
    never blocks the caller.
    """
    try:
        if not session_id or not participant:
            return
        payload = {
            "text": (text or "")[:PROMPT_TEXT_MAX_CHARS],
            "action_taken": action_taken,
        }
        if diagram_type:
            payload["diagram_type"] = diagram_type
        body = {
            "session": session_id,
            "participant": participant,
            "kind": "prompt",
            "payload": payload,
        }
        threading.Thread(
            target=_post_event,
            args=(_telemetry_url(), body),
            name="pilot-telemetry-post",
            daemon=True,
        ).start()
    except Exception as exc:
        logger.debug(f"[Telemetry] emit failed (best-effort): {exc}")
