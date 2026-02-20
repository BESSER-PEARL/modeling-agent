"""Shared test fixtures for the modeling-agent test suite."""

import json
import os
import sys
import pytest
from typing import Any, Dict, Optional

# ── Ensure src/ is importable for bare-style imports ─────────────────────
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ---------------------------------------------------------------------------
# Minimal stub for besser.agent.core.session.Session
# ---------------------------------------------------------------------------

class _FakeEvent:
    """Minimal event stub carrying the JSON payload."""

    def __init__(self, payload: Dict[str, Any]):
        self.json = payload
        self.data = payload
        self.message = payload.get("message", "")
        self.predicted_intent = None


class FakeSession:
    """Lightweight stand-in for ``besser.agent.core.session.Session``.

    Stores key-value pairs in ``_store`` and provides the same
    ``get / set / delete / get_dictionary`` interface that the real
    session exposes.
    """

    def __init__(self, payload: Optional[Dict[str, Any]] = None):
        self._store: Dict[str, Any] = {}
        self._replies: list = []
        self.event = _FakeEvent(payload or {})

    # -- Session data API --------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def get_dictionary(self) -> Dict[str, Any]:
        return dict(self._store)

    # -- Reply capture -----------------------------------------------------

    def reply(self, message: str) -> None:  # type: ignore[override]
        self._replies.append(message)

    @property
    def replies(self) -> list:
        return list(self._replies)

    def last_reply_json(self) -> Optional[Dict[str, Any]]:
        if not self._replies:
            return None
        try:
            return json.loads(self._replies[-1])
        except (json.JSONDecodeError, TypeError):
            return None


# ---------------------------------------------------------------------------
# Fake LLM that returns canned responses
# ---------------------------------------------------------------------------

class FakeLLM:
    """Minimal LLM stub.

    Set ``response`` (or ``responses`` for round-robin) before calling ``predict()``.
    """

    def __init__(self, response: str = '{}'):
        self.responses: list[str] = [response]
        self._call_index = 0
        self.call_log: list[str] = []

    def predict(self, prompt: str) -> str:
        self.call_log.append(prompt)
        result = self.responses[self._call_index % len(self.responses)]
        self._call_index += 1
        return result


# ---------------------------------------------------------------------------
# Protocol / workspace context helpers
# ---------------------------------------------------------------------------

def make_v2_payload(
    message: str,
    diagram_type: str = "ClassDiagram",
    *,
    active_model: Optional[Dict[str, Any]] = None,
    project_snapshot: Optional[Dict[str, Any]] = None,
    action: str = "user_message",
) -> Dict[str, Any]:
    """Build a wrapped v2 assistant protocol payload (as the frontend sends it)."""
    inner = {
        "action": action,
        "protocolVersion": "2.0",
        "clientMode": "workspace",
        "message": message,
        "context": {
            "activeDiagramType": diagram_type,
            **({"activeModel": active_model} if active_model else {}),
            **({"projectSnapshot": project_snapshot} if project_snapshot else {}),
        },
    }
    # BESSER websocket wraps the v2 payload inside `message` as a JSON string.
    return {
        "action": "user_message",
        "user_id": "test_session",
        "message": json.dumps(inner),
    }


def make_session(
    message: str,
    diagram_type: str = "ClassDiagram",
    *,
    active_model: Optional[Dict[str, Any]] = None,
    project_snapshot: Optional[Dict[str, Any]] = None,
) -> FakeSession:
    """Return a ``FakeSession`` pre-loaded with a v2 protocol payload."""
    payload = make_v2_payload(
        message, diagram_type,
        active_model=active_model,
        project_snapshot=project_snapshot,
    )
    return FakeSession(payload)


# ---------------------------------------------------------------------------
# Common model fixtures
# ---------------------------------------------------------------------------

MINIMAL_CLASS_MODEL: Dict[str, Any] = {
    "version": "3.0.0",
    "type": "ClassDiagram",
    "size": {"width": 1200, "height": 800},
    "interactive": {"elements": {}, "relationships": {}},
    "elements": {
        "cls-user-1": {
            "id": "cls-user-1",
            "name": "User",
            "type": "Class",
            "owner": None,
            "bounds": {"x": 100, "y": 100, "width": 200, "height": 150},
            "attributes": {},
            "methods": {},
        }
    },
    "relationships": {},
    "assessments": {},
}

EMPTY_CLASS_MODEL: Dict[str, Any] = {
    "version": "3.0.0",
    "type": "ClassDiagram",
    "size": {"width": 1200, "height": 800},
    "interactive": {"elements": {}, "relationships": {}},
    "elements": {},
    "relationships": {},
    "assessments": {},
}
