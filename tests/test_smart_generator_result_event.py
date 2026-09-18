"""Tests for the smart-generation outcome report over the WebSocket.

The editor now sends ``generator_result`` frontend events with
``metadata.smart = true`` when a smart-generation run finishes (success,
failure, cost cap, cancel). The agent must reply with an outcome-aware
message and record the outcome in conversation memory so follow-up turns
("why did it fail?", "run it again") have context — previously the smart
path was fire-and-forget and the agent learned nothing.
"""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# In full-suite runs an earlier test can leave a plain ``protocol``
# MODULE in sys.modules that shadows the src/protocol PACKAGE (the same
# pre-existing collision that breaks test_unified_classifier). Evict the
# impostor so ``protocol.types`` resolves to the package.
for _name in ("protocol", "protocol.types"):
    _mod = sys.modules.get(_name)
    if _mod is not None and not hasattr(sys.modules.get("protocol"), "__path__"):
        sys.modules.pop(_name, None)

# handlers.generation_handler imports ``baf.core.session`` at module
# level; stub it so the test runs without the BAF stack installed.
if "baf" not in sys.modules:
    baf = types.ModuleType("baf")
    baf_core = types.ModuleType("baf.core")
    baf_session = types.ModuleType("baf.core.session")

    class _StubSession:  # pragma: no cover - placeholder type only
        pass

    baf_session.Session = _StubSession
    baf_core.session = baf_session
    baf.core = baf_core
    sys.modules["baf"] = baf
    sys.modules["baf.core"] = baf_core
    sys.modules["baf.core.session"] = baf_session

from handlers.generation_handler import _handle_frontend_event  # noqa: E402


class _FakeRequest:
    def __init__(self, payload):
        self.action = "frontend_event"
        self.raw_payload = payload
        self.message = ""


class _FakeSession:
    id = "session-smart-test"

    def __init__(self):
        self._data = {}

    def set(self, key, value):
        self._data[key] = value

    def get(self, key):
        return self._data.get(key)


def _smart_event(ok, metadata=None, message=None):
    payload = {
        "eventType": "generator_result",
        "ok": ok,
        "metadata": {"smart": True, **(metadata or {})},
    }
    if message is not None:
        payload["message"] = message
    return _FakeRequest(payload)


def test_smart_success_hides_cost_and_generator(monkeypatch):
    recorded = []

    class _Mem:
        def add_assistant(self, content, **kw):
            recorded.append(content)

    import memory

    monkeypatch.setattr(memory, "get_memory", lambda _sid: _Mem())

    result = _handle_frontend_event(
        _smart_event(
            True,
            {"costUsd": 0.42, "generator_used": "generate_fastapi_backend"},
        ),
        _FakeSession(),
    )

    assert result["action"] == "assistant_message"
    # Cost and the internal scaffold/generator name are intentionally hidden
    # from the user-facing message.
    assert "finished successfully" in result["message"]
    assert "$0.42" not in result["message"]
    assert "generate_fastapi_backend" not in result["message"]
    assert recorded and "smart-generation outcome" in recorded[0]


def test_smart_cost_cap_suggests_retry():
    result = _handle_frontend_event(
        _smart_event(False, {"errorCode": "COST_CAP", "costUsd": 2.01}),
        None,
    )
    assert "cost cap" in result["message"].lower()
    assert result.get("suggestedActions") == ["Retry with refined instructions"]


def test_smart_invalid_key_names_the_problem():
    result = _handle_frontend_event(
        _smart_event(False, {"errorCode": "INVALID_KEY"}),
        None,
    )
    assert "api key" in result["message"].lower()


def test_smart_cancelled_is_not_an_error_tone():
    result = _handle_frontend_event(
        _smart_event(False, {"errorCode": "CANCELLED"}),
        None,
    )
    assert "stopped" in result["message"].lower()


def test_non_smart_generator_result_uses_generator_confirmation():
    """A non-smart generator_result yields a clean, generator-appropriate
    confirmation keyed off ``generatorType`` — it no longer re-echoes the
    frontend message or appends the filename (the download card shows those)."""
    request = _FakeRequest(
        {
            "eventType": "generator_result",
            "ok": True,
            "message": "Generated Django app.",
            "metadata": {"generatorType": "generate_django", "filename": "django.zip"},
        }
    )
    result = _handle_frontend_event(request, None)
    assert result == {
        "action": "assistant_message",
        "message": "Your Django project is generated and ready to download.",
    }


def test_non_smart_generator_result_falls_back_without_type():
    """With no generatorType, a generic 'code is generated' confirmation."""
    request = _FakeRequest(
        {
            "eventType": "generator_result",
            "ok": True,
            "message": "Generated something.",
            "metadata": {"filename": "out.zip"},
        }
    )
    result = _handle_frontend_event(request, None)
    assert result == {
        "action": "assistant_message",
        "message": "Your code is generated and ready to download.",
    }


def test_memory_failure_never_breaks_the_reply(monkeypatch):
    import memory

    def _boom(_sid):
        raise RuntimeError("memory backend down")

    monkeypatch.setattr(memory, "get_memory", _boom)
    session = _FakeSession()
    result = _handle_frontend_event(
        _smart_event(True, {"costUsd": 0.1}),
        session,
    )
    assert result["action"] == "assistant_message"
    # The structured recency signal must survive a memory-stack failure —
    # it is set OUTSIDE the memory try/except precisely for this.
    from session_keys import LAST_SMART_GEN_AT
    assert session.get(LAST_SMART_GEN_AT) is not None
