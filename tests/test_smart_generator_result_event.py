"""Tests for the smart-generation outcome report over the WebSocket.

The editor now sends ``generator_result`` frontend events with
``metadata.smart = true`` when a smart-generation run finishes (success,
failure, cost cap, cancel). The agent must reply with an outcome-aware
message and record the outcome in conversation memory so follow-up turns
("why did it fail?", "run it again") have context — previously the smart
path was fire-and-forget and the agent learned nothing.
"""

import sys
import types

import pytest


# In full-suite runs an earlier test can leave a plain ``protocol``
# MODULE in sys.modules that shadows the src/protocol PACKAGE (the same
# pre-existing collision that breaks test_unified_classifier). Evict the
# impostor so ``protocol.types`` resolves to the package.
for _name in ("protocol", "protocol.types"):
    _mod = sys.modules.get(_name)
    if _mod is not None and not hasattr(sys.modules.get("protocol"), "__path__"):
        sys.modules.pop(_name, None)

# codegen.generation_handler imports ``baf.core.session`` at module
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

from codegen.generation_handler import _handle_frontend_event  # noqa: E402


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


@pytest.mark.parametrize("metadata,expected,unexpected", [
    ({"incomplete": True, "blockerCount": 19}, "19 unresolved blockers", "stopped early"),
    ({"blockerCount": 1}, "1 unresolved blocker;", "stopped early"),
    ({"incomplete": True}, "stopped early", "finished successfully"),
])
def test_smart_partial_output_does_not_claim_success(metadata, expected, unexpected):
    result = _handle_frontend_event(_smart_event(True, metadata), None)
    assert expected in result["message"]
    assert unexpected not in result["message"]
    assert "incomplete" in result["message"]


def test_unresolved_blockers_do_not_call_an_edited_app_incomplete_as_a_whole():
    # The same report follows a first build and a run that edited an existing app.
    result = _handle_frontend_event(_smart_event(True, {"incomplete": True, "blockerCount": 2}), None)
    assert "2 unresolved blockers; the result is incomplete." in result["message"]
    assert "application is incomplete" not in result["message"]


@pytest.mark.parametrize("count, checks", [(1, "1 check could not be verified."), (2, "2 checks could not be verified.")])
def test_a_run_reported_complete_is_not_called_incomplete_for_unverified_checks(count, checks):
    # The editor's incomplete:false wins over the blocker count, as on its run card.
    result = _handle_frontend_event(_smart_event(True, {"incomplete": False, "blockerCount": count}), None)
    assert "finished successfully" in result["message"]
    assert checks in result["message"]
    assert "incomplete" not in result["message"]
    assert "unresolved" not in result["message"]


@pytest.mark.parametrize("metadata, expected", [
    ({"mode": "modify"}, "Spec-Driven Agent finished updating your application."),
    ({"mode": "create"}, "Spec-Driven Agent generation finished successfully."),
    ({}, "Spec-Driven Agent generation finished successfully."),  # an editor that sends no mode
])
def test_a_finished_run_is_reported_by_what_it_did(metadata, expected):
    result = _handle_frontend_event(_smart_event(True, metadata), None)
    assert result["message"].startswith(expected)


@pytest.mark.parametrize("mode, expected", [
    ("modify", "You can start another run to finish the remaining changes."),
    ("create", "You can run the generation again to finish the remaining changes."),
    (None, "You can run the generation again to finish the remaining changes."),
])
def test_an_incomplete_run_suggests_a_next_step_for_what_it_did(mode, expected):
    metadata = {"incomplete": True, "blockerCount": 2, **({"mode": mode} if mode else {})}
    result = _handle_frontend_event(_smart_event(True, metadata), None)
    assert result["message"].endswith(expected)
    if mode == "modify":
        assert "generation" not in result["message"]


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


@pytest.mark.parametrize("generator_type, expected", [
    ("django", "Your Django project is generated and ready to download."),
    ("web_app", "Your web app is generated and ready to download."),
    ("sql", "Your database schema is generated and ready to download."),
    ("jsonschema", "Your JSON Schema is generated and ready to download."),
])
def test_non_smart_generator_result_uses_generator_confirmation(generator_type, expected):
    """A non-smart generator_result yields a clean, generator-appropriate
    confirmation keyed off ``generatorType`` — it no longer re-echoes the
    frontend message or appends the filename (the download card shows those).

    The editor echoes the ``generatorType`` of the ``trigger_generator`` reply
    it ran, so the value is the agent's own generator token."""
    request = _FakeRequest(
        {
            "eventType": "generator_result",
            "ok": True,
            "message": "Generated.",
            "metadata": {"generatorType": generator_type, "filename": "out.zip"},
        }
    )
    result = _handle_frontend_event(request, None)
    assert result == {"action": "assistant_message", "message": expected}


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
    from common.session_keys import LAST_SMART_GEN_AT
    assert session.get(LAST_SMART_GEN_AT) is not None
