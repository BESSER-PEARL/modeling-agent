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


# Exact replies to payloads without notConfirmedCount / unresolvedBlockerCount /
# unverified, as an older editor or backend sends them.
@pytest.mark.parametrize("metadata, expected", [
    ({}, "Spec-Driven Agent generation finished successfully."),
    ({"mode": "modify"}, "Spec-Driven Agent finished updating your application."),
    (
        {"incomplete": False, "blockerCount": 2},
        "Spec-Driven Agent generation finished successfully. 2 checks could not be verified.",
    ),
    (
        {"incomplete": False, "blockerCount": 1, "mode": "modify", "filename": "app.zip",
         "verification": {"verified": 3, "notVerified": 1, "shippedUnenforced": 0}},
        "Spec-Driven Agent finished updating your application. 1 check could not be verified. File: app.zip",
    ),
    (
        {"incomplete": True, "blockerCount": 2, "incompleteReason": "the build failed"},
        "The Spec-Driven Agent finished with 2 unresolved blockers; the result is incomplete: "
        "the build failed. You can run the generation again to finish the remaining changes.",
    ),
    (
        {"blockerCount": 1},
        "The Spec-Driven Agent finished with 1 unresolved blocker; the result is incomplete. "
        "You can run the generation again to finish the remaining changes.",
    ),
    (
        {"incomplete": True, "mode": "modify"},
        "The Spec-Driven Agent produced output, but the run stopped early so it may be incomplete. "
        "You can start another run to finish the remaining changes.",
    ),
    # An incomplete run keeps blockerCount even when the newer counts are sent.
    (
        {"incomplete": True, "blockerCount": 5, "notConfirmedCount": 3, "unresolvedBlockerCount": 2,
         "unverified": False},
        "The Spec-Driven Agent finished with 5 unresolved blockers; the result is incomplete. "
        "You can run the generation again to finish the remaining changes.",
    ),
])
def test_a_payload_without_the_newer_counts_reads_exactly_as_before(metadata, expected):
    result = _handle_frontend_event(_smart_event(True, metadata), None)
    assert result["message"] == expected


_REASON = "The app was never started, so nothing shows it runs."


def test_an_unverified_generation_does_not_claim_success():
    metadata = {"incomplete": False, "blockerCount": 1, "notConfirmedCount": 0,
                "unresolvedBlockerCount": 0, "unverified": True, "unverifiedReason": _REASON}
    result = _handle_frontend_event(_smart_event(True, metadata), None)
    assert result["message"] == (
        f"Spec-Driven Agent generated your application, but it is not verified. {_REASON} "
        "You can download the application to inspect it as-is."
    )


def test_an_unverified_edit_run_keeps_the_reason_as_sent():
    reason = "The app was updated, but it was never started, so nothing shows it runs."
    metadata = {"mode": "modify", "incomplete": False, "blockerCount": 2, "notConfirmedCount": 2,
                "unresolvedBlockerCount": 0, "unverified": True, "unverifiedReason": reason}
    result = _handle_frontend_event(_smart_event(True, metadata), None)
    assert result["message"] == (
        f"Spec-Driven Agent finished updating your application, but the update is not verified. {reason} "
        "You can download the application to inspect it as-is."
    )


def test_an_unverified_run_states_its_real_blocker():
    metadata = {"incomplete": False, "blockerCount": 2, "notConfirmedCount": 0,
                "unresolvedBlockerCount": 1, "unverified": True, "unverifiedReason": _REASON}
    result = _handle_frontend_event(_smart_event(True, metadata), None)
    assert result["message"] == (
        f"Spec-Driven Agent generated your application, but it is not verified. {_REASON} "
        "1 unresolved blocker remains. You can download the application to inspect it as-is."
    )


def test_an_unverified_run_derives_no_blockers_from_the_padded_count():
    # Without unresolvedBlockerCount, blockerCount of an unverified run also counts the checks that never ran.
    metadata = {"incomplete": False, "blockerCount": 3, "notConfirmedCount": 0,
                "unverified": True, "unverifiedReason": _REASON}
    result = _handle_frontend_event(_smart_event(True, metadata), None)
    assert "unresolved" not in result["message"]
    assert "not verified" in result["message"]


def test_findings_not_confirmed_alone_are_a_success_with_a_qualifier():
    metadata = {"incomplete": False, "blockerCount": 3, "notConfirmedCount": 3, "unresolvedBlockerCount": 0}
    result = _handle_frontend_event(_smart_event(True, metadata), None)
    assert result["message"] == (
        "Spec-Driven Agent generation finished successfully. 3 checks could not be confirmed."
    )


@pytest.mark.parametrize("metadata", [
    {"incomplete": False, "blockerCount": 5, "notConfirmedCount": 3, "unresolvedBlockerCount": 2},
    {"incomplete": False, "blockerCount": 5, "notConfirmedCount": 3},  # a backend without the real count
])
def test_real_blockers_and_findings_not_confirmed_are_counted_apart(metadata):
    result = _handle_frontend_event(_smart_event(True, metadata), None)
    assert result["message"] == (
        "The Spec-Driven Agent finished with 2 unresolved blockers. 3 checks could not be confirmed."
    )


def test_a_real_blocker_of_an_edit_run_is_worded_as_an_update():
    metadata = {"mode": "modify", "incomplete": False, "blockerCount": 1, "notConfirmedCount": 0,
                "unresolvedBlockerCount": 1}
    result = _handle_frontend_event(_smart_event(True, metadata), None)
    assert result["message"] == (
        "The Spec-Driven Agent finished updating your application with 1 unresolved blocker."
    )


@pytest.mark.parametrize("metadata", [
    {"blockerCount": 1, "notConfirmedCount": 0, "unresolvedBlockerCount": 0},
    {"incomplete": False, "blockerCount": 1, "notConfirmedCount": 0, "unresolvedBlockerCount": 0, "unverified": False},
])
def test_a_padded_blocker_count_alone_is_a_plain_success(metadata):
    # blockerCount also counts the note of a check switched off on the server.
    result = _handle_frontend_event(_smart_event(True, metadata), None)
    assert result["message"] == "Spec-Driven Agent generation finished successfully."


@pytest.mark.parametrize("failed, sentence", [
    (1, "1 test the agent wrote ran against the app and failed."),
    (2, "2 tests the agent wrote ran against the app and failed."),
])
def test_tests_that_ran_and_failed_are_stated(failed, sentence):
    metadata = {"incomplete": False, "blockerCount": 0, "notConfirmedCount": 0, "unresolvedBlockerCount": 0,
                "verification": {"verified": 4, "notVerified": 0, "shippedUnenforced": 0, "failed": failed}}
    result = _handle_frontend_event(_smart_event(True, metadata), None)
    assert result["message"] == f"Spec-Driven Agent generation finished successfully. {sentence}"


def test_the_outcome_kept_for_the_next_turn_uses_the_same_wording(monkeypatch):
    import memory

    recorded = []

    class _Mem:
        def add_assistant(self, content, **kw):
            recorded.append(content)

    monkeypatch.setattr(memory, "get_memory", lambda _sid: _Mem())
    session = _FakeSession()
    metadata = {"incomplete": False, "blockerCount": 1, "notConfirmedCount": 0,
                "unresolvedBlockerCount": 0, "unverified": True, "unverifiedReason": _REASON}
    result = _handle_frontend_event(_smart_event(True, metadata), session)

    from common.session_keys import LAST_SMART_GEN_SUMMARY
    assert session.get(LAST_SMART_GEN_SUMMARY) == result["message"]
    assert recorded == [f"[smart-generation outcome] {result['message']}"]
    assert "finished successfully" not in recorded[0]


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
