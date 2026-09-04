"""Pilot-experiment telemetry: context propagation + fire-and-forget posting.

During a pilot session the frontend attaches ``context.pilotParticipant`` to
every message. The protocol adapter propagates it into the request, and the
reply choke points in ``session_helpers`` emit ONE ``prompt`` telemetry event
per handled user message (what was asked + what the agent did with it).

The invariants under test:
  - the participant label survives protocol parsing (and invalid labels don't);
  - no participant → no post, ever (regular users produce no telemetry);
  - one event per incoming message even when several replies flow out;
  - a telemetry failure never delays or breaks the actual reply.
"""
import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import session_helpers as sh  # noqa: E402
import telemetry  # noqa: E402
from protocol.adapters import parse_v2_payload  # noqa: E402
from session_keys import TELEMETRY_EMITTED_EVENT_ID  # noqa: E402


class _FakeEvent:
    pass


class _FakeSession:
    def __init__(self):
        self.sent = []
        self.event = _FakeEvent()
        self._store = {}

    def reply(self, raw):
        self.sent.append(json.loads(raw))

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value):
        self._store[key] = value


class _PilotRequest:
    is_v2 = True
    session_id = "session-abc"
    pilot_participant = "P3"
    message = "add a hotel booking system"
    diagram_type = "ClassDiagram"


class _RegularRequest:
    is_v2 = True
    session_id = "session-abc"
    pilot_participant = None
    message = "add a hotel booking system"
    diagram_type = "ClassDiagram"


def _v2_payload(context_extra=None):
    context = {"activeDiagramType": "ClassDiagram"}
    if context_extra:
        context.update(context_extra)
    return {
        "action": "user_message",
        "protocolVersion": "2.0",
        "sessionId": "session-abc",
        "message": "add a hotel booking system",
        "context": context,
    }


# ------------------------------------------------------------------
# Context propagation through protocol parsing
# ------------------------------------------------------------------

def test_pilot_participant_propagates_from_context_into_the_request():
    request = parse_v2_payload(_v2_payload({"pilotParticipant": "P3"}))
    assert request.pilot_participant == "P3"
    assert request.context.pilot_participant == "P3"


def test_pilot_participant_defaults_to_none_when_absent():
    request = parse_v2_payload(_v2_payload())
    assert request.pilot_participant is None
    assert request.context.pilot_participant is None


def test_invalid_pilot_participant_labels_are_dropped_at_the_boundary():
    for bad in ("P 3", "p3!", "a" * 17, "", 42, None, {"label": "P3"}):
        request = parse_v2_payload(_v2_payload({"pilotParticipant": bad}))
        assert request.pilot_participant is None, f"label {bad!r} should be dropped"


# ------------------------------------------------------------------
# The reply choke point
# ------------------------------------------------------------------

def test_reply_payload_emits_one_prompt_event_with_the_reply_action():
    session = _FakeSession()
    with patch.object(sh, "parse_assistant_request", return_value=_PilotRequest()), \
         patch.object(sh, "memory_session_key", return_value="sess-t1"), \
         patch.object(sh, "_record_assistant_response"), \
         patch.object(sh, "emit_prompt_event") as mock_emit:
        sh.reply_payload(session, {"action": "inject_complete_system", "message": "done"})

    assert mock_emit.call_count == 1
    kwargs = mock_emit.call_args.kwargs
    assert kwargs["session_id"] == "session-abc"
    assert kwargs["participant"] == "P3"
    assert kwargs["text"] == "add a hotel booking system"
    assert kwargs["action_taken"] == "inject_complete_system"
    assert kwargs["diagram_type"] == "ClassDiagram"
    # The reply itself still went out.
    assert session.sent[-1]["action"] == "inject_complete_system"


def test_no_participant_means_no_telemetry_post():
    session = _FakeSession()
    with patch.object(sh, "parse_assistant_request", return_value=_RegularRequest()), \
         patch.object(sh, "memory_session_key", return_value="sess-t2"), \
         patch.object(sh, "_record_assistant_response"), \
         patch.object(sh, "emit_prompt_event") as mock_emit:
        sh.reply_payload(session, {"action": "modify_model", "message": "done"})
        sh.reply_message(session, "anything else?")

    assert mock_emit.call_count == 0


def test_only_one_event_per_incoming_message_even_with_multiple_replies():
    session = _FakeSession()
    with patch.object(sh, "parse_assistant_request", return_value=_PilotRequest()), \
         patch.object(sh, "memory_session_key", return_value="sess-t3"), \
         patch.object(sh, "_record_assistant_response"), \
         patch.object(sh, "emit_prompt_event") as mock_emit:
        sh.reply_payload(session, {"action": "trigger_generator", "message": "generating"})
        sh.reply_message(session, "a follow-up line in the same turn")

    assert mock_emit.call_count == 1
    assert mock_emit.call_args.kwargs["action_taken"] == "trigger_generator"

    # A NEW incoming message (fresh event object) emits again.
    session.event = _FakeEvent()
    with patch.object(sh, "parse_assistant_request", return_value=_PilotRequest()), \
         patch.object(sh, "memory_session_key", return_value="sess-t3"), \
         patch.object(sh, "_record_assistant_response"), \
         patch.object(sh, "emit_prompt_event") as mock_emit2:
        sh.reply_message(session, "answered")
    assert mock_emit2.call_count == 1
    assert mock_emit2.call_args.kwargs["action_taken"] == "assistant_message"


def test_progress_frames_never_emit_telemetry():
    session = _FakeSession()
    with patch.object(sh, "parse_assistant_request", return_value=_PilotRequest()), \
         patch.object(sh, "memory_session_key", return_value="sess-t4"), \
         patch.object(sh, "_record_assistant_response"), \
         patch.object(sh, "emit_prompt_event") as mock_emit:
        sh.reply_payload(session, {"action": "progress", "message": "working…"})
    assert mock_emit.call_count == 0
    assert session.get(TELEMETRY_EMITTED_EVENT_ID) is None


def test_stream_done_emits_an_assistant_message_event():
    session = _FakeSession()
    with patch.object(sh, "parse_assistant_request", return_value=_PilotRequest()), \
         patch.object(sh, "emit_prompt_event") as mock_emit:
        sh.reply_stream_done(session, "stream-1", "here is the explanation")
    assert mock_emit.call_count == 1
    assert mock_emit.call_args.kwargs["action_taken"] == "assistant_message"


def test_truncation_warning_is_exempt_and_does_not_claim_the_turns_event():
    session = _FakeSession()
    with patch.object(sh, "parse_assistant_request", return_value=_PilotRequest()), \
         patch.object(sh, "memory_session_key", return_value="sess-t5"), \
         patch.object(sh, "_record_assistant_response"), \
         patch.object(sh, "emit_prompt_event") as mock_emit:
        sh.reply_message(session, "your message was trimmed…", telemetry_exempt=True)
        sh.reply_payload(session, {"action": "modify_model", "message": "done"})

    assert mock_emit.call_count == 1
    assert mock_emit.call_args.kwargs["action_taken"] == "modify_model"


def test_a_raising_telemetry_emit_never_breaks_the_reply():
    session = _FakeSession()
    with patch.object(sh, "parse_assistant_request", return_value=_PilotRequest()), \
         patch.object(sh, "memory_session_key", return_value="sess-t6"), \
         patch.object(sh, "_record_assistant_response"), \
         patch.object(sh, "emit_prompt_event", side_effect=RuntimeError("collector down")):
        sh.reply_payload(session, {"action": "inject_complete_system", "message": "done"})

    assert session.sent[-1]["action"] == "inject_complete_system"


def test_a_session_without_get_set_support_still_replies():
    """The replay tests' bare fake session (no get/set) must keep working —
    the telemetry hook is fully best-effort even against exotic sessions."""

    class _BareSession:
        def __init__(self):
            self.sent = []

        def reply(self, raw):
            self.sent.append(json.loads(raw))

    session = _BareSession()
    with patch.object(sh, "memory_session_key", return_value="sess-t7"), \
         patch.object(sh, "_record_assistant_response"):
        sh.reply_payload(session, {"action": "modify_model", "message": "done"})
    assert session.sent[-1]["action"] == "modify_model"


# ------------------------------------------------------------------
# The telemetry module itself
# ------------------------------------------------------------------

def test_emit_prompt_event_skips_without_participant_or_session():
    with patch.object(telemetry.threading, "Thread") as mock_thread:
        telemetry.emit_prompt_event(None, "P3", "hi", "assistant_message")
        telemetry.emit_prompt_event("session-abc", None, "hi", "assistant_message")
        telemetry.emit_prompt_event("", "", "hi", "assistant_message")
    assert mock_thread.call_count == 0


def test_emit_prompt_event_posts_the_contract_shape_off_thread(monkeypatch):
    monkeypatch.setenv("BESSER_BACKEND_URL", "http://backend:9000")
    captured = {}

    class _InlineThread:
        def __init__(self, target=None, args=(), **_kwargs):
            self._target, self._args = target, args

        def start(self):
            self._target(*self._args)

    def _fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["body"] = json
        captured["timeout"] = timeout

    with patch.object(telemetry.threading, "Thread", _InlineThread), \
         patch.dict(sys.modules, {"requests": type(sys)("requests")}):
        sys.modules["requests"].post = _fake_post
        telemetry.emit_prompt_event(
            "session-abc", "P3", "x" * 5000, "modify_model", diagram_type="BPMN"
        )

    assert captured["url"] == "http://backend:9000/besser_api/telemetry/event"
    assert captured["body"]["session"] == "session-abc"
    assert captured["body"]["participant"] == "P3"
    assert captured["body"]["kind"] == "prompt"
    assert captured["body"]["payload"]["action_taken"] == "modify_model"
    assert captured["body"]["payload"]["diagram_type"] == "BPMN"
    # Contract: prompt text is truncated to 2000 chars.
    assert len(captured["body"]["payload"]["text"]) == telemetry.PROMPT_TEXT_MAX_CHARS
    assert captured["timeout"] == telemetry._POST_TIMEOUT_SECONDS


def test_post_event_swallows_request_failures():
    fake_requests = type(sys)("requests")

    def _exploding_post(*_args, **_kwargs):
        raise ConnectionError("collector unreachable")

    fake_requests.post = _exploding_post
    with patch.dict(sys.modules, {"requests": fake_requests}):
        # Must not raise.
        telemetry._post_event("http://backend:9000/besser_api/telemetry/event", {"k": "v"})
