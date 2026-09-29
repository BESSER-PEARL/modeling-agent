"""Turn-scoped reply replay.

On reconnect the frontend's first heartbeat flushes the platform outbox (the
reply lost to the dead socket) and, in the same tick, the frontend asks for a
replay. The old replay re-sent the session's last terminal reply regardless of
turn, so the client applied the SAME reply twice, or — when the current turn had
not replied yet — the PREVIOUS turn's reply.

Now a v2 ``user_message`` carries a client ``turnId``; every frame of that turn
is stamped with it and a per-turn ``replySeq``, and ``replay_last_response``
with a ``turnId`` re-sends only that turn's terminal replies the client has not
acknowledged (``appliedSeqs``). Requests without a ``turnId`` (older clients)
keep the old wire format and replay behaviour.
"""
import json
import threading
from unittest.mock import patch

import pytest

import session_helpers as sh
from tests.conftest import FakeSession


@pytest.fixture(autouse=True)
def _clean_buffers():
    sh._last_reply_buffer.clear()
    getattr(sh, "_turn_replies", {}).clear()
    with patch.object(sh, "_record_assistant_response"), patch.object(sh, "_emit_prompt_telemetry"):
        yield
    sh._last_reply_buffer.clear()
    getattr(sh, "_turn_replies", {}).clear()


def _turn(message="make a library system", turn_id="t1", session_id="sess-A"):
    payload = {
        "action": "user_message", "protocolVersion": "2.0", "clientMode": "widget",
        "sessionId": session_id, "message": message, "context": {"activeDiagramType": "ClassDiagram"},
    }
    if turn_id is not None:
        payload["turnId"] = turn_id
    return FakeSession(payload)


def _replay(turn_id=None, applied=None, session_id="sess-A"):
    payload = {"action": "replay_last_response", "protocolVersion": "2.0",
               "clientMode": "widget", "sessionId": session_id}
    if turn_id is not None:
        payload["turnId"] = turn_id
    if applied is not None:
        payload["appliedSeqs"] = applied
    return FakeSession(payload)


def _sent(session):
    return [json.loads(raw) for raw in session._replies]


def _system(name):
    return {"action": "inject_complete_system", "diagramType": "ClassDiagram",
            "systemSpec": {"classes": [{"className": name}]}, "message": f"built {name}"}


def test_every_frame_of_a_turn_is_stamped_with_turn_id_and_sequence():
    s = _turn()
    sh.reply_progress(s, "working", 1, 2)
    sh.reply_payload(s, _system("Book"))
    sid = sh.reply_stream_start(s)
    sh.reply_stream_done(s, sid, "done")
    frames = _sent(s)
    assert [f["turnId"] for f in frames] == ["t1"] * 4
    assert [f["replySeq"] for f in frames] == [1, 2, 3, 4]


def test_replay_does_not_resend_a_reply_the_client_already_applied():
    # The outbox flush delivered R1 (seq 1); the replay must not send it again.
    s = _turn()
    sh.reply_payload(s, _system("Book"))
    seq = _sent(s)[0].get("replySeq", 1)

    r = _replay("t1", [seq])
    sh.replay_last_reply(r, sh.parse_assistant_request(r))
    assert _sent(r) == []


def test_replay_for_the_current_turn_never_sends_the_previous_turns_reply():
    t1 = _turn(turn_id="t1")
    sh.reply_payload(t1, _system("Book"))
    # Turn 2 is sent but has produced no terminal reply yet.
    t2 = _turn(message="now add Member", turn_id="t2")
    sh.reply_progress(t2, "working", 1, 1)

    r = _replay("t2", [1])
    did = sh.replay_last_reply(r, sh.parse_assistant_request(r))
    assert did is False
    assert _sent(r) == []


def test_replay_sends_only_the_unacknowledged_terminal_replies_of_the_turn_in_order():
    s = _turn()
    sh.reply_progress(s, "working", 1, 3)                                   # seq 1 (not buffered)
    sh.reply_payload(s, _system("Book"))                                    # seq 2
    sh.reply_payload(s, {"action": "auto_generate_gui", "message": "gui"})  # seq 3
    sh.reply_message(s, "Your app is ready.")                               # seq 4

    r = _replay("t1", [1, 2])
    assert sh.replay_last_reply(r, sh.parse_assistant_request(r)) is True
    frames = _sent(r)
    assert [f["action"] for f in frames] == ["auto_generate_gui", "assistant_message"]
    # Replayed with their ORIGINAL stamps, so the client can dedupe them.
    assert [(f["turnId"], f["replySeq"]) for f in frames] == [("t1", 3), ("t1", 4)]


def test_turn_replies_are_keyed_per_session():
    a = _turn(turn_id="shared", session_id="sess-A")
    sh.reply_payload(a, _system("Book"))
    r = _replay("shared", [], session_id="sess-B")
    assert sh.replay_last_reply(r, sh.parse_assistant_request(r)) is False
    assert _sent(r) == []


def test_sequence_numbers_are_unique_under_concurrent_replies():
    # Planned operations reply from a thread pool within one turn.
    s = _turn()
    threads = [threading.Thread(target=sh.reply_payload, args=(s, {"action": "modify_model", "message": str(i)}))
               for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(f["replySeq"] for f in _sent(s)) == list(range(1, 21))


def test_turn_buffer_is_bounded():
    for i in range(sh._REPLY_BUFFER_MAX + 5):
        sh.reply_payload(_turn(turn_id=f"t{i}"), _system("Book"))
    assert len(sh._turn_replies) == sh._REPLY_BUFFER_MAX


@pytest.mark.parametrize("action", [
    "inject_element", "trigger_smart_generator", "agent_error",
    "trigger_export", "trigger_deploy", "create_diagram_tab",
])
def test_every_final_reply_type_is_replayable_within_its_turn(action):
    s = _turn()
    sh.reply_payload(s, {"action": action, "message": "x"})
    r = _replay("t1", [])
    assert sh.replay_last_reply(r, sh.parse_assistant_request(r)) is True
    assert _sent(r)[0]["action"] == action


def test_side_effect_trigger_is_replayed_when_missed_and_never_when_applied():
    s = _turn()
    sh.reply_payload(s, {"action": "trigger_smart_generator", "message": "starting"})
    seq = _sent(s)[0]["replySeq"]

    missed = _replay("t1", [])
    sh.replay_last_reply(missed, sh.parse_assistant_request(missed))
    assert [f["action"] for f in _sent(missed)] == ["trigger_smart_generator"]

    applied = _replay("t1", [seq])
    sh.replay_last_reply(applied, sh.parse_assistant_request(applied))
    assert _sent(applied) == []


def test_new_types_stay_out_of_the_legacy_replay():
    # A legacy client has no dedupe: replaying a paid-run trigger could start it twice.
    s = _turn(turn_id=None)
    sh.reply_payload(s, {"action": "trigger_smart_generator", "message": "starting"})
    r = _replay()
    assert sh.replay_last_reply(r, sh.parse_assistant_request(r)) is False


def test_a_streamed_reply_is_replayable_by_its_stream_done():
    s = _turn(message="describe my model")
    sid = sh.reply_stream_start(s)                 # seq 1
    sh.reply_stream_chunk(s, "The ", sid)          # seq 2
    sh.reply_stream_done(s, sid, "The model.")     # seq 3 — lost to the reconnect
    r = _replay("t1", [1, 2])
    assert sh.replay_last_reply(r, sh.parse_assistant_request(r)) is True
    frames = _sent(r)
    assert [(f["action"], f["fullText"], f["replySeq"]) for f in frames] == [("stream_done", "The model.", 3)]


# ── Compatibility ──────────────────────────────────────────────────────────

def test_old_client_without_turn_id_gets_unstamped_frames():
    s = _turn(turn_id=None)
    sh.reply_progress(s, "working", 1, 1)
    sh.reply_payload(s, _system("Book"))
    sh.reply_message(s, "done")
    for frame in _sent(s):
        assert "turnId" not in frame and "replySeq" not in frame


def test_old_client_replay_without_turn_id_resends_the_last_terminal_reply():
    s = _turn(turn_id=None)
    sh.reply_payload(s, _system("Book"))
    sh.reply_message(s, "anything else?")

    r = _replay()
    assert sh.replay_last_reply(r, sh.parse_assistant_request(r)) is True
    assert _sent(r) == [{"action": "assistant_message", "message": "anything else?"}]


def test_new_client_turnless_replay_is_answered_with_the_stamped_reply():
    # A voice turn has no turnId, so its replay is the legacy one; the buffered
    # reply keeps its stamps, letting the client drop it if already applied.
    s = _turn(turn_id="t1")
    sh.reply_payload(s, _system("Book"))
    r = _replay()
    assert sh.replay_last_reply(r, sh.parse_assistant_request(r)) is True
    assert (_sent(r)[0]["turnId"], _sent(r)[0]["replySeq"]) == ("t1", 1)


def test_turn_id_is_parsed_from_the_wrapped_wire_envelope():
    from protocol.adapters import parse_assistant_request
    inner = {"action": "user_message", "protocolVersion": "2.0", "sessionId": "s",
             "message": "hi", "turnId": "t9", "context": {}}
    wrapped = FakeSession({"action": "user_message", "user_id": "u", "message": json.dumps(inner)})
    assert parse_assistant_request(wrapped).turn_id == "t9"
    # Non-string / oversized values are ignored rather than echoed.
    assert parse_assistant_request(_turn(turn_id="x" * 500)).turn_id is None
