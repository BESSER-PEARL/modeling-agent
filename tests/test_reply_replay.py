"""Reconnect recovery: buffer the last terminal reply and replay it on demand.

A long generation can outlive its WebSocket connection — the socket reconnects
mid-flight and the final reply is routed to the dead socket, leaving the UI stuck
on "still working…". The agent buffers the last terminal reply per stable session
key and re-sends it when the frontend fires a ``replay_last_response`` control
message on reconnect. These tests cover the buffer + replay directly.
"""
import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import session_helpers as sh  # noqa: E402


class _FakeSession:
    def __init__(self):
        self.sent = []

    def reply(self, raw):
        self.sent.append(json.loads(raw))


def _reset():
    sh._last_reply_buffer.clear()


def test_terminal_reply_is_buffered_and_replays_on_a_reconnected_session():
    _reset()
    session = _FakeSession()
    payload = {"action": "inject_complete_system", "diagramType": "ClassDiagram",
               "systemSpec": {"classes": [{"className": "Menu"}]}, "message": "spec ready"}
    with patch.object(sh, "memory_session_key", return_value="sess-1"), \
         patch.object(sh, "_record_assistant_response"):
        sh.reply_payload(session, payload)

    # It was buffered under the stable session key…
    assert sh._last_reply_buffer.get("sess-1")["action"] == "inject_complete_system"

    # …and a FRESH (reconnected) session replays the same terminal payload.
    reconnected = _FakeSession()
    with patch.object(sh, "memory_session_key", return_value="sess-1"):
        did = sh.replay_last_reply(reconnected, request=None)
    assert did is True
    assert reconnected.sent[-1]["action"] == "inject_complete_system"
    assert reconnected.sent[-1]["systemSpec"]["classes"][0]["className"] == "Menu"


def test_trigger_github_import_is_a_terminal_reply_and_replays():
    """The continue-from-GitHub action is the turn's terminal reply — a
    reconnect that drops it must be able to replay it, exactly like
    trigger_generator (live bug class 2026-09-03)."""
    _reset()
    session = _FakeSession()
    payload = {"action": "trigger_github_import", "owner": "armen",
               "repo": "hotel-app", "branch": None, "message": "Importing…"}
    with patch.object(sh, "memory_session_key", return_value="sess-gh"), \
         patch.object(sh, "_record_assistant_response"):
        sh.reply_payload(session, payload)

    assert sh._last_reply_buffer.get("sess-gh")["action"] == "trigger_github_import"

    reconnected = _FakeSession()
    with patch.object(sh, "memory_session_key", return_value="sess-gh"):
        did = sh.replay_last_reply(reconnected, request=None)
    assert did is True
    assert reconnected.sent[-1]["owner"] == "armen"
    assert reconnected.sent[-1]["repo"] == "hotel-app"


def test_non_terminal_frames_are_not_buffered():
    _reset()
    session = _FakeSession()
    with patch.object(sh, "memory_session_key", return_value="sess-2"), \
         patch.object(sh, "_record_assistant_response"):
        sh.reply_payload(session, {"action": "progress", "message": "working…"})
    assert "sess-2" not in sh._last_reply_buffer


def test_assistant_message_is_buffered_via_reply_message():
    _reset()
    session = _FakeSession()

    class _V2Req:
        is_v2 = True

    with patch.object(sh, "parse_assistant_request", return_value=_V2Req()), \
         patch.object(sh, "memory_session_key", return_value="sess-3"), \
         patch.object(sh, "_record_assistant_response"):
        sh.reply_message(session, "here is your clarification")
    buffered = sh._last_reply_buffer.get("sess-3")
    assert buffered and buffered["action"] == "assistant_message"


def test_replay_is_a_safe_noop_when_nothing_is_buffered():
    _reset()
    session = _FakeSession()
    with patch.object(sh, "memory_session_key", return_value="never-seen"):
        did = sh.replay_last_reply(session)
    assert did is False
    assert session.sent == []


def test_buffer_is_bounded_lru():
    _reset()
    session = _FakeSession()
    with patch.object(sh, "_record_assistant_response"):
        for i in range(sh._REPLY_BUFFER_MAX + 25):
            with patch.object(sh, "memory_session_key", return_value=f"s{i}"):
                sh.reply_payload(session, {"action": "modify_model", "message": f"m{i}"})
    # never grows past the cap; oldest evicted, newest kept
    assert len(sh._last_reply_buffer) == sh._REPLY_BUFFER_MAX
    assert "s0" not in sh._last_reply_buffer
    assert f"s{sh._REPLY_BUFFER_MAX + 24}" in sh._last_reply_buffer


def test_common_preamble_intercepts_replay_and_never_runs_generation():
    import state_bodies as sb

    class _Req:
        action = "replay_last_response"

    session = _FakeSession()
    with patch.object(sb, "parse_assistant_request", return_value=_Req()), \
         patch.object(sb, "replay_last_reply") as mock_replay, \
         patch.object(sb, "handle_pending_gui_choice") as mock_gui, \
         patch.object(sb, "handle_pending_system_confirmation") as mock_conf:
        result = sb._common_preamble(session)
    assert result is None                 # short-circuits — state body does nothing
    assert mock_replay.called             # the buffered reply is re-sent
    assert not mock_gui.called            # pending flows are never consumed
    assert not mock_conf.called
