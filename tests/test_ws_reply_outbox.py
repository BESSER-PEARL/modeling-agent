"""Regression tests for the WebSocket reply-outbox (the "generate did nothing" bug).

Root cause: ``WebSocketPlatform._send`` routed each session's replies to a single
connection slot that could hold a stale/half-open socket after a reconnect;
``conn.send`` on a half-open TCP fails silently, so a fast (~4s) reply produced
during the reconnect gap was lost with no error. The fix buffers undeliverable
replies and flushes them on the next slot reclaim (message / heartbeat).

These exercise ``_send`` / ``_buffer_reply`` / ``_flush_outbox`` directly against
a lightweight shim (no real websocket server), so they run fast and hermetically.
"""
import threading
import types

import pytest

# The patched platform imports heavy BAF/streamlit deps — skip cleanly if the
# environment can't import it rather than failing collection.
wp = pytest.importorskip("patches.websocket_platform")
WebSocketPlatform = wp.WebSocketPlatform

from baf.platforms.payload import Payload, PayloadAction  # noqa: E402


class _FakeConn:
    def __init__(self, conn_id="c1", fail=False):
        self.id = conn_id
        self.fail = fail
        self.sent = []

    def send(self, data):
        if self.fail:
            raise OSError("half-open socket")
        self.sent.append(data)


def _shim():
    s = types.SimpleNamespace()
    s._connections = {}
    s._outbox = {}
    s._outbox_lock = threading.Lock()
    s._outbox_max = 50
    # Bind the real methods so ``self._buffer_reply`` / ``self._flush_outbox``
    # resolve when ``_send`` calls them.
    s._send = types.MethodType(WebSocketPlatform._send, s)
    s._buffer_reply = types.MethodType(WebSocketPlatform._buffer_reply, s)
    s._flush_outbox = types.MethodType(WebSocketPlatform._flush_outbox, s)
    return s


def _payload(msg="hi"):
    return Payload(action=PayloadAction.AGENT_REPLY_STR, message=msg)


def test_send_to_live_conn_delivers_and_does_not_buffer():
    s = _shim()
    conn = _FakeConn()
    s._connections["sess"] = conn
    WebSocketPlatform._send(s, "sess", _payload("ready"))
    assert len(conn.sent) == 1
    assert s._outbox.get("sess") in (None, [])


def test_send_with_no_conn_buffers_for_redelivery():
    s = _shim()  # no connection registered
    WebSocketPlatform._send(s, "sess", _payload("ready to run"))
    assert len(s._outbox.get("sess", [])) == 1


def test_send_to_failing_conn_buffers_and_prunes_slot():
    s = _shim()
    dead = _FakeConn(fail=True)
    s._connections["sess"] = dead
    WebSocketPlatform._send(s, "sess", _payload())
    # Reply buffered, and the dead slot pruned so it isn't retried blindly.
    assert len(s._outbox.get("sess", [])) == 1
    assert "sess" not in s._connections


def test_flush_redelivers_buffered_replies_on_reclaim():
    s = _shim()
    # Two replies buffered while the client was gone / stale.
    WebSocketPlatform._send(s, "sess", _payload("first"))
    WebSocketPlatform._send(s, "sess", _payload("second"))
    assert len(s._outbox.get("sess", [])) == 2
    # Client reconnects -> a live conn reclaims the slot -> flush.
    fresh = _FakeConn(conn_id="c2")
    WebSocketPlatform._flush_outbox(s, "sess", fresh)
    assert len(fresh.sent) == 2  # both recovered, in order
    assert s._outbox.get("sess") in (None, [])


def test_flush_rebuffers_when_the_fresh_conn_also_fails():
    s = _shim()
    WebSocketPlatform._buffer_reply(s, "sess", _payload("x"))
    still_dead = _FakeConn(fail=True)
    WebSocketPlatform._flush_outbox(s, "sess", still_dead)
    # Nothing delivered; the reply stays buffered for the next reclaim.
    assert len(s._outbox.get("sess", [])) == 1


def test_buffer_is_bounded_per_session():
    s = _shim()
    s._outbox_max = 3
    for i in range(6):
        WebSocketPlatform._buffer_reply(s, "sess", _payload(str(i)))
    assert len(s._outbox["sess"]) == 3  # capped, oldest dropped
