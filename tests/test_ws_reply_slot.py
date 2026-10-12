"""Reply-slot ownership across browser tabs.

Live incident 2026-10-12 13:17 UTC (prod): every editor tab of one browser
connects with the same ``?user_id=``, so they share one reply slot. Tab A sent
"generate web app" (REPLY_ROUTE -> conn cbe7ff08); two seconds later a heartbeat
from another open tab re-claimed the slot and the reply was sent to conn
480b9712. Tab A only ever showed the client's "Still working on this" notice.
"""
import json
import threading
import types

import pytest

wp = pytest.importorskip("patches.websocket_platform")
WebSocketPlatform = wp.WebSocketPlatform

from baf.platforms.payload import Payload, PayloadAction  # noqa: E402


class _FakeConn:
    def __init__(self, conn_id):
        self.id = conn_id
        self.sent = []

    def send(self, data):
        self.sent.append(data)


def _shim():
    s = types.SimpleNamespace()
    s._connections = {}
    s._slot_tab = {}
    s._outbox = {}
    s._outbox_lock = threading.Lock()
    s._outbox_max = 50
    s._claim_slot = types.MethodType(WebSocketPlatform._claim_slot, s)
    s._send = types.MethodType(WebSocketPlatform._send, s)
    s._buffer_reply = types.MethodType(WebSocketPlatform._buffer_reply, s)
    s._flush_outbox = types.MethodType(WebSocketPlatform._flush_outbox, s)
    return s


KEY = "bfacd108-5abb-495e-9044-ca753345bd31"


def test_heartbeat_from_another_tab_does_not_steal_an_in_flight_reply():
    s = _shim()
    tab_a, tab_b = _FakeConn("cbe7ff08"), _FakeConn("480b9712")
    s._claim_slot(KEY, tab_a, "tab-A", sender=True)       # "generate web app"
    assert s._claim_slot(KEY, tab_b, "tab-B", sender=False) is False  # other tab's beat
    s._send(KEY, Payload(action=PayloadAction.AGENT_REPLY_STR, message="BESSER will generate…"))
    assert len(tab_a.sent) == 1 and tab_b.sent == []


def test_same_tab_reconnect_reclaims_the_slot():
    s = _shim()
    old, new = _FakeConn("old"), _FakeConn("new")
    s._claim_slot(KEY, old, "tab-A", sender=True)
    assert s._claim_slot(KEY, new, "tab-A", sender=False) is True
    assert s._connections[KEY] is new


def test_heartbeat_claims_an_empty_slot_from_any_tab():
    s = _shim()
    conn = _FakeConn("c")
    assert s._claim_slot(KEY, conn, "tab-B", sender=False) is True
    assert s._connections[KEY] is conn


def test_a_message_always_claims_the_slot():
    s = _shim()
    a, b = _FakeConn("a"), _FakeConn("b")
    s._claim_slot(KEY, a, "tab-A", sender=True)
    assert s._claim_slot(KEY, b, "tab-B", sender=True) is True
    assert s._connections[KEY] is b


def test_unknown_owner_or_sender_tab_keeps_the_old_reclaim_behaviour():
    # Older clients send no per-tab id; their heartbeats still reclaim.
    s = _shim()
    a, b = _FakeConn("a"), _FakeConn("b")
    s._claim_slot(KEY, a, None, sender=True)
    assert s._claim_slot(KEY, b, "tab-B", sender=False) is True


def test_tab_id_is_read_from_the_raw_payload():
    raw = json.dumps({"action": "user_set_variable", "user_id": "tab-A", "message": {"__heartbeat": 1}})
    assert wp._tab_id_of(raw) == "tab-A"
    assert wp._tab_id_of(json.dumps({"action": "x", "message": "m"})) is None
    assert wp._tab_id_of("not json") is None
