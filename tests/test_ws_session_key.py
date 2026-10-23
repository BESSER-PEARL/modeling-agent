"""Session keying on connect.

The editor connects with only ``?user_id=<stable id>`` (AssistantClient.ts).
Stock BAF 4.5 keys a session by ``?session_id=`` and otherwise by the
connection id, so every reconnect would open a new, empty session and the
reply slot the heartbeat reclaims would never match.
"""
import types

import pytest

wp = pytest.importorskip("patches.websocket_platform")
WebSocketPlatform = wp.WebSocketPlatform


class _Agent:
    def __init__(self):
        self.keys = []

    def get_or_create_session(self, session_key, platform, username=None, session_name=None):
        self.keys.append(session_key)
        return types.SimpleNamespace(id=session_key)


class _Conn:
    def __init__(self, conn_id, path):
        self.id = conn_id
        self.request = types.SimpleNamespace(path=path, headers={})

    def __iter__(self):
        return iter(())


def _connect(platform, conn_id, path):
    platform._message_handler(_Conn(conn_id, path))


def test_reconnects_with_the_same_user_id_share_one_session():
    agent = _Agent()
    platform = WebSocketPlatform(agent, use_ui=False)
    _connect(platform, "conn-1", "/?user_id=bfacd108")
    _connect(platform, "conn-2", "/?user_id=bfacd108")
    assert agent.keys == ["bfacd108", "bfacd108"]


def test_an_explicit_session_id_still_wins():
    agent = _Agent()
    platform = WebSocketPlatform(agent, use_ui=False)
    _connect(platform, "conn-1", "/?user_id=bfacd108&session_id=s-42")
    assert agent.keys == ["s-42"]


def test_no_identity_falls_back_to_the_connection():
    agent = _Agent()
    platform = WebSocketPlatform(agent, use_ui=False)
    _connect(platform, "conn-1", "/")
    assert agent.keys == ["conn-1"]
