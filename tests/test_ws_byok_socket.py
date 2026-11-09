"""The user's API key lives with the socket that armed it, never on the session.

The session is never deleted on close (it keeps conversation state across
reconnects) and every tab of a browser shares it, so a key stored there stayed
in the agent's memory after the tab closed, a keyless tab could spend it, and a
new keyless tab wiped it. Every (re)connect re-arms the key from the tab's
sessionStorage, so the socket is all the storage it needs.

These drive the real ``message_handler`` with fake sockets.
"""
import json
import threading
import types

import pytest

wp = pytest.importorskip("patches.websocket_platform")

from baf.platforms.payload import PayloadAction  # noqa: E402
from llm.byok_config import current_byok  # noqa: E402
from tests.conftest import FakeSession  # noqa: E402

KEY = "bfacd108-5abb-495e-9044-ca753345bd31"
ARM = {
    "user_api_key": "sk-test-not-a-real-key",
    "user_api_provider": "openai",
    "user_api_model": "gpt-test",
}


def _set_variable(values):
    return {"action": PayloadAction.USER_SET_VARIABLE.value, "message": values}


USER_MESSAGE = {"action": PayloadAction.USER_MESSAGE.value, "message": "hello"}


class _Session(FakeSession):
    id = KEY


class _Agent:
    """Records the BYOK config each turn runs with, as the LLM layer would read it."""

    def __init__(self):
        self.session = _Session()
        self.turns = []

    def get_or_create_session(self, session_id, platform, username=None, session_name=None):
        return self.session

    def receive_event(self, event):
        self.turns.append(current_byok.get())


class _Conn:
    def __init__(self, messages):
        self.id = object()
        self.request = types.SimpleNamespace(path=f"/?user_id={KEY}", headers={})
        self._messages = [json.dumps(message) for message in messages]

    def send(self, data):
        pass

    def __iter__(self):
        return iter(self._messages)


@pytest.fixture
def platform(monkeypatch):
    monkeypatch.setattr(wp.ReceiveMessageEvent, "create_event_from", lambda **_: object())
    p = wp.WebSocketPlatform(_Agent(), use_ui=False)
    p.running = True
    return p


def _run_socket(platform, *messages):
    platform._message_handler(_Conn(messages))


def test_a_turn_on_the_arming_socket_runs_with_the_key(platform):
    _run_socket(platform, _set_variable(ARM), USER_MESSAGE)

    config = platform._agent.turns[0]
    assert (config.api_key, config.provider, config.model) == (ARM["user_api_key"], "openai", "gpt-test")


def test_the_key_never_reaches_the_session(platform):
    """BAF writes every session variable to its monitoring database, as plain JSON."""
    _run_socket(platform, _set_variable(ARM), USER_MESSAGE)

    assert not set(ARM) & set(platform._agent.session.get_dictionary())


def test_a_later_socket_of_the_session_has_no_key(platform):
    """A tab that closed: its key is gone, and a keyless tab cannot spend it."""
    _run_socket(platform, _set_variable(ARM), USER_MESSAGE)
    _run_socket(platform, USER_MESSAGE)

    assert platform._agent.turns[1] is None


def test_a_reconnect_rearms_the_key(platform):
    _run_socket(platform, _set_variable(ARM))
    _run_socket(platform, _set_variable(ARM), USER_MESSAGE)

    assert platform._agent.turns[0].api_key == ARM["user_api_key"]


def test_a_keyless_tab_does_not_wipe_another_tabs_key(platform):
    """The client's re-arm sends ``user_api_key: ''`` on a tab with no key."""
    tab_a = threading.Event()
    tab_b_done = threading.Event()

    class _HeldOpen(_Conn):
        def __iter__(self):
            yield json.dumps(_set_variable(ARM))
            tab_a.set()
            tab_b_done.wait(timeout=5)
            yield json.dumps(USER_MESSAGE)

    thread = threading.Thread(target=platform._message_handler, args=(_HeldOpen([]),), daemon=True)
    thread.start()
    assert tab_a.wait(timeout=5)

    _run_socket(platform, _set_variable({"user_api_key": ""}))
    tab_b_done.set()
    thread.join(timeout=5)

    assert platform._agent.turns[0].api_key == ARM["user_api_key"]


def test_removing_the_key_on_a_live_socket_takes_effect(platform):
    _run_socket(platform, _set_variable(ARM), _set_variable({"user_api_key": ""}), USER_MESSAGE)

    assert platform._agent.turns[0] is None


def test_other_variables_stay_on_the_session(platform):
    _run_socket(platform, _set_variable({**ARM, "_voice_context": "kept"}))

    assert platform._agent.session.get("_voice_context") == "kept"


def test_a_heartbeat_keeps_the_tabs_conversation_in_use(platform):
    """Issue #745: the heartbeat's ``user_id`` is the tab's memory key."""
    from memory.conversation_memory import cleanup_stale_memories, get_memory, remove_memory

    tab = "tab-session-745"
    mem = get_memory(tab)
    mem.add_user("my specification")
    with mem._lock:
        mem._messages[-1].timestamp -= 2 * 3600
    try:
        _run_socket(platform, {
            "action": PayloadAction.USER_SET_VARIABLE.value,
            "user_id": tab,
            "message": {"__heartbeat": 1},
        })
        cleanup_stale_memories(3600)
        assert get_memory(tab) is mem
    finally:
        remove_memory(tab)
