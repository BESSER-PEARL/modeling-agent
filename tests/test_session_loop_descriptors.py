"""Closed sessions give their file descriptors back.

BAF 4.5.3 runs each session on its own asyncio loop and never closes it when
the session ends: the loop's epoll descriptor and self-pipe sockets stayed
open until a full garbage collection, which CPython seldom runs in the agent's
large heap. Production reached the 1,024-descriptor limit after six days and
new WebSocket connections hung. ``app.session_loops`` closes the loop; these
tests open and close real sessions through the vendored WebSocket platform
with the collector off (as it effectively is in production) and check the
descriptor count stays flat.
"""
import gc
import json
import os
import threading
import time

import pytest

pytest.importorskip("websockets.sync.client")
wp = pytest.importorskip("patches.websocket_platform")

from baf import CHECK_TRANSITIONS_DELAY  # noqa: E402
from baf.core.agent import Agent  # noqa: E402
from baf.core.session import Session  # noqa: E402
from baf.platforms import websocket  # noqa: E402
from websockets.sync.client import connect  # noqa: E402

from app import session_loops  # noqa: E402

SESSIONS = 40


def _idle_body(session: Session) -> None:
    pass


def _open_fds() -> int:
    return len(os.listdir("/dev/fd"))


@pytest.fixture
def patched_sessions(monkeypatch):
    # Saved so the test restores BAF's methods afterwards.
    monkeypatch.setattr(Session, "_run_event_thread", Session._run_event_thread)
    monkeypatch.setattr(Session, "_stop_event_thread", Session._stop_event_thread)
    session_loops.install()


@pytest.fixture
def no_gc():
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


@pytest.fixture
def served_agent(patched_sessions):
    agent = Agent("fd_leak_test")
    state = agent.new_state("idle", initial=True)
    state.set_body(_idle_body)
    agent.set_property(websocket.WEBSOCKET_HOST, "localhost")
    agent.set_property(websocket.WEBSOCKET_PORT, 0)  # any free port
    platform = wp.WebSocketPlatform(agent, use_ui=False)
    agent._platforms.append(platform)
    platform.initialize()
    port = platform._websocket_server.socket.getsockname()[1]
    server = threading.Thread(target=platform.start, daemon=True)
    server.start()
    deadline = time.monotonic() + 5
    while not platform.running and time.monotonic() < deadline:
        time.sleep(0.01)
    yield agent, platform, port
    for sid in list(agent._sessions):
        agent.close_session(sid)
    platform.stop()
    server.join(5)


def _open_and_close_session(port: int, user_id: str) -> None:
    with connect(f"ws://localhost:{port}/?user_id={user_id}", origin="http://localhost:3000") as ws:
        ws.send(json.dumps({"action": "user_set_variable", "message": {"__heartbeat": 1}}))


def _wait_for(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def test_reaped_sessions_release_their_descriptors(served_agent, no_gc):
    agent, platform, port = served_agent
    baseline = _open_fds()

    for rnd in range(2):
        for i in range(SESSIONS):
            _open_and_close_session(port, f"user-{rnd}-{i}")
        assert _wait_for(lambda: len(agent._sessions) == SESSIONS and not platform._connections)
        # Each open session holds a loop: one selector and two self-pipe sockets.
        assert _open_fds() >= baseline + 2 * SESSIONS
        # What the session reaper in modeling_agent.py does once the grace period ends.
        for sid in list(agent._sessions):
            agent.close_session(sid)
        assert _wait_for(lambda: _open_fds() <= baseline + 3), (
            f"round {rnd}: {_open_fds()} descriptors open after closing {SESSIONS} sessions, "
            f"{baseline} before"
        )


def test_stop_wakes_a_loop_waiting_on_a_long_timer(served_agent):
    agent, _platform, port = served_agent
    # BAF's loop.stop() from another thread waits for the loop's next timer.
    agent.set_property(CHECK_TRANSITIONS_DELAY, 60.0)
    _open_and_close_session(port, "sleepy")
    assert _wait_for(lambda: "sleepy" in agent._sessions)
    time.sleep(0.2)  # let the loop settle into its 60 s wait

    started = time.monotonic()
    agent.close_session("sleepy")
    assert time.monotonic() - started < 5


def test_stopping_from_the_session_thread_does_not_deadlock(patched_sessions):
    agent = Agent("fd_leak_self_stop")
    state = agent.new_state("idle", initial=True)
    state.set_body(_idle_body)
    platform = wp.WebSocketPlatform(agent, use_ui=False)
    platform.running = True
    agent._platforms.append(platform)
    session = agent.get_or_create_session("self", platform)
    loop, thread = session._event_loop, session._event_thread

    loop.call_soon_threadsafe(session._stop_event_thread)
    thread.join(5)
    assert not thread.is_alive()
    assert loop.is_closed()
