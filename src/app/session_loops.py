"""Close each BAF session's event loop when the session ends.

BAF 4.5.3 gives every session its own asyncio loop on its own thread
(``Session._run_event_thread``). ``Session._stop_event_thread``, which
``Agent.close_session`` (the session reaper), ``Agent.delete_session``
(``reset``) and ``Agent.stop`` call, stops and joins that thread but never
closes the loop. A stopped loop still holds its selector (one epoll
descriptor on Linux) and its self-pipe (two sockets). It is freed only when
the garbage collector breaks the cycle session -> timer handle -> loop ->
handle, and CPython seldom runs a full collection in a process this large.
Production reached the 1,024-descriptor limit after six days (329 epoll
descriptors, 673 sockets), and new WebSocket connections hung.

``install()`` replaces the two methods: the session thread closes its loop
when the loop stops, and stopping wakes the loop at once
(``call_soon_threadsafe``) instead of waiting for its next timer, with a
bounded join so a session busy in a long turn cannot block the reaper. Remove
once BAF closes the loop itself; re-check if the BAF pin changes.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

from baf.core.session import Session

logger = logging.getLogger(__name__)

# How long a stop waits for the session thread. A thread still running a turn
# after this closes its loop itself when the turn returns.
STOP_JOIN_TIMEOUT_SECONDS = 30.0

# BAF's own wait for the platform before a session's loop starts.
_PLATFORM_START_TIMEOUT_SECONDS = 30.0
_PLATFORM_POLL_SECONDS = 0.05


def _close_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Close a loop that is not running; releases its selector and self-pipe."""
    if loop.is_closed():
        return
    try:
        loop.close()
    except Exception as exc:  # never let cleanup kill the session thread noisily
        logger.warning("[session-loops] closing a session loop failed: %s", exc)


def _run_event_thread(self: Session) -> None:
    """BAF's ``Session._run_event_thread``, with the loop closed when it stops."""
    loop = asyncio.new_event_loop()
    self._event_loop = loop

    def start_event_loop() -> None:
        try:
            waited = 0.0
            while not self._platform.running:
                if waited >= _PLATFORM_START_TIMEOUT_SECONDS:
                    logger.error(
                        "Platform did not start within %ss for session %s. Aborting session start.",
                        _PLATFORM_START_TIMEOUT_SECONDS, self.id,
                    )
                    return
                time.sleep(_PLATFORM_POLL_SECONDS)
                waited += _PLATFORM_POLL_SECONDS
            asyncio.set_event_loop(loop)
            loop.call_soon(self.manage_transition)
            loop.run_forever()
        finally:
            asyncio.set_event_loop(None)
            _close_loop(loop)

    thread = threading.Thread(target=start_event_loop, name=f"baf-session-{self.id}")
    self._event_thread = thread
    thread.start()


def _stop_event_thread(self: Session) -> None:
    """BAF's ``Session._stop_event_thread``: wake and stop the loop, join briefly."""
    loop, thread = self._event_loop, self._event_thread
    handle = self._timer_handle
    self._timer_handle = None
    if handle is not None:
        handle.cancel()  # drops the handle's reference to the session
    if loop is not None:
        try:
            # loop.stop() from another thread only sets a flag; the loop would
            # sleep in select() until its next timer (or forever without one).
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass  # already closed
    if thread is not None and thread is not threading.current_thread():
        thread.join(STOP_JOIN_TIMEOUT_SECONDS)
        if thread.is_alive():
            logger.warning(
                "[session-loops] session %s is still running a turn after %ss; "
                "its loop closes when the turn returns.",
                self.id, STOP_JOIN_TIMEOUT_SECONDS,
            )
    elif thread is None and loop is not None and not loop.is_running():
        _close_loop(loop)  # created but its thread never started
    self._event_loop = None
    self._event_thread = None


def install() -> None:
    """Patch BAF's ``Session`` in place; calling it again changes nothing."""
    Session._run_event_thread = _run_event_thread
    Session._stop_event_thread = _stop_event_thread
