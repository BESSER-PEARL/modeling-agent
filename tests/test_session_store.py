"""The session store: the per-session stores that expire entries register their
expiry, and the reaper's one ``expire`` runs them all."""

import logging
import time

import pytest

from common.session_store import SessionStore, sessions
from memory import conversation_memory as cm


@pytest.fixture(autouse=True)
def _clean_memories():
    cm._memories.clear()
    yield
    cm._memories.clear()


def test_the_memory_is_the_only_expiring_store():
    # The reply buffers are bounded LRUs; token usage per session is never filled.
    assert sessions.names == ["conversation_memory"]


def test_expire_drops_stale_memories_only():
    cm.get_memory("old").add_user("hello")
    cm.get_memory("new").add_user("hello")
    cm._memories["old"]._messages[-1].timestamp = time.time() - 7200

    sessions.expire(3600)

    assert sorted(cm._memories) == ["new"]


def test_a_failing_store_is_logged_at_warning_and_does_not_stop_the_others(caplog):
    store = SessionStore()
    expired = []

    def broken(age):
        raise RuntimeError("boom")

    store.register("a", expire=broken)
    store.register("b", expire=expired.append)

    with caplog.at_level(logging.DEBUG, logger="common.session_store"):
        store.expire(60)

    assert expired == [60]
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (logging.WARNING, "[SessionStore] a: expiry failed: boom"),
    ]
