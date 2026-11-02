"""The session store: every module-level per-session store is registered, and
``close`` leaves no entry for the session in any of them."""

import time

import pytest

from common.session_store import SessionStore, sessions
from conversation import session_helpers as sh
from llm.tracking import token_tracker
from memory import conversation_memory as cm

STORES = ["conversation_memory", "last_reply", "token_usage", "turn_replies"]


@pytest.fixture(autouse=True)
def _clean_stores():
    def clear():
        cm._memories.clear()
        sh._last_reply_buffer.clear()
        sh._turn_replies.clear()
        token_tracker.get_tracker().reset_all()

    clear()
    yield
    clear()


def _fill(key):
    cm.get_memory(key).add_user("hello")
    sh._last_reply_buffer[key] = {"action": "assistant_message"}
    sh._turn_replies[(key, "t1")] = {"seq": 1, "replies": []}
    sh._turn_replies[(key, "t2")] = {"seq": 1, "replies": []}
    token_tracker.get_tracker().record(10, 5, model="gpt-4o", session_id=key)


def test_every_per_session_store_is_registered():
    assert sessions.names == STORES


def test_close_leaves_no_entry_in_any_store():
    _fill("s1")
    _fill("s2")
    assert sessions.holding("s1") == STORES

    sessions.close("s1")

    assert sessions.holding("s1") == []
    assert sessions.holding("s2") == STORES
    assert token_tracker.get_tracker().summary()["call_count"] == 2  # global totals stay


def test_close_of_an_unknown_session_is_a_no_op():
    _fill("s1")
    sessions.close("nobody")
    assert sessions.holding("s1") == STORES


def test_expire_runs_the_memory_expiry_only():
    _fill("old")
    _fill("new")
    cm._memories["old"]._messages[-1].timestamp = time.time() - 7200

    sessions.expire(3600)

    assert sessions.holding("old") == ["last_reply", "token_usage", "turn_replies"]
    assert sessions.holding("new") == STORES


def test_a_failing_store_does_not_stop_the_others():
    store = SessionStore()
    dropped = []

    def broken(key):
        raise RuntimeError("boom")

    store.register("a", discard=broken, keys=lambda: [], expire=lambda age: broken(age))
    store.register("b", discard=dropped.append, keys=lambda: [], expire=lambda age: dropped.append(age))

    store.close("s1")
    store.expire(60)

    assert dropped == ["s1", 60]
