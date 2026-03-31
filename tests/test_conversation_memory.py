"""Tests for conversation memory -- sliding window, summarization, cleanup."""
import pytest
import sys
import os
import time
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from memory.conversation_memory import (
    ConversationMemory,
    get_memory,
    remove_memory,
    cleanup_stale_memories,
    _memories,
    _memories_lock,
    _SUMMARY_TRIGGER,
    _DEFAULT_WINDOW_SIZE,
)


@pytest.fixture(autouse=True)
def _clear_registry():
    """Ensure the global memory registry is clean before and after each test."""
    with _memories_lock:
        _memories.clear()
    yield
    with _memories_lock:
        _memories.clear()


# ------------------------------------------------------------------
# 1. Basic add / retrieve
# ------------------------------------------------------------------

def test_basic_add_and_retrieve():
    mem = ConversationMemory("s-basic")
    mem.add_user("Hello")
    mem.add_assistant("Hi there")
    mem.add_user("Create a class diagram")

    ctx = mem.build_context()
    assert len(ctx) == 3
    assert ctx[0] == {"role": "user", "content": "Hello"}
    assert ctx[1] == {"role": "assistant", "content": "Hi there"}
    assert ctx[2] == {"role": "user", "content": "Create a class diagram"}


# ------------------------------------------------------------------
# 2. Sliding window
# ------------------------------------------------------------------

def test_sliding_window():
    mem = ConversationMemory("s-window", window_size=20)
    for i in range(25):
        mem.add_user(f"msg-{i}")

    # Total messages added is 25
    assert mem.message_count == 25
    # Window only keeps the most recent 20
    assert mem.window_count == 20

    ctx = mem.build_context()
    # Without a summarizer, no summary is prepended -- only window messages
    contents = [m["content"] for m in ctx]
    assert contents[0] == "msg-5"
    assert contents[-1] == "msg-24"


# ------------------------------------------------------------------
# 3. get_last_n
# ------------------------------------------------------------------

def test_get_last_n():
    mem = ConversationMemory("s-lastn")
    for i in range(10):
        mem.add_user(f"msg-{i}")

    result = mem.get_last_n(3)
    assert len(result) == 3
    assert result[0]["content"] == "msg-7"
    assert result[1]["content"] == "msg-8"
    assert result[2]["content"] == "msg-9"


# ------------------------------------------------------------------
# 4. Summary generation with mock summarizer
# ------------------------------------------------------------------

def test_summary_generation():
    mock_summarizer = lambda text: "SUMMARY"  # noqa: E731
    mem = ConversationMemory("s-summary", window_size=20, summarizer=mock_summarizer)

    # Fill enough to trigger compression: need len >= _SUMMARY_TRIGGER (16)
    # AND len >= window_size - 2 (18).  With window_size=20 the deque maxlen
    # is 20, so filling 19 messages means the 19th add sees len==18 which
    # satisfies both conditions.
    for i in range(19):
        mem.add_user(f"msg-{i}")

    assert mem.get_summary() == "SUMMARY"


# ------------------------------------------------------------------
# 5. Summary without summarizer -- no crash
# ------------------------------------------------------------------

def test_summary_without_summarizer():
    mem = ConversationMemory("s-no-summarizer", window_size=20)
    for i in range(25):
        mem.add_user(f"msg-{i}")

    # Should not crash, and no summary should be set
    assert mem.get_summary() == ""
    assert mem.window_count == 20


# ------------------------------------------------------------------
# 6. build_context with summary prepended
# ------------------------------------------------------------------

def test_build_context_with_summary():
    mock_summarizer = lambda text: "SUMMARY OF CONVERSATION"  # noqa: E731
    mem = ConversationMemory("s-ctx-summary", window_size=20, summarizer=mock_summarizer)

    # Trigger compression
    for i in range(19):
        mem.add_user(f"msg-{i}")

    ctx = mem.build_context()

    # First message should be the summary as a system message
    assert ctx[0]["role"] == "system"
    assert "SUMMARY OF CONVERSATION" in ctx[0]["content"]

    # Remaining messages are the window contents
    non_system = [m for m in ctx if m["role"] != "system"]
    assert len(non_system) > 0
    assert all(m["role"] == "user" for m in non_system)


# ------------------------------------------------------------------
# 7. clear()
# ------------------------------------------------------------------

def test_clear():
    mem = ConversationMemory("s-clear")
    mem.add_user("hello")
    mem.add_assistant("hi")
    assert mem.message_count == 2

    mem.clear()
    assert mem.message_count == 0
    assert mem.window_count == 0
    assert mem.get_summary() == ""
    assert mem.build_context() == []


# ------------------------------------------------------------------
# 8. to_dict()
# ------------------------------------------------------------------

def test_to_dict():
    mem = ConversationMemory("s-dict")
    mem.add_user("hello")
    mem.add_assistant("hi")

    d = mem.to_dict()
    assert d["session_id"] == "s-dict"
    assert d["total_messages"] == 2
    assert d["window_count"] == 2
    assert d["has_summary"] is False
    assert d["summary"] == ""
    assert isinstance(d["recent_messages"], list)
    assert len(d["recent_messages"]) == 2
    assert d["recent_messages"][0]["role"] == "user"
    assert d["recent_messages"][1]["role"] == "assistant"


# ------------------------------------------------------------------
# 9. get_memory registry -- same instance returned
# ------------------------------------------------------------------

def test_get_memory_same_instance():
    m1 = get_memory("s1")
    m2 = get_memory("s1")
    assert m1 is m2


# ------------------------------------------------------------------
# 10. remove_memory -- new instance after removal
# ------------------------------------------------------------------

def test_remove_memory():
    m1 = get_memory("s1")
    m1.add_user("important data")

    remove_memory("s1")

    m2 = get_memory("s1")
    assert m2 is not m1
    assert m2.message_count == 0


# ------------------------------------------------------------------
# 11. cleanup_stale_memories
# ------------------------------------------------------------------

def test_cleanup_stale_memories():
    mem = get_memory("stale-session")
    mem.add_user("old message")

    # Manually set the last message's timestamp to 2 days ago
    with mem._lock:
        mem._messages[-1].timestamp = time.time() - 2 * 86400

    cleaned = cleanup_stale_memories(max_age_seconds=86400)
    assert cleaned == 1

    # The session should be gone -- get_memory returns a fresh instance
    m2 = get_memory("stale-session")
    assert m2.message_count == 0


# ------------------------------------------------------------------
# 12. Thread safety
# ------------------------------------------------------------------

def test_thread_safety():
    mem = ConversationMemory("s-threaded", window_size=200)
    num_threads = 10
    msgs_per_thread = 50

    def add_messages(thread_id):
        for i in range(msgs_per_thread):
            mem.add_user(f"t{thread_id}-msg-{i}")

    threads = [threading.Thread(target=add_messages, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert mem.message_count == num_threads * msgs_per_thread
    assert mem.window_count == 200  # window capped at 200


# ------------------------------------------------------------------
# 13. Empty memory
# ------------------------------------------------------------------

def test_empty_memory():
    mem = ConversationMemory("s-empty")
    assert mem.build_context() == []
    assert mem.message_count == 0
    assert mem.window_count == 0
    assert mem.get_summary() == ""


# ------------------------------------------------------------------
# 14. Message metadata
# ------------------------------------------------------------------

def test_message_metadata():
    mem = ConversationMemory("s-meta")
    mem.add_user("hello", diagram_type="class", version=2)

    d = mem.to_dict()
    assert d["total_messages"] == 1

    # Verify metadata stored on the internal _Message object
    with mem._lock:
        msg = mem._messages[0]
        assert msg.metadata["diagram_type"] == "class"
        assert msg.metadata["version"] == 2
