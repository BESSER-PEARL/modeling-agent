"""Tests for token tracking -- recording, summarization, cost estimation."""
import pytest
import sys
import os
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tracking.token_tracker import TokenTracker, get_tracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tracker():
    """Return a fresh TokenTracker for each test."""
    return TokenTracker()


class _MockUsage:
    """Mimics an OpenAI CompletionUsage object."""
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


# ---------------------------------------------------------------------------
# 1. Basic record
# ---------------------------------------------------------------------------

def test_basic_record(tracker):
    """Record tokens once and verify the summary reflects correct totals."""
    tracker.record(prompt_tokens=100, completion_tokens=50, model="gpt-4.1-mini")
    s = tracker.summary()
    assert s["prompt_tokens"] == 100
    assert s["completion_tokens"] == 50
    assert s["total_tokens"] == 150
    assert s["call_count"] == 1


# ---------------------------------------------------------------------------
# 2. Multiple records accumulate
# ---------------------------------------------------------------------------

def test_multiple_records_accumulate(tracker):
    """Recording 3 times should accumulate token counts."""
    tracker.record(prompt_tokens=100, completion_tokens=50)
    tracker.record(prompt_tokens=200, completion_tokens=100)
    tracker.record(prompt_tokens=300, completion_tokens=150)
    s = tracker.summary()
    assert s["prompt_tokens"] == 600
    assert s["completion_tokens"] == 300
    assert s["total_tokens"] == 900
    assert s["call_count"] == 3


# ---------------------------------------------------------------------------
# 3. Per-session tracking
# ---------------------------------------------------------------------------

def test_per_session_tracking(tracker):
    """Session summaries should be isolated to their own session_id."""
    tracker.record(prompt_tokens=100, completion_tokens=50, session_id="s1")
    tracker.record(prompt_tokens=200, completion_tokens=100, session_id="s2")
    tracker.record(prompt_tokens=300, completion_tokens=150, session_id="s1")

    s1 = tracker.session_summary("s1")
    s2 = tracker.session_summary("s2")

    assert s1["prompt_tokens"] == 400
    assert s1["completion_tokens"] == 200
    assert s1["total_tokens"] == 600
    assert s1["call_count"] == 2

    assert s2["prompt_tokens"] == 200
    assert s2["completion_tokens"] == 100
    assert s2["total_tokens"] == 300
    assert s2["call_count"] == 1

    # Global totals should include everything
    g = tracker.summary()
    assert g["total_tokens"] == 900


# ---------------------------------------------------------------------------
# 4. Cost estimation
# ---------------------------------------------------------------------------

def test_cost_estimation_gpt4_1_mini(tracker):
    """Verify cost for gpt-4.1-mini: $0.0004/1k prompt + $0.0016/1k completion."""
    tracker.record(prompt_tokens=1000, completion_tokens=500, model="gpt-4.1-mini")
    s = tracker.summary()
    # Expected: (1000/1000)*0.0004 + (500/1000)*0.0016 = 0.0004 + 0.0008 = 0.0012
    assert s["estimated_cost_usd"] == pytest.approx(0.0012, abs=1e-7)


# ---------------------------------------------------------------------------
# 5. Unknown model
# ---------------------------------------------------------------------------

def test_unknown_model_no_crash(tracker):
    """Recording tokens for an unrecognized model should not crash and should
    fall back to the default cost table."""
    tracker.record(prompt_tokens=100, completion_tokens=50, model="unknown-model")
    s = tracker.summary()
    assert s["total_tokens"] == 150
    # Default cost: $0.001/1k prompt + $0.004/1k completion
    # (100/1000)*0.001 + (50/1000)*0.004 = 0.0001 + 0.0002 = 0.0003
    assert s["estimated_cost_usd"] == pytest.approx(0.0003, abs=1e-7)


# ---------------------------------------------------------------------------
# 6. record_from_usage
# ---------------------------------------------------------------------------

def test_record_from_usage(tracker):
    """record_from_usage should extract tokens from an object with
    prompt_tokens and completion_tokens attributes."""
    usage = _MockUsage(prompt_tokens=250, completion_tokens=100)
    tracker.record_from_usage(usage, model="gpt-4.1-mini")
    s = tracker.summary()
    assert s["prompt_tokens"] == 250
    assert s["completion_tokens"] == 100
    assert s["total_tokens"] == 350
    assert s["call_count"] == 1


def test_record_from_usage_none(tracker):
    """record_from_usage(None) should silently do nothing."""
    tracker.record_from_usage(None)
    s = tracker.summary()
    assert s["total_tokens"] == 0
    assert s["call_count"] == 0


# ---------------------------------------------------------------------------
# 7. reset_session
# ---------------------------------------------------------------------------

def test_reset_session(tracker):
    """After resetting a session, its summary should show zeros."""
    tracker.record(prompt_tokens=500, completion_tokens=200, session_id="s1")
    tracker.reset_session("s1")
    s1 = tracker.session_summary("s1")
    assert s1["total_tokens"] == 0
    assert s1["call_count"] == 0

    # Global totals should still retain the data
    g = tracker.summary()
    assert g["total_tokens"] == 700


# ---------------------------------------------------------------------------
# 8. summary format
# ---------------------------------------------------------------------------

def test_summary_format(tracker):
    """summary() should return a dict with all expected keys."""
    tracker.record(prompt_tokens=10, completion_tokens=5)
    s = tracker.summary()
    expected_keys = {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "call_count",
        "cache_hits",
        "active_sessions",
    }
    assert set(s.keys()) == expected_keys


# ---------------------------------------------------------------------------
# 9. get_tracker singleton
# ---------------------------------------------------------------------------

def test_get_tracker_singleton():
    """get_tracker() should return the same instance on repeated calls."""
    t1 = get_tracker()
    t2 = get_tracker()
    assert t1 is t2


# ---------------------------------------------------------------------------
# 10. Thread safety
# ---------------------------------------------------------------------------

def test_thread_safety(tracker):
    """Concurrent record() calls should not lose data."""
    num_threads = 10
    calls_per_thread = 100
    barrier = threading.Barrier(num_threads)

    def worker():
        barrier.wait()
        for _ in range(calls_per_thread):
            tracker.record(prompt_tokens=1, completion_tokens=1)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    s = tracker.summary()
    expected_total = num_threads * calls_per_thread * 2  # 1 prompt + 1 completion each
    assert s["total_tokens"] == expected_total
    assert s["call_count"] == num_threads * calls_per_thread


# ---------------------------------------------------------------------------
# 11. Zero tokens
# ---------------------------------------------------------------------------

def test_zero_tokens(tracker):
    """Recording 0 prompt + 0 completion tokens should not crash."""
    tracker.record(prompt_tokens=0, completion_tokens=0)
    s = tracker.summary()
    assert s["total_tokens"] == 0
    assert s["estimated_cost_usd"] == 0.0
    assert s["call_count"] == 1
