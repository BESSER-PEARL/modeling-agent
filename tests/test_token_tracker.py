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
    tracker.record(prompt_tokens=100, completion_tokens=50, model="gpt-4.1-mini")
    tracker.record(prompt_tokens=200, completion_tokens=100, model="gpt-4.1-mini")
    tracker.record(prompt_tokens=300, completion_tokens=150, model="gpt-4.1-mini")
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
    tracker.record(prompt_tokens=100, completion_tokens=50, session_id="s1", model="gpt-4.1-mini")
    tracker.record(prompt_tokens=200, completion_tokens=100, session_id="s2", model="gpt-4.1-mini")
    tracker.record(prompt_tokens=300, completion_tokens=150, session_id="s1", model="gpt-4.1-mini")

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
    tracker.record_from_usage(None, model="gpt-4.1-mini")
    s = tracker.summary()
    assert s["total_tokens"] == 0
    assert s["call_count"] == 0


# ---------------------------------------------------------------------------
# 7. reset_session
# ---------------------------------------------------------------------------

def test_reset_session(tracker):
    """After resetting a session, its summary should show zeros."""
    tracker.record(prompt_tokens=500, completion_tokens=200, session_id="s1", model="gpt-4.1-mini")
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
    tracker.record(prompt_tokens=10, completion_tokens=5, model="gpt-4.1-mini")
    s = tracker.summary()
    expected_keys = {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_prompt_tokens",
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
            tracker.record(prompt_tokens=1, completion_tokens=1, model="gpt-4.1-mini")

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
    tracker.record(prompt_tokens=0, completion_tokens=0, model="gpt-4.1-mini")
    s = tracker.summary()
    assert s["total_tokens"] == 0
    assert s["estimated_cost_usd"] == 0.0
    assert s["call_count"] == 1


# ---------------------------------------------------------------------------
# Price coverage
#
# A model nothing prices still records tokens but prices them from a
# placeholder, so every reported cost for it is wrong without any error.
# ---------------------------------------------------------------------------

def test_unknown_model_warns_once(caplog):
    import logging
    from tracking import token_tracker as tt

    tt._unknown_models_seen.discard("totally-made-up-model")
    tracker = tt.get_tracker()
    with caplog.at_level(logging.WARNING):
        tracker.record(prompt_tokens=10, completion_tokens=5, model="totally-made-up-model")
        tracker.record(prompt_tokens=10, completion_tokens=5, model="totally-made-up-model")

    warnings = [r for r in caplog.records if "totally-made-up-model" in r.getMessage()]
    assert len(warnings) == 1, f"expected exactly one warning, got {len(warnings)}"


def test_every_model_a_user_key_can_route_to_has_a_price():
    """BYOK maps each call to a provider tier model; an unpriced one reports
    placeholder costs, as happened for the Nebius Qwen model and for
    gpt-5-mini."""
    from byok import _PROVIDER_TIER_MODELS
    from tracking.token_tracker import _vendor_cost_table

    routed = {model for tiers in _PROVIDER_TIER_MODELS.values() for model in tiers.values()}
    for model in routed | {"gpt-5-mini"}:
        assert _vendor_cost_table(model) is not None, f"no price for {model!r}"


# USD per 1M tokens, input / output / cache read. The 29 rates the agent typed
# by hand before it read the price file; each must come out unchanged.
PRICES = {
    "gpt-4.1-mini": (0.4, 1.6, 0.1),
    "gpt-4.1": (2.0, 8.0, 0.5),
    "gpt-4o-mini": (0.15, 0.6, 0.075),
    "gpt-4o": (2.5, 10.0, 1.25),
    "gpt-5": (1.25, 10.0, 0.125),
    "gpt-5-mini": (0.25, 2.0, 0.025),
    "gpt-5.5": (5.0, 30.0, 0.5),
    "gpt-5.4-mini": (0.75, 4.5, 0.075),
    "gpt-5.4-nano": (0.2, 1.25, 0.02),
    # The 5.6 entries once carried the wrong list prices (sol 5/30, terra 2.5/15, luna 1/6).
    "gpt-5.6-sol": (4.0, 20.0, 0.4),
    "gpt-5.6-terra": (2.0, 12.0, 0.2),
    "gpt-5.6-luna": (0.2, 1.2, 0.02),
    "gpt-6-astra": (10.0, 50.0, 1.0),
    "gpt-6-sol": (2.0, 10.0, 0.2),
    "gpt-6-luna": (0.1, 0.5, 0.01),
    "claude-fable-5-1": (10.0, 50.0, 0.25),
    "claude-fable-5": (10.0, 50.0, 1.0),
    "claude-opus-5-5": (4.0, 20.0, 0.2),
    "claude-opus-5": (5.0, 25.0, 0.5),
    "claude-sonnet-5": (2.0, 10.0, 0.2),
    "claude-opus-4-8": (5.0, 25.0, 0.5),
    "claude-opus-4-7": (5.0, 25.0, 0.5),
    "claude-opus-4-6": (5.0, 25.0, 0.5),
    "claude-sonnet-4-6": (3.0, 15.0, 0.3),
    "claude-haiku-4-5": (1.0, 5.0, 0.1),
    # The hand table had no cache rate for these and billed cached input at the
    # full rate; the price file lists Mistral's discount.
    "mistral-large-latest": (0.5, 1.5, 0.05),
    "mistral-small-latest": (0.15, 0.6, 0.015),
    "Qwen/Qwen3-30B-A3B-Instruct-2507": (0.1, 0.3, 0.1),
    "text-embedding-3-small": (0.02, 0.0, 0.02),
}


@pytest.mark.parametrize("model,price", sorted(PRICES.items()))
def test_model_has_its_list_price(model, price):
    from tracking.token_tracker import _vendor_cost_table

    entry = _vendor_cost_table(model)
    assert entry is not None, f"{model} would be costed from placeholder pricing"
    rates = (entry["prompt"], entry["completion"], entry.get("cached", entry["prompt"]))
    assert tuple(rate * 1000 for rate in rates) == pytest.approx(price)


def test_unpublished_rows_are_not_in_the_price_file():
    """_UNPUBLISHED only bridges ids the price file lacks; once a refresh lists
    one, its row would shadow nothing and should be deleted."""
    from tracking.token_tracker import _UNPUBLISHED, _price_file

    assert _price_file(), "the price file did not load"
    listed = [model for model in _UNPUBLISHED if model in _price_file()]
    assert not listed, f"now in the price file, delete from _UNPUBLISHED: {listed}"


def test_a_missing_price_file_never_reads_as_free(monkeypatch, tmp_path):
    from tracking import token_tracker as tt

    monkeypatch.setattr(tt, "_PRICES_PATH", tmp_path / "missing.json")
    tt._price_file.cache_clear()
    tt._vendor_cost_table.cache_clear()
    try:
        assert tt._vendor_cost_table("gpt-4o") is None
    finally:
        tt._price_file.cache_clear()
        tt._vendor_cost_table.cache_clear()
