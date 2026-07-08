"""Retry classification — permanent billing/quota errors must fail fast.

Regression for the incident where a depleted OpenAI key (``insufficient_quota``,
delivered as a 429 ``RateLimitError``) was retried in a tight loop; under
concurrency the pile-up of workers parked on a doomed backoff took the whole
service unresponsive. Permanent codes must now short-circuit the retry.
"""
import pytest

from utilities import llm_retry


class _FakeErr(Exception):
    """Stand-in for an SDK error carrying a structured code / body / status."""

    def __init__(self, msg="err", code=None, body=None, status_code=None):
        super().__init__(msg)
        if code is not None:
            self.code = code
        if body is not None:
            self.body = body
        if status_code is not None:
            self.status_code = status_code


# ── classification ────────────────────────────────────────────────────────

def test_insufficient_quota_via_code_is_permanent():
    e = _FakeErr(code="insufficient_quota")
    assert llm_retry._permanent_error_code(e) == "insufficient_quota"
    assert llm_retry._is_transient(e) is False


def test_insufficient_quota_via_body_is_permanent():
    e = _FakeErr(body={"error": {"code": "insufficient_quota"}})
    assert llm_retry._permanent_error_code(e) == "insufficient_quota"
    assert llm_retry._is_transient(e) is False


def test_insufficient_quota_message_fallback_is_permanent():
    e = _FakeErr("Error code: 429 - You exceeded your current quota (insufficient_quota)")
    assert llm_retry._permanent_error_code(e) == "insufficient_quota"
    assert llm_retry._is_transient(e) is False


def test_invalid_api_key_is_permanent():
    assert llm_retry._is_transient(_FakeErr(code="invalid_api_key")) is False


def test_plain_429_without_permanent_code_stays_transient():
    # a genuine rate-limit (not a billing wall) must still be retried
    e = _FakeErr("rate limited", status_code=429)
    assert llm_retry._permanent_error_code(e) is None
    assert llm_retry._is_transient(e) is True


def test_5xx_stays_transient():
    assert llm_retry._is_transient(_FakeErr("server error", status_code=503)) is True


# ── with_retry behaviour ──────────────────────────────────────────────────

def test_with_retry_fails_fast_on_permanent(monkeypatch):
    monkeypatch.setattr(llm_retry.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def f():
        calls["n"] += 1
        raise _FakeErr(code="insufficient_quota")

    wrapped = llm_retry.with_retry(f, label="test")
    with pytest.raises(_FakeErr):
        wrapped()
    assert calls["n"] == 1  # called once, never retried


def test_with_retry_retries_transient_up_to_cap(monkeypatch):
    monkeypatch.setattr(llm_retry.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def f():
        calls["n"] += 1
        raise _FakeErr("rate limited", status_code=429)

    wrapped = llm_retry.with_retry(f, label="test")
    with pytest.raises(_FakeErr):
        wrapped()
    assert calls["n"] == llm_retry.MAX_ATTEMPTS
