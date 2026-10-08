"""Token/cost accounting matches what the providers bill.

Before the fix: cached prompt tokens were billed at the full rate, a BYOK
free-text call was counted twice (real usage plus a len/4 estimate), the
shared client's free-text calls were only ever estimated, a truncated
classifier parse lost its usage, dated / gateway model ids fell back to
placeholder pricing, and nothing reported the totals in the logs.
"""

import logging
from types import SimpleNamespace

import openai
import pytest
from openai.types import CompletionUsage
from openai.types.completion_usage import PromptTokensDetails
from pydantic import BaseModel

import byok
import tracking
from tracking import token_tracker
from tracking.token_tracker import TokenTracker


def _placeholder_cost(prompt, completion):
    rates = token_tracker._DEFAULT_COST
    return prompt / 1000 * rates["prompt"] + completion / 1000 * rates["completion"]


@pytest.fixture
def tracker(monkeypatch):
    fresh = TokenTracker()
    monkeypatch.setattr(tracking, "get_tracker", lambda: fresh)
    return fresh


def _usage(prompt, completion, cached=0):
    return CompletionUsage(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=cached),
    )


def _completion(text, usage):
    message = SimpleNamespace(content=text, refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage)


def test_cached_prompt_tokens_are_billed_at_the_cached_rate():
    t = TokenTracker()
    # gpt-4o-mini: $0.15/M input, $0.075/M cached, $0.60/M output.
    t.record_from_usage(_usage(10_000, 1_000, cached=8_000), model="gpt-4o-mini")

    expected = 2_000 * 0.15e-6 + 8_000 * 0.075e-6 + 1_000 * 0.60e-6
    assert t.summary()["estimated_cost_usd"] == pytest.approx(expected)
    assert t.summary()["cached_prompt_tokens"] == 8_000


def test_anthropic_cache_reads_and_writes_are_billed_separately(tracker):
    usage = SimpleNamespace(
        input_tokens=200, output_tokens=100,
        cache_read_input_tokens=9_000, cache_creation_input_tokens=0,
    )
    byok.BYOKClient._track_anthropic(usage, "claude-haiku-4-5")
    write = SimpleNamespace(
        input_tokens=200, output_tokens=100,
        cache_read_input_tokens=0, cache_creation_input_tokens=9_000,
    )
    byok.BYOKClient._track_anthropic(write, "claude-haiku-4-5")

    # haiku 4.5: $1/M input, $0.10/M cache read, 1.25x input for a cache write, $5/M output.
    read_cost = 200e-6 + 9_000 * 0.1e-6 + 100 * 5e-6
    write_cost = 200e-6 + 9_000 * 1.25e-6 + 100 * 5e-6
    summary = tracker.summary()
    assert summary["prompt_tokens"] == 2 * 9_200
    assert summary["estimated_cost_usd"] == pytest.approx(read_cost + write_cost)


@pytest.mark.parametrize("model, base", [
    ("claude-haiku-4-5-20251001", "claude-haiku-4-5"),
    ("us.anthropic.claude-sonnet-5", "claude-sonnet-5"),
    ("anthropic.claude-haiku-4-5-20251001-v1:0", "claude-haiku-4-5"),
    ("gpt-4o-mini-2024-07-18", "gpt-4o-mini"),
    ("text-embedding-3-small", "text-embedding-3-small"),
])
def test_dated_and_gateway_ids_price_as_their_base_model(model, base):
    dated, canonical = TokenTracker(), TokenTracker()
    dated.record(prompt_tokens=10_000, completion_tokens=1_000, model=model)
    canonical.record(prompt_tokens=10_000, completion_tokens=1_000, model=base)

    assert dated.summary()["estimated_cost_usd"] == canonical.summary()["estimated_cost_usd"]
    assert canonical.summary()["estimated_cost_usd"] != _placeholder_cost(10_000, 1_000)


def test_a_different_variant_does_not_inherit_its_base_price():
    variant = TokenTracker()
    variant.record(prompt_tokens=10_000, completion_tokens=1_000, model="gpt-5.5-pro")
    assert variant.summary()["estimated_cost_usd"] == _placeholder_cost(10_000, 1_000)


class _FakeOpenAI:
    """Stands in for ``openai.OpenAI`` inside the BYOK client."""

    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: _completion("a reply", _usage(1_234, 56)),
        ))


def test_byok_free_text_call_is_counted_once_with_real_usage(tracker, monkeypatch):
    from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    token = byok.set_current("openai", "sk-test")
    try:
        ClassDiagramHandler(SimpleNamespace(name="gpt-4o-mini", client=object())).predict_with_retry(
            "x" * 4000, model="gpt-5-mini",
        )
    finally:
        byok.reset_current(token)

    summary = tracker.summary()
    assert summary["call_count"] == 1
    assert summary["prompt_tokens"] == 1_234


def test_shared_free_text_call_records_real_usage(tracker):
    from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: _completion("a reply", _usage(1_234, 56, cached=1_024)),
    )))
    handler = ClassDiagramHandler(SimpleNamespace(name="gpt-4o-mini", client=client))

    handler.predict_with_retry("x" * 4000, model="gpt-5-mini")

    summary = tracker.summary()
    assert summary["call_count"] == 1
    assert (summary["prompt_tokens"], summary["cached_prompt_tokens"]) == (1_234, 1_024)


def test_truncated_classifier_parse_still_records_usage(tracker):
    from llm.provider import LLMProvider

    class _Out(BaseModel):
        intent: str

    def parse(**kwargs):
        raise openai.LengthFinishReasonError(completion=SimpleNamespace(usage=_usage(9_000, 512)))

    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(parse=parse),
    )))
    provider = LLMProvider(SimpleNamespace(client=client))
    provider.tracker = tracker

    with pytest.raises(openai.LengthFinishReasonError):
        provider.parse([{"role": "user", "content": "hi"}], _Out)

    assert tracker.summary()["completion_tokens"] == 512


def test_totals_are_logged_periodically(monkeypatch, caplog):
    monkeypatch.setattr(token_tracker, "_SUMMARY_INTERVAL_SECONDS", 60.0)
    now = [1_000.0]
    monkeypatch.setattr(token_tracker.time, "time", lambda: now[0])
    t = TokenTracker()

    with caplog.at_level(logging.INFO, logger=token_tracker.__name__):
        now[0] = 1_010.0
        t.record(prompt_tokens=100, completion_tokens=10, model="gpt-4o-mini")
        now[0] = 1_070.0
        t.record(prompt_tokens=100, completion_tokens=10, model="gpt-4o-mini")

    lines = [r.getMessage() for r in caplog.records if "totals since start" in r.getMessage()]
    assert len(lines) == 1
    assert "calls=2" in lines[0]
