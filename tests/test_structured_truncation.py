"""Truncated structured output: retry once with the concise instruction.

``beta.chat.completions.parse`` raises ``openai.LengthFinishReasonError`` on
``finish_reason == "length"`` instead of returning the completion. Before the
fix the concise-retry branch never ran, the retry re-sent the same prompt and
the truncated call's (billed) usage was never recorded.
"""

from types import SimpleNamespace

import openai
import pytest
from openai.types import CompletionUsage
from pydantic import BaseModel

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler
from errors import LLMPredictionError, non_recoverable_cause
import tracking


class _Spec(BaseModel):
    name: str


def _truncation():
    usage = CompletionUsage(prompt_tokens=1000, completion_tokens=4000, total_tokens=5000)
    return openai.LengthFinishReasonError(completion=SimpleNamespace(usage=usage))


def _ok():
    message = SimpleNamespace(parsed=_Spec(name="Book"), content='{"name":"Book"}', refusal=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=1000, completion_tokens=10, total_tokens=1010),
    )


class _LLM:
    name = "gpt-4o-mini"

    def __init__(self, outcomes):
        self.requests = []
        outcomes = list(outcomes)

        def parse(**kwargs):
            self.requests.append(kwargs)
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        self.client = SimpleNamespace(
            beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))),
        )


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    tracker = tracking.TokenTracker()
    monkeypatch.setattr(tracking, "get_tracker", lambda: tracker)
    return tracker


def test_truncation_retries_with_concise_instruction_and_records_usage(_isolated):
    llm = _LLM([_truncation(), _ok()])
    handler = ClassDiagramHandler(llm)

    result = handler.predict_structured("Create a Book class", _Spec)

    assert result.name == "Book"
    assert len(llm.requests) == 2
    assert "Be CONCISE" not in llm.requests[0]["messages"][-1]["content"]
    assert "Be CONCISE" in llm.requests[1]["messages"][-1]["content"]
    # Both calls are billed: the truncated one included.
    assert _isolated.summary()["completion_tokens"] == 4010
    assert _isolated.summary()["call_count"] == 2


def test_repeated_truncation_fails_fast_with_cause():
    llm = _LLM([_truncation(), _truncation(), _ok()])
    handler = ClassDiagramHandler(llm)

    with pytest.raises(LLMPredictionError) as info:
        handler.predict_structured("Create a Book class", _Spec, max_retries=2)

    assert len(llm.requests) == 2
    # The cause stays attached so complete-system generation skips its fallback.
    assert isinstance(non_recoverable_cause(info.value), openai.LengthFinishReasonError)
