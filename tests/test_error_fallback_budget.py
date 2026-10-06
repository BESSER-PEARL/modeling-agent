"""A provider error must not fan out into the per-class incremental fallback.

Before the fix, an ``insufficient_quota`` 429 on class-diagram complete-system
generation fell into ``_incremental_system_fallback``: one extraction call plus
one call per class (7 calls for a five-entity request), stub classes injected,
and the rate-limit error the caller turns into "add your API key" was lost.
"""

import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler
from errors import ErrorCode, classify_error

REQUEST = "Create a library system with books, members, loans, authors and branches"
_QUOTA_BODY = {"error": {
    "message": "You exceeded your current quota. See "
               "https://platform.openai.com/docs/guides/error-codes/api-errors.",
    "type": "insufficient_quota", "code": "insufficient_quota",
}}


def _quota_error() -> openai.RateLimitError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.RateLimitError(
        f"Error code: 429 - {_QUOTA_BODY}",
        response=httpx.Response(429, request=request), body=_QUOTA_BODY,
    )


class _CountingLLM:
    """BAF-style LLM without a client: every call goes through ``predict``."""

    name = "gpt-4o-mini"

    def __init__(self, error=None, response=""):
        self.calls = 0
        self._error = error
        self._response = response

    def predict(self, prompt):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._response


class _ParseClientLLM:
    """LLM whose OpenAI client exposes ``beta.chat.completions.parse``."""

    name = "gpt-4o-mini"

    def __init__(self, error):
        self.calls = 0

        def parse(**kwargs):
            self.calls += 1
            raise error

        def create(**kwargs):
            self.calls += 1
            raise error

        self.client = SimpleNamespace(
            beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))),
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)


@pytest.mark.parametrize("llm_factory", [_CountingLLM, _ParseClientLLM])
def test_quota_error_surfaces_after_one_call(llm_factory):
    llm = llm_factory(error=_quota_error())
    handler = ClassDiagramHandler(llm)

    with pytest.raises(openai.RateLimitError):
        handler.generate_complete_system(REQUEST, raw_request=REQUEST)

    assert llm.calls == 1


def test_quota_error_on_two_pass_reasoning_is_not_retried_single_pass():
    # Long requests take the two-pass path; a failed reasoning pass used to
    # fall back to a single-pass call that hit the same 429.
    long_request = REQUEST + ". " + "Members borrow books from branches and pay fines. " * 6
    llm = _CountingLLM(error=_quota_error())
    handler = ClassDiagramHandler(llm)

    with pytest.raises(openai.RateLimitError):
        handler.generate_complete_system(long_request, raw_request=long_request)

    assert llm.calls == 1


def test_quota_error_is_classified_as_rate_limit():
    # model_operations keys the "add your own API key" reply on this code.
    assert classify_error(_quota_error()) == ErrorCode.RATE_LIMIT


def test_authentication_error_is_classified_as_auth_error():
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    exc = openai.AuthenticationError(
        "Error code: 401 - Incorrect API key provided",
        response=httpx.Response(401, request=request), body=None,
    )
    assert classify_error(exc) == ErrorCode.AUTH_ERROR


def test_content_failure_still_uses_incremental_fallback():
    # A response that never parses is a content failure: the fallback stays.
    llm = _CountingLLM(response=json.dumps(["Book", "Member"]))
    handler = ClassDiagramHandler(llm)

    result = handler.generate_complete_system(REQUEST, raw_request=REQUEST)

    assert result["action"] == "inject_complete_system"
    assert {c["className"] for c in result["systemSpec"]["classes"]} == {"Book", "Member"}
