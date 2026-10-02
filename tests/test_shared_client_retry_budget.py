"""The shared client's transport retries must not nest under llm_retry.

The SDK default (``max_retries=2``) under the wrapper's 4 attempts turned one
transient 5xx into up to 12 HTTP requests per logical call.
"""

import httpx
import openai
import pytest

import agent_setup
from utilities import llm_retry


class _FakeLLMOpenAI:
    """Mimics BAF's LLMOpenAI: the client only exists after initialize()."""

    def __init__(self, http_client):
        self.client = None
        self._http_client = http_client

    def initialize(self):
        self.client = openai.OpenAI(api_key="test-key", http_client=self._http_client)


def test_one_transient_failure_costs_at_most_the_wrapper_attempts(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(openai.OpenAI, "_sleep_for_retry", lambda *a, **k: None, raising=False)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(500, json={"error": {"message": "upstream"}})

    llm = _FakeLLMOpenAI(httpx.Client(transport=httpx.MockTransport(handler)))
    agent_setup._enable_shared_llm_retry(llm, label="test")
    llm.initialize()

    with pytest.raises(openai.InternalServerError):
        llm.client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}],
        )

    assert len(requests) == llm_retry.MAX_ATTEMPTS
    assert llm.client.max_retries == 0
