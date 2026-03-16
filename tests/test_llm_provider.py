"""Tests for the LLM provider abstraction."""
import pytest
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm.provider import LLMProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_llm(name: str = "gpt-4.1-mini", has_client: bool = True) -> MagicMock:
    """Build a MagicMock that looks like a BESSER LLMOpenAI instance."""
    mock_llm = MagicMock()
    mock_llm.predict.return_value = "test response"
    mock_llm.name = name
    if has_client:
        mock_llm.client = MagicMock()
    else:
        # Ensure getattr(mock, 'client', None) returns None
        del mock_llm.client
    return mock_llm


# ---------------------------------------------------------------------------
# 1. Constructor
# ---------------------------------------------------------------------------

class TestConstructor:
    """LLMProvider.__init__ stores the LLM instance and model name."""

    def test_stores_model_name(self):
        mock_llm = _make_mock_llm()
        provider = LLMProvider(mock_llm, model_name="gpt-4o")
        assert provider.model_name == "gpt-4o"

    def test_default_model_name(self):
        mock_llm = _make_mock_llm()
        provider = LLMProvider(mock_llm)
        assert provider.model_name == "gpt-4.1-mini"

    def test_tracker_is_set(self):
        mock_llm = _make_mock_llm()
        provider = LLMProvider(mock_llm)
        assert provider.tracker is not None


# ---------------------------------------------------------------------------
# 2. predict() – happy path
# ---------------------------------------------------------------------------

class TestPredict:
    """predict() delegates to the underlying LLM's predict method."""

    def test_returns_llm_response(self):
        mock_llm = _make_mock_llm()
        mock_llm.predict.return_value = "hello"
        provider = LLMProvider(mock_llm)

        result = provider.predict("test")
        assert result == "hello"
        mock_llm.predict.assert_called_once_with("test")

    def test_passes_kwargs_through(self):
        mock_llm = _make_mock_llm()
        mock_llm.predict.return_value = "ok"
        provider = LLMProvider(mock_llm)

        provider.predict("prompt", temperature=0.5)
        mock_llm.predict.assert_called_once_with("prompt", temperature=0.5)

    def test_records_token_usage(self):
        mock_llm = _make_mock_llm()
        mock_llm.predict.return_value = "short"
        provider = LLMProvider(mock_llm)

        initial_count = provider.tracker.summary()["call_count"]
        provider.predict("some prompt text here")
        after_count = provider.tracker.summary()["call_count"]
        assert after_count > initial_count


# ---------------------------------------------------------------------------
# 3. predict() – error handling
# ---------------------------------------------------------------------------

class TestPredictErrors:
    """Exceptions from the underlying LLM propagate through predict()."""

    def test_exception_propagates(self):
        mock_llm = _make_mock_llm()
        mock_llm.predict.side_effect = RuntimeError("API is down")
        provider = LLMProvider(mock_llm)

        with pytest.raises(RuntimeError, match="API is down"):
            provider.predict("test")

    def test_connection_error_propagates(self):
        mock_llm = _make_mock_llm()
        mock_llm.predict.side_effect = ConnectionError("timeout")
        provider = LLMProvider(mock_llm)

        with pytest.raises(ConnectionError, match="timeout"):
            provider.predict("test")


# ---------------------------------------------------------------------------
# 4. client property
# ---------------------------------------------------------------------------

class TestClientProperty:
    """The client property exposes the LLM's underlying client."""

    def test_returns_client_attribute(self):
        mock_llm = _make_mock_llm(has_client=True)
        provider = LLMProvider(mock_llm)
        assert provider.client is mock_llm.client

    def test_returns_none_when_no_client(self):
        mock_llm = _make_mock_llm(has_client=False)
        provider = LLMProvider(mock_llm)
        assert provider.client is None


# ---------------------------------------------------------------------------
# 5. model_name property
# ---------------------------------------------------------------------------

class TestModelNameProperty:
    """model_name returns the configured model string."""

    def test_returns_configured_name(self):
        mock_llm = _make_mock_llm()
        provider = LLMProvider(mock_llm, model_name="gpt-4o")
        assert provider.model_name == "gpt-4o"

    def test_different_model_names(self):
        for name in ("gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1", "custom-model"):
            provider = LLMProvider(_make_mock_llm(), model_name=name)
            assert provider.model_name == name


# ---------------------------------------------------------------------------
# 6. parse() – fallback when client is missing
# ---------------------------------------------------------------------------

class TestParseFallback:
    """parse() falls back to predict + model_validate_json when no client."""

    def test_fallback_predict_called(self):
        """Without a client, parse() concatenates messages and calls predict."""
        mock_llm = _make_mock_llm(has_client=False)
        mock_llm.predict.return_value = '{"value": 42}'
        provider = LLMProvider(mock_llm)

        from pydantic import BaseModel

        class SimpleSchema(BaseModel):
            value: int

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Give me JSON."},
        ]
        result = provider.parse(messages, schema=SimpleSchema)
        assert result.value == 42
        # predict should have been called with the concatenated content
        call_args = mock_llm.predict.call_args[0][0]
        assert "You are helpful." in call_args
        assert "Give me JSON." in call_args

    def test_fallback_invalid_json_raises(self):
        """If predict returns non-JSON, pydantic validation should raise."""
        mock_llm = _make_mock_llm(has_client=False)
        mock_llm.predict.return_value = "not valid json"
        provider = LLMProvider(mock_llm)

        from pydantic import BaseModel

        class SimpleSchema(BaseModel):
            value: int

        with pytest.raises(Exception):  # ValidationError or JSONDecodeError
            provider.parse(
                [{"role": "user", "content": "hi"}],
                schema=SimpleSchema,
            )

    def test_fallback_when_client_has_no_beta(self):
        """Client exists but has no 'beta' attribute -> fallback path."""
        mock_llm = _make_mock_llm(has_client=True)
        # Remove beta so hasattr(client, 'beta') is False
        del mock_llm.client.beta
        mock_llm.predict.return_value = '{"value": 7}'
        provider = LLMProvider(mock_llm)

        from pydantic import BaseModel

        class SimpleSchema(BaseModel):
            value: int

        result = provider.parse(
            [{"role": "user", "content": "test"}],
            schema=SimpleSchema,
        )
        assert result.value == 7


# ---------------------------------------------------------------------------
# 7. stream() – fallback when client is missing
# ---------------------------------------------------------------------------

class TestStreamFallback:
    """stream() falls back to a single-chunk predict when no client."""

    def test_fallback_yields_single_chunk(self):
        mock_llm = _make_mock_llm(has_client=False)
        mock_llm.predict.return_value = "streamed response"
        provider = LLMProvider(mock_llm)

        messages = [{"role": "user", "content": "hello"}]
        chunks = list(provider.stream(messages))
        assert chunks == ["streamed response"]

    def test_fallback_when_client_has_no_chat(self):
        """Client exists but has no 'chat' attribute -> fallback path."""
        mock_llm = _make_mock_llm(has_client=True)
        del mock_llm.client.chat
        mock_llm.predict.return_value = "fallback stream"
        provider = LLMProvider(mock_llm)

        messages = [{"role": "user", "content": "test"}]
        chunks = list(provider.stream(messages))
        assert chunks == ["fallback stream"]

    def test_fallback_predict_receives_concatenated_messages(self):
        mock_llm = _make_mock_llm(has_client=False)
        mock_llm.predict.return_value = "ok"
        provider = LLMProvider(mock_llm)

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ]
        list(provider.stream(messages))
        call_args = mock_llm.predict.call_args[0][0]
        assert "sys" in call_args
        assert "usr" in call_args
