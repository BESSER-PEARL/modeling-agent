"""A model that declines a request must be reported as a refusal, not a failure.

Claude (Fable 5.1, Opus 5.5) answers a declined request with HTTP 200,
``stop_reason: "refusal"``, an optional ``stop_details.category`` and possibly
empty content; OpenAI sets ``message.refusal`` or stops with
``finish_reason: "content_filter"``. Before the fix those came back as an empty
string or a parse error: the agent retried the doomed call and the user saw
"I couldn't generate that ...", with no hint that the model had declined.

The assertions check behaviour (exception name, call counts, user text) rather
than importing the new class, so they fail against the pre-fix code for the
reason they exist, not with an ImportError.
"""
import os
import sys
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import byok  # noqa: E402
from diagram_handlers.core.base_handler import LLMPredictionError  # noqa: E402
from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler  # noqa: E402

CYBER_MESSAGE = (
    "The AI model declined this request (safety category: cyber). Please try "
    "rephrasing it, or switch to a different model in the assistant settings."
)


class _Answer(BaseModel):
    name: str


# ---------------------------------------------------------------------------
# Response fakes shaped like the SDK objects
# ---------------------------------------------------------------------------

def _anthropic_message(*, stop_reason="end_turn", text="", category=None):
    content = [SimpleNamespace(type="text", text=text)] if text else []
    stop_details = (
        SimpleNamespace(type="refusal", category=category, explanation="declined by policy")
        if stop_reason == "refusal" else None
    )
    usage = SimpleNamespace(input_tokens=10, output_tokens=0)
    return SimpleNamespace(
        stop_reason=stop_reason, stop_details=stop_details, content=content, usage=usage,
    )


class _AnthropicSDK:
    def __init__(self, message):
        self.calls = 0

        def create(**_kwargs):
            self.calls += 1
            return message

        self.messages = SimpleNamespace(create=create)


def _anthropic_byok(message):
    client = object.__new__(byok.BYOKClient)
    client.provider = "anthropic"
    client._user_model = "claude-opus-5-5"
    client._base_url = None
    client._client = _AnthropicSDK(message)
    return client


def _openai_choice(*, content=None, refusal=None, finish_reason="stop", parsed=None):
    message = SimpleNamespace(content=content, refusal=refusal, parsed=parsed)
    return SimpleNamespace(message=message, finish_reason=finish_reason)


class _OpenAISDK:
    """Chat-completions client (``create``) plus structured ``beta...parse``."""

    def __init__(self, choice):
        self.calls = 0
        completion = SimpleNamespace(choices=[choice], usage=None)

        def call(**_kwargs):
            self.calls += 1
            return completion

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=call))
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=call)))


def _openai_byok(choice):
    client = object.__new__(byok.BYOKClient)
    client.provider = "openai"
    client._user_model = "gpt-5.5"
    client._base_url = None
    client._client = _OpenAISDK(choice)
    return client


class _ServerLLM:
    name = "gpt-5.5"

    def __init__(self, client=None):
        self.client = client

    def predict(self, prompt):  # pragma: no cover - not reached in these tests
        raise AssertionError("the BAF predict path must not be used")


@pytest.fixture
def with_byok(monkeypatch):
    tokens = []

    def _activate(client):
        tokens.append(byok.set_current(client.provider, "sk-user", None))
        monkeypatch.setattr(byok, "get_active_client", lambda: client)

    yield _activate
    for token in tokens:
        byok.reset_current(token)


# ---------------------------------------------------------------------------
# Anthropic (BYOK messages API)
# ---------------------------------------------------------------------------

def test_anthropic_refusal_raises_with_its_category():
    client = _anthropic_byok(_anthropic_message(stop_reason="refusal", category="cyber"))

    with pytest.raises(LLMPredictionError) as info:
        client.predict_raw("design a port scanner", json_mode=True)

    assert type(info.value).__name__ == "ModelRefusal"
    assert info.value.category == "cyber"
    assert info.value.provider == "anthropic"


def test_anthropic_refusal_with_partial_text_is_still_a_refusal():
    client = _anthropic_byok(
        _anthropic_message(stop_reason="refusal", text='{"name": "par', category=None)
    )

    with pytest.raises(LLMPredictionError) as info:
        client.predict_raw("x", json_mode=True)

    assert type(info.value).__name__ == "ModelRefusal"
    assert info.value.category is None


def test_anthropic_stop_details_as_plain_dict_is_read():
    # An SDK that does not model stop_details yet exposes the raw dict.
    message = _anthropic_message(stop_reason="refusal")
    message.stop_details = {"type": "refusal", "category": "bio", "explanation": None}
    client = _anthropic_byok(message)

    with pytest.raises(LLMPredictionError) as info:
        client.predict_raw("x")

    assert info.value.category == "bio"


def test_anthropic_normal_json_is_unchanged():
    client = _anthropic_byok(_anthropic_message(text='```json\n{"name": "Library"}\n```'))

    assert client.predict_raw("x", json_mode=True) == '{"name": "Library"}'


def test_anthropic_refusal_is_not_retried_and_reaches_the_user(with_byok):
    client = _anthropic_byok(_anthropic_message(stop_reason="refusal", category="cyber"))
    with_byok(client)

    result = ClassDiagramHandler(_ServerLLM()).generate_single_element("a Malware class")

    assert client._client.calls == 1, "a refusal must not be retried"
    assert result["action"] == "assistant_message"
    assert result["errorCode"] == "model_refusal"
    assert result["retryable"] is False
    assert result["message"] == CYBER_MESSAGE


def test_anthropic_normal_structured_response_is_unchanged(with_byok):
    client = _anthropic_byok(_anthropic_message(text='{"name": "Library"}'))
    with_byok(client)

    parsed = ClassDiagramHandler(_ServerLLM()).predict_structured("x", _Answer, max_retries=0)

    assert parsed.name == "Library"
    assert client._client.calls == 1


# ---------------------------------------------------------------------------
# OpenAI (structured outputs + chat completions)
# ---------------------------------------------------------------------------

def test_openai_structured_refusal_reaches_the_user_as_a_refusal():
    sdk = _OpenAISDK(_openai_choice(refusal="I'm sorry, I can't help with that."))

    result = ClassDiagramHandler(_ServerLLM(sdk)).generate_single_element("a Weapon class")

    assert sdk.calls == 1
    assert result["errorCode"] == "model_refusal"
    assert result["message"].startswith("The AI model declined this request.")
    assert "I'm sorry" not in result["message"], "raw model output must not leak"


def test_openai_content_filter_is_a_refusal():
    client = _openai_byok(_openai_choice(content="", finish_reason="content_filter"))

    with pytest.raises(LLMPredictionError) as info:
        client.predict_raw("x")

    assert type(info.value).__name__ == "ModelRefusal"
    assert info.value.category == "content_filter"


def test_openai_message_refusal_is_a_refusal():
    client = _openai_byok(_openai_choice(refusal="I can't assist with that."))

    with pytest.raises(LLMPredictionError) as info:
        client.predict_raw("x")

    assert type(info.value).__name__ == "ModelRefusal"
    assert info.value.detail == "I can't assist with that."


def test_openai_normal_json_is_unchanged():
    client = _openai_byok(_openai_choice(content='{"name": "Library"}'))

    assert client.predict_raw("x", json_mode=True) == '{"name": "Library"}'


def test_openai_structured_normal_response_is_unchanged():
    sdk = _OpenAISDK(_openai_choice(parsed=_Answer(name="Library"), content='{"name": "Library"}'))

    parsed = ClassDiagramHandler(_ServerLLM(sdk)).predict_structured("x", _Answer, max_retries=0)

    assert parsed.name == "Library"


def test_classifier_parse_raises_a_refusal_not_a_value_error():
    from llm.provider import LLMProvider

    sdk = _OpenAISDK(_openai_choice(refusal="No."))
    provider = LLMProvider(_ServerLLM(sdk), model_name="gpt-4o-mini")

    with pytest.raises(Exception) as info:
        provider.parse([{"role": "user", "content": "hi"}], schema=_Answer)

    assert type(info.value).__name__ == "ModelRefusal"


def test_modification_refusal_skips_the_incremental_fallback():
    # A refused complete-system request used to fall through to the incremental
    # fallback, which re-sends the same request as several more calls.
    sdk = _OpenAISDK(_openai_choice(refusal="No."))
    handler = ClassDiagramHandler(_ServerLLM(sdk))

    result = handler.generate_complete_system("a botnet command server " * 20)

    assert result["errorCode"] == "model_refusal"
    assert sdk.calls == 1


# ---------------------------------------------------------------------------
# Streaming conversational replies
# ---------------------------------------------------------------------------

def _stream_client(events):
    def create(**_kwargs):
        return iter(events)

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _event(*, content=None, refusal=None, finish_reason=None):
    delta = SimpleNamespace(content=content, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)], usage=None)


@pytest.mark.parametrize("events", [
    [_event(refusal="I can't "), _event(refusal="help with that."), _event(finish_reason="stop")],
    [_event(finish_reason="content_filter")],
], ids=["delta-refusal", "content-filter"])
def test_streamed_refusal_tells_the_user_the_model_declined(events):
    from session_helpers import stream_llm_response
    from tests.conftest import FakeSession

    llm = SimpleNamespace(client=_stream_client(events), name="gpt-4o-mini")
    text = stream_llm_response(FakeSession(), llm, "describe my model")

    assert text.startswith("The AI model declined this request")
    assert "I can't" not in text


def test_streamed_normal_reply_is_unchanged():
    from session_helpers import stream_llm_response
    from tests.conftest import FakeSession

    events = [_event(content="Your model has "), _event(content="3 classes."), _event(finish_reason="stop")]
    llm = SimpleNamespace(client=_stream_client(events), name="gpt-4o-mini")

    assert stream_llm_response(FakeSession(), llm, "describe") == "Your model has 3 classes."


# ---------------------------------------------------------------------------
# Retry policy, taxonomy and message text
# ---------------------------------------------------------------------------

def test_sdk_retry_wrapper_does_not_retry_a_refusal():
    from errors import ModelRefusal
    from utilities.llm_retry import with_retry

    calls = []

    def declined():
        calls.append(1)
        raise ModelRefusal("anthropic", "cyber")

    with pytest.raises(ModelRefusal):
        with_retry(declined)()
    assert len(calls) == 1


def test_classify_error_maps_a_refusal_to_its_own_code():
    from errors import ErrorCode, ModelRefusal, classify_error

    assert classify_error(ModelRefusal("openai", "content_filter")) is ErrorCode.MODEL_REFUSAL


def test_user_message_text():
    from errors import ModelRefusal

    assert ModelRefusal("anthropic", "cyber").user_message() == CYBER_MESSAGE
    assert ModelRefusal("anthropic", "reasoning_extraction").user_message().startswith(
        "The AI model declined this request (safety category: reasoning extraction)."
    )
    assert ModelRefusal("openai").user_message() == (
        "The AI model declined this request. Please try rephrasing it, or switch "
        "to a different model in the assistant settings."
    )


def test_vision_refusal_is_not_retried(monkeypatch):
    import requests as http_requests
    from handlers.file_conversion_handler import convert_file_to_class_spec

    posts = []

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"finish_reason": "stop",
                                 "message": {"content": None, "refusal": "I can't help."}}]}

    def _post(*_args, **_kwargs):
        posts.append(1)
        return _Resp()

    monkeypatch.setattr(http_requests, "post", _post)
    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    result = convert_file_to_class_spec(
        file_content_b64=png, filename="mockup.png",
        llm_predict=lambda _p: "{}", openai_api_key="test-key",
    )

    assert "declined" in result["message"].lower()
    assert len(posts) == 1, "a declined image must not be sent again"


def test_file_text_refusal_reaches_the_user():
    import base64

    from errors import ModelRefusal
    from handlers.file_conversion_handler import convert_file_to_class_spec

    def declined(_prompt):
        raise ModelRefusal("anthropic", "cyber")

    result = convert_file_to_class_spec(
        file_content_b64=base64.b64encode(b"class Exploit: pass").decode(),
        filename="model.py", llm_predict=declined,
    )

    assert result["message"] == CYBER_MESSAGE
