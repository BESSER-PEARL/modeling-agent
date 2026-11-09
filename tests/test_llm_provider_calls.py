"""Every LLM call goes through ``llm.provider.LLMProvider`` and sends what it sent before.

The call sites used to build their own requests: the vision path posted JSON to
``https://api.openai.com/v1/chat/completions`` with ``requests`` (no retry,
tracking, gateway or non-OpenAI keys), and ``gpt_predict_json``, the handlers'
free-text call and the reply stream each called the shared client directly.
The ``_old_*`` builders below are frozen copies of those request bodies; the
tests assert the provider still sends exactly them.
"""

import base64
import io
from pathlib import Path
from types import SimpleNamespace

import pypdfium2
import pytest
from openai.types import CompletionUsage
from pydantic import BaseModel

from common.errors import ImagesNotSupported, ModelRefusal
from conversation import session_helpers
from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler
from ingestion import file_conversion_handler as fc
from llm import byok, provider as provider_module, tracking
from llm.model_config import reasoning_effort_for, supports_custom_temperature
from llm.provider import LLMProvider
from tests.conftest import FakeSession

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="


# -- frozen request bodies of the pre-provider call sites ---------------------

def _old_sampling(model, temperature):
    if supports_custom_temperature(model):
        return {"temperature": temperature}
    effort = reasoning_effort_for(model)
    return {"reasoning_effort": effort} if effort else {}


def _old_vision_sampling_params(vision_model):
    if supports_custom_temperature(vision_model):
        return {"max_tokens": 8192, "temperature": 0.1}
    kwargs = {"max_completion_tokens": 8192}
    if reasoning_effort_for(vision_model):
        kwargs["reasoning_effort"] = reasoning_effort_for(vision_model)
    return kwargs


def _old_image_payload(vision_model, vision_prompt, mime_type, image_b64):
    return {
        "model": vision_model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": vision_prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
        ]}],
        "response_format": {"type": "json_object"},
        **_old_vision_sampling_params(vision_model),
    }


def _old_pdf_payload(vision_model, vision_prompt, images_b64):
    content_blocks = [{"type": "text", "text": vision_prompt}]
    for img_b64 in images_b64:
        content_blocks.append({
            "type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"},
        })
    return {
        "model": vision_model,
        "messages": [{"role": "user", "content": content_blocks}],
        "response_format": {"type": "json_object"},
        **_old_vision_sampling_params(vision_model),
    }


def _old_gpt_predict_json_kwargs(model, prompt):
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        **_old_sampling(model, 0.2),
        "max_completion_tokens": 8192,
        "response_format": {"type": "json_object"},
    }


def _old_predict_raw_kwargs(model, content, max_tokens=None):
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_completion_tokens": max_tokens or 8192,
        **_old_sampling(model, 0.2),
    }


def _old_stream_kwargs(model, prompt, system_prompt):
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    messages.append({"role": "user", "content": prompt})
    return {
        "model": model,
        "messages": messages,
        "max_completion_tokens": 4096,
        "stream": True,
        "stream_options": {"include_usage": True},
        **_old_sampling(model, 0.4),
    }


# -- fakes ---------------------------------------------------------------------

def _usage(prompt=100, completion=10):
    return CompletionUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion)


def _completion(content='{"ok": true}', usage=None, refusal=None, finish_reason="stop"):
    message = SimpleNamespace(content=content, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)], usage=usage)


class _Client:
    """An OpenAI-shaped client recording ``chat.completions.create`` calls."""

    def __init__(self, reply=None):
        self.calls = []
        self._reply = reply or (lambda **kw: _completion())
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._reply(**kwargs)


class _ServerLLM:
    """Stands in for BAF's ``LLMOpenAI``: a model name, a client, ``predict``."""

    def __init__(self, name="gpt-4o-mini", client=None):
        self.name = name
        self.client = client if client is not None else _Client()
        self.predict_calls = []

    def predict(self, prompt, parameters=None):
        self.predict_calls.append((prompt, parameters))
        return '{"ok": true}'


@pytest.fixture
def tracker(monkeypatch):
    fresh = tracking.TokenTracker()
    monkeypatch.setattr(tracking, "get_tracker", lambda: fresh)
    return fresh


@pytest.fixture
def no_byok(monkeypatch):
    monkeypatch.setattr(byok, "get_active_client", lambda: None)


# -- vision: the request is unchanged, it just no longer bypasses the client ----

@pytest.mark.parametrize("vision_model", ["gpt-4o", "gpt-6-luna"])
def test_image_vision_request_is_the_old_payload(monkeypatch, no_byok, vision_model):
    monkeypatch.setattr(provider_module, "MODEL_VISION", vision_model)
    server = _ServerLLM()

    fc.convert_file_to_diagram_spec(
        file_content_b64=PNG, filename="diagram.jpg", llm_predict=lambda p: "{}",
        llm_vision_predict=LLMProvider(server).vision,
    )

    sent = dict(server.client.calls[0])
    assert sent.pop("timeout") == 60
    assert sent == _old_image_payload(
        vision_model, fc._build_image_prompt(None, detect_mockup=True), "image/jpeg", PNG,
    )


def test_scanned_pdf_vision_request_is_the_old_payload(no_byok):
    doc = pypdfium2.PdfDocument.new()
    doc.new_page(595, 842).close()
    saved = io.BytesIO()
    doc.save(saved)
    pdf = saved.getvalue()
    page_png = base64.b64encode(fc._page_png(doc, 0)).decode("ascii")
    doc.close()
    server = _ServerLLM()

    fc.convert_file_to_diagram_spec(
        file_content_b64=base64.b64encode(pdf).decode(), filename="scan.pdf",
        llm_predict=lambda p: "{}", llm_vision_predict=LLMProvider(server).vision,
    )

    sent = dict(server.client.calls[0])
    assert sent.pop("timeout") == 90
    assert sent == _old_pdf_payload(provider_module.MODEL_VISION, fc._build_image_prompt(), [page_png])


def test_vision_is_tracked_once_with_its_real_usage(tracker, no_byok):
    server = _ServerLLM(client=_Client(lambda **kw: _completion(usage=_usage(900, 40))))

    LLMProvider(server).vision("read it", ["data:image/png;base64,AAAA"], timeout=60)

    summary = tracker.summary()
    assert (summary["call_count"], summary["prompt_tokens"], summary["completion_tokens"]) == (1, 900, 40)


def test_a_vision_refusal_raises_model_refusal(no_byok):
    server = _ServerLLM(client=_Client(lambda **kw: _completion(content=None, refusal="No.")))

    with pytest.raises(ModelRefusal):
        LLMProvider(server).vision("read it", ["data:image/png;base64,AAAA"], timeout=60)


def test_a_users_openai_key_gets_the_same_vision_request(monkeypatch):
    user_client, server = _Client(), _ServerLLM()
    monkeypatch.setattr(byok, "get_active_client", lambda: SimpleNamespace(
        provider="openai", openai_client=user_client, vision_model=provider_module.MODEL_VISION,
    ))

    LLMProvider(server).vision("read it", ["data:image/png;base64,AAAA"], timeout=60)

    assert server.client.calls == [], "the server key must not pay for it"
    sent = dict(user_client.calls[0])
    sent.pop("timeout")
    assert sent == _old_image_payload(provider_module.MODEL_VISION, "read it", "image/png", "AAAA")


@pytest.fixture
def user_anthropic_sdk(monkeypatch):
    """Records the ``messages.create`` calls an Anthropic key makes."""
    import anthropic

    sent = []

    class _FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **kwargs):
            sent.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text='{"ok": true}')],
                usage=None, stop_reason="end_turn",
            )

    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    return sent


def test_an_anthropic_key_now_reads_images_on_the_users_key(monkeypatch):
    """Before, a non-OpenAI key fell back to the server key for vision. Every
    current Claude model reads images, so the user's pick is kept, and the
    call's timeout reaches the SDK."""
    import anthropic

    sent = {}

    class _FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **kwargs):
            sent.update(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text='{"ok": true}')],
                usage=None, stop_reason="end_turn",
            )

    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    server = _ServerLLM()
    token = byok.set_current("anthropic", "sk-ant-user", "claude-opus-5")
    try:
        text = LLMProvider(server).vision("read it", ["data:image/png;base64,AAAA"], timeout=60)
    finally:
        byok.reset_current(token)

    assert text == '{"ok": true}'
    assert server.client.calls == []
    assert sent["model"] == "claude-opus-5"
    assert sent["timeout"] == 60
    image, prompt = sent["messages"][0]["content"]
    assert image["source"] == {"type": "base64", "media_type": "image/png", "data": "AAAA"}
    assert prompt["text"].startswith("read it")


@pytest.fixture
def user_openai_sdk(monkeypatch):
    """Records the OpenAI-SDK clients a BYOK key builds and the calls they make."""
    import openai

    record = {"clients": [], "calls": []}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            record["clients"].append(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            record["calls"].append(kwargs)
            return _completion()

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    return record


def _vision_on(provider, key, model=None, base_url=None):
    server = _ServerLLM()
    token = byok.set_current(provider, key, model, base_url)
    try:
        LLMProvider(server).vision("read it", [f"data:image/png;base64,{PNG}"], timeout=60)
    finally:
        byok.reset_current(token)
    return server


@pytest.mark.parametrize("provider, key, base_url", [
    # Nebius's tier model, Qwen3-30B-A3B-Instruct-2507, is text-only.
    ("nebius", "nb-user", None),
    ("openai", "gw-key", "https://gateway.example/v1"),
])
def test_a_key_without_an_image_model_reads_images_on_the_server_key(
    monkeypatch, user_openai_sdk, provider, key, base_url,
):
    """Before the vision fallback, a Nebius key sent the images to a text-only
    model and the user was told to send a clearer diagram."""
    monkeypatch.setenv("BESSER_AGENT_ALLOW_CUSTOM_BASE_URL", "true")

    server = _vision_on(provider, key, base_url=base_url)

    assert user_openai_sdk["calls"] == []
    assert server.client.calls == [
        {**_old_image_payload(provider_module.MODEL_VISION, "read it", "image/png", PNG), "timeout": 60},
    ]


def test_an_openai_key_reads_images_on_the_users_model(user_openai_sdk):
    """Every current OpenAI chat model reads images: the pick is kept."""
    server = _vision_on("openai", "sk-user", "gpt-6.1-sol")

    assert server.client.calls == []
    assert "base_url" not in user_openai_sdk["clients"][0]
    assert user_openai_sdk["calls"] == [
        {**_old_image_payload("gpt-6.1-sol", "read it", "image/png", PNG), "timeout": 60},
    ]


@pytest.mark.parametrize("provider, key, tier_model", [
    ("openai", "sk-user", "gpt-6-sol"),
    ("anthropic", "sk-ant-user", "claude-sonnet-5-5"),
])
def test_without_a_pick_images_run_on_the_tier_model(
    user_openai_sdk, user_anthropic_sdk, provider, key, tier_model,
):
    _vision_on(provider, key)

    sent = user_openai_sdk["calls"] or user_anthropic_sdk
    assert [call["model"] for call in sent] == [tier_model]


def test_a_mistral_key_reads_images_on_its_image_model(user_openai_sdk):
    server = _vision_on("mistral", "ms-user", "codestral-latest")

    assert server.client.calls == []
    assert user_openai_sdk["clients"][0]["base_url"] == byok.MISTRAL_BASE_URL
    sent = user_openai_sdk["calls"][0]
    assert sent["model"] == "mistral-large-latest"
    assert sent["timeout"] == 60
    assert sent["messages"][0]["content"][1]["image_url"]["url"] == f"data:image/png;base64,{PNG}"
    assert sent["response_format"] == {"type": "json_object"}


# -- images with a prompt (the GUI mockup path): the vision rule ----------------

def _images_on(provider, key, model=None, base_url=None):
    """A handler's free-text call with a mockup, as the GUI reasoning pass and
    the JSON-mode structured fallback send it."""
    server = _ServerLLM()
    token = byok.set_current(provider, key, model, base_url)
    try:
        ClassDiagramHandler(server).predict_with_retry(
            "design it", max_retries=0, model="gpt-5-mini", images=[f"data:image/png;base64,{PNG}"],
        )
    finally:
        byok.reset_current(token)
    return server


@pytest.mark.parametrize("provider, key, base_url", [
    ("nebius", "nb-user", None),
    ("openai", "gw-key", "https://gateway.example/v1"),
])
def test_a_mockup_on_a_key_without_an_image_model_runs_on_the_server_key(
    monkeypatch, user_openai_sdk, provider, key, base_url,
):
    """A Nebius key sent the mockup to its text-only Qwen tier model."""
    monkeypatch.setenv("BESSER_AGENT_ALLOW_CUSTOM_BASE_URL", "true")

    server = _images_on(provider, key, base_url=base_url)

    assert user_openai_sdk["calls"] == []
    content = [
        {"type": "text", "text": "design it"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
    ]
    assert server.client.calls == [_old_predict_raw_kwargs("gpt-5-mini", content)]


def test_a_mockup_on_a_mistral_key_runs_on_its_image_model(user_openai_sdk):
    """The user's pick (codestral) may be text-only, as on the vision path."""
    server = _images_on("mistral", "ms-user", "codestral-latest")

    assert server.client.calls == []
    sent = user_openai_sdk["calls"][0]
    assert sent["model"] == "mistral-large-latest"
    assert sent["messages"][0]["content"][1]["image_url"]["url"] == f"data:image/png;base64,{PNG}"


@pytest.mark.parametrize("provider, key, model", [
    ("openai", "sk-user", "gpt-6.1-sol"),
    ("anthropic", "sk-ant-user", "claude-opus-5-5"),
])
def test_a_mockup_keeps_the_users_model_where_every_model_reads_images(
    user_openai_sdk, user_anthropic_sdk, provider, key, model,
):
    server = _images_on(provider, key, model)

    assert server.client.calls == []
    sent = user_openai_sdk["calls"] or user_anthropic_sdk
    assert [call["model"] for call in sent] == [model]


def test_text_on_a_key_keeps_the_users_model(user_openai_sdk):
    server = _ServerLLM()
    token = byok.set_current("mistral", "ms-user", "codestral-latest")
    try:
        LLMProvider(server).complete("plan it", model="gpt-5-mini")
    finally:
        byok.reset_current(token)
    assert user_openai_sdk["calls"][0]["model"] == "codestral-latest"


def _bad_request(message, code=None):
    """An ``openai.BadRequestError`` as the SDK raises it: the text carries the body."""
    import httpx
    import openai

    request = httpx.Request("POST", "https://api.example/v1/chat/completions")
    body = {"message": message, "type": "invalid_request_error", "param": None, "code": code}
    return openai.BadRequestError(
        f"Error code: 400 - {{'error': {body}}}",
        response=httpx.Response(400, request=request), body=body,
    )


def _convert_png(vision):
    return fc.convert_file_to_diagram_spec(
        file_content_b64=PNG, filename="diagram.png",
        llm_predict=lambda p: "{}", llm_vision_predict=vision,
    )


def test_a_400_on_images_says_the_model_cannot_read_them(no_byok, caplog):
    def _reject(**kw):
        raise _bad_request("Invalid content type. image_url is only supported by certain models.")

    server = _ServerLLM(client=_Client(_reject))

    with pytest.raises(ImagesNotSupported) as raised:
        LLMProvider(server).vision("read it", ["data:image/png;base64,AAAA"], timeout=60)
    result = _convert_png(LLMProvider(server).vision)

    assert isinstance(raised.value.__cause__, Exception)
    assert "can't read this image" in result["message"]
    assert provider_module.MODEL_VISION in result["message"]
    # The provider's reason reaches the log, not only "rejected the image request".
    assert "image_url is only supported by certain models" in caplog.text


@pytest.mark.parametrize("message, code", [
    ("This model's maximum context length is 128000 tokens. However, your messages "
     "resulted in 131072 tokens. Please reduce the length of the messages.",
     "context_length_exceeded"),
    ("Unsupported parameter: 'max_tokens' is not supported with this model. "
     "Use 'max_completion_tokens' instead.", "unsupported_parameter"),
])
def test_a_400_not_about_the_image_keeps_its_own_error(no_byok, caplog, message, code):
    """Every vision 400 used to become "can't read this image", hiding the
    provider's reason (a context overflow, a bad parameter)."""
    def _reject(**kw):
        raise _bad_request(message, code)

    server = _ServerLLM(client=_Client(_reject))

    with pytest.raises(Exception) as raised:
        LLMProvider(server).vision("read it", ["data:image/png;base64,AAAA"], timeout=60)
    result = _convert_png(LLMProvider(server).vision)

    assert not isinstance(raised.value, ImagesNotSupported)
    assert "can't read this image" not in result["message"]
    assert code in caplog.text


def test_an_anthropic_400_about_the_image_says_the_model_cannot_read_it(monkeypatch, caplog):
    import anthropic
    import httpx

    error = {"type": "invalid_request_error",
             "message": "messages.0.content.0.image.source.base64: image exceeds 5 MB maximum"}

    class _FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **kwargs):
            request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            body = {"type": "error", "error": error}
            raise anthropic.BadRequestError(
                f"Error code: 400 - {body}", response=httpx.Response(400, request=request), body=body,
            )

    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    token = byok.set_current("anthropic", "sk-ant-user", "claude-opus-5-5")
    try:
        result = _convert_png(LLMProvider(_ServerLLM()).vision)
    finally:
        byok.reset_current(token)

    assert "can't read this image" in result["message"]
    assert "claude-opus-5-5" in result["message"]
    assert "image exceeds 5 MB maximum" in caplog.text


@pytest.mark.parametrize("provider, key, base_url", [
    ("nebius", "nb-user", None),
    ("openai", "gw-key", "https://gateway.example/v1"),
])
def test_images_moved_to_the_server_key_are_a_warning(
    monkeypatch, caplog, user_openai_sdk, provider, key, base_url,
):
    """The user's key does not pay for these images, the deployment's does."""
    monkeypatch.setenv("BESSER_AGENT_ALLOW_CUSTOM_BASE_URL", "true")

    with caplog.at_level("INFO", logger=provider_module.logger.name):
        _vision_on(provider, key, base_url=base_url)

    moved = [r for r in caplog.records if "using the server key" in r.getMessage()]
    assert [r.levelname for r in moved] == ["WARNING"]


# -- the other migrated call sites --------------------------------------------

@pytest.mark.parametrize("model", ["gpt-4o", "gpt-6-luna"])
def test_gpt_predict_json_request_is_unchanged(no_byok, model):
    server = _ServerLLM()

    LLMProvider(server).complete("convert this", model=model, json_mode=True)

    assert server.client.calls == [_old_gpt_predict_json_kwargs(model, "convert this")]


def test_gpt_predict_json_without_a_model_uses_bafs_predict_with_json_params(no_byok):
    server = _ServerLLM(name="gpt-6-luna")

    LLMProvider(server).complete("convert this", json_mode=True)

    params = {k: v for k, v in _old_gpt_predict_json_kwargs("gpt-6-luna", "").items()
              if k not in ("model", "messages")}
    assert server.predict_calls == [("convert this", params)]
    assert server.client.calls == []


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-5-mini"])
@pytest.mark.parametrize("max_tokens", [None, 32_000])
def test_handler_free_text_request_is_unchanged(no_byok, model, max_tokens):
    server = _ServerLLM()

    ClassDiagramHandler(server)._predict_raw("plan it", model=model, max_tokens=max_tokens)

    assert server.client.calls == [_old_predict_raw_kwargs(model, "plan it", max_tokens)]


def test_handler_image_request_uses_the_instance_model(no_byok):
    server = _ServerLLM(name="gpt-4o")
    image = "data:image/png;base64,AAAA"

    ClassDiagramHandler(server)._predict_raw("plan it", images=[image])

    content = [{"type": "text", "text": "plan it"}, {"type": "image_url", "image_url": {"url": image}}]
    assert server.client.calls == [_old_predict_raw_kwargs("gpt-4o", content)]


def test_handler_without_a_model_uses_bafs_predict(no_byok, tracker):
    server = _ServerLLM()

    ClassDiagramHandler(server)._predict_raw("plan it")

    assert server.predict_calls == [("plan it", None)]
    assert tracker.summary()["call_count"] == 1


@pytest.mark.parametrize(("model", "system_prompt"), [("gpt-4o", ""), ("gpt-6-luna", "Be brief.")])
def test_stream_request_is_unchanged(monkeypatch, no_byok, model, system_prompt):
    monkeypatch.setattr(session_helpers, "language_instruction", lambda s: "")
    server = _ServerLLM(client=_Client(lambda **kw: []))

    session_helpers.stream_llm_response(FakeSession(), server, "describe", system_prompt, model=model)

    assert server.client.calls == [_old_stream_kwargs(model, "describe", system_prompt)]


def test_a_stream_is_tracked_once_with_its_real_usage(no_byok, tracker):
    def events(**kw):
        delta = SimpleNamespace(content="Hi.", refusal=None)
        return [
            SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)], usage=None),
            SimpleNamespace(choices=[], usage=_usage(300, 2)),
        ]

    server = _ServerLLM(client=_Client(events))

    assert "".join(LLMProvider(server).stream("hello")) == "Hi."
    summary = tracker.summary()
    assert (summary["call_count"], summary["prompt_tokens"]) == (1, 300)


def test_a_failed_stream_request_raises_before_iteration(no_byok):
    """The reply helper turns a raise into the recovery message; a stream
    that only failed on first ``next()`` would have been read as empty."""
    def boom(**kw):
        raise RuntimeError("401")

    with pytest.raises(RuntimeError):
        LLMProvider(_ServerLLM(client=_Client(boom))).stream("hello")


def test_a_free_text_call_is_tracked_once_with_its_real_usage(no_byok, tracker):
    server = _ServerLLM(client=_Client(lambda **kw: _completion(usage=_usage(1_234, 56))))

    LLMProvider(server).complete("hello", model="gpt-4o")

    summary = tracker.summary()
    assert (summary["call_count"], summary["prompt_tokens"]) == (1, 1_234)


class _Answer(BaseModel):
    name: str


def test_handler_and_classifier_send_the_same_structured_request(no_byok):
    """Both structured paths now share one SDK call."""
    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(parsed=_Answer(name="x"), content="{}", refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None)

    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    server = _ServerLLM(name="gpt-6-luna", client=client)
    messages = [{"role": "user", "content": "make it"}]

    ClassDiagramHandler(server).predict_structured("make it", _Answer, max_retries=0)
    LLMProvider(server).parse(messages, _Answer, temperature=0.2, max_tokens=8192)

    expected = {
        "model": "gpt-6-luna", "messages": messages, "response_format": _Answer,
        "max_completion_tokens": 8192, **_old_sampling("gpt-6-luna", 0.2),
    }
    assert calls == [expected, expected]


# -- one door to the SDKs -------------------------------------------------------

def test_no_module_outside_llm_names_an_llm_api_host():
    src = Path(__file__).resolve().parents[1] / "src"
    hosts = ("api.openai.com", "api.anthropic.com", "api.mistral.ai")
    offenders = [
        str(path.relative_to(src))
        for path in src.rglob("*.py")
        if path.relative_to(src).parts[0] != "llm"
        and any(host in path.read_text(encoding="utf-8") for host in hosts)
    ]
    assert offenders == []
