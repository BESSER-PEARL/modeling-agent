"""A 429 on the user's own API key says the key is the cause, wherever it happens.

A depleted key answered "The AI service is currently busy. Please wait a moment"
when streaming and "try again or rephrase" in the handlers, though waiting can
never help it. The two cases read differently, because only a real rate limit
passes with time. A limit on the shared server key keeps its own replies.

The editor's "Try again" chip sends ``suggestedRecovery`` as the next prompt, so
no reply may offer the chip with a recovery sentence.
"""

import anthropic
import httpx
import openai
import pytest

from app import state_bodies
from conversation import agent_context as ctx
from conversation.session_helpers import stream_llm_response
from diagram_handlers.registry.factory import DiagramHandlerFactory, HANDLER_CLASSES
from execution.model_operations import execute_model_operation
from ingestion import file_conversion_handler as conversion
from llm import byok
from protocol.adapters import parse_assistant_request
from tests.conftest import FakeSession, make_session
from tests.test_byok_rejected_key import (
    ANTHROPIC_401, FAKE_KEY, _BUILD, _ENTRY_POINTS, _IMAGE, _NO_LLM_SINGLE_ELEMENT, _PLANTUML,
    _assert_key_rejected, _replies, _sdk_clients, _server_llm, _status_error,
)

# Shaped like the providers' answers; "org-secret" stands for what must not leak.
OPENAI_RATE_LIMIT = _status_error(
    openai.RateLimitError, 429,
    "Error code: 429 - {'error': {'message': 'Rate limit reached for gpt-4o-mini in "
    "organization org-secret on requests per min (RPM): Limit 3, Used 3, Requested 1.', "
    "'type': 'requests', 'code': 'rate_limit_exceeded'}}",
)
ANTHROPIC_RATE_LIMIT = _status_error(
    anthropic.RateLimitError, 429,
    "Error code: 429 - {'type': 'error', 'error': {'type': 'rate_limit_error', 'message': "
    "'This request would exceed the rate limit for your organization org-secret.'}}",
)
OPENAI_NO_QUOTA = _status_error(
    openai.RateLimitError, 429,
    "Error code: 429 - {'error': {'message': 'You exceeded your current quota, please check "
    "your plan and billing details.', 'type': 'insufficient_quota', 'code': 'insufficient_quota'}}",
)
# A spend cap keeps the type and has its own code.
OPENAI_SPEND_CAP = _status_error(
    openai.RateLimitError, 429,
    "Error code: 429 - {'error': {'message': 'Project spend limit reached', "
    "'type': 'insufficient_quota', 'code': 'project_spend_limit_exceeded'}}",
)
# Anthropic reports a depleted or capped account as a 400.
ANTHROPIC_NO_CREDIT = _status_error(
    anthropic.BadRequestError, 400,
    "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': "
    "'Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing "
    "to upgrade or purchase credits.'}}",
)
ANTHROPIC_SPEND_CAP = _status_error(
    anthropic.BadRequestError, 400,
    "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': "
    "'You have reached your specified API usage limits. You will regain access on 2026-12-01.'}}",
)


def _assert_no_prompt_chip(reply: dict) -> None:
    # useAssistantLogic.ts: retryable && suggestedRecovery makes the "Try again"
    # chip, whose prompt is that text.
    assert not (reply["retryable"] and reply.get("suggestedRecovery"))


def _assert_rate_limited(reply: dict) -> None:
    assert reply["action"] == "agent_error"
    # The editor offers its API key button on this code.
    assert reply["errorCode"] == "rate_limit"
    assert "Your API key hit its rate limit" in reply["message"]
    assert "Wait a minute" in reply["message"]
    assert "key icon" in reply["message"]
    # Waiting helps here, and only here.
    assert reply["retryable"] is True
    _assert_no_prompt_chip(reply)


def _assert_exhausted(reply: dict) -> None:
    assert reply["action"] == "agent_error"
    assert reply["errorCode"] == "rate_limit"
    assert "Your API key has run out of credit" in reply["message"]
    assert "key icon" in reply["message"]
    # Waiting cannot refill an account.
    assert "wait" not in reply["message"].lower()
    assert reply["retryable"] is False
    _assert_no_prompt_chip(reply)


CASES = [
    pytest.param(("openai", OPENAI_RATE_LIMIT, _assert_rate_limited), id="openai-rate-limit"),
    pytest.param(("anthropic", ANTHROPIC_RATE_LIMIT, _assert_rate_limited), id="anthropic-rate-limit"),
    pytest.param(("openai", OPENAI_NO_QUOTA, _assert_exhausted), id="openai-no-quota"),
    pytest.param(("openai", OPENAI_SPEND_CAP, _assert_exhausted), id="openai-spend-cap"),
    pytest.param(("anthropic", ANTHROPIC_NO_CREDIT, _assert_exhausted), id="anthropic-no-credit"),
    pytest.param(("anthropic", ANTHROPIC_SPEND_CAP, _assert_exhausted), id="anthropic-spend-cap"),
]


@pytest.fixture(params=CASES)
def limited_key(request, monkeypatch):
    """A request on the user's key, which the provider refuses; yields the check
    its reply must pass."""
    provider, error, check = request.param
    _sdk_clients(monkeypatch, error)
    token = byok.set_current(provider, FAKE_KEY)
    yield check
    byok.reset_current(token)


@pytest.fixture
def active_key():
    token = byok.set_current("anthropic", FAKE_KEY)
    yield
    byok.reset_current(token)


# ---------------------------------------------------------------------------
# Telling the cases apart, on errors the real SDKs build from a response
# ---------------------------------------------------------------------------

def _openai_error(status: int, error: dict) -> Exception:
    transport = httpx.MockTransport(lambda _: httpx.Response(status, json={"error": error}))
    client = openai.OpenAI(api_key=FAKE_KEY, max_retries=0, http_client=httpx.Client(transport=transport))
    with pytest.raises(openai.APIStatusError) as info:
        client.chat.completions.create(model="m", messages=[])
    return info.value


def _anthropic_error(status: int, error: dict) -> Exception:
    transport = httpx.MockTransport(
        lambda _: httpx.Response(status, json={"type": "error", "error": error}),
    )
    client = anthropic.Anthropic(api_key=FAKE_KEY, max_retries=0, http_client=httpx.Client(transport=transport))
    with pytest.raises(anthropic.APIStatusError) as info:
        client.messages.create(model="m", max_tokens=1, messages=[])
    return info.value


@pytest.mark.parametrize("make_error, status, error, fault", [
    (_openai_error, 429, {"message": "Rate limit reached", "type": "requests", "code": "rate_limit_exceeded"},
     byok.KeyFault.RATE_LIMITED),
    (_openai_error, 429, {"message": "Quota", "type": "insufficient_quota", "code": "insufficient_quota"},
     byok.KeyFault.EXHAUSTED),
    (_openai_error, 429, {"message": "Cap", "type": "insufficient_quota", "code": "project_spend_limit_exceeded"},
     byok.KeyFault.EXHAUSTED),
    (_openai_error, 429, {"message": "Cap", "type": "insufficient_quota", "code": "organization_usage_limit_exceeded"},
     byok.KeyFault.EXHAUSTED),
    (_openai_error, 400, {"message": "Billing hard limit has been reached", "type": "invalid_request_error",
                          "code": "billing_hard_limit_reached"},
     byok.KeyFault.EXHAUSTED),
    (_openai_error, 400, {"message": "Bad field", "type": "invalid_request_error", "code": None}, None),
    (_anthropic_error, 429, {"type": "rate_limit_error", "message": "Over the rate limit"},
     byok.KeyFault.RATE_LIMITED),
    (_anthropic_error, 400, {"type": "invalid_request_error",
                             "message": "Your credit balance is too low to access the Anthropic API."},
     byok.KeyFault.EXHAUSTED),
    (_anthropic_error, 400, {"type": "invalid_request_error",
                             "message": "You have reached your specified API usage limits."},
     byok.KeyFault.EXHAUSTED),
    (_anthropic_error, 400, {"type": "invalid_request_error", "message": "max_tokens: bad"}, None),
])
def test_real_sdk_errors_are_told_apart(active_key, make_error, status, error, fault):
    assert byok.user_key_fault(make_error(status, error)) is fault


# ---------------------------------------------------------------------------
# Streaming replies (help, fallback, describe)
# ---------------------------------------------------------------------------

def test_streamed_reply_names_the_key(limited_key):
    session = FakeSession()

    text = stream_llm_response(session, _server_llm(), "what can you do?")

    replies = _replies(session)
    errors = [r for r in replies if r["action"] == "agent_error"]
    assert len(errors) == 1
    limited_key(errors[0])
    # The stream closes empty, so the editor shows only the error bubble.
    assert text == ""
    assert [r["action"] for r in replies if r["action"].startswith("stream_")] == [
        "stream_start", "stream_done",
    ]


def test_reply_never_contains_the_key_or_the_providers_text(limited_key):
    session = FakeSession()

    stream_llm_response(session, _server_llm(), "what can you do?")

    sent = " ".join(session.replies)
    assert FAKE_KEY not in sent
    assert "org-secret" not in sent
    assert "Error code" not in sent


@pytest.mark.parametrize("error", [OPENAI_RATE_LIMIT, OPENAI_NO_QUOTA], ids=["rate-limit", "no-quota"])
def test_a_limit_on_the_server_key_does_not_blame_the_user(error):
    session = FakeSession()

    text = stream_llm_response(session, _server_llm(error), "what can you do?")

    assert text == "The AI service is currently busy. Please wait a moment and try again."
    assert not [r for r in _replies(session) if r["action"] == "agent_error"]


def test_an_unrelated_bad_request_on_the_users_key_does_not_blame_the_key(monkeypatch, active_key):
    _sdk_clients(monkeypatch, _status_error(
        anthropic.BadRequestError, 400,
        "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
        "'message': 'max_tokens: must be positive'}}",
    ))
    session = FakeSession()

    stream_llm_response(session, _server_llm(), "what can you do?")

    assert not [r for r in _replies(session) if r["action"] == "agent_error"]


# ---------------------------------------------------------------------------
# Diagram generation: every handler, every entry point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("handler_class", HANDLER_CLASSES, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("entry", _ENTRY_POINTS, ids=[e[0] for e in _ENTRY_POINTS])
def test_handlers_hand_a_limited_key_up_instead_of_replying(limited_key, handler_class, entry):
    if entry[0] == "single_element" and handler_class.__name__ in _NO_LLM_SINGLE_ELEMENT:
        pytest.skip("makes no model call")
    # Not "Please try again or rephrase", and no fallback starter diagram.
    with pytest.raises((openai.RateLimitError, anthropic.RateLimitError, anthropic.BadRequestError)):
        entry[1](handler_class(_server_llm()))


@pytest.mark.parametrize("mode", ["complete_system", "modify_model"])
def test_a_generation_turn_names_the_key(limited_key, monkeypatch, mode):
    monkeypatch.setattr(ctx, "diagram_factory", DiagramHandlerFactory(_server_llm()))
    session = make_session("create a library system", "StateMachineDiagram")

    result = execute_model_operation(
        session,
        parse_assistant_request(session),
        {"type": "model", "diagramType": "StateMachineDiagram", "mode": mode, "request": _BUILD},
        default_mode=mode,
    )

    assert result is None
    terminal = [r for r in _replies(session) if r["action"] != "progress"]
    assert len(terminal) == 1
    limited_key(terminal[0])


class _RaisingFactory:
    """A diagram factory whose handlers raise *error* from any generation call."""

    def __init__(self, error):
        self._error = error

    def get_handler(self, *_args, **_kwargs):
        return self

    def generate_complete_system(self, *_args, **_kwargs):
        raise self._error


def _run_model_operation(monkeypatch, error) -> list:
    monkeypatch.setattr(ctx, "diagram_factory", _RaisingFactory(error))
    session = make_session("create a library system", "StateMachineDiagram")
    execute_model_operation(
        session,
        parse_assistant_request(session),
        {"type": "model", "diagramType": "StateMachineDiagram", "mode": "complete_system", "request": _BUILD},
        default_mode="complete_system",
    )
    return [r for r in _replies(session) if r["action"] != "progress"]


class RateLimitedProxy(Exception):
    """Reads as a rate limit by its name, but is no SDK error: nothing says whose."""


def test_an_unattributed_limit_on_a_user_key_offers_no_prompt_chip(monkeypatch, active_key):
    terminal = _run_model_operation(monkeypatch, RateLimitedProxy("slow down"))

    assert len(terminal) == 1
    assert "API key" in terminal[0]["message"]
    _assert_no_prompt_chip(terminal[0])


def test_a_limit_on_the_server_key_still_offers_the_users_own_key(monkeypatch):
    terminal = _run_model_operation(monkeypatch, OPENAI_RATE_LIMIT)

    assert len(terminal) == 1
    assert terminal[0]["errorCode"] == "rate_limit"
    assert "Add your own API key" in terminal[0]["message"]


# ---------------------------------------------------------------------------
# File conversion
# ---------------------------------------------------------------------------

@pytest.fixture(params=CASES)
def failing_predict(request):
    provider, error, check = request.param
    token = byok.set_current(provider, FAKE_KEY)

    def _predict(*_args, **_kwargs):
        raise error

    yield _predict, check
    byok.reset_current(token)


def test_text_file_conversion_names_the_key(failing_predict):
    predict, check = failing_predict

    check(conversion.convert_file_to_diagram_spec(_PLANTUML, "model.puml", llm_predict=predict))


def test_image_conversion_names_the_key(failing_predict):
    predict, check = failing_predict

    check(conversion.convert_file_to_diagram_spec(
        _IMAGE, "model.png", llm_predict=predict, llm_vision_predict=predict,
    ))


def test_image_conversion_on_the_server_key_does_not_blame_the_users_key():
    # Nebius cannot read images, so this vision call ran on the server key.
    token = byok.set_current("nebius", FAKE_KEY)

    def _predict(*_args, **_kwargs):
        raise OPENAI_RATE_LIMIT

    try:
        reply = conversion.convert_file_to_diagram_spec(
            _IMAGE, "model.png", llm_predict=_predict, llm_vision_predict=_predict,
        )
    finally:
        byok.reset_current(token)

    assert "API key" not in reply["message"]


# ---------------------------------------------------------------------------
# The UML answer: nothing around these calls catches, and BAF only logs what a
# state body raises, so the user used to get no reply at all
# ---------------------------------------------------------------------------

def _ask_uml(monkeypatch, rag) -> list:
    monkeypatch.setattr(ctx, "gpt_text", _server_llm())
    monkeypatch.setattr(ctx, "uml_rag", rag)
    session = make_session("what is an association class?")
    session.run_rag = _server_llm(RuntimeError("no index")).predict
    state_bodies.uml_rag_body(session)
    return [r for r in _replies(session) if r["action"] == "agent_error"]


@pytest.mark.parametrize("rag", [None, object()], ids=["no-rag", "rag-fails"])
def test_uml_answer_names_the_key(limited_key, monkeypatch, rag):
    errors = _ask_uml(monkeypatch, rag)

    assert len(errors) == 1
    limited_key(errors[0])


def test_uml_answer_reports_a_rejected_key_too(monkeypatch, active_key):
    _sdk_clients(monkeypatch, ANTHROPIC_401)

    errors = _ask_uml(monkeypatch, None)

    assert len(errors) == 1
    _assert_key_rejected(errors[0])
