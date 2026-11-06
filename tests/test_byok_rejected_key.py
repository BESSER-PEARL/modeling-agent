"""A provider rejecting the user's own API key says so, wherever it happens.

Live on experimental, 2026-11-05: an invalid Anthropic key got a 401
``invalid x-api-key`` and the user read "The AI service is temporarily
unavailable. Please please try again later.": a retry that can never work, and
no hint that the key was at fault. The diagram handlers hid the same 401 behind
"Please try again or rephrase your request". A 401 on the server's own key is
a real outage and keeps the generic message.
"""

import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx
import openai
import pytest

from common.errors import ErrorCode, LLMPredictionError, classify_error
from conversation import agent_context as ctx
from conversation.session_helpers import stream_llm_response
from diagram_handlers.registry.factory import DiagramHandlerFactory, HANDLER_CLASSES
from execution.model_operations import execute_model_operation
from ingestion import file_conversion_handler as conversion
from llm import byok
from protocol.adapters import parse_assistant_request
from tests.conftest import FakeSession, make_session

FAKE_KEY = "sk-ant-test-not-a-real-key"
GENERIC_OUTAGE = "The AI service is temporarily unavailable. Please try again later."

_REQUEST = httpx.Request("POST", "https://provider.test/v1")


def _status_error(sdk_class, status: int, message: str) -> Exception:
    return sdk_class(message, response=httpx.Response(status, request=_REQUEST), body=None)


# The error texts are the ones the providers answered a made-up key with.
ANTHROPIC_401 = _status_error(
    anthropic.AuthenticationError, 401,
    "Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', "
    "'message': 'invalid x-api-key'}}",
)
OPENAI_401 = _status_error(
    openai.AuthenticationError, 401,
    "Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-test-*****0000.', "
    "'code': 'invalid_api_key'}}",
)


def _sdk_clients(monkeypatch, error: Exception) -> None:
    """Make both SDKs fail every call with *error*."""
    openai_client = MagicMock()
    openai_client.chat.completions.create.side_effect = error
    openai_client.chat.completions.parse.side_effect = error
    openai_client.beta.chat.completions.parse.side_effect = error
    monkeypatch.setattr(openai, "OpenAI", lambda **_: openai_client)
    monkeypatch.setattr(
        anthropic, "Anthropic",
        lambda **_: SimpleNamespace(messages=SimpleNamespace(create=MagicMock(side_effect=error))),
    )


@pytest.fixture
def user_key():
    """Activate *FAKE_KEY* for one provider, for the test's request."""
    tokens = []

    def _activate(provider: str) -> None:
        tokens.append(byok.set_current(provider, FAKE_KEY))

    yield _activate
    for token in reversed(tokens):
        byok.reset_current(token)


@pytest.fixture(params=[("anthropic", ANTHROPIC_401), ("openai", OPENAI_401)], ids=["anthropic", "openai"])
def rejected_key(request, monkeypatch, user_key):
    """A request on the user's key, which the provider refuses with a 401."""
    provider, error = request.param
    _sdk_clients(monkeypatch, error)
    user_key(provider)


def _server_llm(error: Exception = None):
    """The shared LLM; *error* makes its client fail, as a bad server key would."""
    client = MagicMock()
    client.chat.completions.create.side_effect = error
    client.beta.chat.completions.parse.side_effect = error
    return SimpleNamespace(client=client, name="gpt-4o-mini", predict=MagicMock(side_effect=error))


def _replies(session: FakeSession) -> list:
    return [json.loads(reply) for reply in session.replies]


def _assert_key_rejected(reply: dict) -> None:
    assert reply["action"] == "agent_error"
    # The editor offers its "Add your API key" button on this code.
    assert reply["errorCode"] == "auth_error"
    assert "rejected your API key" in reply["message"]
    assert "key icon" in reply["message"]
    # A key that was refused is refused again: no "Try again" chip.
    assert reply["retryable"] is False


# ---------------------------------------------------------------------------
# Streaming replies (help, fallback, describe), the path a bad key hits first:
# the classifier's own 401 turns the turn into a fallback reply.
# ---------------------------------------------------------------------------

def test_streamed_reply_tells_the_user_their_key_was_rejected(rejected_key):
    session = FakeSession()

    text = stream_llm_response(session, _server_llm(), "what can you do?")

    replies = _replies(session)
    errors = [r for r in replies if r["action"] == "agent_error"]
    assert len(errors) == 1
    _assert_key_rejected(errors[0])
    # The stream closes empty, so the editor shows only the error bubble.
    assert text == ""
    assert [r["action"] for r in replies if r["action"].startswith("stream_")] == [
        "stream_start", "stream_done",
    ]
    assert replies[-1]["fullText"] == ""


def test_rejected_key_reply_never_contains_the_key_or_the_providers_text(rejected_key):
    session = FakeSession()

    stream_llm_response(session, _server_llm(), "what can you do?")

    sent = " ".join(session.replies)
    assert FAKE_KEY not in sent
    assert "x-api-key" not in sent
    assert "sk-test-" not in sent


def test_forbidden_key_is_a_rejected_key_too(monkeypatch, user_key):
    # Anthropic's 403 permission_error: the key exists but may not do this.
    _sdk_clients(monkeypatch, _status_error(
        anthropic.PermissionDeniedError, 403,
        "Error code: 403 - {'error': {'type': 'permission_error'}}",
    ))
    user_key("anthropic")
    session = FakeSession()

    stream_llm_response(session, _server_llm(), "what can you do?")

    _assert_key_rejected(next(r for r in _replies(session) if r["action"] == "agent_error"))


def test_server_key_401_stays_the_generic_outage_message():
    session = FakeSession()

    text = stream_llm_response(session, _server_llm(ANTHROPIC_401), "what can you do?")

    assert text == GENERIC_OUTAGE
    assert not [r for r in _replies(session) if r["action"] == "agent_error"]


@pytest.mark.parametrize("error", [
    _status_error(anthropic.InternalServerError, 500, "Error code: 500 - overloaded"),
    anthropic.APIConnectionError(request=_REQUEST),
], ids=["provider-500", "connection"])
def test_outage_on_the_users_key_does_not_blame_the_key(monkeypatch, user_key, error):
    _sdk_clients(monkeypatch, error)
    user_key("anthropic")
    session = FakeSession()

    text = stream_llm_response(session, _server_llm(), "what can you do?")

    assert "API key" not in text
    assert not [r for r in _replies(session) if r["action"] == "agent_error"]


@pytest.mark.parametrize("error, expected", [
    (ANTHROPIC_401, GENERIC_OUTAGE),
    (
        _status_error(openai.RateLimitError, 429, "Error code: 429"),
        "The AI service is currently busy. Please wait a moment and try again.",
    ),
], ids=["auth", "rate-limit"])
def test_streamed_error_says_please_once(error, expected):
    # Two recovery hints carried their own "please" into "Please {recovery}."
    assert stream_llm_response(FakeSession(), _server_llm(error), "hi") == expected


# ---------------------------------------------------------------------------
# classify_error
# ---------------------------------------------------------------------------

def test_a_wrapped_401_is_an_auth_error():
    # Read as text, "invalid x-api-key" used to make this a validation error.
    try:
        try:
            raise ANTHROPIC_401
        except anthropic.AuthenticationError as exc:
            raise LLMPredictionError(f"LLM call failed (non-retryable): {exc}") from exc
    except LLMPredictionError as wrapped:
        assert classify_error(wrapped) == ErrorCode.AUTH_ERROR


# ---------------------------------------------------------------------------
# Diagram generation: every handler, every entry point
# ---------------------------------------------------------------------------

_ENTRY_POINTS = [
    ("complete_system", lambda handler: handler.generate_complete_system(_BUILD, raw_request=_BUILD)),
    ("single_element", lambda handler: handler.generate_single_element("add a thing")),
    ("modify", lambda handler: handler.generate_modification("rename it", {"elements": {}})),
]
_BUILD = "a library system with books and members"


# Their single element is a fixed starter: no model call, so nothing to reject.
_NO_LLM_SINGLE_ELEMENT = {"BPMNDiagramHandler", "ComponentDiagramHandler", "DeploymentDiagramHandler"}


@pytest.mark.parametrize("handler_class", HANDLER_CLASSES, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("entry", _ENTRY_POINTS, ids=[e[0] for e in _ENTRY_POINTS])
def test_handlers_hand_a_rejected_key_up_instead_of_replying(rejected_key, handler_class, entry):
    if entry[0] == "single_element" and handler_class.__name__ in _NO_LLM_SINGLE_ELEMENT:
        pytest.skip("makes no model call")
    # Not "Please try again or rephrase", and no fallback starter diagram.
    with pytest.raises((anthropic.AuthenticationError, openai.AuthenticationError)):
        entry[1](handler_class(_server_llm()))


@pytest.mark.parametrize("mode", ["complete_system", "modify_model"])
def test_a_generation_turn_tells_the_user_their_key_was_rejected(rejected_key, monkeypatch, mode):
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
    _assert_key_rejected(terminal[0])


def test_handler_with_a_bad_server_key_keeps_its_own_reply():
    handler = HANDLER_CLASSES[2](_server_llm(ANTHROPIC_401))

    reply = handler.generate_complete_system(_BUILD, raw_request=_BUILD)

    assert reply["action"] == "assistant_message"
    assert "API key" not in reply["message"]


# ---------------------------------------------------------------------------
# File conversion: it answers on its own, and a bad key's turn still reaches it
# ---------------------------------------------------------------------------

def _reject(*_args, **_kwargs):
    raise ANTHROPIC_401


_PLANTUML = base64.b64encode(b"@startuml\nclass Book\n@enduml").decode()
_IMAGE = base64.b64encode(b"not really a png").decode()


def test_text_file_conversion_reports_the_rejected_key(user_key):
    user_key("anthropic")

    reply = conversion.convert_file_to_diagram_spec(
        _PLANTUML, "model.puml", llm_predict=_reject,
    )

    _assert_key_rejected(reply)


def test_image_conversion_reports_the_rejected_key(user_key):
    user_key("anthropic")

    reply = conversion.convert_file_to_diagram_spec(
        _IMAGE, "model.png", llm_predict=_reject, llm_vision_predict=_reject,
    )

    _assert_key_rejected(reply)


def test_image_conversion_on_the_server_key_does_not_blame_the_users_key(user_key):
    # Nebius cannot read images, so this vision call ran on the server key.
    user_key("nebius")

    reply = conversion.convert_file_to_diagram_spec(
        _IMAGE, "model.png", llm_predict=_reject, llm_vision_predict=_reject,
    )

    assert "API key" not in reply["message"]
