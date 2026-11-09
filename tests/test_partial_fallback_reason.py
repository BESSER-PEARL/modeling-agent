"""The class diagram's per-class fallback says why its result is partial.

Issue #744: "I had a bit of trouble building everything at once, but I set up 8
item(s): ... Want me to connect them together?" gave no reason.
"""
import pytest

from common.errors import LLMPredictionError, partial_result_reason
from conversation import reply_language
from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler
from llm.provider import OutputTruncated
from protocol.actions import Action
from tests.conftest import FakeSession


class APITimeoutError(Exception):
    """Named like the SDK's timeout error."""


def _wrapped(cause: BaseException) -> LLMPredictionError:
    try:
        raise LLMPredictionError(str(cause)) from cause
    except LLMPredictionError as exc:
        return exc


@pytest.mark.parametrize("error, reason", [
    (_wrapped(APITimeoutError("Request timed out.")), "the AI model took too long to answer"),
    (LLMPredictionError("Structured parse failed: invalid JSON"), "the AI model's answer was incomplete or malformed"),
    (_wrapped(OutputTruncated("hit the completion cap")), "the AI model's answer was cut off before it was complete"),
    (LLMPredictionError("LLM returned empty response"), "I couldn't get a complete answer from the AI service"),
    (KeyError("classes"), "something went wrong while I was putting the full diagram together"),
    (None, "something went wrong while I was putting the full diagram together"),
])
def test_the_reason_names_the_failure_category(error, reason):
    assert partial_result_reason(error) == reason


def test_the_reason_never_echoes_the_raw_error():
    error = LLMPredictionError("Structured parse failed: sk-secret-key at https://provider/v1")
    assert "sk-secret" not in partial_result_reason(error)
    assert "provider" not in partial_result_reason(error)


def test_the_reply_says_why_and_still_offers_to_connect():
    classes = [{"className": "Brand"}, {"className": "Car"}]
    message = ClassDiagramHandler._partial_system_message(
        classes, _wrapped(APITimeoutError("Request timed out.")),
    )
    assert message == (
        "I couldn't build the whole diagram in one go because the AI model took too long to "
        "answer, so I created the 2 classes I could identify, one by one: **Brand**, **Car**. "
        "They aren't connected yet. Want me to connect them together?"
    )


def test_a_single_class_is_not_offered_a_connection():
    message = ClassDiagramHandler._partial_system_message([{"className": "Brand"}], None)
    assert "connect" not in message
    assert "the class I could identify: **Brand**" in message


def test_the_reply_is_translated_into_the_users_language(monkeypatch):
    """It is an inject_complete_system reply, not an error, so the reply-language
    choke point translates it like any templated reply."""
    message = ClassDiagramHandler._partial_system_message([{"className": "A"}, {"className": "B"}], None)
    monkeypatch.setattr(reply_language, "translate_texts", lambda texts, code: [f"[{code}] {t}" for t in texts])
    session = FakeSession()
    session.set(reply_language.REPLY_LANGUAGE, "es")
    payload = {"action": Action.INJECT_COMPLETE_SYSTEM, "systemSpec": {}, "message": message}
    assert reply_language.localize_payload(session, payload)["message"] == f"[es] {message}"
