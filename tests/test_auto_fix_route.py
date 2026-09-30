"""[auto-fix] protocol marker: the editor's validate-and-repair loop sends
machine-generated repair requests. They must route to the modify flow
DETERMINISTICALLY — no classifier LLM call (works during an outage, costs
nothing) and never interpreted as an answer to a pending question.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from protocol.types import AssistantRequest, WorkspaceContext  # noqa: E402
from unified_classifier import AUTO_FIX_PREFIX, classify_message  # noqa: E402


class _ExplodingProvider:
    """A provider that fails the test if the classifier actually calls it."""

    model_name = "test-model"

    def parse(self, **_kwargs):
        raise AssertionError("[auto-fix] messages must not reach the LLM")


def _request(text):
    return AssistantRequest(message=text, context=WorkspaceContext())


class TestAutoFixRoute:
    def test_routes_to_modify_without_llm(self):
        message = (
            f"{AUTO_FIX_PREFIX} The last change left the diagram with "
            "validation errors. Fix exactly these:\n"
            "- Invalid type 'str?' for the parameter 'description'"
        )
        result = classify_message(_request(message), _ExplodingProvider())
        assert result.intent == "modify_model_intent"
        assert result.model_disposition == "extend_existing"

    def test_not_treated_as_pending_flow_answer(self):
        pending = {"question": "Replace or keep?", "answers": ["replace", "keep"]}
        result = classify_message(
            _request(f"{AUTO_FIX_PREFIX} Fix: duplicate attribute 'id'"),
            _ExplodingProvider(),
            pending_flow=pending,
        )
        assert result.pending_flow_action == "new_request"
        assert result.pending_flow_answer is None

    def test_plain_messages_unaffected(self):
        # A normal message must still go to the provider (here: the exploding
        # one raises, and classify_message converts that into the fallback).
        result = classify_message(_request("add a Payment class"), _ExplodingProvider())
        assert result.intent == "fallback_intent"
