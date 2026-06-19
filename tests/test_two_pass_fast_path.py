"""Tests for the two-pass structured fast path with context-enriched prompts.

Regression tests for the bug where conversation memory defeated the fast-path
length check in ``predict_two_pass_structured``: the enriched modeling prompt
is built as ``{conversation_context}{operation_request}\\n\\n{workspace_block}``
(history as a PREFIX), so the old string-splitting recovery could never strip
the history. Once a session had any memory, even trivial requests like
"add a class X" paid an extra reasoning LLM round-trip.

The fix threads the raw user message separately (``raw_request``) so the
fast-path decision is based on what the user actually typed.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler
from schemas import SystemClassSpec


# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------

class RecordingFakeLLM:
    """LLM stub that records every prompt and returns canned responses round-robin."""

    def __init__(self, responses):
        self.responses = list(responses)
        self._call_index = 0
        self.call_log = []

    def predict(self, prompt: str) -> str:
        self.call_log.append(prompt)
        result = self.responses[self._call_index % len(self.responses)]
        self._call_index += 1
        return result


VALID_SYSTEM_SPEC_JSON = json.dumps({
    "systemName": "TestSystem",
    "classes": [
        {
            "className": "Customer",
            "attributes": [{"name": "name", "type": "String"}],
            "methods": [],
        }
    ],
    "relationships": [],
})

RAW_REQUEST = "add a class Customer"

# Mirrors how src/execution/model_operations.py enriches the modeling prompt:
# conversation history PREFIX + raw request + workspace block SUFFIX.
CONVERSATION_CONTEXT = (
    "Recent conversation context (use this to understand what the user has been working on):\n"
    "  user: create a complete e-commerce system with products, orders and payments\n"
    "  assistant: I created an e-commerce class diagram with Product, Order and Payment classes.\n"
    "  user: now add shipping and inventory tracking with warehouses\n"
    "  assistant: Added Shipping, Inventory and Warehouse classes with associations."
    "\n\n"
)

WORKSPACE_BLOCK = (
    "Workspace context:\n"
    "Target diagram type: ClassDiagram\n"
    "Current model: 7 classes (Product, Order, Payment, Shipping, Inventory, Warehouse, User)\n"
    "Other diagrams in project:\n"
    "- StateMachineDiagram: 5 states"
)


def build_enriched_prompt(raw_request: str) -> str:
    """Build the prompt exactly the way model_operations.py does."""
    return f"{CONVERSATION_CONTEXT}{raw_request}\n\n{WORKSPACE_BLOCK}"


# ---------------------------------------------------------------------------
# Fast path: trivial request + session memory => exactly ONE LLM call
# ---------------------------------------------------------------------------

class TestTwoPassFastPath:
    def test_trivial_request_with_conversation_memory_single_llm_call(self):
        """A <250-char request must skip the reasoning pass even when the
        session has conversation memory inflating the enriched prompt."""
        llm = RecordingFakeLLM([VALID_SYSTEM_SPEC_JSON])
        handler = ClassDiagramHandler(llm)

        enriched = build_enriched_prompt(RAW_REQUEST)
        # Sanity: memory pushes the enriched prompt over the threshold while
        # the raw request stays well under it — the exact bug scenario.
        assert len(enriched) >= handler._TWO_PASS_MIN_LENGTH
        assert len(RAW_REQUEST) < handler._TWO_PASS_MIN_LENGTH

        result = handler.generate_complete_system(enriched, raw_request=RAW_REQUEST)

        assert len(llm.call_log) == 1, (
            f"Expected exactly one LLM call (fast path), got {len(llm.call_log)}: "
            f"{[p[:80] for p in llm.call_log]}"
        )
        assert result["action"] == "inject_complete_system"
        # The single structured call still receives the full enriched context.
        assert "Workspace context:" in llm.call_log[0]

    def test_fast_path_decision_uses_raw_request_directly(self):
        """predict_two_pass_structured keys the fast path on raw_request,
        not on any string-splitting of the enriched prompt."""
        llm = RecordingFakeLLM([VALID_SYSTEM_SPEC_JSON])
        handler = ClassDiagramHandler(llm)

        parsed = handler.predict_two_pass_structured(
            user_request=build_enriched_prompt(RAW_REQUEST),
            system_prompt="You are a UML expert.",
            reasoning_prompt="Think about the design.",
            response_schema=SystemClassSpec,
            raw_request=RAW_REQUEST,
        )

        assert len(llm.call_log) == 1
        assert parsed.classes[0].className == "Customer"

    def test_without_raw_request_falls_back_to_full_prompt_length(self):
        """Legacy callers that don't thread raw_request keep the two-pass
        behavior for long enriched prompts (no silent behavior change)."""
        llm = RecordingFakeLLM([
            "Reasoning: a Customer class with name attribute is needed.",
            VALID_SYSTEM_SPEC_JSON,
        ])
        handler = ClassDiagramHandler(llm)

        result = handler.generate_complete_system(build_enriched_prompt(RAW_REQUEST))

        assert len(llm.call_log) == 2  # reasoning pass + structured pass
        assert result["action"] == "inject_complete_system"


# ---------------------------------------------------------------------------
# Slow path: enriched context must not be duplicated across the two passes
# ---------------------------------------------------------------------------

class TestTwoPassContextDeduplication:
    LONG_RAW_REQUEST = (
        "Create a complete hospital management system with patients, doctors, "
        "nurses, appointments, prescriptions, medical records, billing, "
        "insurance claims, departments, wards, beds and lab tests. Include "
        "inheritance for staff roles and associations with multiplicities "
        "between all the entities."
    )

    def test_reasoning_pass_gets_raw_request_structured_pass_gets_context_once(self):
        llm = RecordingFakeLLM([
            "Reasoning: model Patient, Doctor, Appointment and related entities.",
            VALID_SYSTEM_SPEC_JSON,
        ])
        handler = ClassDiagramHandler(llm)
        assert len(self.LONG_RAW_REQUEST) >= handler._TWO_PASS_MIN_LENGTH

        enriched = build_enriched_prompt(self.LONG_RAW_REQUEST)
        result = handler.generate_complete_system(
            enriched, raw_request=self.LONG_RAW_REQUEST,
        )

        assert len(llm.call_log) == 2
        reasoning_prompt, structured_prompt = llm.call_log

        # Pass 1 reasons over the raw request only — the enriched context
        # (history + workspace block) must not be duplicated into it.
        assert self.LONG_RAW_REQUEST in reasoning_prompt
        assert "Recent conversation context" not in reasoning_prompt
        assert "Workspace context:" not in reasoning_prompt

        # Pass 2 receives the enriched context exactly once.
        assert structured_prompt.count("Workspace context:") == 1
        assert structured_prompt.count("Recent conversation context") == 1
        assert result["action"] == "inject_complete_system"
