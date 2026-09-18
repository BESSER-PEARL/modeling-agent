"""Tests for OCL constraint capture in class-diagram generation (#46).

The agent should capture business rules the user EXPLICITLY states (uniqueness,
multiplicity-beyond-cardinality, value ranges) as OCL invariants in the
generated systemSpec — and capture NOTHING when the user states no such rule
(no hallucinated constraints).

Covers:
- The ``OCLConstraintSpec`` schema and the ``SystemClassSpec.constraints`` field.
- The ``_validate_constraints`` guard that drops constraints with an unknown
  context class.
- End-to-end: a structured spec containing a constraint flows through
  ``generate_complete_system`` into the ``inject_complete_system`` payload; an
  empty-constraints spec yields an empty list.
"""

import json
import os
import sys

import pytest


@pytest.fixture(autouse=True)
def _canonical_schema(monkeypatch):
    """These tests pin two-pass / guard MECHANICS with fixtures that emit the
    canonical SystemClassSpec wire shape. Force the canonical schema so the
    compact-output flag (BESSER_AGENT_COMPACT_SPEC) doesn't change the wire
    format under them — the compact chain has its own tests in
    test_compact_spec.py."""
    import diagram_handlers.types.class_diagram_handler as _m
    monkeypatch.setattr(_m, "COMPACT_SPEC_ENABLED", False, raising=True)


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schemas import SystemClassSpec, OCLConstraintSpec
from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler


class RecordingFakeLLM:
    """LLM stub returning canned JSON responses round-robin."""

    def __init__(self, responses):
        self.responses = list(responses)
        self._call_index = 0
        self.call_log = []

    def predict(self, prompt: str) -> str:
        self.call_log.append(prompt)
        result = self.responses[self._call_index % len(self.responses)]
        self._call_index += 1
        return result


@pytest.fixture
def handler():
    return ClassDiagramHandler(llm=None)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestOCLConstraintSchema:
    def test_systemspec_constraints_default_empty(self):
        spec = SystemClassSpec(classes=[{"className": "A", "attributes": []}])
        assert spec.constraints == []

    def test_constraint_roundtrips_through_model_dump(self):
        spec = SystemClassSpec(
            systemName="Conf",
            classes=[
                {"className": "Speaker", "attributes": []},
                {"className": "Session", "attributes": []},
            ],
            constraints=[
                {
                    "context": "Speaker",
                    "expression": "context Speaker inv: self.sessions->size() <= 1",
                    "name": "oneSession",
                }
            ],
        )
        dumped = spec.model_dump()
        assert "constraints" in dumped
        assert len(dumped["constraints"]) == 1
        assert dumped["constraints"][0]["context"] == "Speaker"
        assert "inv" in dumped["constraints"][0]["expression"]

    def test_constraint_name_optional(self):
        c = OCLConstraintSpec(context="Account", expression="context Account inv: self.balance >= 0")
        assert c.name is None


# ---------------------------------------------------------------------------
# _validate_constraints guard
# ---------------------------------------------------------------------------

class TestValidateConstraints:
    def test_keeps_constraint_with_known_context(self, handler):
        spec = {
            "classes": [{"className": "Account", "attributes": []}],
            "relationships": [],
            "constraints": [
                {"context": "Account", "expression": "context Account inv: self.balance >= 0"}
            ],
        }
        handler._validate_constraints(spec)
        assert len(spec["constraints"]) == 1

    def test_drops_constraint_with_unknown_context(self, handler):
        spec = {
            "classes": [{"className": "Account", "attributes": []}],
            "relationships": [],
            "constraints": [
                {"context": "Ghost", "expression": "context Ghost inv: self.x > 0"}
            ],
        }
        handler._validate_constraints(spec)
        assert spec["constraints"] == []

    def test_drops_constraint_with_empty_expression(self, handler):
        spec = {
            "classes": [{"className": "Account", "attributes": []}],
            "constraints": [{"context": "Account", "expression": "   "}],
        }
        handler._validate_constraints(spec)
        assert spec["constraints"] == []

    def test_empty_constraints_is_noop(self, handler):
        spec = {"classes": [{"className": "A", "attributes": []}], "constraints": []}
        handler._validate_constraints(spec)
        assert spec["constraints"] == []


# ---------------------------------------------------------------------------
# End-to-end via generate_complete_system
# ---------------------------------------------------------------------------

class TestConstraintCaptureEndToEnd:
    def _spec_json(self, constraints):
        return json.dumps({
            "systemName": "Conference",
            "classes": [
                {"className": "Speaker", "attributes": [{"name": "name", "type": "String"}], "methods": []},
                {"className": "Session", "attributes": [{"name": "timeSlot", "type": "String"}], "methods": []},
            ],
            "relationships": [
                {"type": "Association", "source": "Speaker", "target": "Session"}
            ],
            "constraints": constraints,
        })

    def test_stated_constraint_is_captured(self):
        """A stated rule ('at most one session per time slot') appears in the
        injected systemSpec as an OCL constraint."""
        constraint = {
            "context": "Speaker",
            "expression": (
                "context Speaker inv oneSessionPerSlot: "
                "self.sessions->forAll(s1, s2 | s1 <> s2 implies s1.timeSlot <> s2.timeSlot)"
            ),
            "name": "oneSessionPerSlot",
        }
        llm = RecordingFakeLLM([self._spec_json([constraint])])
        handler = ClassDiagramHandler(llm)

        result = handler.generate_complete_system(
            "Conference with speakers and sessions; a speaker can present at most "
            "one session per time slot",
            raw_request="conference speakers sessions, one session per slot",
        )

        assert result["action"] == "inject_complete_system"
        constraints = result["systemSpec"]["constraints"]
        assert len(constraints) == 1
        assert constraints[0]["context"] == "Speaker"
        assert "forAll" in constraints[0]["expression"]
        # The success message tells the user the stated rule was noted (worded
        # in plain, non-technical language — see FIX 3 wording softening).
        msg = result["message"].lower()
        assert "rule" in msg
        assert "aren't shown" in msg or "not shown" in msg

    def test_no_constraint_stated_yields_empty(self):
        """When the spec carries no constraints, the systemSpec constraints
        list is empty — nothing is hallucinated."""
        llm = RecordingFakeLLM([self._spec_json([])])
        handler = ClassDiagramHandler(llm)

        result = handler.generate_complete_system(
            "Conference with speakers and sessions",
            raw_request="conference speakers sessions",
        )

        assert result["action"] == "inject_complete_system"
        assert result["systemSpec"]["constraints"] == []
        # No constraint mention in the message when none were captured.
        assert "constraint" not in result["message"].lower()

    def test_constraint_with_unknown_context_is_dropped_end_to_end(self):
        """A constraint whose context class doesn't exist is stripped before
        it can reach the editor."""
        bad_constraint = {
            "context": "Nonexistent",
            "expression": "context Nonexistent inv: self.x > 0",
        }
        llm = RecordingFakeLLM([self._spec_json([bad_constraint])])
        handler = ClassDiagramHandler(llm)

        result = handler.generate_complete_system(
            "Conference", raw_request="conference",
        )
        assert result["systemSpec"]["constraints"] == []
