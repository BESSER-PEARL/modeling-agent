"""Tests for the identifier-name sanitizer guard.

A class named with characters BUML rejects as an identifier (hyphens, spaces,
dots) makes metamodel validation fail, and the editor's post-injection auto-fix
loop can never repair it — it emits a "rename X" request, but X was never
accepted as a class, so the modify handler can't find it and the loop spins
(a real pilot incident: a user stuck ~25 min on an app named with hyphens).

The guard rewrites such names to valid identifiers BEFORE the spec is injected,
and rewrites every reference to a renamed class/enum (relationship endpoints,
class-typed attribute types, OCL contexts) so the spec stays consistent.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler


@pytest.fixture
def handler():
    return ClassDiagramHandler(llm=None)


def _spec(classes, relationships=None, constraints=None):
    return {
        "systemName": "T",
        "classes": classes,
        "relationships": relationships or [],
        "constraints": constraints or [],
    }


class TestIdentifierSanitize:
    def test_to_identifier_pascal_and_camel(self, handler):
        assert handler._to_identifier("risk-awareness-recommendation", pascal=True) == "RiskAwarenessRecommendation"
        assert handler._to_identifier("user profile", pascal=True) == "UserProfile"
        assert handler._to_identifier("first-name", pascal=False) == "firstName"
        # leading digit gets an underscore prefix
        assert handler._to_identifier("123-abc", pascal=True) == "_123Abc"
        # nothing usable -> empty (caller leaves the original alone)
        assert handler._to_identifier("---", pascal=True) == ""

    def test_hyphenated_class_name_is_pascalcased(self, handler):
        spec = _spec([
            {"className": "risk-awareness-recommendation", "isEnumeration": False,
             "attributes": [{"name": "title", "type": "String"}]},
        ])
        handler._sanitize_identifier_names(spec)
        assert spec["classes"][0]["className"] == "RiskAwarenessRecommendation"

    def test_references_to_renamed_class_are_rewritten(self, handler):
        spec = _spec(
            classes=[
                {"className": "risk-awareness", "isEnumeration": False, "attributes": []},
                {"className": "User", "isEnumeration": False, "attributes": []},
            ],
            relationships=[
                {"type": "ClassBidirectional", "source": "User", "target": "risk-awareness"},
            ],
            constraints=[{"context": "risk-awareness", "expression": "inv: true"}],
        )
        handler._sanitize_identifier_names(spec)
        rel = spec["relationships"][0]
        assert rel["source"] == "User"
        assert rel["target"] == "RiskAwareness"
        assert spec["constraints"][0]["context"] == "RiskAwareness"

    def test_enum_typed_attribute_reference_rewritten(self, handler):
        spec = _spec([
            {"className": "Order", "isEnumeration": False,
             "attributes": [{"name": "status", "type": "order-status"}]},
            {"className": "order-status", "isEnumeration": True,
             "attributes": [{"name": "OPEN"}, {"name": "CLOSED"}]},
        ])
        handler._sanitize_identifier_names(spec)
        enum = next(c for c in spec["classes"] if c.get("isEnumeration"))
        assert enum["className"] == "OrderStatus"
        order = next(c for c in spec["classes"] if c["className"] == "Order")
        assert order["attributes"][0]["type"] == "OrderStatus"

    def test_member_names_fixed_in_place(self, handler):
        spec = _spec([
            {"className": "User", "isEnumeration": False,
             "attributes": [{"name": "first-name", "type": "String"}],
             "methods": [{"name": "do-thing", "returnType": "void",
                          "parameters": [{"name": "raw value", "type": "String"}]}]},
        ])
        handler._sanitize_identifier_names(spec)
        cls = spec["classes"][0]
        assert cls["attributes"][0]["name"] == "firstName"
        assert cls["methods"][0]["name"] == "doThing"
        assert cls["methods"][0]["parameters"][0]["name"] == "rawValue"

    def test_valid_names_are_left_untouched(self, handler):
        spec = _spec(
            classes=[
                {"className": "User", "isEnumeration": False,
                 "attributes": [{"name": "email", "type": "String"}]},
                {"className": "Order", "isEnumeration": False, "attributes": []},
            ],
            relationships=[{"type": "ClassBidirectional", "source": "User", "target": "Order"}],
        )
        before = repr(spec)
        handler._sanitize_identifier_names(spec)
        assert repr(spec) == before  # no changes when everything is already valid

    def test_empty_or_malformed_spec_does_not_crash(self, handler):
        handler._sanitize_identifier_names({})
        handler._sanitize_identifier_names({"classes": None})
        handler._sanitize_identifier_names({"classes": ["not-a-dict", 5]})
        # all no-ops, no exception
