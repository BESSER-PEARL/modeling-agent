"""A closed value set must survive the model forgetting to declare its enum.

Some models (e.g. Qwen3-30B-A3B) type an attribute with an enumeration name
but omit the enumeration from ``classes``; ``_rewrite_class_typed_attributes``
- whose job is to kill hallucinated type references - then coerced that name
to ``String``, so every closed value set was silently lost::

    [ClassDiagram] Coerced unknown attribute type Ticket.status : StatusEnum -> String

The missing enumerations were this coercion, not a modelling choice, so the
fix is deterministic and belongs here, not in a firmer prompt sentence.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler
from schemas.class_diagram import RecoveredEnumerationsSpec


TICKETING_SPEC = (
    "The status of a ticket describes how far it has got. It can only be one "
    "of five things: new, triaged, in progress, resolved, or closed. The "
    "urgency describes how pressing it is. It can only be one of four "
    "things: low, medium, high, or critical."
)


class _Recovering(ClassDiagramHandler):
    """Handler whose recovery call is answered from a canned table."""

    def __init__(self, answers, *, explode=False):
        super().__init__(llm=None)
        self.answers = answers
        self.explode = explode
        self.calls = 0

    def predict_structured(self, prompt, response_schema, **kwargs):
        self.calls += 1
        self.last_prompt = prompt
        if self.explode:
            raise RuntimeError("provider is down")
        return RecoveredEnumerationsSpec(enumerations=[
            {"typeName": name, "literals": literals}
            for name, literals in self.answers.items()
        ])


def _ticket_spec():
    """The observed output: the enum is referenced, never declared."""
    return {
        "systemName": "Ticketing",
        "classes": [
            {"className": "Ticket", "isEnumeration": False, "methods": [],
             "attributes": [
                 {"name": "reference", "type": "str"},
                 {"name": "status", "type": "StatusEnum"},
                 {"name": "urgency", "type": "UrgencyEnum"},
             ]},
        ],
        "relationships": [],
    }


def _enum(spec, name):
    return next((c for c in spec["classes"]
                 if c.get("isEnumeration") and c["className"] == name), None)


def _literals(spec, name):
    return [a["name"] for a in _enum(spec, name)["attributes"]]


def _attr_type(spec, cls, attr):
    owner = next(c for c in spec["classes"] if c["className"] == cls)
    return next(a["type"] for a in owner["attributes"] if a["name"] == attr)


class TestTheReferencedEnumerationIsDeclared:

    def test_without_the_guard_the_closed_set_becomes_a_string(self):
        """The behaviour this exists to stop — pinned so it cannot come back."""
        spec = _ticket_spec()
        ClassDiagramHandler(llm=None)._rewrite_class_typed_attributes(spec)

        assert _attr_type(spec, "Ticket", "status") == "String"
        assert _attr_type(spec, "Ticket", "urgency") == "String"
        assert not [c for c in spec["classes"] if c.get("isEnumeration")]

    def test_the_enumeration_is_declared_and_the_attribute_keeps_its_type(self):
        handler = _Recovering({
            "StatusEnum": ["NEW", "TRIAGED", "IN_PROGRESS", "RESOLVED", "CLOSED"],
            "UrgencyEnum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        })
        spec = _ticket_spec()
        handler._declare_referenced_enumerations(spec, TICKETING_SPEC)
        handler._rewrite_class_typed_attributes(spec)

        assert _attr_type(spec, "Ticket", "status") == "StatusEnum"
        assert _attr_type(spec, "Ticket", "urgency") == "UrgencyEnum"
        assert _literals(spec, "StatusEnum") == [
            "NEW", "TRIAGED", "IN_PROGRESS", "RESOLVED", "CLOSED"]
        assert _literals(spec, "UrgencyEnum") == ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_declaration_order_is_the_order_the_spec_lists(self):
        """"Every new ticket starts out new" — the first literal is a decision."""
        handler = _Recovering({"StatusEnum": ["NEW", "TRIAGED", "CLOSED"]})
        spec = _ticket_spec()
        handler._declare_referenced_enumerations(spec, TICKETING_SPEC)

        assert _literals(spec, "StatusEnum")[0] == "NEW"

    def test_literals_keep_their_underscores(self):
        handler = _Recovering({"StatusEnum": ["not arrived", "checked-in", "CHECKED_OUT"]})
        spec = _ticket_spec()
        handler._declare_referenced_enumerations(spec, TICKETING_SPEC)

        assert _literals(spec, "StatusEnum") == [
            "NOT_ARRIVED", "CHECKED_IN", "CHECKED_OUT"]

    def test_the_enumeration_class_is_shaped_like_every_other_one(self):
        handler = _Recovering({"StatusEnum": ["NEW", "CLOSED"]})
        spec = _ticket_spec()
        handler._declare_referenced_enumerations(spec, TICKETING_SPEC)

        enum = _enum(spec, "StatusEnum")
        assert enum["methods"] == [] and enum["isAbstract"] is False
        assert all(a["type"] is None and a["visibility"] == "public"
                   for a in enum["attributes"])


class TestItFailsClosed:

    def test_a_declared_model_makes_no_call_at_all(self):
        """A model that declares its enums must pay nothing for this guard."""
        handler = _Recovering({})
        spec = {
            "systemName": "T", "relationships": [],
            "classes": [
                {"className": "Ticket", "isEnumeration": False, "methods": [],
                 "attributes": [{"name": "status", "type": "StatusEnum"}]},
                {"className": "StatusEnum", "isEnumeration": True, "methods": [],
                 "attributes": [{"name": "NEW", "type": None}]},
            ],
        }
        handler._declare_referenced_enumerations(spec, TICKETING_SPEC)

        assert handler.calls == 0

    def test_a_plain_primitive_model_makes_no_call(self):
        handler = _Recovering({})
        spec = {"systemName": "T", "relationships": [], "classes": [
            {"className": "Ticket", "isEnumeration": False, "methods": [],
             "attributes": [{"name": "title", "type": "str"}]}]}
        handler._declare_referenced_enumerations(spec, TICKETING_SPEC)

        assert handler.calls == 0

    def test_a_failed_recovery_leaves_the_spec_exactly_as_it_was(self):
        handler = _Recovering({}, explode=True)
        spec = _ticket_spec()
        before = repr(spec)
        handler._declare_referenced_enumerations(spec, TICKETING_SPEC)

        assert repr(spec) == before

    def test_an_empty_answer_is_not_turned_into_an_enumeration(self):
        """"Return an EMPTY list when the spec closes nothing" must be obeyed."""
        handler = _Recovering({"StatusEnum": [], "UrgencyEnum": ["ONLY_ONE"]})
        spec = _ticket_spec()
        handler._declare_referenced_enumerations(spec, TICKETING_SPEC)

        assert not [c for c in spec["classes"] if c.get("isEnumeration")]

    def test_a_type_naming_a_real_class_is_left_to_the_association_guard(self):
        handler = _Recovering({"Location": ["A", "B"]})
        spec = {"systemName": "T", "relationships": [], "classes": [
            {"className": "Trip", "isEnumeration": False, "methods": [],
             "attributes": [{"name": "startState", "type": "Location"}]},
            {"className": "Location", "isEnumeration": False, "methods": [],
             "attributes": [{"name": "lat", "type": "float"}]}]}
        handler._declare_referenced_enumerations(spec, TICKETING_SPEC)

        assert handler.calls == 0
        assert not [c for c in spec["classes"] if c.get("isEnumeration")]

    def test_a_hallucinated_type_can_still_be_declined(self):
        """The call decides; being an unknown PascalCase type is not enough."""
        handler = _Recovering({"Money": []})
        spec = {"systemName": "T", "relationships": [], "classes": [
            {"className": "Bill", "isEnumeration": False, "methods": [],
             "attributes": [{"name": "amount", "type": "Money"}]}]}
        handler._declare_referenced_enumerations(spec, TICKETING_SPEC)
        handler._rewrite_class_typed_attributes(spec)

        assert _attr_type(spec, "Bill", "amount") == "String"

    def test_the_request_text_is_what_the_call_is_asked_about(self):
        handler = _Recovering({"StatusEnum": ["NEW", "CLOSED"]})
        handler._declare_referenced_enumerations(_ticket_spec(), TICKETING_SPEC)

        assert "StatusEnum" in handler.last_prompt
        assert "Ticket.status" in handler.last_prompt
        assert TICKETING_SPEC in handler.last_prompt

    @pytest.mark.parametrize("text", ["", None])
    def test_no_request_text_means_no_call(self, text):
        handler = _Recovering({"StatusEnum": ["NEW", "CLOSED"]})
        handler._declare_referenced_enumerations(_ticket_spec(), text)

        assert handler.calls == 0

    def test_a_type_that_is_not_an_identifier_is_never_made_a_class(self):
        """_sanitize_identifier_names has already run and does not touch an
        unknown TYPE, so a spaced name here would become an invalid class."""
        handler = _Recovering({"Order Status": ["A", "B"]})
        spec = {"systemName": "T", "relationships": [], "classes": [
            {"className": "Order", "isEnumeration": False, "methods": [],
             "attributes": [{"name": "status", "type": "Order Status"}]}]}
        handler._declare_referenced_enumerations(spec, TICKETING_SPEC)

        assert handler.calls == 0
        assert not [c for c in spec["classes"] if c.get("isEnumeration")]

    def test_an_enum_name_with_no_status_ish_suffix_is_still_considered(self):
        """"never free text: one of Monday..Friday" — DayOfWeek is an enum too."""
        handler = _Recovering({"DayOfWeek": ["MONDAY", "TUESDAY", "FRIDAY"]})
        spec = {"systemName": "T", "relationships": [], "classes": [
            {"className": "TimeSlot", "isEnumeration": False, "methods": [],
             "attributes": [{"name": "day", "type": "DayOfWeek"}]}]}
        handler._declare_referenced_enumerations(spec, TICKETING_SPEC)

        assert _literals(spec, "DayOfWeek") == ["MONDAY", "TUESDAY", "FRIDAY"]

    def test_a_lowercase_unknown_type_is_left_alone(self):
        handler = _Recovering({"weirdthing": ["A", "B"]})
        spec = {"systemName": "T", "relationships": [], "classes": [
            {"className": "X", "isEnumeration": False, "methods": [],
             "attributes": [{"name": "y", "type": "weirdthing"}]}]}
        handler._declare_referenced_enumerations(spec, TICKETING_SPEC)

        assert handler.calls == 0
