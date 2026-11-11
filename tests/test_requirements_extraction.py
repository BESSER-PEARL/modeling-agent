"""Requirements extracted from a complete-system request, and the elements that realize them.

The class diagram handler extracts the requirements of the user's own message,
lists the ones a class diagram can realize in its structured pass, and reports
which elements realize each one.
The editor records both in the project trace, so every reference must name an
element of the FINAL spec and every requirement must quote the message.
"""

import logging

import pytest

import diagram_handlers.types.class_diagram_handler as handler_mod
from common.agent_config import MAX_REQUIREMENT_CHARS, MAX_REQUIREMENTS
from common.errors import LLMPredictionError
from llm import byok
from llm.model_config import MODEL_GENERATION_SMALL
from llm.provider import OutputTruncated
from diagram_handlers.types.class_diagram_requirements import (
    REALIZATION_TOKENS_PER_REQUIREMENT,
    extract_requirements,
    extraction_model,
    normalise_for_quote,
    resolve_realizations,
)
from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler
from prompts import class_diagram as class_prompts
from schemas import RealizationSpec, RequirementsExtractionSpec
from schemas.compact_class_diagram import (
    CompactClassSpec,
    CompactRelationshipSpec,
    CompactSystemClassSpec,
    expand_compact_spec,
)

REQUEST = (
    "A hotel system for a small family hotel on the coast. Guests book one room per stay and the "
    "receptionist checks them out when they leave. Each room is identified by its   number. The "
    "number of guests on a booking must not exceed the capacity of its room. Use FastAPI."
)


def _extraction(*items):
    return RequirementsExtractionSpec.model_validate({"requirements": [
        {"text": text, "quote": quote, "kind": kind} for text, quote, kind in items
    ]})


class _Handler:
    """Answers predict_structured with a canned extraction and records the call."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def predict_structured(self, prompt, schema, **kwargs):
        self.calls.append((prompt, schema, kwargs))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

class TestExtraction:
    def test_requirements_are_numbered_in_order_with_their_quote(self):
        handler = _Handler(_extraction(
            ("A room is identified by its number", "each room is identified by its number", "uniqueness"),
            ("Guests never exceed the room's capacity", "must not exceed the capacity of its room", "rule"),
            ("The backend uses FastAPI", "Use FastAPI", "technical"),
        ))
        requirements = extract_requirements(handler, REQUEST)
        assert [(r["id"], r["kind"]) for r in requirements] == [
            ("R1", "uniqueness"), ("R2", "rule"), ("R3", "technical")]
        assert requirements[0]["quote"] == "each room is identified by its number"
        prompt, schema, kwargs = handler.calls[0]
        assert REQUEST in prompt and schema is RequirementsExtractionSpec
        assert kwargs["model"] == handler_mod.MODEL_GENERATION_SMALL

    def test_a_requirement_the_request_does_not_state_is_dropped(self, caplog):
        """A quote that is not in the message means the item was not stated there."""
        handler = _Handler(_extraction(
            ("Users log in", "users must log in", "action"),
            ("The backend uses FastAPI", "Use FastAPI", "technical"),
        ))
        with caplog.at_level(logging.WARNING):
            requirements = extract_requirements(handler, REQUEST)
        assert [(r["id"], r["text"]) for r in requirements] == [("R1", "The backend uses FastAPI")]
        assert "users must log in" in caplog.text

    def test_an_overlong_requirement_is_dropped(self):
        long_text = "x" * (MAX_REQUIREMENT_CHARS + 1)
        handler = _Handler(_extraction((long_text, "Use FastAPI", "technical")))
        assert extract_requirements(handler, REQUEST) == []

    def test_an_overlong_quote_is_dropped(self):
        """A quote is a phrase or a sentence; the request it must be found in can be 64k characters."""
        request = "x" * (MAX_REQUIREMENT_CHARS + 1)
        handler = _Handler(_extraction(("Everything", request, "rule")))
        assert extract_requirements(handler, request) == []

    def test_the_cap_applies_to_the_requirements_kept(self):
        """Items dropped first: a list cut at the cap reaches it, so the Spec-Driven Agent reads it as full."""
        dropped = [("Users log in", "users must log in", "action")] * 3
        kept = [("The backend uses FastAPI", "Use FastAPI", "technical")] * (MAX_REQUIREMENTS + 2)
        requirements = extract_requirements(_Handler(_extraction(*dropped, *kept)), REQUEST)
        assert len(requirements) == MAX_REQUIREMENTS
        assert requirements[-1]["id"] == f"R{MAX_REQUIREMENTS}"

    def test_a_failed_extraction_builds_the_diagram_without_requirements(self, caplog):
        handler = _Handler(LLMPredictionError("no answer"))
        with caplog.at_level(logging.WARNING):
            assert extract_requirements(handler, REQUEST) == []
        assert "extraction failed" in caplog.text

    def test_a_list_too_long_for_the_token_cap_builds_the_diagram_without_requirements(self, caplog):
        """A truncated reply is not a provider fault: the diagram's own call can still succeed."""
        class LengthFinishReasonError(Exception):
            pass

        truncated = OutputTruncated("cut at max_tokens")
        truncated.__cause__ = LengthFinishReasonError()
        failure = LLMPredictionError("cut at max_tokens")
        failure.__cause__ = truncated
        with caplog.at_level(logging.WARNING):
            assert extract_requirements(_Handler(failure), REQUEST) == []
        assert "extraction failed" in caplog.text

    def test_a_quota_or_key_error_is_raised_not_skipped(self):
        """The diagram's own call would only fail the same way, after the user waited for it."""
        class RateLimitError(Exception):
            pass

        failure = LLMPredictionError("quota")
        failure.__cause__ = RateLimitError("429")
        with pytest.raises(LLMPredictionError):
            extract_requirements(_Handler(failure), REQUEST)

    def test_quotes_compare_without_case_or_whitespace_differences(self):
        assert normalise_for_quote("  Each room\n is  IDENTIFIED ") == "each room is identified"


# ---------------------------------------------------------------------------
# Realizations, resolved against the final spec
# ---------------------------------------------------------------------------

SPEC = {
    "classes": [
        {"className": "Room", "attributes": [{"name": "number"}, {"name": "capacity"}], "methods": []},
        {"className": "Booking", "attributes": [{"name": "guestCount"}], "methods": [{"name": "checkOut"}]},
        {"className": "BookingStatus", "isEnumeration": True,
         "attributes": [{"name": "PENDING"}, {"name": "CONFIRMED"}]},
    ],
    "relationships": [
        {"type": "Association", "source": "Booking", "target": "Room", "name": "rooms", "sourceRole": "bookings"},
        {"type": "Inheritance", "source": "Suite", "target": "Room", "name": None},
    ],
    "constraints": [
        {"context": "Booking", "expression": "context Booking inv guestsWithinCapacity: self.guestCount <= 4"},
        # The editor names an invariant by its expression, never by this field.
        {"context": "Room", "expression": "context Room inv positiveCapacity: self.capacity > 0",
         "name": "capacityIsPositive"},
        {"context": "Room", "expression": "context Room inv: self.number <> ''", "name": "numbered"},
    ],
}
IDS = {"R1", "R2", "R3"}


def _resolve(*elements, requirement="R1"):
    resolved = resolve_realizations([{"requirement": requirement, "elements": list(elements)}], SPEC, IDS)
    return resolved[0]["elements"] if resolved else []


class TestRealizations:
    @pytest.mark.parametrize("reference, element", [
        ("Room", {"type": "Class", "name": "Room"}),
        ("Room.number", {"type": "Property", "name": "number", "owner": "Room"}),
        ("BookingStatus.CONFIRMED", {"type": "EnumerationLiteral", "name": "CONFIRMED", "owner": "BookingStatus"}),
        ("Booking.checkOut()", {"type": "Method", "name": "checkOut", "owner": "Booking"}),
        ("Booking.rooms", {"type": "BinaryAssociation", "name": "rooms", "owner": "Booking"}),
        ("Room.bookings", {"type": "BinaryAssociation", "name": "rooms", "owner": "Booking"}),
        ("inv guestsWithinCapacity", {"type": "Constraint", "name": "guestsWithinCapacity", "owner": "Booking"}),
        ("inv positiveCapacity", {"type": "Constraint", "name": "positiveCapacity", "owner": "Room"}),
    ])
    def test_each_kind_of_reference_names_one_element(self, reference, element):
        assert _resolve(reference) == [element]

    @pytest.mark.parametrize("reference", [
        "Guest", "Room.price", "Room.checkOut()", "inv missing", "Room -> Booking", "Suite.room",
        "inv capacityIsPositive", "inv numbered",
    ])
    def test_a_reference_to_no_element_is_left_out(self, reference, caplog):
        with caplog.at_level(logging.WARNING):
            assert _resolve(reference) == []
        assert reference in caplog.text

    def test_an_enumeration_named_whole_brings_its_literals(self):
        """'A booking is pending, confirmed or cancelled' is realized by the enumeration and its values."""
        assert _resolve("BookingStatus") == [
            {"type": "Enumeration", "name": "BookingStatus"},
            {"type": "EnumerationLiteral", "name": "PENDING", "owner": "BookingStatus"},
            {"type": "EnumerationLiteral", "name": "CONFIRMED", "owner": "BookingStatus"},
        ]

    def test_an_enumeration_with_named_literals_brings_only_those(self):
        """'Only a pending booking can be confirmed' does not claim every status."""
        assert _resolve("BookingStatus", "BookingStatus.PENDING") == [
            {"type": "Enumeration", "name": "BookingStatus"},
            {"type": "EnumerationLiteral", "name": "PENDING", "owner": "BookingStatus"},
        ]

    def test_a_realization_may_name_no_element(self):
        """An empty entry is dropped later; it must not fail the whole diagram's validation."""
        assert RealizationSpec.model_validate({"requirement": "R1", "elements": []}).elements == []

    def test_an_unknown_requirement_is_left_out(self):
        assert resolve_realizations([{"requirement": "R9", "elements": ["Room"]}], SPEC, IDS) == []

    def test_repeated_entries_merge_without_duplicates(self):
        resolved = resolve_realizations([
            {"requirement": "R2", "elements": ["Room.capacity", "inv guestsWithinCapacity"]},
            {"requirement": "R2", "elements": ["Room.capacity"]},
        ], SPEC, IDS)
        assert resolved == [{"requirement": "R2", "elements": [
            {"type": "Property", "name": "capacity", "owner": "Room"},
            {"type": "Constraint", "name": "guestsWithinCapacity", "owner": "Booking"},
        ]}]


# ---------------------------------------------------------------------------
# Through the handler
# ---------------------------------------------------------------------------

def _hotel_compact(realizations=()):
    return CompactSystemClassSpec(
        name="Hotel",
        classes=[
            CompactClassSpec(n="Room", a=["roomNumber: str!", "capacity: int"], m=[], k=""),
            CompactClassSpec(n="Booking", a=["guestCount: int"], m=["checkOut()"], k=""),
        ],
        rels=[CompactRelationshipSpec(f="Booking", t="Room", k="assoc", how_many_TARGET_for_one_SOURCE="1",
                                      how_many_SOURCE_for_one_TARGET="0..*", l="room", ls="bookings")],
        ocl=["context Booking inv guestsWithinCapacity: self.guestCount <= self.room.capacity"],
        realizations=list(realizations),
    )


@pytest.fixture
def compact_handler(monkeypatch):
    monkeypatch.setattr(handler_mod, "COMPACT_SPEC_ENABLED", True, raising=True)
    return ClassDiagramHandler(None)


class TestHandler:
    def test_the_reply_carries_the_requirements_and_what_realizes_them(self, compact_handler, monkeypatch):
        extracted = _extraction(
            ("A room is identified by its number", "each room is identified by its number", "uniqueness"),
            ("Guests never exceed the room's capacity", "must not exceed the capacity of its room", "rule"),
            ("The backend uses FastAPI", "Use FastAPI", "technical"),
        )
        captured = {}

        def fake_two_pass(**kwargs):
            captured.update(kwargs)
            return _hotel_compact([
                {"requirement": "R1", "elements": ["Room.roomNumber"]},
                {"requirement": "R2", "elements": ["inv guestsWithinCapacity", "Booking.guestCount", "Guest.age"]},
                {"requirement": "R3", "elements": ["Room"]},
            ])

        monkeypatch.setattr(compact_handler, "predict_structured", lambda *a, **k: extracted)
        monkeypatch.setattr(compact_handler, "predict_two_pass_structured", fake_two_pass)
        result = compact_handler.generate_complete_system(f"User Request: {REQUEST}", raw_request=REQUEST)

        assert [r["id"] for r in result["requirements"]] == ["R1", "R2", "R3"]
        assert result["realizations"] == [
            {"requirement": "R1", "elements": [{"type": "Property", "name": "roomNumber", "owner": "Room"}]},
            {"requirement": "R2", "elements": [
                {"type": "Constraint", "name": "guestsWithinCapacity", "owner": "Booking"},
                {"type": "Property", "name": "guestCount", "owner": "Booking"},
            ]},
        ]
        assert result["requirementsModel"] == {"name": MODEL_GENERATION_SMALL, "provider": "openai"}
        assert "realizations" not in result["systemSpec"], "the editor converts systemSpec into the model"
        assert "R2 [rule]. Guests never exceed the room's capacity" in captured["user_request"]
        assert "R2 [rule]" not in captured["reasoning_prompt"], "the reasoning pass keeps the raw request"
        # A technology is realized by the application, not a class: not listed, so not realized.
        assert "R3 [technical]" not in captured["user_request"]
        assert captured["max_tokens"] == (compact_handler._structured_max_tokens(CompactSystemClassSpec)
                                          + 2 * REALIZATION_TOKENS_PER_REQUIREMENT)

    def test_without_the_users_own_words_nothing_is_extracted(self, compact_handler, monkeypatch):
        def no_extraction(*args, **kwargs):
            raise AssertionError("an enriched request has nothing to quote")

        monkeypatch.setattr(compact_handler, "predict_structured", no_extraction)
        monkeypatch.setattr(compact_handler, "predict_two_pass_structured", lambda **kwargs: _hotel_compact())
        result = compact_handler.generate_complete_system("hotel")
        assert "requirements" not in result and "realizations" not in result
        assert "realizations" not in result["systemSpec"]

    def test_a_request_stating_nothing_sends_no_requirements(self, compact_handler, monkeypatch):
        monkeypatch.setattr(compact_handler, "predict_structured", lambda *a, **k: _extraction())
        monkeypatch.setattr(compact_handler, "predict_two_pass_structured", lambda **kwargs: _hotel_compact())
        result = compact_handler.generate_complete_system(REQUEST, raw_request=REQUEST)
        assert "requirements" not in result and "requirementsModel" not in result

    def test_a_trivial_request_stays_one_call(self, compact_handler, monkeypatch):
        """Below the reasoning pass's threshold a request is trivial: no extraction call."""
        def no_extraction(*args, **kwargs):
            raise AssertionError("a trivial request must not pay for an extraction")

        short = "a todo app with tasks"
        assert len(short) < compact_handler._TWO_PASS_MIN_LENGTH
        monkeypatch.setattr(compact_handler, "predict_structured", no_extraction)
        monkeypatch.setattr(compact_handler, "predict_two_pass_structured", lambda **kwargs: _hotel_compact())
        assert "requirements" not in compact_handler.generate_complete_system(short, raw_request=short)


class TestExtractionModel:
    """The editor records the extraction as a call to this LLM, so it must be the one that ran."""

    def test_the_server_key_runs_the_small_tier(self):
        assert extraction_model(ClassDiagramHandler(None)) == {"name": MODEL_GENERATION_SMALL, "provider": "openai"}

    def test_a_users_key_runs_its_providers_small_model(self):
        token = byok.set_current("anthropic", "sk-test")
        try:
            assert extraction_model(ClassDiagramHandler(None)) == {
                "name": byok.resolve_model("anthropic", MODEL_GENERATION_SMALL, None), "provider": "anthropic"}
        finally:
            byok.reset_current(token)

    def test_a_model_the_user_picked_runs_every_call(self):
        token = byok.set_current("openai", "sk-test", "my-model")
        try:
            assert extraction_model(ClassDiagramHandler(None)) == {"name": "my-model", "provider": "openai"}
        finally:
            byok.reset_current(token)


def test_the_compact_spec_keeps_its_realizations():
    realization = {"requirement": "R1", "elements": ["Room.roomNumber"]}
    expanded = expand_compact_spec(_hotel_compact([realization]))
    assert [r.model_dump() for r in expanded.realizations] == [realization]


def test_the_requirements_block_lists_each_requirement_with_its_kind():
    block = class_prompts.requirements_block([
        {"id": "R1", "kind": "uniqueness", "text": "A room is identified by its number"},
    ])
    assert "R1 [uniqueness]. A room is identified by its number" in block
    assert "stays the authority" in block


def test_the_requirements_block_says_what_each_kind_is_realized_by():
    """An element left out of every realization is traced to the prompt alone, not to its requirement."""
    block = class_prompts.requirements_block([{"id": "R1", "kind": "computed", "text": "Total = nights x price"}])
    assert "EVERY element you modeled because of it" in block
    for kind in ("data", "uniqueness, validation", "rule", "computed", "transition", "action"):
        assert f"\n- {kind}: " in block
    assert "the attribute holding the value, the attributes it is computed from" in block
