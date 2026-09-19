"""Class-diagram fidelity rules from the 2026-09-18 hotel run (Qwen3-30B).

Three stated requirements were lost between the request and the generated
app, and each one traced back to the compact schema or the generation prompt:

* "Every room is identified by its room number" landed as a plain
  ``roomNumber`` attribute — two rooms numbered "101" were both accepted.
  BESSER's SQLAlchemy template only emits ``unique=True`` for an attribute
  whose ``is_external_id`` is set, and the compact grammar had no way to say
  that, while rule 13 sent uniqueness to OCL instead.
* "An email address must have the usual shape" became the unanchored
  ``self.email.matches('.+@.+\\..+')`` that rule 13 itself taught, so
  "spaces in@email.com" passed.
* "the price actually agreed for that room in that booking ... and any extra
  charges recorded against those rooms" gave ReservedRoom only
  ``agreedPrice``; the extra charges were dropped.
"""

import sys
from pathlib import Path

# Make src/ importable when running pytest from the repo root.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import diagram_handlers.types.class_diagram_handler as handler_mod  # noqa: E402
from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler  # noqa: E402
from schemas.class_diagram import AttributeSpec  # noqa: E402
from schemas.compact_class_diagram import (  # noqa: E402
    COMPACT_ENCODING_RULES,
    CompactClassSpec,
    CompactRelationshipSpec,
    CompactSystemClassSpec,
    _parse_attribute,
)

PROMPT = ClassDiagramHandler(None)._get_system_generation_prompt()

# The unanchored example the live run copied verbatim (Python literal for
# the prompt text  .+@.+\..+ ).
UNANCHORED_EMAIL = ".+@.+\\..+"
ANCHORED_EMAIL = "^[^\\s@]+@[^\\s@]+\\.[A-Za-z]{2,}$"
ANCHORED_PHONE = "^\\+?[0-9]{7,15}$"


# ---------------------------------------------------------------------------
# (1) compact grammar: '!' marks a natural/external identifier
# ---------------------------------------------------------------------------

class TestExternalIdDecoration:
    def test_bang_suffix_sets_external_id_and_is_stripped(self):
        a = _parse_attribute("roomNumber: str!", False)
        assert (a.name, a.type) == ("roomNumber", "str")
        assert a.isExternalId is True
        assert a.isOptional is False

    def test_bang_suffix_on_the_entry_not_glued_to_the_type(self):
        a = _parse_attribute("roomNumber: str !", False)
        assert (a.name, a.type, a.isExternalId) == ("roomNumber", "str", True)

    def test_question_mark_still_means_optional_only(self):
        a = _parse_attribute("nickname: str?", False)
        assert a.isOptional is True
        assert a.isExternalId is False

    def test_plain_attribute_is_not_an_external_id(self):
        assert _parse_attribute("price: float", False).isExternalId is False

    def test_grammar_documents_the_marker(self):
        assert "'!'" in CompactClassSpec.model_fields["a"].description
        assert "'!'" in COMPACT_ENCODING_RULES
        assert "isExternalId" in COMPACT_ENCODING_RULES


# ---------------------------------------------------------------------------
# (2) the flag reaches the injected payload the same way isOptional does
# ---------------------------------------------------------------------------

def _hotel_compact():
    return CompactSystemClassSpec(
        name="Hotel",
        classes=[
            CompactClassSpec(n="Room", a=["id: int", "roomNumber: str!",
                                          "capacity: int", "notes: str?"],
                             m=[], k=""),
            CompactClassSpec(n="Booking", a=["id: int", "arrivalDate: date"],
                             m=[], k=""),
        ],
        rels=[CompactRelationshipSpec(
            f="Booking", t="Room", k="assoc",
            how_many_TARGET_for_one_SOURCE="1..*",
            how_many_SOURCE_for_one_TARGET="0..*", l="rooms")],
        ocl=[],
    )


def _attrs(result, class_name):
    cls = next(c for c in result["systemSpec"]["classes"]
               if c["className"] == class_name)
    return {a["name"]: a for a in cls["attributes"]}


class TestExternalIdReachesPayload:
    def test_canonical_schema_carries_the_flag(self):
        dumped = AttributeSpec(name="roomNumber", type="str",
                               isExternalId=True).model_dump()
        assert dumped["isExternalId"] is True
        assert AttributeSpec(name="x").model_dump()["isExternalId"] is False

    def test_inject_payload_attribute_has_boolean_flag(self, monkeypatch):
        """Through generate_complete_system on the compact path, after every
        guard: the attribute dict the frontend converter receives carries
        ``isExternalId`` as a real boolean, exactly like ``isOptional``."""
        monkeypatch.setattr(handler_mod, "COMPACT_SPEC_ENABLED", True,
                            raising=True)
        handler = ClassDiagramHandler(None)
        monkeypatch.setattr(handler, "predict_two_pass_structured",
                            lambda **kwargs: _hotel_compact())

        result = handler.generate_complete_system(
            "hotel; every room is identified by its room number")

        assert result["action"] == "inject_complete_system"
        room = _attrs(result, "Room")
        assert room["roomNumber"]["isExternalId"] is True
        assert room["roomNumber"]["type"] == "str"
        assert room["capacity"]["isExternalId"] is False
        assert room["notes"]["isOptional"] is True
        assert room["notes"]["isExternalId"] is False


# ---------------------------------------------------------------------------
# (3) rule 13: shape constraints are anchored and reject whitespace
# ---------------------------------------------------------------------------

class TestShapeConstraintExamples:
    def test_unanchored_email_example_is_gone(self):
        assert UNANCHORED_EMAIL not in PROMPT

    def test_anchored_email_and_phone_examples(self):
        assert ANCHORED_EMAIL in PROMPT
        assert ANCHORED_PHONE in PROMPT

    def test_states_the_anchoring_rule(self):
        low = PROMPT.lower()
        assert "anchored" in low
        assert "whitespace" in low


# ---------------------------------------------------------------------------
# (4) rule 13: single-attribute uniqueness is the external-id flag, not OCL
# ---------------------------------------------------------------------------

class TestUniquenessIsExternalId:
    def test_rule_names_the_flag_for_uniqueness(self):
        low = PROMPT.lower()
        assert "isexternalid" in low
        assert "identified by its room number" in low
        assert "emails must be unique" in low

    def test_uniqueness_is_no_longer_an_ocl_trigger(self):
        assert 'express — uniqueness ("emails must be unique")' not in PROMPT

    def test_reasoning_pass_does_not_plan_ocl_for_uniqueness(self, monkeypatch):
        """The reasoning prompt is built inside generate_complete_system;
        capture it and check item 6 no longer lists uniqueness as an OCL
        reason."""
        captured = {}
        handler = ClassDiagramHandler(None)

        def fake_two_pass(**kwargs):
            captured.update(kwargs)
            return _hotel_compact()

        monkeypatch.setattr(handler_mod, "COMPACT_SPEC_ENABLED", True,
                            raising=True)
        monkeypatch.setattr(handler, "predict_two_pass_structured",
                            fake_two_pass)
        handler.generate_complete_system("hotel")
        reasoning = captured["reasoning_prompt"]
        assert "constraint (uniqueness," not in reasoning
        assert "isExternalId" in reasoning


# ---------------------------------------------------------------------------
# (5) rule 17: further per-link amounts stay on the link class
# ---------------------------------------------------------------------------

class TestLinkClassExtraAmounts:
    def test_extra_charges_are_a_link_attribute(self):
        assert "associationClass" in PROMPT
        low = PROMPT.lower()
        assert "agreed price and extra charges" in low
        assert "all those per-link attributes" in low
        assert "do not drop amounts" in low
        assert "do not replace it with two ordinary links" in low
