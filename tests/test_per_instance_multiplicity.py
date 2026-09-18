"""Per-instance multiplicity encoding in the compact class-diagram spec.

The compact schema used to ask for "source multiplicity" / "target
multiplicity". The model answered the ORM has-many question and wrote the
count on the near end, so every association came out inverted: observed live
2026-09-17, where "each booking is handled by exactly one employee" became
Booking 1 -- 0..* Employee and made Employee.booking_id NOT NULL, so no
employee could be created before a booking existed.

The fields now ask two separate per-instance questions. These tests pin the
mapping onto UML's far-end convention, which everything downstream assumes
(ClassDiagramConverter.ts and class_diagram_processor.py both type the source
Property by the source class using source.multiplicity).
"""
import pytest

from src.schemas.compact_class_diagram import (
    CompactClassSpec,
    CompactRelationshipSpec,
    CompactSystemClassSpec,
    expand_compact_spec,
)


def _cls(name):
    return CompactClassSpec(n=name, a=[], m=[], k="")


def _expand(*rels, classes=("Booking", "Employee", "Guest", "Person", "Room")):
    spec = CompactSystemClassSpec(
        name="HotelBookingSystem",
        classes=[_cls(c) for c in classes],
        rels=list(rels),
        ocl=[],
    )
    return expand_compact_spec(spec)


def _rel(f, t, t_per_f, f_per_t, k="assoc", l=""):
    return CompactRelationshipSpec(
        f=f, t=t, k=k, l=l,
        how_many_TARGET_for_one_SOURCE=t_per_f,
        how_many_SOURCE_for_one_TARGET=f_per_t,
    )


# -- the far-end mapping ------------------------------------------------
def test_source_multiplicity_counts_sources_per_target():
    out = _expand(_rel("Booking", "Person", "1", "0..*"))
    rel = out.relationships[0]
    assert rel.sourceMultiplicity == "0..*"
    assert rel.targetMultiplicity == "1"


def test_the_contact_sentence_no_longer_inverts():
    """"each booking has one person acting as its contact" +
    "any individual may hold several bookings"."""
    out = _expand(_rel("Booking", "Person", "1", "0..*", l="contact"))
    rel = out.relationships[0]
    # For ONE Person there are 0..* Bookings -> the Booking end reads 0..*.
    assert (rel.source, rel.sourceMultiplicity) == ("Booking", "0..*")
    # For ONE Booking there is exactly 1 contact Person.
    assert (rel.target, rel.targetMultiplicity) == ("Person", "1")


def test_the_employee_sentence_no_longer_inverts():
    """"each booking is handled by exactly one employee" +
    "an employee may be responsible for many bookings"."""
    out = _expand(_rel("Booking", "Employee", "1", "0..*", l="handledBy"))
    rel = out.relationships[0]
    assert rel.sourceMultiplicity == "0..*", "many bookings per employee"
    assert rel.targetMultiplicity == "1", "one employee per booking"


def test_the_guest_sentence_stays_many_to_many():
    """"a given guest may appear on several bookings" +
    "each booking lists at least one guest"."""
    out = _expand(_rel("Booking", "Guest", "1..*", "0..*", l="guests"))
    rel = out.relationships[0]
    assert rel.sourceMultiplicity == "0..*"
    assert rel.targetMultiplicity == "1..*"


def test_a_genuine_one_to_many_is_preserved():
    """A ReservedRoom really does belong to one Booking."""
    out = _expand(_rel("Booking", "Room", "1..*", "1", l="rooms"))
    rel = out.relationships[0]
    assert rel.sourceMultiplicity == "1"
    assert rel.targetMultiplicity == "1..*"


# -- normalization and fallbacks ---------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("many", "0..*"),
    ("N", "0..*"),
    ("*", "0..*"),
    ("0..n", "0..*"),
    ("1..n", "1..*"),
    ("1..many", "1..*"),
    ("+", "1..*"),
])
def test_loose_values_are_normalized(raw, expected):
    out = _expand(_rel("Booking", "Guest", raw, "1"))
    assert out.relationships[0].targetMultiplicity == expected


def test_empty_source_side_falls_back_to_one():
    out = _expand(_rel("Booking", "Guest", "0..*", ""))
    assert out.relationships[0].sourceMultiplicity == "1"


def test_inheritance_ends_are_not_multiplicities():
    out = _expand(_rel("Guest", "Person", "", "", k="inher"))
    rel = out.relationships[0]
    assert rel.type == "Inheritance"
    assert (rel.source, rel.target) == ("Guest", "Person")
    assert rel.targetMultiplicity == "1"


def test_whitespace_is_stripped():
    out = _expand(_rel("Booking", "Guest", "  1..*  ", "  0..*  "))
    rel = out.relationships[0]
    assert rel.sourceMultiplicity == "0..*"
    assert rel.targetMultiplicity == "1..*"


def test_relationship_name_survives():
    out = _expand(_rel("Booking", "Guest", "1..*", "0..*", l="guests"))
    assert out.relationships[0].name == "guests"


def test_blank_name_becomes_none():
    out = _expand(_rel("Booking", "Guest", "1..*", "0..*", l=""))
    assert out.relationships[0].name is None


# -- the schema must not offer the old framing -------------------------
def test_old_multiplicity_fields_are_gone():
    fields = CompactRelationshipSpec.model_fields
    assert "sm" not in fields and "tm" not in fields
    assert "how_many_TARGET_for_one_SOURCE" in fields and "how_many_SOURCE_for_one_TARGET" in fields


def test_both_directions_are_asked_as_questions():
    fields = CompactRelationshipSpec.model_fields
    for name in ("how_many_TARGET_for_one_SOURCE", "how_many_SOURCE_for_one_TARGET"):
        description = fields[name].description
        assert "Take ONE" in description, f"{name} must ask a per-instance question"
        assert "?" in description


def test_the_two_fields_ask_opposite_directions():
    fields = CompactRelationshipSpec.model_fields
    assert "ONE <f>" in fields["how_many_TARGET_for_one_SOURCE"].description
    assert "ONE <t>" in fields["how_many_SOURCE_for_one_TARGET"].description


def test_both_multiplicity_fields_are_required():
    """A default is what let the model skip the question and inherit '1 -> *'."""
    fields = CompactRelationshipSpec.model_fields
    assert fields["how_many_TARGET_for_one_SOURCE"].is_required()
    assert fields["how_many_SOURCE_for_one_TARGET"].is_required()


# -- a whole hotel model round-trips -----------------------------------
def test_full_hotel_model_has_no_inverted_end():
    out = _expand(
        _rel("Guest", "Person", "", "", k="inher"),
        _rel("Employee", "Person", "", "", k="inher"),
        _rel("Booking", "Person", "1", "0..*", l="contact"),
        _rel("Booking", "Employee", "1", "0..*", l="handledBy"),
        _rel("Booking", "Guest", "1..*", "0..*", l="guests"),
        _rel("Booking", "Room", "1..*", "0..*", l="rooms"),
    )
    assocs = [r for r in out.relationships if r.type == "Association"]
    assert len(assocs) == 4

    # Nothing on the Person/Employee/Guest/Room side may be a mandatory single
    # end, or that class gets a NOT NULL FK and stops being creatable alone.
    for rel in assocs:
        assert rel.sourceMultiplicity != "1", (
            f"{rel.source} would need a {rel.target} before it could be created")


# -- the prompt text must describe the fields that actually exist ---------
def test_encoding_rules_do_not_advertise_the_old_field_names():
    """The rules text is sent to the model alongside the schema. After the
    rename it still said 'sm/tm', so the prose and the schema disagreed."""
    from src.schemas.compact_class_diagram import COMPACT_ENCODING_RULES
    assert "sm/tm" not in COMPACT_ENCODING_RULES
    assert "how_many_TARGET_for_one_SOURCE" in COMPACT_ENCODING_RULES
    assert "how_many_SOURCE_for_one_TARGET" in COMPACT_ENCODING_RULES


def test_encoding_rules_forbid_a_two_part_relationship_label():
    """Live 2026-09-18: the model wrote 'contact / bookingsAsContact' into l,
    which fails the editor's quality check on the space."""
    from src.schemas.compact_class_diagram import COMPACT_ENCODING_RULES
    assert "never two names" in COMPACT_ENCODING_RULES
    assert "slash" in COMPACT_ENCODING_RULES


def test_every_schema_field_is_named_in_the_encoding_rules():
    from src.schemas.compact_class_diagram import (
        COMPACT_ENCODING_RULES, CompactRelationshipSpec)
    for name in CompactRelationshipSpec.model_fields:
        if len(name) > 1:  # single letters appear as 'f:', 't:' etc.
            assert name in COMPACT_ENCODING_RULES, f"{name} undocumented"
