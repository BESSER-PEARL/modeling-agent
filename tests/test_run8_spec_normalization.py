"""End-to-end normalization of a hotel spec as the LLM actually produced it.

The relationship set below is read back from a generated
``sql_alchemy.py``: every FK, join table and
``relationship()`` in that file is represented here. Running the real
normalizer pipeline over it must remove the duplicate links that made the
create schemas unsatisfiable, and must not disturb anything else.
"""
import pytest

from src.diagram_handlers.types.class_diagram_handler import ClassDiagramHandler

UNBOUNDED = ClassDiagramHandler._SPEC_UNBOUNDED


def _cls(name, attrs=(), abstract=False):
    return {
        "className": name,
        "attributes": [{"name": a, "type": "str"} for a in attrs],
        "methods": [],
        "isAbstract": abstract,
        "isEnumeration": False,
    }


@pytest.fixture
def run8_spec():
    """Reconstructed from the run's sql_alchemy.py.

    Bill has booking_id AND forBooking_id (both NOT NULL + UNIQUE);
    Room.reservedroom_id is a NOT NULL FK while a `reservations` join table
    also links the same pair.
    """
    return {
        "systemName": "HotelBookingSystem",
        "classes": [
            _cls("Person", ["personId", "firstName", "familyName", "phone", "email"]),
            _cls("Guest"),
            _cls("Employee"),
            _cls("Booking", ["bookingNumber", "arrivalDate", "departureDate", "totalPrice"]),
            _cls("Room", ["roomNumber", "capacity", "description", "standardPrice"]),
            _cls("ReservedRoom", ["agreedPrice"]),
            _cls("Bill", ["billNumber", "issueDate", "totalAmount", "settled"]),
        ],
        "relationships": [
            {"type": "Inheritance", "source": "Guest", "target": "Person"},
            {"type": "Inheritance", "source": "Employee", "target": "Person"},
            # -- the duplicate pair: one fact, two links --
            {"type": "Association", "source": "Bill", "target": "Booking",
             "sourceMultiplicity": "1", "targetMultiplicity": "1",
             "name": "forBooking"},
            {"type": "Association", "source": "Bill", "target": "Booking",
             "sourceMultiplicity": "1", "targetMultiplicity": "1"},
            # -- the other duplicate pair: 1:N FK *and* an N:M join table --
            {"type": "Association", "source": "ReservedRoom", "target": "Room",
             "sourceMultiplicity": "1", "targetMultiplicity": "0..*"},
            {"type": "Association", "source": "ReservedRoom", "target": "Room",
             "sourceMultiplicity": "0..*", "targetMultiplicity": "0..*",
             "name": "reservations"},
            # -- singletons --
            {"type": "Association", "source": "Booking", "target": "ReservedRoom",
             "sourceMultiplicity": "1", "targetMultiplicity": "0..*",
             "name": "reservedRooms"},
            {"type": "Association", "source": "Booking", "target": "Employee",
             "sourceMultiplicity": "1", "targetMultiplicity": "0..*",
             "name": "handledBy"},
            {"type": "Association", "source": "Booking", "target": "Guest",
             "sourceMultiplicity": "1", "targetMultiplicity": "0..*",
             "name": "guests"},
            {"type": "Association", "source": "Booking", "target": "Person",
             "sourceMultiplicity": "1", "targetMultiplicity": "0..*",
             "name": "contact"},
        ],
        "constraints": [],
    }


@pytest.fixture
def normalized(run8_spec):
    """Run the real pipeline in the same order generate_complete_system does."""
    h = ClassDiagramHandler.__new__(ClassDiagramHandler)
    h._sanitize_identifier_names(run8_spec)
    h._rewrite_enum_relationships(run8_spec)
    h._rewrite_class_typed_attributes(run8_spec)
    h._merge_redundant_parallel_associations(run8_spec)
    h._dedupe_relationship_names(run8_spec)
    h._ensure_unique_association_ends(run8_spec)
    h._strip_shadowed_attributes(run8_spec)
    h._sanitize_member_types(run8_spec)
    h._validate_constraints(run8_spec)
    return run8_spec


def _assocs(spec):
    return [r for r in spec["relationships"]
            if ClassDiagramHandler._spec_rel_type(r) not in
            ClassDiagramHandler._SPEC_NON_END_REL_TYPES]


def _pair_counts(spec):
    counts = {}
    for r in _assocs(spec):
        key = frozenset((r["source"], r["target"]))
        counts[key] = counts.get(key, 0) + 1
    return counts


def test_pipeline_leaves_no_duplicated_pair(normalized):
    dupes = {tuple(sorted(k)): v for k, v in _pair_counts(normalized).items() if v > 1}
    assert not dupes, f"one fact still stated twice: {dupes}"


def test_bill_booking_collapses_to_a_single_link(normalized):
    links = [r for r in _assocs(normalized)
             if {r["source"], r["target"]} == {"Bill", "Booking"}]
    assert len(links) == 1


def test_room_reservedroom_collapses_to_a_single_link(normalized):
    links = [r for r in _assocs(normalized)
             if {r["source"], r["target"]} == {"Room", "ReservedRoom"}]
    assert len(links) == 1


def test_room_end_is_no_longer_a_mandatory_fk(normalized):
    """Room.reservedroom_id NOT NULL is what made a Room uncreatable alone."""
    link = [r for r in _assocs(normalized)
            if {r["source"], r["target"]} == {"Room", "ReservedRoom"}][0]
    room_end = ("sourceMultiplicity" if link["source"] == "Room"
                else "targetMultiplicity")
    other_end = ("targetMultiplicity" if link["source"] == "Room"
                 else "sourceMultiplicity")
    lower, _ = ClassDiagramHandler._parse_multiplicity(link[other_end])
    assert lower == 0, (
        "the ReservedRoom end must be optional, else Room still needs one on create")
    assert link[room_end]


def test_no_class_carries_two_mandatory_single_ends_to_the_same_partner(normalized):
    """The BillCreate defect: two required ids for one relationship."""
    mandatory = {}
    for r in _assocs(normalized):
        for owner_key, partner_key, mult_key in (
            ("source", "target", "targetMultiplicity"),
            ("target", "source", "sourceMultiplicity"),
        ):
            low, up = ClassDiagramHandler._parse_multiplicity(r.get(mult_key))
            if (low, up) == (1, 1):
                key = (r[owner_key], r[partner_key])
                mandatory[key] = mandatory.get(key, 0) + 1
    offenders = {k: v for k, v in mandatory.items() if v > 1}
    assert not offenders, f"two mandatory ends to one partner: {offenders}"


def test_no_underscore_one_suffixes_are_needed(normalized):
    """`booking_1`/`room_1`/`bill_1` were the visible symptom of the duplicates."""
    names = [r.get("name") for r in _assocs(normalized) if r.get("name")]
    assert not [n for n in names if n.endswith("_1") or n.endswith("_2")], names


def test_relationship_count_drops_by_exactly_the_two_duplicates(run8_spec, normalized):
    # 10 in, 2 duplicates removed -> 8
    assert len(normalized["relationships"]) == 8


def test_the_genuine_links_all_survive(normalized):
    surviving = {tuple(sorted((r["source"], r["target"]))) for r in _assocs(normalized)}
    assert surviving == {
        ("Bill", "Booking"),
        ("ReservedRoom", "Room"),
        ("Booking", "ReservedRoom"),
        ("Booking", "Employee"),
        ("Booking", "Guest"),
        ("Booking", "Person"),
    }


def test_inheritance_survives_intact(normalized):
    inh = [r for r in normalized["relationships"]
           if ClassDiagramHandler._spec_rel_type(r) == "inheritance"]
    assert {(r["source"], r["target"]) for r in inh} == {
        ("Guest", "Person"), ("Employee", "Person")}


def test_all_seven_classes_survive(normalized):
    assert {c["className"] for c in normalized["classes"]} == {
        "Person", "Guest", "Employee", "Booking", "Room", "ReservedRoom", "Bill"}


def test_pipeline_is_idempotent(normalized):
    """Running it twice must not change anything further."""
    h = ClassDiagramHandler.__new__(ClassDiagramHandler)
    before = [dict(r) for r in normalized["relationships"]]
    h._merge_redundant_parallel_associations(normalized)
    h._dedupe_relationship_names(normalized)
    h._ensure_unique_association_ends(normalized)
    assert [dict(r) for r in normalized["relationships"]] == before


def test_manual_model_shape_is_left_untouched():
    """The hand-built model that already generates a working app must survive.

    Ends and multiplicities are those of Downloads/model.json, the model the
    deterministic generator turns into a usable app.
    """
    h = ClassDiagramHandler.__new__(ClassDiagramHandler)
    spec = {
        "systemName": "HotelManual",
        "classes": [_cls(n) for n in
                    ("Person", "Guest", "Employee", "Booking", "Room",
                     "ReservedRoom", "Invoice")],
        "relationships": [
            {"type": "Inheritance", "source": "Guest", "target": "Person"},
            {"type": "Inheritance", "source": "Employee", "target": "Person"},
            {"type": "Association", "source": "Booking", "target": "Invoice",
             "sourceMultiplicity": "1", "targetMultiplicity": "0..1", "name": "invoice"},
            {"type": "Association", "source": "Employee", "target": "Booking",
             "sourceMultiplicity": "1", "targetMultiplicity": "0..*", "name": "manages"},
            {"type": "Association", "source": "Booking", "target": "Guest",
             "sourceMultiplicity": "0..*", "targetMultiplicity": "1..*", "name": "guests"},
            {"type": "Association", "source": "Booking", "target": "Person",
             "sourceMultiplicity": "0..*", "targetMultiplicity": "1",
             "name": "booking_contact"},
            {"type": "Association", "source": "Room", "target": "Booking",
             "sourceMultiplicity": "1..*", "targetMultiplicity": "0..*", "name": "rooms"},
        ],
        "constraints": [],
    }
    before = [dict(r) for r in spec["relationships"]]
    h._merge_redundant_parallel_associations(spec)
    assert [dict(r) for r in spec["relationships"]] == before, (
        "a correct model must pass through unchanged")
