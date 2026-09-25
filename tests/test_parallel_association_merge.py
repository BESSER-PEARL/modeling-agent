"""Duplicate-association merge.

A hotel-booking generation produced a Bill carrying BOTH ``booking_id`` and
``forBooking_id`` (each NOT NULL + UNIQUE) for the single "a booking has one
bill" fact, and linked Room to ReservedRoom twice - once as a mandatory FK,
once through a join table. Both made the generated create schemas demand ids
that no client could supply. These tests pin the shapes taken from that run's
generated sql_alchemy.py and assert the merge collapses them without touching genuinely
distinct parallel links.
"""
import pytest

from src.diagram_handlers.types.class_diagram_handler import ClassDiagramHandler


@pytest.fixture
def handler():
    return ClassDiagramHandler.__new__(ClassDiagramHandler)


def _rels(spec):
    return spec["relationships"]


def _pairs(spec):
    return [(r["source"], r["target"], r.get("sourceMultiplicity"),
             r.get("targetMultiplicity"), r.get("name")) for r in _rels(spec)]


# -- multiplicity parsing ------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("1", (1, 1)),
    ("0..1", (0, 1)),
    ("0..*", (0, 9999)),
    ("1..*", (1, 9999)),
    ("*", (9999, 9999)),
    ("  0..*  ", (0, 9999)),
    ("", (1, 1)),
    (None, (1, 1)),
    ("garbage", (1, 1)),
    ("2..5", (2, 5)),
])
def test_parse_multiplicity(handler, text, expected):
    assert handler._parse_multiplicity(text) == expected


@pytest.mark.parametrize("low,up,expected", [
    (1, 1, "1"),
    (0, 1, "0..1"),
    (0, 9999, "0..*"),
    (1, 9999, "1..*"),
    (2, 5, "2..5"),
])
def test_format_multiplicity(handler, low, up, expected):
    assert handler._format_multiplicity(low, up) == expected


def test_parse_format_roundtrip(handler):
    for text in ("1", "0..1", "0..*", "1..*", "2..5"):
        assert handler._format_multiplicity(*handler._parse_multiplicity(text)) == text


# -- the Bill<->Booking defect -------------------------------------------
def test_bill_booking_stated_twice_becomes_one_link(handler):
    """Observed shape: Bill.booking_id AND Bill.forBooking_id, both 1:1."""
    spec = {"relationships": [
        {"type": "Association", "source": "Bill", "target": "Booking",
         "sourceMultiplicity": "1", "targetMultiplicity": "1", "name": "forBooking"},
        {"type": "Association", "source": "Booking", "target": "Bill",
         "sourceMultiplicity": "1", "targetMultiplicity": "0..1"},
    ]}
    handler._merge_redundant_parallel_associations(spec)
    assert len(_rels(spec)) == 1, "one fact must yield one relationship"
    rel = _rels(spec)[0]
    assert (rel["source"], rel["target"]) == ("Bill", "Booking")
    assert rel["name"] == "forBooking", "the named end survives"
    # Bill end: union of 1 (this) and 0..1 (the other, reoriented) -> 0..1
    assert rel["sourceMultiplicity"] == "0..1"
    assert rel["targetMultiplicity"] == "1"


def test_merged_bill_end_is_no_longer_mandatory_both_ways(handler):
    """The union must not leave BOTH ends mandatory-single (the deadlock)."""
    spec = {"relationships": [
        {"type": "Association", "source": "Bill", "target": "Booking",
         "sourceMultiplicity": "1", "targetMultiplicity": "1", "name": "forBooking"},
        {"type": "Association", "source": "Booking", "target": "Bill",
         "sourceMultiplicity": "1", "targetMultiplicity": "0..1"},
    ]}
    handler._merge_redundant_parallel_associations(spec)
    rel = _rels(spec)[0]
    mandatory_single = [
        m for m in (rel["sourceMultiplicity"], rel["targetMultiplicity"])
        if handler._parse_multiplicity(m) == (1, 1)
    ]
    assert len(mandatory_single) < 2


# -- the Room<->ReservedRoom defect --------------------------------------
def test_room_reservedroom_keeps_the_permissive_end(handler):
    """Observed shape: a 1:N mandatory FK *and* an N:M join table for one fact."""
    spec = {"relationships": [
        {"type": "Association", "source": "ReservedRoom", "target": "Room",
         "sourceMultiplicity": "1", "targetMultiplicity": "0..*"},
        {"type": "Association", "source": "ReservedRoom", "target": "Room",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "0..*",
         "name": "reservations"},
    ]}
    handler._merge_redundant_parallel_associations(spec)
    assert len(_rels(spec)) == 1
    rel = _rels(spec)[0]
    # min lower bound wins: 1 and 0..* union to 0..*, killing the NOT NULL FK
    assert rel["sourceMultiplicity"] == "0..*"
    assert rel["targetMultiplicity"] == "0..*"
    assert rel["name"] == "reservations", "the unnamed duplicate must not win the name"


def test_union_widens_upper_bound_too(handler):
    spec = {"relationships": [
        {"type": "Association", "source": "A", "target": "B",
         "sourceMultiplicity": "1", "targetMultiplicity": "1"},
        {"type": "Association", "source": "A", "target": "B",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "1..*"},
    ]}
    handler._merge_redundant_parallel_associations(spec)
    rel = _rels(spec)[0]
    assert rel["sourceMultiplicity"] == "0..*"
    assert rel["targetMultiplicity"] == "1..*"


# -- what must NOT be merged ---------------------------------------------
def test_distinctly_named_parallels_are_left_alone(handler):
    """homeAddress / workAddress are two real links, not one stated twice."""
    spec = {"relationships": [
        {"type": "Association", "source": "Person", "target": "Address",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "1", "name": "homeAddress"},
        {"type": "Association", "source": "Person", "target": "Address",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "1", "name": "workAddress"},
    ]}
    before = _pairs(spec)
    handler._merge_redundant_parallel_associations(spec)
    assert _pairs(spec) == before


def test_three_distinctly_named_parallels_survive(handler):
    spec = {"relationships": [
        {"type": "Association", "source": "Flight", "target": "Airport",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "1", "name": "origin"},
        {"type": "Association", "source": "Flight", "target": "Airport",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "1", "name": "destination"},
        {"type": "Association", "source": "Flight", "target": "Airport",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "0..1", "name": "diversion"},
    ]}
    handler._merge_redundant_parallel_associations(spec)
    assert len(_rels(spec)) == 3


def test_repeated_name_is_a_duplicate_not_a_role(handler):
    """Two links both called 'booking' are the same fact, not two roles."""
    spec = {"relationships": [
        {"type": "Association", "source": "A", "target": "B",
         "sourceMultiplicity": "1", "targetMultiplicity": "1", "name": "booking"},
        {"type": "Association", "source": "A", "target": "B",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "1", "name": "booking"},
    ]}
    handler._merge_redundant_parallel_associations(spec)
    assert len(_rels(spec)) == 1


def test_composition_is_never_merged(handler):
    """Existential dependency must not be dissolved into a plain association."""
    spec = {"relationships": [
        {"type": "Composition", "source": "Order", "target": "OrderLine",
         "sourceMultiplicity": "1", "targetMultiplicity": "1..*"},
        {"type": "Association", "source": "Order", "target": "OrderLine",
         "sourceMultiplicity": "1", "targetMultiplicity": "0..*"},
    ]}
    handler._merge_redundant_parallel_associations(spec)
    assert len(_rels(spec)) == 2
    assert any(r["type"] == "Composition" for r in _rels(spec))


def test_aggregation_is_never_merged(handler):
    spec = {"relationships": [
        {"type": "Aggregation", "source": "Team", "target": "Player",
         "sourceMultiplicity": "1", "targetMultiplicity": "0..*"},
        {"type": "Aggregation", "source": "Team", "target": "Player",
         "sourceMultiplicity": "1", "targetMultiplicity": "0..*"},
    ]}
    handler._merge_redundant_parallel_associations(spec)
    assert len(_rels(spec)) == 2


def test_inheritance_is_untouched(handler):
    spec = {"relationships": [
        {"type": "Inheritance", "source": "Guest", "target": "Person"},
        {"type": "Inheritance", "source": "Employee", "target": "Person"},
    ]}
    handler._merge_redundant_parallel_associations(spec)
    assert len(_rels(spec)) == 2


def test_self_association_is_untouched(handler):
    """Employee-manages-Employee twice is legitimate (manager / mentor)."""
    spec = {"relationships": [
        {"type": "Association", "source": "Employee", "target": "Employee",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "0..1"},
        {"type": "Association", "source": "Employee", "target": "Employee",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "0..1"},
    ]}
    handler._merge_redundant_parallel_associations(spec)
    assert len(_rels(spec)) == 2


def test_different_pairs_are_independent(handler):
    spec = {"relationships": [
        {"type": "Association", "source": "A", "target": "B",
         "sourceMultiplicity": "1", "targetMultiplicity": "1"},
        {"type": "Association", "source": "A", "target": "B",
         "sourceMultiplicity": "1", "targetMultiplicity": "1"},
        {"type": "Association", "source": "C", "target": "D",
         "sourceMultiplicity": "1", "targetMultiplicity": "1"},
    ]}
    handler._merge_redundant_parallel_associations(spec)
    assert len(_rels(spec)) == 2
    assert {(r["source"], r["target"]) for r in _rels(spec)} == {("A", "B"), ("C", "D")}


# -- robustness ----------------------------------------------------------
@pytest.mark.parametrize("spec", [
    {},
    {"relationships": None},
    {"relationships": []},
    {"relationships": [{"type": "Association", "source": "A", "target": "B"}]},
])
def test_degenerate_specs_are_noops(handler, spec):
    handler._merge_redundant_parallel_associations(spec)  # must not raise


def test_malformed_entries_are_skipped(handler):
    spec = {"relationships": [
        "not a dict",
        {"type": "Association", "source": "A"},
        {"type": "Association", "target": "B"},
        {"type": "Association", "source": "", "target": "B"},
        {"type": "Association", "source": "A", "target": "B",
         "sourceMultiplicity": "1", "targetMultiplicity": "1"},
    ]}
    handler._merge_redundant_parallel_associations(spec)
    assert len(_rels(spec)) == 5, "nothing to merge; malformed entries preserved"


def test_missing_multiplicities_default_to_one(handler):
    spec = {"relationships": [
        {"type": "Association", "source": "A", "target": "B"},
        {"type": "Association", "source": "A", "target": "B"},
    ]}
    handler._merge_redundant_parallel_associations(spec)
    assert len(_rels(spec)) == 1
    rel = _rels(spec)[0]
    assert rel["sourceMultiplicity"] == "1"
    assert rel["targetMultiplicity"] == "1"


def test_missing_type_defaults_to_association_and_merges(handler):
    spec = {"relationships": [
        {"source": "A", "target": "B", "sourceMultiplicity": "1", "targetMultiplicity": "1"},
        {"source": "A", "target": "B", "sourceMultiplicity": "0..*", "targetMultiplicity": "1"},
    ]}
    handler._merge_redundant_parallel_associations(spec)
    assert len(_rels(spec)) == 1


def test_order_of_surviving_relationships_is_stable(handler):
    spec = {"relationships": [
        {"type": "Association", "source": "X", "target": "Y",
         "sourceMultiplicity": "1", "targetMultiplicity": "1", "name": "first"},
        {"type": "Association", "source": "A", "target": "B",
         "sourceMultiplicity": "1", "targetMultiplicity": "1"},
        {"type": "Association", "source": "A", "target": "B",
         "sourceMultiplicity": "1", "targetMultiplicity": "1"},
        {"type": "Association", "source": "P", "target": "Q",
         "sourceMultiplicity": "1", "targetMultiplicity": "1", "name": "last"},
    ]}
    handler._merge_redundant_parallel_associations(spec)
    assert [(r["source"], r["target"]) for r in _rels(spec)] == [
        ("X", "Y"), ("A", "B"), ("P", "Q")]


def test_four_way_duplicate_collapses_to_one(handler):
    spec = {"relationships": [
        {"type": "Association", "source": "A", "target": "B",
         "sourceMultiplicity": "1", "targetMultiplicity": "1"},
        {"type": "Association", "source": "B", "target": "A",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "1"},
        {"type": "Association", "source": "A", "target": "B",
         "sourceMultiplicity": "1", "targetMultiplicity": "0..*"},
        {"type": "Association", "source": "B", "target": "A",
         "sourceMultiplicity": "1", "targetMultiplicity": "0..1"},
    ]}
    handler._merge_redundant_parallel_associations(spec)
    assert len(_rels(spec)) == 1
    rel = _rels(spec)[0]
    assert (rel["source"], rel["target"]) == ("A", "B")
    # A end: lower bounds seen are 1,1,1,0 -> 0; upper bounds all 1 -> 1.
    assert rel["sourceMultiplicity"] == "0..1"
    # B end: lower bounds 1,0,0,1 -> 0; uppers 1,*,*,1 -> *.
    assert rel["targetMultiplicity"] == "0..*"


# -- reciprocal pairs: one fact the request stated from both sides -------
def test_live_booking_bookingroom_reciprocal_is_merged(handler):
    """Observed shape: Booking--bookingRooms/booking-->BookingRoom
    alongside BookingRoom--booking/bookingRooms-->Booking. Each side named
    both ends, so the swapped roles prove it is one fact."""
    spec = {"relationships": [
        {"type": "Association", "source": "Booking", "target": "BookingRoom",
         "sourceMultiplicity": "1", "targetMultiplicity": "1..*",
         "name": "bookingRooms", "sourceRole": "booking"},
        {"type": "Association", "source": "BookingRoom", "target": "Booking",
         "sourceMultiplicity": "1..*", "targetMultiplicity": "1",
         "name": "booking", "sourceRole": "bookingRooms"},
    ]}
    handler._merge_redundant_parallel_associations(spec)
    assert len(_rels(spec)) == 1
    rel = _rels(spec)[0]
    assert (rel["source"], rel["target"]) == ("Booking", "BookingRoom")
    assert rel["name"] == "bookingRooms"
    assert rel["sourceMultiplicity"] == "1"
    assert rel["targetMultiplicity"] == "1..*"


def test_reciprocal_pair_with_unrelated_roles_is_kept(handler):
    """Person--owns-->Car and Car--insuredBy-->Person are two real facts."""
    spec = {"relationships": [
        {"type": "Association", "source": "Person", "target": "Car",
         "sourceMultiplicity": "1", "targetMultiplicity": "0..*", "name": "owns"},
        {"type": "Association", "source": "Car", "target": "Person",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "1..*",
         "name": "insuredBy"},
    ]}
    before = _pairs(spec)
    handler._merge_redundant_parallel_associations(spec)
    assert _pairs(spec) == before


def test_three_way_group_is_not_treated_as_reciprocal(handler):
    """A reciprocal pair plus a third distinct link stays distinct-named."""
    spec = {"relationships": [
        {"type": "Association", "source": "A", "target": "B",
         "sourceMultiplicity": "1", "targetMultiplicity": "0..*", "name": "one"},
        {"type": "Association", "source": "B", "target": "A",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "1", "name": "two"},
        {"type": "Association", "source": "A", "target": "B",
         "sourceMultiplicity": "1", "targetMultiplicity": "1", "name": "three"},
    ]}
    handler._merge_redundant_parallel_associations(spec)
    assert len(_rels(spec)) == 3


def test_same_direction_distinct_names_still_survive(handler):
    """The reciprocal rule must not weaken the homeAddress/workAddress case."""
    spec = {"relationships": [
        {"type": "Association", "source": "Person", "target": "Address",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "1", "name": "homeAddress"},
        {"type": "Association", "source": "Person", "target": "Address",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "1", "name": "workAddress"},
    ]}
    handler._merge_redundant_parallel_associations(spec)
    assert len(_rels(spec)) == 2



# -- dual-role labels: "targetRole / sourceRole" ---------------------------
def test_dual_role_label_is_split_into_name_and_source_role(handler):
    """The blocker: 'contact / bookingsAsContact' failed the editor's
    quality check with "Name cannot contain spaces"."""
    spec = {"relationships": [
        {"type": "Association", "source": "Booking", "target": "Person",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "1",
         "name": "contact / bookingsAsContact"},
    ]}
    handler._split_dual_role_names(spec)
    rel = _rels(spec)[0]
    assert rel["name"] == "contact"
    assert rel["sourceRole"] == "bookingsAsContact"


def test_every_live_run9_label_becomes_a_valid_identifier(handler):
    labels = [
        "contact / bookingsAsContact", "guests / bookingsAsGuest",
        "handledBy / handledBookings", "reservedRooms / booking",
        "room / reservations", "bill / booking", "reservations / room",
    ]
    spec = {"relationships": [
        {"type": "Association", "source": "A", "target": "B",
         "sourceMultiplicity": "1", "targetMultiplicity": "1", "name": x}
        for x in labels
    ]}
    handler._split_dual_role_names(spec)
    import re
    for rel in _rels(spec):
        assert re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", rel["name"]), rel["name"]
        assert " " not in rel["name"]


def test_plain_label_is_untouched(handler):
    spec = {"relationships": [
        {"type": "Association", "source": "A", "target": "B",
         "sourceMultiplicity": "1", "targetMultiplicity": "1", "name": "guests"},
    ]}
    handler._split_dual_role_names(spec)
    rel = _rels(spec)[0]
    assert rel["name"] == "guests"
    assert "sourceRole" not in rel


def test_label_with_spaces_but_no_slash_is_coerced(handler):
    spec = {"relationships": [
        {"type": "Association", "source": "A", "target": "B",
         "sourceMultiplicity": "1", "targetMultiplicity": "1",
         "name": "booking contact person"},
    ]}
    handler._split_dual_role_names(spec)
    assert _rels(spec)[0]["name"] == "bookingContactPerson"


def test_existing_source_role_is_not_overwritten(handler):
    spec = {"relationships": [
        {"type": "Association", "source": "A", "target": "B",
         "sourceMultiplicity": "1", "targetMultiplicity": "1",
         "name": "x / y", "sourceRole": "alreadySet"},
    ]}
    handler._split_dual_role_names(spec)
    assert _rels(spec)[0]["sourceRole"] == "alreadySet"


@pytest.mark.parametrize("spec", [
    {}, {"relationships": None}, {"relationships": []},
    {"relationships": ["junk", {"type": "Association"}]},
])
def test_split_handles_degenerate_specs(handler, spec):
    handler._split_dual_role_names(spec)


def test_live_room_reservedroom_crosswise_duplicate_is_merged(handler):
    """Observed shape: ReservedRoom--room/reservations-->Room alongside
    Room--reservations/room-->ReservedRoom. The bounds disagree (1 vs 1..*)
    so the mirror test cannot catch it; the swapped roles can."""
    spec = {"relationships": [
        {"type": "Association", "source": "ReservedRoom", "target": "Room",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "1",
         "name": "room / reservations"},
        {"type": "Association", "source": "Room", "target": "ReservedRoom",
         "sourceMultiplicity": "1..*", "targetMultiplicity": "0..*",
         "name": "reservations / room"},
    ]}
    handler._split_dual_role_names(spec)
    handler._merge_redundant_parallel_associations(spec)
    assert len(_rels(spec)) == 1, "one fact stated from both sides"
    assert _rels(spec)[0]["name"] == "room"


def test_crosswise_rule_does_not_merge_unrelated_roles(handler):
    """Person--owns/ownedBy-->Car and Car--insuredBy/insurer-->Person are
    two facts: the roles are not each other's."""
    spec = {"relationships": [
        {"type": "Association", "source": "Person", "target": "Car",
         "sourceMultiplicity": "1", "targetMultiplicity": "0..*",
         "name": "owns / ownedBy"},
        {"type": "Association", "source": "Car", "target": "Person",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "1",
         "name": "insuredBy / insurer"},
    ]}
    handler._split_dual_role_names(spec)
    handler._merge_redundant_parallel_associations(spec)
    assert len(_rels(spec)) == 2


def test_crosswise_helper_requires_both_roles(handler):
    assert not handler._is_crosswise_reciprocal([
        {"source": "A", "target": "B", "name": "x"},
        {"source": "B", "target": "A", "name": "y"},
    ])


def test_keeping_a_duplicate_is_logged_not_silent(handler, caplog):
    """EF Core and SQLAlchemy surface this ambiguity rather than resolving it
    silently; downstream it shows up only as a '_1' suffix that reads like a
    bug, so the decision must be visible in the log."""
    import logging
    spec = {"relationships": [
        {"type": "Association", "source": "Person", "target": "Address",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "1", "name": "homeAddress"},
        {"type": "Association", "source": "Person", "target": "Address",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "1", "name": "workAddress"},
    ]}
    with caplog.at_level(logging.INFO):
        handler._merge_redundant_parallel_associations(spec)
    assert len(_rels(spec)) == 2
    assert "Kept 2 parallel" in caplog.text
    assert "homeAddress" in caplog.text and "workAddress" in caplog.text


# -- a label that only repeats the target class is not a role -------------
def test_live_pair_named_after_its_own_target_from_both_sides_is_merged(handler):
    """Booking--reservedRooms-->ReservedRoom beside ReservedRoom--booking-->
    Booking: 'booking' is what the converter derives for an unlabelled end,
    so the pair is one fact stated twice. Kept, it surfaced downstream as a
    'booking_1' end and a second foreign key."""
    spec = {"relationships": [
        {"type": "Association", "source": "Booking", "target": "ReservedRoom",
         "sourceMultiplicity": "1", "targetMultiplicity": "1..*", "name": "reservedRooms"},
        {"type": "Association", "source": "ReservedRoom", "target": "Booking",
         "sourceMultiplicity": "1..*", "targetMultiplicity": "1", "name": "booking"},
    ]}
    handler._merge_redundant_parallel_associations(spec)
    assert len(_rels(spec)) == 1
    rel = _rels(spec)[0]
    assert (rel["source"], rel["target"]) == ("Booking", "ReservedRoom")
    assert rel["name"] == "reservedRooms"
    assert rel["sourceMultiplicity"] == "1"
    assert rel["targetMultiplicity"] == "1..*"


def test_live_room_reservedroom_default_name_is_merged(handler):
    """ReservedRoom--room-->Room beside Room--reservations-->ReservedRoom
    (downstream: 'room_1' / 'reservations')."""
    spec = {"relationships": [
        {"type": "Association", "source": "ReservedRoom", "target": "Room",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "1", "name": "room"},
        {"type": "Association", "source": "Room", "target": "ReservedRoom",
         "sourceMultiplicity": "1", "targetMultiplicity": "0..*", "name": "reservations"},
    ]}
    handler._merge_redundant_parallel_associations(spec)
    assert len(_rels(spec)) == 1
    rel = _rels(spec)[0]
    assert (rel["source"], rel["target"]) == ("ReservedRoom", "Room")
    assert rel["sourceMultiplicity"] == "0..*"
    assert rel["targetMultiplicity"] == "1"


def test_live_bill_booking_default_names_are_merged(handler):
    """Booking--bill-->Bill beside Bill--booking-->Booking (downstream:
    'bill_1' / 'billBooking', the latter minted by the name dedupe)."""
    spec = {"relationships": [
        {"type": "Association", "source": "Booking", "target": "Bill",
         "sourceMultiplicity": "1", "targetMultiplicity": "0..1", "name": "bill"},
        {"type": "Association", "source": "Bill", "target": "Booking",
         "sourceMultiplicity": "0..1", "targetMultiplicity": "1", "name": "booking"},
    ]}
    handler._merge_redundant_parallel_associations(spec)
    assert len(_rels(spec)) == 1
    rel = _rels(spec)[0]
    assert rel["sourceMultiplicity"] == "1"
    assert rel["targetMultiplicity"] == "0..1"


def test_collision_suffixed_default_name_is_not_a_role(handler):
    spec = {"relationships": [
        {"type": "Association", "source": "Booking", "target": "ReservedRoom",
         "sourceMultiplicity": "1", "targetMultiplicity": "1..*", "name": "reservedRooms"},
        {"type": "Association", "source": "ReservedRoom", "target": "Booking",
         "sourceMultiplicity": "1..*", "targetMultiplicity": "1", "name": "booking_1"},
    ]}
    handler._merge_redundant_parallel_associations(spec)
    assert len(_rels(spec)) == 1


def test_plural_default_names_on_both_sides_are_merged(handler):
    spec = {"relationships": [
        {"type": "Association", "source": "Customer", "target": "Order",
         "sourceMultiplicity": "1", "targetMultiplicity": "0..*", "name": "orders"},
        {"type": "Association", "source": "Order", "target": "Customer",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "1", "name": "customer"},
    ]}
    handler._merge_redundant_parallel_associations(spec)
    assert len(_rels(spec)) == 1


@pytest.mark.parametrize("name,target,expected", [
    ("booking", "Booking", True),
    ("Booking", "Booking", True),
    ("bookings", "Booking", True),
    ("addresses", "Address", True),
    ("booking_1", "Booking", True),
    ("reservedRooms", "ReservedRoom", True),
    ("homeAddress", "Address", False),
    ("owns", "Car", False),
    ("contact", "Person", False),
    ("reservations", "ReservedRoom", False),
    ("", "Booking", False),
    (None, "Booking", False),
])
def test_default_end_name_helper(handler, name, target, expected):
    assert handler._is_default_end_name(name, target) is expected
