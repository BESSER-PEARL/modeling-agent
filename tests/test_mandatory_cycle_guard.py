"""Mandatory creation cycles (live runs 4efe04ff / 9a6063ed, 2026-09-18).

Both runs rendered "a booking covers at least one room and may cover several"
as ``Booking --[1..*]--> BookedRoom`` beside the link class's own
``BookedRoom --[1]--> Booking``. Neither instance can be created first, so the
generated ``BookingCreate`` demanded a ``BookedRoom`` id and
``BookedRoomCreate`` a ``Booking`` id; the delivered app served 69 routes and
passed 2 of 15 workflow checks. BESSER's Phase 0 model-contract check now
rejects the shape (``Mandatory creation cycle: Booking -> BookedRoom ->
Booking``).

``ClassDiagramHandler._break_mandatory_cycles`` relaxes one end per cycle to
an optional bound and keeps the rule as an OCL invariant on the class that
stated it. These tests pin which end is relaxed and what the invariant says;
``test_hotel_model_contract.py`` runs the repaired spec through BESSER's own
checks.
"""
import copy

import pytest

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler


@pytest.fixture
def handler():
    return ClassDiagramHandler(None)


def _assoc(src, tgt, sm, tm, name=None, type_="Association"):
    rel = {"type": type_, "source": src, "target": tgt,
           "sourceMultiplicity": sm, "targetMultiplicity": tm}
    if name is not None:
        rel["name"] = name
    return rel


def _needs(spec):
    """Mirror of BESSER's ``DomainModel._mandatory_dependencies`` on a spec:
    an end with min >= 1 means the class at the other end needs its type."""
    needs = {}
    for rel in spec["relationships"]:
        if str(rel.get("type", "")).lower() in ("inheritance", "generalization"):
            continue
        for side, needer, needed in (("target", rel["source"], rel["target"]),
                                     ("source", rel["target"], rel["source"])):
            low, _ = ClassDiagramHandler._parse_multiplicity(rel.get(f"{side}Multiplicity"))
            if low >= 1:
                needs.setdefault(needer, set()).add(needed)
    return needs


def _has_cycle(spec):
    needs = _needs(spec)

    def visit(node, path):
        return any(nxt in path or visit(nxt, path | {nxt}) for nxt in needs.get(node, ()))

    return any(visit(n, {n}) for n in needs)


def _constraints(spec):
    return spec.get("constraints") or []


# -- the live defect --------------------------------------------------------
def test_live_two_association_shape_becomes_constructible(handler):
    """4efe04ff / 9a6063ed: Booking 1..* BookedRoom and BookedRoom 1 Booking as
    two links. Both Booking-side "at least one" ends give way; the link's
    mandatory booking reference (the NOT NULL foreign key) survives."""
    spec = {"relationships": [
        _assoc("Booking", "BookedRoom", "1", "1..*", name="bookedRooms"),
        _assoc("BookedRoom", "Booking", "1..*", "1", name="booking"),
    ]}
    assert _has_cycle(spec)
    handler._break_mandatory_cycles(spec)
    assert not _has_cycle(spec)
    first, second = spec["relationships"]
    assert first["targetMultiplicity"] == "0..*"
    assert first["sourceMultiplicity"] == "1"
    assert second["sourceMultiplicity"] == "0..*"
    assert second["targetMultiplicity"] == "1"
    assert [c["context"] for c in _constraints(spec)] == ["Booking", "Booking"]


def test_single_association_mandatory_both_ways_relaxes_the_many_end(handler):
    """The post-merge shape: one link, 1 on the booking end, 1..* on the rooms
    end. The rule moves into an invariant on the class that stated it."""
    spec = {"relationships": [
        _assoc("Booking", "BookedRoom", "1", "1..*", name="bookedRooms"),
    ]}
    handler._break_mandatory_cycles(spec)
    rel = spec["relationships"][0]
    assert rel["targetMultiplicity"] == "0..*"
    assert rel["sourceMultiplicity"] == "1", "the link still needs its booking"
    assert _constraints(spec) == [{
        "context": "Booking",
        "expression": "context Booking inv bookedRooms_at_least_1: "
                      "self.bookedRooms->size() >= 1",
        "name": "bookedRooms_at_least_1",
    }]


def test_many_end_is_relaxed_even_when_it_is_the_source_end(handler):
    """Stated from the link's side: BookedRoom --[1..* ; 1]--> Booking. The
    source end has no label, so the invariant navigates the name the
    converter derives for it (the lowercased source class)."""
    spec = {"relationships": [
        _assoc("BookedRoom", "Booking", "1..*", "1", name="booking"),
    ]}
    handler._break_mandatory_cycles(spec)
    rel = spec["relationships"][0]
    assert rel["sourceMultiplicity"] == "0..*"
    assert rel["targetMultiplicity"] == "1"
    con = _constraints(spec)[0]
    assert con["context"] == "Booking"
    assert con["expression"].endswith("self.bookedroom->size() >= 1")


def test_source_end_invariant_uses_the_source_role(handler):
    """The invariant must navigate the end the converter will actually build.
    With ``sourceRole='bookedRooms'`` the guard used to emit
    ``self.bookedroom`` against a real end called ``bookedRooms`` — an OCL
    rule BESSER cannot resolve, self-inflicted by the repair."""
    rel = _assoc("BookedRoom", "Booking", "1..*", "1", name="booking")
    rel["sourceRole"] = "bookedRooms"
    spec = {"relationships": [rel]}
    handler._break_mandatory_cycles(spec)
    con = _constraints(spec)[0]
    assert con["context"] == "Booking"
    assert con["expression"].endswith("self.bookedRooms->size() >= 1")
    assert con["name"] == "bookedRooms_at_least_1"


def test_one_to_one_relaxes_the_target_end(handler):
    """Both single-valued: the named, navigable end gives way and the source's
    reference stays mandatory."""
    spec = {"relationships": [_assoc("Booking", "Bill", "1", "1", name="bill")]}
    handler._break_mandatory_cycles(spec)
    rel = spec["relationships"][0]
    assert rel["targetMultiplicity"] == "0..1"
    assert rel["sourceMultiplicity"] == "1"
    assert _constraints(spec)[0]["expression"] == (
        "context Booking inv bill_at_least_1: self.bill->size() >= 1"
    )


def test_unlabelled_target_end_uses_the_lowercased_class(handler):
    spec = {"relationships": [_assoc("Booking", "Bill", "1", "1")]}
    handler._break_mandatory_cycles(spec)
    assert _constraints(spec)[0]["expression"].endswith("self.bill->size() >= 1")


def test_mandatory_self_association_is_a_cycle_of_length_one(handler):
    spec = {"relationships": [
        _assoc("Employee", "Employee", "0..*", "1", name="manager"),
    ]}
    handler._break_mandatory_cycles(spec)
    rel = spec["relationships"][0]
    assert rel["targetMultiplicity"] == "0..1"
    assert rel["sourceMultiplicity"] == "0..*"
    assert _constraints(spec)[0]["context"] == "Employee"
    assert "self.manager->size() >= 1" in _constraints(spec)[0]["expression"]


def test_optional_self_association_is_left_alone(handler):
    spec = {"relationships": [
        _assoc("Employee", "Employee", "0..*", "0..1", name="manager"),
    ]}
    before = copy.deepcopy(spec)
    handler._break_mandatory_cycles(spec)
    assert spec == before


def test_three_hop_cycle_relaxes_exactly_one_end(handler):
    spec = {"relationships": [
        _assoc("A", "B", "0..*", "1"),
        _assoc("B", "C", "0..*", "1"),
        _assoc("C", "A", "0..*", "1"),
    ]}
    handler._break_mandatory_cycles(spec)
    assert not _has_cycle(spec)
    relaxed = [r for r in spec["relationships"] if r["targetMultiplicity"] == "0..1"]
    assert len(relaxed) == 1
    assert relaxed[0] is spec["relationships"][0], "ties fall to document order"
    assert len(_constraints(spec)) == 1


def test_composition_keeps_its_whole_mandatory(handler):
    spec = {"relationships": [
        _assoc("Order", "OrderLine", "1", "1..*", name="lines", type_="Composition"),
    ]}
    handler._break_mandatory_cycles(spec)
    rel = spec["relationships"][0]
    assert rel["sourceMultiplicity"] == "1", "a part still needs its whole"
    assert rel["targetMultiplicity"] == "0..*"
    assert rel["type"] == "Composition"


def test_lower_bound_above_one_survives_in_the_invariant(handler):
    spec = {"relationships": [_assoc("Team", "Player", "1", "2..*", name="players")]}
    handler._break_mandatory_cycles(spec)
    assert spec["relationships"][0]["targetMultiplicity"] == "0..*"
    assert _constraints(spec) == [{
        "context": "Team",
        "expression": "context Team inv players_at_least_2: self.players->size() >= 2",
        "name": "players_at_least_2",
    }]


# -- what must NOT change ---------------------------------------------------
def test_acyclic_spec_is_untouched(handler):
    """Mandatory ends that point at independent classes (a booking needs a
    guest and a room; neither needs a booking) are exactly what the reference
    hotel model ships and must survive."""
    spec = {"relationships": [
        _assoc("Booking", "Guest", "0..*", "1..*", name="guests"),
        _assoc("Booking", "Room", "0..*", "1", name="room"),
        _assoc("Booking", "Person", "0..*", "1", name="contact"),
    ]}
    before = copy.deepcopy(spec)
    handler._break_mandatory_cycles(spec)
    assert spec == before
    assert "constraints" not in spec


def test_bare_star_is_an_optional_end(handler):
    """The schema default ``"*"`` reaches the guard unnormalised and parses as
    (star, star); the frontend and BESSER read it as ``0..*``. Speaker 1 -- *
    Session is not a cycle (caught by test_ocl_constraint_capture)."""
    spec = {"relationships": [_assoc("Speaker", "Session", "1", "*")]}
    before = copy.deepcopy(spec)
    handler._break_mandatory_cycles(spec)
    assert spec == before


def test_inheritance_creates_no_dependency(handler):
    spec = {"relationships": [
        {"type": "Inheritance", "source": "Guest", "target": "Person"},
        {"type": "Inheritance", "source": "Person", "target": "Guest"},
    ]}
    before = copy.deepcopy(spec)
    handler._break_mandatory_cycles(spec)
    assert spec == before


def test_existing_constraints_are_kept_and_names_stay_unique(handler):
    spec = {
        "relationships": [_assoc("Booking", "BookedRoom", "1", "1..*", name="bookedRooms")],
        "constraints": [{
            "context": "Booking",
            "expression": "context Booking inv bookedRooms_at_least_1: "
                          "self.bookedRooms->size() >= 1",
            "name": None,
        }],
    }
    handler._break_mandatory_cycles(spec)
    assert [c["name"] for c in spec["constraints"]] == [None, "bookedRooms_at_least_1_2"]
    assert "inv bookedRooms_at_least_1_2:" in spec["constraints"][1]["expression"]


def test_invariant_survives_the_context_check(handler):
    spec = {
        "classes": [{"className": "Booking"}, {"className": "BookedRoom"}],
        "relationships": [_assoc("Booking", "BookedRoom", "1", "1..*", name="bookedRooms")],
        "constraints": [],
    }
    handler._break_mandatory_cycles(spec)
    handler._validate_constraints(spec)
    assert len(spec["constraints"]) == 1


@pytest.mark.parametrize("spec", [
    {},
    {"relationships": None},
    {"relationships": []},
    {"relationships": ["junk", {"type": "Association", "source": "A"}]},
])
def test_degenerate_specs_are_noops(handler, spec):
    handler._break_mandatory_cycles(spec)


def test_repair_terminates_on_a_tangle_of_cycles(handler):
    spec = {"relationships": [
        _assoc("A", "B", "1", "1"),
        _assoc("B", "A", "1", "1"),
        _assoc("B", "C", "1..*", "1..*"),
        _assoc("C", "A", "1", "1..*"),
        _assoc("A", "A", "1", "1"),
    ]}
    handler._break_mandatory_cycles(spec)
    assert not _has_cycle(spec)
