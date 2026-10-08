"""The hotel prose end to end: agent spec -> editor JSON -> BUML -> model contract.

Generation runs took a hotel description and shipped a model BESSER's
model-contract check rejects: a mandatory
creation cycle (``Booking -> BookedRoom -> Booking``) and three class pairs
connected twice. The prose, in the shape that matters here: a booking covers
at least one room and may cover several; each room is tied to zero or
multiple booking records; the agreed price is fixed per booking on the link;
at least one guest must be listed; a bill belongs to exactly one booking, and
a booking may have a bill raised against it, but never more than one.

Two calibration targets:

1. The agent's own spec for that prose — the shape the LLM produced,
   replayed through ``generate_complete_system`` with a canned LLM — must
   build a ``DomainModel`` that passes ``_validate_mandatory_cycles`` and
   ``_validate_duplicate_associations`` with no warning, and its relaxed
   "at least one" must come back as a parsed OCL invariant.
2. The human-modelled reference for the same prose
   (BESSER-PEARL/Agentic-Low-Code-Benchmark,
   ``hotel-booking/2-validation-and-business-rules/low-code-model/model.json``,
   stored under ``tests/fixtures/``) must validate with zero warnings. It pins
   the checks themselves against a known-good model — ``ReservedRoom``
   included, as a native association class.

The native case also exercises both wire schemas, the REAL TypeScript editor
converter, BUML conversion, and SQLite/FastAPI HTTP creation. It requires the
sibling frontend checkout with its installed Node dependencies.

The BESSER side runs in ``besser_contract_probe.py`` (its own interpreter, the
workspace sibling ``../BESSER`` first on the path when present); the probe
skips these tests when the BESSER it finds lacks the checks.
"""
import json
from copy import deepcopy
import os
import subprocess
import sys

import pytest

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROBE = os.path.join(_HERE, "besser_contract_probe.py")
_WORKSPACE_BESSER = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "BESSER")
_FIXTURE = os.path.join(_HERE, "fixtures", "hotel_booking_reference_model.json")

_PROSE = (
    "A hotel takes bookings. A booking covers at least one room and may cover "
    "several, and each room is tied to zero or multiple booking records. The "
    "price agreed for a room is fixed per booking. At least one guest must be "
    "listed on a booking. A bill belongs to exactly one booking; a booking may "
    "have a bill raised against it, but never more than one."
)

# The spec the LLM produced, reduced to the classes the prose above
# names. Every relationship is stated the way the runs stated it.
_LIVE_SPEC = {
    "systemName": "HotelBooking",
    "classes": [
        {"className": "Guest", "attributes": [{"name": "name", "type": "String"}], "methods": []},
        {"className": "Room", "attributes": [{"name": "number", "type": "String"}], "methods": []},
        {"className": "Booking", "attributes": [{"name": "checkIn", "type": "Date"}], "methods": []},
        {"className": "ReservedRoom", "attributes": [{"name": "agreedPrice", "type": "Float"}], "methods": []},
        {"className": "Bill", "attributes": [{"name": "amount", "type": "Float"}], "methods": []},
    ],
    "relationships": [
        # "covers at least one room and may cover several" as a structural bound
        {"type": "Association", "source": "Booking", "target": "ReservedRoom",
         "sourceMultiplicity": "1", "targetMultiplicity": "1..*", "name": "reservedRooms"},
        # the link's own booking, stated again from its side
        {"type": "Association", "source": "ReservedRoom", "target": "Booking",
         "sourceMultiplicity": "1..*", "targetMultiplicity": "1", "name": "booking"},
        {"type": "Association", "source": "ReservedRoom", "target": "Room",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "1", "name": "room"},
        # "each room is tied to zero or multiple booking records"
        {"type": "Association", "source": "Room", "target": "ReservedRoom",
         "sourceMultiplicity": "1", "targetMultiplicity": "0..*", "name": "reservations"},
        # "at least one guest must be listed"
        {"type": "Association", "source": "Booking", "target": "Guest",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "1..*", "name": "guests"},
        # "may have a bill raised against it, but never more than one"
        {"type": "Association", "source": "Booking", "target": "Bill",
         "sourceMultiplicity": "1", "targetMultiplicity": "0..1", "name": "bill"},
        # "belongs to exactly one booking"
        {"type": "Association", "source": "Bill", "target": "Booking",
         "sourceMultiplicity": "0..1", "targetMultiplicity": "1", "name": "booking"},
    ],
    "constraints": [],
}


class _CannedLLM:
    def __init__(self, response):
        self.response = response

    def predict(self, prompt):
        return self.response


def _besser_report(kind, data):
    args = [sys.executable, _PROBE]
    if os.path.isdir(os.path.join(_WORKSPACE_BESSER, "besser")):
        args.append(_WORKSPACE_BESSER)
    proc = subprocess.run(
        args, input=json.dumps({"kind": kind, "data": data}), capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = json.loads(proc.stdout)
    if "skip" in report:
        pytest.skip(report["skip"])
    return report


def _cycle_warnings(report):
    return [w for w in report["validate"]["warnings"] if "Mandatory creation cycle" in w]


def _duplicate_warnings(report):
    return [w for w in report["validate"]["warnings"] if "are connected by" in w]


def _pair(spec, a, b):
    rels = [r for r in spec["relationships"] if {r["source"], r["target"]} == {a, b}]
    assert len(rels) == 1, f"{a}-{b} must be connected exactly once, got {len(rels)}"
    return rels[0]


def _end(rel, cls):
    """Multiplicity of the end typed *cls* — the bound the OTHER class sees."""
    return rel["targetMultiplicity"] if rel["target"] == cls else rel["sourceMultiplicity"]


@pytest.fixture(scope="module")
def agent_spec():
    import diagram_handlers.types.class_diagram_handler as module
    saved = module.COMPACT_SPEC_ENABLED
    module.COMPACT_SPEC_ENABLED = False  # the canned LLM answers in the canonical wire shape
    try:
        handler = ClassDiagramHandler(_CannedLLM(json.dumps(_LIVE_SPEC)))
        result = handler.generate_complete_system(_PROSE, raw_request=_PROSE)
    finally:
        module.COMPACT_SPEC_ENABLED = saved
    assert result["action"] == "inject_complete_system"
    return result["systemSpec"]


@pytest.fixture(scope="module")
def agent_report(agent_spec):
    return _besser_report("spec", agent_spec)


@pytest.fixture(scope="module")
def live_shape_report():
    return _besser_report("spec", _LIVE_SPEC)


@pytest.fixture(scope="module")
def reference_report():
    with open(_FIXTURE, encoding="utf-8") as fh:
        export = json.load(fh)
    return _besser_report("diagram", export["project"]["diagrams"]["ClassDiagram"][0])


# -- reproduction ----------------------------------------------------------
def test_the_live_shape_trips_both_checks(live_shape_report):
    """The raw LLM spec, converted as-is, is what the agent produced and
    what the model-contract check refuses — proof the checks see the defect."""
    assert any("Booking" in w and "ReservedRoom" in w
               for w in _cycle_warnings(live_shape_report))
    assert len(_duplicate_warnings(live_shape_report)) == 3


# -- the agent's repaired spec ---------------------------------------------
def test_agent_output_passes_the_model_contract(agent_report):
    assert agent_report["validate"]["success"], agent_report["validate"]["errors"]
    assert _cycle_warnings(agent_report) == []
    assert _duplicate_warnings(agent_report) == []


def test_agent_output_connects_each_pair_once(agent_spec):
    pairs = [frozenset((r["source"], r["target"])) for r in agent_spec["relationships"]]
    assert len(pairs) == len(set(pairs)) == 4


def test_at_least_one_room_becomes_a_parsed_invariant(agent_spec, agent_report):
    rel = _pair(agent_spec, "Booking", "ReservedRoom")
    assert _end(rel, "ReservedRoom") == "0..*"
    assert _end(rel, "Booking") == "1", "the link still needs its booking"
    assert agent_spec["constraints"] == [{
        "context": "Booking",
        "expression": "context Booking inv reservedRooms_at_least_1: "
                      "self.reservedRooms->size() >= 1",
        "name": "reservedRooms_at_least_1",
    }]
    assert agent_report["ocl_warnings"] == []
    assert "reservedRooms_at_least_1" in agent_report["constraints"]
    assert agent_report["classes"]["Booking"]["ends"]["reservedRooms"][:2] == ["ReservedRoom", 0]


def test_bounds_that_do_not_cycle_are_kept(agent_spec):
    """Matches the reference: guests 1..*, room 1 per reserved room, bill 0..1."""
    assert _end(_pair(agent_spec, "Booking", "Guest"), "Guest") == "1..*"
    assert _end(_pair(agent_spec, "ReservedRoom", "Room"), "Room") == "1"
    bill = _pair(agent_spec, "Booking", "Bill")
    assert _end(bill, "Bill") == "0..1"
    assert _end(bill, "Booking") == "1"


# -- the human reference ---------------------------------------------------
def test_reference_model_validates_with_zero_warnings(reference_report):
    assert reference_report["validate"] == {"success": True, "errors": [], "warnings": []}
    assert reference_report["ocl_warnings"] == []
    assert reference_report["classes"]["ReservedRoom"]["kind"] == "AssociationClass", \
        "ClassLinkRel must round-trip into an AssociationClass"
    assert reference_report["classes"]["Booking"]["ends"]["rooms"][:2] == ["Room", 1], \
        "'at least one room' lives on the direct Booking-Room association"


_NATIVE_COMPACT = {
    "name": "HotelBooking", "ocl": [],
    "classes": [
        {"n": "Booking", "a": ["number: int!"], "m": [], "k": ""},
        {"n": "Room", "a": ["roomNumber: str!"], "m": [], "k": ""},
        {"n": "Guest", "a": ["name: str"], "m": [], "k": ""},
        {"n": "ReservedRoom", "a": ["agreedPrice: float", "extraCharges: float"], "m": [], "k": ""},
    ],
    "rels": [
        {"f": "Booking", "t": "Room", "k": "assoc", "l": "rooms", "ac": "ReservedRoom",
         "how_many_SOURCE_for_one_TARGET": "0..*", "how_many_TARGET_for_one_SOURCE": "1..*"},
        {"f": "Booking", "t": "Guest", "k": "assoc", "l": "guests",
         "how_many_SOURCE_for_one_TARGET": "0..*", "how_many_TARGET_for_one_SOURCE": "1..*"},
    ],
}


@pytest.mark.parametrize("compact", [True, False])
def test_native_agent_spec_creates_attributed_links_without_relaxing_bounds(monkeypatch, compact):
    import diagram_handlers.types.class_diagram_handler as module
    from schemas.compact_class_diagram import CompactSystemClassSpec, expand_compact_spec

    response = CompactSystemClassSpec(**_NATIVE_COMPACT)
    if not compact:
        response = expand_compact_spec(response)
    monkeypatch.setattr(module, "COMPACT_SPEC_ENABLED", compact)
    handler = ClassDiagramHandler(_CannedLLM(response.model_dump_json()))
    result = handler.generate_complete_system(_PROSE, raw_request=_PROSE)
    assert result["action"] == "inject_complete_system"
    spec = result["systemSpec"]
    assert _pair(spec, "Booking", "Room")["associationClass"] == "ReservedRoom"
    assert _end(_pair(spec, "Booking", "Room"), "Room") == "1..*"
    assert spec["constraints"] == [], "native links need no construction-cycle relaxation"
    report = _besser_report("native", spec)
    assert report["validate"] == {"success": True, "errors": [], "warnings": []}
    assert report["classes"]["ReservedRoom"]["kind"] == "AssociationClass"
    assert report["classes"]["Booking"]["ends"]["rooms"][:2] == ["Room", 1]
    assert report["http"]["links"] == [[100.0, 12.5], [200.0, 0.0]]
    assert report["http"]["empty_rooms_status"] == 400
    assert report["http"]["duplicate_room_status"] == 409


def test_native_attachment_survives_guards_and_rejects_invalid_references():
    from pydantic import ValidationError
    from schemas.class_diagram import SystemClassSpec
    from schemas.compact_class_diagram import CompactSystemClassSpec, expand_compact_spec

    spec = expand_compact_spec(CompactSystemClassSpec(**_NATIVE_COMPACT)).model_dump()
    spec["classes"][-1]["className"] = "reserved-room"
    spec["relationships"][0]["associationClass"] = "reserved-room"
    handler = ClassDiagramHandler(None)
    handler._sanitize_identifier_names(spec)
    assert spec["relationships"][0]["associationClass"] == "ReservedRoom"
    # A distinct plain link sharing the same endpoints cannot consume the attachment.
    plain = {**spec["relationships"][0], "associationClass": None}
    spec["relationships"].insert(0, plain)
    handler._merge_redundant_parallel_associations(spec)
    assert len(spec["relationships"]) == 3
    for invalid in ("Missing", "Booking"):
        broken = deepcopy(spec)
        broken["relationships"][1]["associationClass"] = invalid
        with pytest.raises(ValidationError):
            SystemClassSpec(**broken)
    broken = deepcopy(spec)
    broken["relationships"][0]["associationClass"] = "ReservedRoom"
    with pytest.raises(ValidationError, match="exactly one association"):
        SystemClassSpec(**broken)


# ---------------------------------------------------------------------------
# Both ends of every relationship, end to end (the inherited-end clash)
# ---------------------------------------------------------------------------
# The hotel prose states two rules OCL has to carry: "the total number of
# guests must not exceed the combined capacity of the rooms booked" and "a
# room cannot be double-booked". Both navigate association ends by name, so
# the repair that makes the Person/Employee/Guest ends unique decides whether
# they survive.
#
# The old repair SWAPPED the endpoints of Booking->Guest and
# Booking->Employee. ``name`` carries the TARGET end's role, so the swap moved
# 'guests' onto Guest's side and left Booking navigating 'guest' -- BESSER
# answered "Property 'guests' not found in context 'Booking' (did you mean
# 'self.guest'?)" and dropped the invariant. Naming the source end instead
# keeps every end reading correctly from both sides.

_BOTH_ENDS_PROSE = (
    "A hotel takes bookings. Each booking has a contact person, lists at least "
    "one guest and covers at least one room, and is handled by an employee. "
    "Employees and guests are people. The total number of guests must not "
    "exceed the combined capacity of the rooms booked, and a room cannot be "
    "double-booked for overlapping dates."
)

_BOTH_ENDS_OCL = [
    "context Booking inv guestsWithinCapacity: self.guests->size() <= "
    "self.rooms->collect(maxOccupancy)->sum()",
    "context Room inv noOverlappingBookings: self.bookings->forAll(b1, b2 | "
    "b1 <> b2 implies b1.departureDate <= b2.arrivalDate or "
    "b2.departureDate <= b1.arrivalDate)",
]

_BOTH_ENDS_CLASSES = [
    {"n": "Person", "a": ["name: str"], "m": [], "k": ""},
    {"n": "Employee", "a": ["staffNumber: str!"], "m": [], "k": ""},
    {"n": "Guest", "a": ["loyaltyId: str"], "m": [], "k": ""},
    {"n": "Room", "a": ["roomNumber: str!", "maxOccupancy: int"], "m": [], "k": ""},
    {"n": "Booking", "a": ["arrivalDate: date", "departureDate: date"], "m": [], "k": ""},
]

# (f, t, kind, l, how_many_TARGET_for_one_SOURCE, how_many_SOURCE_for_one_TARGET, ls)
_BOTH_ENDS_RELS = [
    ("Employee", "Person", "inher", "", "", "", ""),
    ("Guest", "Person", "inher", "", "", "", ""),
    ("Booking", "Person", "assoc", "contact", "1", "0..*", "bookingsAsContact"),
    ("Booking", "Guest", "assoc", "guests", "1..*", "0..*", "bookings"),
    ("Booking", "Employee", "assoc", "handledBy", "1", "0..*", "bookingsHandled"),
    ("Booking", "Room", "assoc", "rooms", "1..*", "0..*", "bookings"),
]


def _both_ends_compact(with_source_roles):
    return {
        "name": "HotelBooking", "classes": _BOTH_ENDS_CLASSES,
        "ocl": _BOTH_ENDS_OCL,
        "rels": [
            {"f": f, "t": t, "k": k, "l": l, "ac": "",
             "ls": ls if with_source_roles else "",
             "how_many_TARGET_for_one_SOURCE": tgt,
             "how_many_SOURCE_for_one_TARGET": src}
            for f, t, k, l, tgt, src, ls in _BOTH_ENDS_RELS
        ],
    }


def _both_ends_spec(with_source_roles):
    import diagram_handlers.types.class_diagram_handler as module
    from schemas.compact_class_diagram import CompactSystemClassSpec

    response = CompactSystemClassSpec(**_both_ends_compact(with_source_roles))
    saved = module.COMPACT_SPEC_ENABLED
    module.COMPACT_SPEC_ENABLED = True
    try:
        handler = ClassDiagramHandler(_CannedLLM(response.model_dump_json()))
        result = handler.generate_complete_system(_BOTH_ENDS_PROSE,
                                                  raw_request=_BOTH_ENDS_PROSE)
    finally:
        module.COMPACT_SPEC_ENABLED = saved
    assert result["action"] == "inject_complete_system"
    return result["systemSpec"]


@pytest.fixture(scope="module")
def both_ends_report():
    # The REAL TypeScript converter: the OCL navigates attributes as well as
    # ends, and the contract-only shim leaves attributes out.
    return _besser_report("real", _both_ends_spec(True))


def test_the_repair_never_swaps_endpoints(monkeypatch):
    """A swap moves both role names to the opposite end."""
    spec = _both_ends_spec(True)
    assert [(r["source"], r["target"]) for r in spec["relationships"]] == [
        (f, t) for f, t, *_ in _BOTH_ENDS_RELS]


def test_named_source_ends_survive_the_guards(monkeypatch):
    spec = _both_ends_spec(True)
    roles = {(r["source"], r["target"]): r.get("sourceRole")
             for r in spec["relationships"]}
    assert roles[("Booking", "Room")] == "bookings"
    assert roles[("Booking", "Guest")] == "bookings"
    assert roles[("Booking", "Person")] == "bookingsAsContact"


def test_both_stated_rules_parse_against_besser(both_ends_report):
    """The acceptance criterion: zero warnings and both invariants recovered."""
    assert both_ends_report["validate"] == {
        "success": True, "errors": [], "warnings": []}
    assert both_ends_report["ocl_warnings"] == []
    assert both_ends_report["constraints"] == [
        "guestsWithinCapacity", "noOverlappingBookings"]


def test_every_end_reads_correctly_from_both_sides(both_ends_report):
    classes = both_ends_report["classes"]
    assert sorted(classes["Booking"]["ends"]) == [
        "contact", "guests", "handledBy", "rooms"]
    assert sorted(classes["Room"]["ends"]) == ["bookings"]
    assert sorted(classes["Person"]["ends"]) == ["bookingsAsContact"]
    assert sorted(classes["Guest"]["ends"]) == ["bookings"]
    assert sorted(classes["Employee"]["ends"]) == ["bookingsHandled"]


def test_unnamed_source_ends_are_repaired_without_losing_the_named_ones():
    """Same prose, source ends left blank: the inherited clash still has to be
    resolved, and the OCL-navigable target names must survive it."""
    spec = _both_ends_spec(False)
    report = _besser_report("real", spec)
    assert report["validate"]["errors"] == []
    assert sorted(report["classes"]["Booking"]["ends"]) == [
        "contact", "guests", "handledBy", "rooms"]
    # Employee inherits Person's end, so the three 'booking' ends must differ.
    inherited = (list(report["classes"]["Person"]["ends"])
                 + list(report["classes"]["Employee"]["ends"])
                 + list(report["classes"]["Guest"]["ends"]))
    assert len(inherited) == len(set(inherited)) == 3
    assert "guestsWithinCapacity" in report["constraints"]


def test_besser_resolves_the_inherited_clash_the_removed_guard_existed_for():
    """The guard was added for "The class 'Employee' cannot have two
    association ends with the same name: 'booking'". BESSER's own
    ``_dedupe_end_name`` now resolves that, so the raw
    unrepaired shape must convert cleanly on its own."""
    raw = {
        "systemName": "HotelBooking",
        "classes": [{"className": n, "attributes": [], "methods": []}
                    for n in ("Person", "Employee", "Guest", "Booking")],
        "relationships": [
            {"type": "Inheritance", "source": "Employee", "target": "Person"},
            {"type": "Inheritance", "source": "Guest", "target": "Person"},
            {"type": "Association", "source": "Booking", "target": "Person",
             "sourceMultiplicity": "0..*", "targetMultiplicity": "1", "name": "contact"},
            {"type": "Association", "source": "Booking", "target": "Guest",
             "sourceMultiplicity": "0..*", "targetMultiplicity": "1..*", "name": "guests"},
            {"type": "Association", "source": "Booking", "target": "Employee",
             "sourceMultiplicity": "0..*", "targetMultiplicity": "1", "name": "handledBy"},
        ],
        "constraints": [],
    }
    report = _besser_report("real", raw)
    assert report["validate"] == {"success": True, "errors": [], "warnings": []}
    assert sorted(report["classes"]["Booking"]["ends"]) == [
        "contact", "guests", "handledBy"]
    for cls in ("Employee", "Guest"):
        ends = list(report["classes"][cls]["ends"])
        assert len(ends) == len(set(ends)), ends
