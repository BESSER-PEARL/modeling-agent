"""The hotel prose end to end: agent spec -> editor JSON -> BUML -> model contract.

Live runs 4efe04ff / 9a6063ed (2026-09-18) took a hotel description and
shipped a model BESSER's Phase 0 model-contract check now rejects: a mandatory
creation cycle (``Booking -> BookedRoom -> Booking``) and three class pairs
connected twice. The prose, in the shape that matters here: a booking covers
at least one room and may cover several; each room is tied to zero or
multiple booking records; the agreed price is fixed per booking on the link;
at least one guest must be listed; a bill belongs to exactly one booking, and
a booking may have a bill raised against it, but never more than one.

Two calibration targets:

1. The agent's own spec for that prose — the shape the LLM produced live,
   replayed through ``generate_complete_system`` with a canned LLM — must
   build a ``DomainModel`` that passes ``_validate_mandatory_cycles`` and
   ``_validate_duplicate_associations`` with no warning, and its relaxed
   "at least one" must come back as a parsed OCL invariant.
2. The human-modelled reference for the same prose
   (BESSER-PEARL/Agentic-Low-Code-Benchmark,
   ``hotel-booking/2-validation-and-business-rules/low-code-model/model.json``,
   stored under ``tests/fixtures/``) must validate with zero warnings. It pins
   the checks themselves against a known-good model — ``ReservedRoom``
   included, as the association class the agent's spec cannot yet express.

The BESSER side runs in ``besser_contract_probe.py`` (its own interpreter, the
workspace sibling ``../BESSER`` first on the path when present); the probe
skips these tests when the BESSER it finds lacks the checks.
"""
import json
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

# The spec the LLM produced live, reduced to the classes the prose above
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
    """The raw LLM spec, converted as-is, is what the two runs shipped and
    what Phase 0 now refuses — proof the checks see the defect."""
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
