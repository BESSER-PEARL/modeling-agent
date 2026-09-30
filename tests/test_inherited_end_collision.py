"""Association ends that collide through an inheritance hierarchy.

A generation run died before a single file was generated:

    ValueError: The class 'Employee' cannot have two association ends
    with the same name: 'booking'

The agent names one end of each association and leaves the other blank, and
an unnamed end takes the lowercased class name. So Booking->Person[contact]
gives Person an end called 'booking', Booking->Employee[handledBy] gives
Employee one too, and Employee inherits Person's. Guest is identical.

``_ensure_unique_association_ends`` DETECTED this - it logged the clash four
times - and then declined to act: it believed a source-derived end could not
be named. It can: the converter writes ``source.role = rel.sourceRole``. The
guard now names the colliding source ends instead of swapping endpoints,
which relabelled the OPPOSITE end and cost the model its OCL
(``test_source_role_survives_to_besser`` in test_hotel_model_contract.py).
"""
import copy

import pytest

from src.diagram_handlers.types.class_diagram_handler import ClassDiagramHandler


def _hotel_spec():
    return {
        "classes": [{"className": n} for n in
                    ["Person", "Employee", "Guest", "Room", "Booking", "Bill"]],
        "relationships": [
            {"source": "Booking", "target": "Person", "type": "association", "name": "contact"},
            {"source": "Booking", "target": "Guest", "type": "association", "name": "guests"},
            {"source": "Booking", "target": "Employee", "type": "association", "name": "handledBy"},
            {"source": "Employee", "target": "Person", "type": "inheritance"},
            {"source": "Guest", "target": "Person", "type": "inheritance"},
        ],
    }


def _end_names_per_class(spec):
    """The names the validator derives, counting inherited ends."""
    inheritance = [(r["source"], r["target"]) for r in spec["relationships"]
                   if r["type"] == "inheritance"]
    ends = {}
    for r in spec["relationships"]:
        if r["type"] == "inheritance":
            continue
        ends.setdefault(r["source"], []).append(r.get("name") or r["target"].lower())
        ends.setdefault(r["target"], []).append(
            r.get("sourceRole") or r["source"].lower())

    def chain(cls):
        out, changed = {cls}, True
        while changed:
            changed = False
            for a, b in inheritance:
                if a in out and b not in out:
                    out.add(b)
                    changed = True
        return out

    return {c: [n for k in chain(c) for n in ends.get(k, [])]
            for c in {r["source"] for r in spec["relationships"]} |
                     {r["target"] for r in spec["relationships"]}}


def test_hierarchy_sharing_one_association_target_has_unique_ends():
    spec = _hotel_spec()
    ClassDiagramHandler(None)._ensure_unique_association_ends(spec)
    for cls, names in _end_names_per_class(spec).items():
        assert len(names) == len(set(names)), f"{cls} has duplicate ends: {sorted(names)}"


def test_the_unrepaired_spec_really_does_collide():
    """Guards the test itself: without the repair, Employee and Guest clash."""
    clashing = [c for c, n in _end_names_per_class(_hotel_spec()).items()
                if len(n) != len(set(n))]
    assert "Employee" in clashing and "Guest" in clashing


def test_no_relationship_is_ever_reoriented():
    """Endpoints carry the roles: swapping them relabels both ends. The repair
    is naming, never reorientation — for compositions, whose direction is
    semantic, and for plain associations alike."""
    spec = _hotel_spec()
    for r in spec["relationships"]:
        if r.get("name") == "handledBy":
            r["type"] = "composition"
    before = [(r["source"], r["target"]) for r in spec["relationships"]]
    ClassDiagramHandler(None)._ensure_unique_association_ends(spec)
    assert [(r["source"], r["target"]) for r in spec["relationships"]] == before


def test_the_repair_names_source_ends_and_leaves_target_names_alone():
    """The names the OCL navigates (contact/guests/handledBy) are untouched;
    the colliding source-derived ends get the suffix instead."""
    spec = _hotel_spec()
    ClassDiagramHandler(None)._ensure_unique_association_ends(spec)
    by_target = {r["target"]: r for r in spec["relationships"]
                 if r["type"] == "association"}
    assert [r["name"] for r in spec["relationships"] if r["type"] == "association"] == [
        "contact", "guests", "handledBy"]
    assert by_target["Person"].get("sourceRole") in (None, "booking")
    assert by_target["Guest"]["sourceRole"] == "booking_1"
    assert by_target["Employee"]["sourceRole"] == "booking_2"


def test_clean_spec_is_left_alone():
    spec = {
        "classes": [{"className": n} for n in ["Author", "Book"]],
        "relationships": [
            {"source": "Author", "target": "Book", "type": "association", "name": "writes"},
        ],
    }
    before = copy.deepcopy(spec["relationships"])
    ClassDiagramHandler(None)._ensure_unique_association_ends(spec)
    assert spec["relationships"] == before


def test_prompt_tells_the_model_to_name_both_ends():
    p = ClassDiagramHandler(None)._get_system_generation_prompt()
    assert "NAME BOTH ENDS OF EVERY RELATIONSHIP" in p
    # The rule must name fields the schemas actually have, or the model
    # invents a syntax that is silently dropped.
    assert "sourceRole" in p and "compact encoding: l and ls" in p
    assert "defaults to the lowercased class name" in p
    assert "INCLUDING the names it inherits" in p
