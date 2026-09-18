"""Association ends that collide through an inheritance hierarchy.

Run 3a4cab06 (2026-09-17) died before a single file was generated:

    ValueError: The class 'Employee' cannot have two association ends
    with the same name: 'booking'

The agent names one end of each association and leaves the other blank, and
an unnamed end takes the lowercased class name. So Booking->Person[contact]
gives Person an end called 'booking', Booking->Employee[handledBy] gives
Employee one too, and Employee inherits Person's. Guest is identical.

``_ensure_unique_association_ends`` DETECTED this - it logged the clash four
times - and then declined to act, because its reorientation phase only fires
for parallel associations between the *identical* pair
(``pair_direction_counts[(src, tgt)] > 1``). Here the pairs differ and only
meet through inheritance, so nothing flipped and the run aborted with an
opaque BAD_REQUEST.
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
        ends.setdefault(r["target"], []).append(r["source"].lower())

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


def test_only_plain_associations_are_flipped():
    """A composition carries direction meaning and must never be reoriented."""
    spec = _hotel_spec()
    for r in spec["relationships"]:
        if r.get("name") == "handledBy":
            r["type"] = "composition"
    before = copy.deepcopy(spec["relationships"])
    ClassDiagramHandler(None)._ensure_unique_association_ends(spec)
    after = next(r for r in spec["relationships"] if r.get("name") == "handledBy")
    orig = next(r for r in before if r.get("name") == "handledBy")
    assert (after["source"], after["target"]) == (orig["source"], orig["target"])


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
    assert "defaults to the lowercased class name" in p
    assert "INCLUDING the names it inherits" in p
