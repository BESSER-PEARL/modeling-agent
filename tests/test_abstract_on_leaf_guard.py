"""``isAbstract`` on a leaf class instead of on the base.

Measured on a 10-run live batch of the hotel benchmark prompt: 3 of 10 runs
marked a SUBCLASS abstract while its base stayed concrete --

    run03  Guest abstract,  Person concrete
    run04  Employee abstract, Person concrete
    run07  BOTH Employee and Guest abstract, Person concrete

An abstract leaf cannot be instantiated, so the delivered app could not create a
guest at all, and nothing downstream objected: BUML happily holds an abstract
class with no subclasses beneath it.

The guard only CLEARS the flag from a class that is the source of an inheritance
and never its target. It does not promote the base, because that would be a
guess -- the benchmark's own known-good hotel model marks no class abstract at
all, so these specs do not require an abstract base.
"""
import pytest

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler


@pytest.fixture
def handler():
    return ClassDiagramHandler(None)


def _spec(classes, rels):
    return {"classes": [dict(c) for c in classes],
            "relationships": [dict(r) for r in rels]}


def _abstract(spec):
    return sorted(c["className"] for c in spec["classes"] if c.get("isAbstract"))


HIERARCHY = [{"type": "Inheritance", "source": "Guest", "target": "Person"},
             {"type": "Inheritance", "source": "Employee", "target": "Person"}]


@pytest.mark.parametrize("abstract_leaves", [["Guest"], ["Employee"],
                                             ["Employee", "Guest"]])
def test_the_flag_is_cleared_from_a_leaf(handler, abstract_leaves):
    """runs 03, 04 and 07 of the live batch."""
    spec = _spec([{"className": "Person"},
                  {"className": "Guest", "isAbstract": "Guest" in abstract_leaves},
                  {"className": "Employee", "isAbstract": "Employee" in abstract_leaves}],
                 HIERARCHY)

    handler._fix_misplaced_abstract(spec)

    assert _abstract(spec) == []


def test_an_abstract_base_is_kept(handler):
    """The legitimate placement must survive untouched."""
    spec = _spec([{"className": "Person", "isAbstract": True},
                  {"className": "Guest"}, {"className": "Employee"}], HIERARCHY)

    handler._fix_misplaced_abstract(spec)

    assert _abstract(spec) == ["Person"]


def test_a_middle_class_in_a_chain_is_kept(handler):
    """B is a subclass AND a base; abstract is meaningful there."""
    spec = _spec([{"className": "A"}, {"className": "B", "isAbstract": True},
                  {"className": "C"}],
                 [{"type": "Inheritance", "source": "B", "target": "A"},
                  {"type": "Inheritance", "source": "C", "target": "B"}])

    handler._fix_misplaced_abstract(spec)

    assert _abstract(spec) == ["B"]


def test_a_class_outside_any_hierarchy_is_untouched(handler):
    """An abstract class with no inheritance at all is not this guard's business."""
    spec = _spec([{"className": "Person"}, {"className": "Guest"},
                  {"className": "Loner", "isAbstract": True}], HIERARCHY)

    handler._fix_misplaced_abstract(spec)

    assert _abstract(spec) == ["Loner"]


def test_no_inheritance_means_no_change(handler):
    spec = _spec([{"className": "A", "isAbstract": True}], [])

    handler._fix_misplaced_abstract(spec)

    assert _abstract(spec) == ["A"]


def test_it_does_not_promote_the_base(handler):
    """Clearing a leaf must not silently make the base abstract instead --
    the benchmark model marks nothing abstract."""
    spec = _spec([{"className": "Person"},
                  {"className": "Guest", "isAbstract": True},
                  {"className": "Employee"}], HIERARCHY)

    handler._fix_misplaced_abstract(spec)

    assert _abstract(spec) == []


def test_a_malformed_relationship_is_survived(handler):
    spec = _spec([{"className": "Guest", "isAbstract": True}],
                 [{"type": "Inheritance", "source": "Guest", "target": "Missing"},
                  {"type": "Inheritance"}])

    handler._fix_misplaced_abstract(spec)   # must not raise
