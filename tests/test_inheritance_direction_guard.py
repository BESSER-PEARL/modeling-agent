"""An abstract base declared as the CHILD of its own subclasses.

On the hotel benchmark prompt the spec shipped

    Inheritance  Person -> Guest
    Inheritance  Person -> Employee

with Person marked abstract. The converter reads source as the SUBCLASS, so the
domain model came out as "Person extends Guest" and "Person extends Employee":
an abstract class multiply inheriting from two concrete ones, exactly backwards
from the spec's "two specialised kinds of persons exist".

Nothing caught it. ``DomainModel.validate()`` checks that both ends EXIST, never
that the direction is sensible, so the model validated CLEAN and reached code
generation inverted. The benchmark's own known-good model has it the right way
round (source=Guest -> target=Person), which is the control for the convention.

The signal is the SHARED SOURCE. A base extended by two kinds appears as two
links sharing a TARGET; the inverted form shares a SOURCE, i.e. multiple
inheritance, which these specs never ask for.

Keying on ``isAbstract`` is wrong: a spec can emit the direction CORRECTLY
(``Employee -> Person``) while marking *Employee* abstract and Person concrete
-- an abstract-child rule would invert a correct hierarchy. Pinned below.
"""
import pytest

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler


@pytest.fixture
def handler():
    return ClassDiagramHandler(None)


def _spec(classes, rels):
    return {"classes": [dict(c) for c in classes],
            "relationships": [dict(r) for r in rels]}


def _pairs(spec):
    return [(r["source"], r["target"]) for r in spec["relationships"]]


PERSON = {"className": "Person", "isAbstract": True}
GUEST = {"className": "Guest"}
EMPLOYEE = {"className": "Employee"}


def test_the_live_hotel_inversion_is_corrected(handler):
    """The regression, verbatim from the reported run."""
    spec = _spec([PERSON, GUEST, EMPLOYEE], [
        {"type": "Inheritance", "source": "Person", "target": "Guest"},
        {"type": "Inheritance", "source": "Person", "target": "Employee"},
    ])

    handler._fix_inverted_inheritance(spec)

    assert _pairs(spec) == [("Guest", "Person"), ("Employee", "Person")]


def test_a_correct_hierarchy_is_left_alone(handler):
    """Must not flip what the benchmark model already gets right."""
    spec = _spec([PERSON, GUEST, EMPLOYEE], [
        {"type": "Inheritance", "source": "Guest", "target": "Person"},
        {"type": "Inheritance", "source": "Employee", "target": "Person"},
    ])

    handler._fix_inverted_inheritance(spec)

    assert _pairs(spec) == [("Guest", "Person"), ("Employee", "Person")]


def test_the_abstract_flag_does_not_arbitrate_direction(handler):
    """Observed spec, verbatim: correct direction, flag on the WRONG
    class. An abstract-child rule would invert this. It must not move."""
    spec = _spec([{"className": "Person"},
                  {"className": "Employee", "isAbstract": True},
                  {"className": "Guest"}],
                 [{"type": "Inheritance", "source": "Employee", "target": "Person"},
                  {"type": "Inheritance", "source": "Guest", "target": "Person"}])

    handler._fix_inverted_inheritance(spec)

    assert _pairs(spec) == [("Employee", "Person"), ("Guest", "Person")]


@pytest.mark.parametrize("classes, why", [
    ([{"className": "A"}, {"className": "B"}],
     "a lone link has no shared end to judge by"),
    ([{"className": "A", "isAbstract": True}, {"className": "B"}],
     "still a lone link -- the abstract flag is not evidence"),
])
def test_a_lone_link_is_left_alone(handler, classes, why):
    spec = _spec(classes, [{"type": "Inheritance", "source": "A", "target": "B"}])

    handler._fix_inverted_inheritance(spec)

    assert _pairs(spec) == [("A", "B")], why


def test_a_genuine_diamond_is_left_alone(handler):
    """Two links sharing a TARGET is the normal shape and must never flip."""
    spec = _spec([{"className": "P"}, {"className": "X"}, {"className": "Y"}],
                 [{"type": "Inheritance", "source": "X", "target": "P"},
                  {"type": "Inheritance", "source": "Y", "target": "P"}])

    handler._fix_inverted_inheritance(spec)

    assert _pairs(spec) == [("X", "P"), ("Y", "P")]


def test_associations_are_not_inheritance(handler):
    """An abstract class on either end of an ASSOCIATION is ordinary."""
    spec = _spec([PERSON, GUEST], [
        {"type": "Association", "source": "Person", "target": "Guest", "name": "contact"},
    ])

    handler._fix_inverted_inheritance(spec)

    assert _pairs(spec) == [("Person", "Guest")]


@pytest.mark.parametrize("rels", [
    [{"type": "Inheritance", "source": "Person", "target": "Nope"}],
    [{"type": "Inheritance", "source": "Person", "target": "Person"}],
    [{"type": "Inheritance"}],
])
def test_a_malformed_relationship_is_survived(handler, rels):
    """A dangling or self-referential end must not raise; a later guard owns it."""
    spec = _spec([PERSON, GUEST], rels)

    handler._fix_inverted_inheritance(spec)   # must not raise


def test_no_classes_is_survived(handler):
    handler._fix_inverted_inheritance({"classes": [], "relationships": []})
    handler._fix_inverted_inheritance({})
