"""The prompt must tell the model to read cardinality out of the user's prose.

Observed across three live runs of one very explicit spec (2026-09-16/17):
the spec said "a given guest may appear on several bookings" and "an employee
may be responsible for many bookings", and two of the three runs still emitted
Booking 1--1 Guest. The generator faithfully turns a 1--1 association into a
UNIQUE foreign key, so the shipped app then refuses the user's real data.

The prompt used to say only "ALWAYS include multiplicities on relationships",
which asks for them to be PRESENT, not for them to be CORRECT.
"""
import re

from src.diagram_handlers.types.class_diagram_handler import ClassDiagramHandler

SOURCE = __import__("inspect").getsource(
    __import__("src.diagram_handlers.types.class_diagram_handler",
               fromlist=["x"]))


def test_prompt_maps_plural_language_to_many_multiplicity():
    """The phrases users actually write must be named in the guidance."""
    for phrase in ("several", "many", "exactly one", "never more than one"):
        assert phrase in SOURCE, f"guidance does not mention {phrase!r}"


def test_prompt_warns_against_defaulting_to_one_to_one():
    low = SOURCE.lower()
    assert "never default to 1-to-1" in low or "1--1 association is rare" in low, (
        "nothing tells the model that 1--1 is the exceptional case"
    )


def test_prompt_carries_a_worked_cardinality_example():
    """An abstract rule is ignored more often than a worked example."""
    assert "0..* -- 1..* Guest" in SOURCE or "Booking 0..* -- 1..* Guest" in SOURCE


def test_prompt_states_the_downstream_consequence():
    """Saying WHY it matters measurably improves compliance."""
    low = SOURCE.lower()
    assert "unique foreign key" in low, (
        "the guidance should say a wrong 1--1 becomes a UNIQUE FK and breaks the app"
    )


def test_multiplicity_rule_still_lists_the_legal_values():
    assert re.search(r"1, 0\.\.1, 0\.\.\*, 1\.\.\*", SOURCE), (
        "the legal multiplicity vocabulary must remain in the prompt"
    )
