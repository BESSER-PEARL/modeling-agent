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


# ----------------------------------------------------------------------
# Which END carries which number
#
# Run 8c5a087a, 2026-09-17: the agent read every cardinality in the spec
# correctly and then attached them to the wrong ends -
#   emitted   Booking [1]    -> Person [0..*] contact
#   correct   Booking [0..*] -> Person [1]    booking_contact
# The first says every Person has exactly one Booking. The generator
# faithfully produced PersonCreate.booking as a mandatory N:1 field, so a
# Person needed a Booking and a Booking needed a contact Person, and
# nothing in the app could be created at all.
#
# Rule 7 already said to transcribe the user's words; its worked examples
# used the shorthand "Employee 1 -- 0..* Booking", which never states
# which side of the emitted JSON each number lands on.
# ----------------------------------------------------------------------

from src.diagram_handlers.types.class_diagram_handler import ClassDiagramHandler

PROMPT = ClassDiagramHandler(None)._get_system_generation_prompt()


def test_prompt_states_what_a_multiplicity_at_an_end_means():
    low = PROMPT.lower()
    assert "how many of that end's class" in low
    assert "put the 'many' number on the class you have many of" in low


def test_prompt_gives_the_encoding_not_just_the_dash_shorthand():
    """The dash form is ambiguous about source vs target; the fix is to
    name them."""
    assert 'source=Booking with multiplicity "0..*"' in PROMPT
    assert 'target=Person with multiplicity "1"' in PROMPT


def test_prompt_shows_the_inverted_form_as_wrong():
    assert 'source=Booking "1", target=Person "0..*"' in PROMPT
    assert "opposite of what the user wrote" in PROMPT


def test_prompt_names_the_live_consequence():
    """Naming the actual breakage is what made the other rules land."""
    low = PROMPT.lower()
    assert "personcreate require a booking" in low
    assert "unusable" in low


def test_prompt_gives_a_self_check_question():
    assert "for ONE <target class>" in PROMPT
    assert "that number is the SOURCE multiplicity" in PROMPT
