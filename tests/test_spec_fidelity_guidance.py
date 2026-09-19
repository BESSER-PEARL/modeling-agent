"""The prompt must carry over what the user actually stated.

One explicit hotel spec, run live 2026-09-17. Four things the user wrote in
plain prose never reached the model:

  * "A booking offers five actions: produce the bill, check the guest in,
    check the guest out, cancel the booking, and compute the amount due."
    -> got 2 methods (checkIn, calculateTotal), plus three INVENTED ones on
    classes the user described no behaviour for (Employee.assignBooking,
    Guest.registerStay, Room.isAvailable).

  * "a commercial status: awaiting payment, confirmed or cancelled" AND
    "a physical status: not arrived, checked in or checked out"
    -> got ONE enum, BookingStatus(BOOKED, CHECKED_IN, CHECKED_OUT,
    CANCELLED, NO_SHOW): two user values dropped, two invented.

  * four business rules stated in prose -> constraints list EMPTY.

  * "the price actually agreed for that room in that booking, which may
    differ from the standard price" -> no link class anywhere.

Each had a prompt rule that actively worked against it: methods were capped
at "1-2 per class MAX", and OCL was gated on rules being "EXPLICITLY stated",
which the model read as "written as a formal constraint".

These are prompt-contract tests. They cannot prove the model complies — only
a live run does that — but each one pins a specific instruction whose absence
is the documented cause of a specific live failure.
"""
import inspect

from src.diagram_handlers.types import class_diagram_handler
from src.diagram_handlers.types.class_diagram_handler import ClassDiagramHandler

SOURCE = inspect.getsource(class_diagram_handler)
PROMPT = ClassDiagramHandler(None)._get_system_generation_prompt()


# ----------------------------------------------------------------------
# Methods: an explicit list must beat the cap
# ----------------------------------------------------------------------


def test_explicit_action_list_overrides_the_method_cap():
    """The old rule said '1-2 core domain methods per class MAX' with no
    exception, so a spec naming five actions could only lose three."""
    low = PROMPT.lower()
    assert "overrides the cap" in low
    assert "five stated actions means five methods" in low


def test_method_rule_names_the_phrasings_that_count_as_a_request():
    for phrase in ("can be cancelled", "produces a bill", "offers five actions"):
        assert phrase in PROMPT, f"method rule does not mention {phrase!r}"


def test_method_rule_forbids_inventing_methods_on_silent_classes():
    """The other half of the failure: three methods appeared on classes the
    user described no behaviour for."""
    low = PROMPT.lower()
    assert "never invent a method on a class the user said nothing about" in low
    assert "employee, guest and room get no methods" in low


def test_method_rule_keeps_the_no_behaviour_default():
    """Widening case (a) must not turn every class into a method farm."""
    low = PROMPT.lower()
    assert "the user says nothing about behaviour" in low
    assert "at most 1-2" in low
    assert "getters/setters" in low


# ----------------------------------------------------------------------
# Status dimensions: N described dimensions => N enumerations
# ----------------------------------------------------------------------


def test_prompt_forbids_merging_independent_status_dimensions():
    low = PROMPT.lower()
    assert "one enumeration per dimension" in low
    assert "never merge them into a single status" in low


def test_prompt_carries_the_two_dimension_worked_example():
    assert "BookingCommercialStatus" in PROMPT
    assert "BookingPhysicalStatus" in PROMPT
    assert "AWAITING_PAYMENT" in PROMPT


def test_prompt_names_the_invented_members_as_the_failure():
    """Naming the actual wrong output is what makes the rule concrete."""
    assert "NO_SHOW" in PROMPT and "BOOKED" in PROMPT
    assert "never mentioned" in PROMPT.lower()


def test_prompt_gives_the_reason_merging_is_wrong():
    """Co-occurring states are unrepresentable once merged."""
    low = PROMPT.lower()
    assert "co-occur" in low
    assert "confirmed" in low and "not_arrived" in low


def test_prompt_lists_the_tell_tale_second_dimension_wording():
    for phrase in ("separately", "independently", "at the same time"):
        assert phrase in PROMPT.lower(), f"missing tell-tale {phrase!r}"


# ----------------------------------------------------------------------
# OCL: prose rules are stated rules
# ----------------------------------------------------------------------


def test_prose_phrasings_count_as_explicitly_stated():
    low = PROMPT.lower()
    assert "stated in ordinary prose" in low
    for phrase in ("must not exceed", "may not overlap", "must be a valid",
                   "cannot be double-booked", "must not be before/after"):
        assert phrase in low, f"OCL rule does not list {phrase!r}"


def test_ocl_rule_keeps_the_no_invention_clause():
    """Widening what counts as 'stated' must not license invention."""
    assert "NEVER invent constraints" in PROMPT


def test_ocl_rule_carries_prose_to_ocl_worked_examples():
    for fragment in ("guestsWithinCapacity", "noOverlappingBookings",
                     "arrivalBeforeDeparture"):
        assert fragment in PROMPT, f"no worked example named {fragment!r}"


def test_ocl_rule_calls_an_empty_list_a_failure():
    """The model read caution as the safe default; it isn't."""
    low = PROMPT.lower()
    assert "empty constraints list is a failure" in low


# ----------------------------------------------------------------------
# Link attributes
# ----------------------------------------------------------------------


def test_prompt_covers_attributes_that_belong_to_the_relationship():
    low = PROMPT.lower()
    assert "belongs to the link" in low
    assert "class for the link itself" in low


def test_link_rule_carries_the_agreed_price_example():
    assert "ReservedRoom" in PROMPT
    assert "agreedPrice" in PROMPT
    assert 'Booking--Room with associationClass="ReservedRoom"' in PROMPT
    assert "ReservedRoom.extraCharges" in PROMPT
    assert "Do NOT replace it with two ordinary links" in PROMPT
    assert "Booking 1 -- 0..* ReservedRoom 0..* -- 1 Room" not in PROMPT


def test_link_rule_explains_why_neither_class_works():
    """Without the reason, the model puts it on whichever class it met first."""
    low = PROMPT.lower()
    assert "overwrites it for every other booking" in low
    assert "cannot distinguish two different rooms" in low


def test_link_rule_lists_the_tell_tale_wording():
    for phrase in ("for that <a> in that <b>", "may differ from the standard",
                   "per <a> per <b>"):
        assert phrase in PROMPT.lower(), f"missing tell-tale {phrase!r}"


# ----------------------------------------------------------------------
# The rules have to actually be in the prompt the generator uses
# ----------------------------------------------------------------------


def test_rules_reach_the_complete_system_prompt_not_just_the_source():
    """`get_system_prompt()` (single element) and
    `_get_system_generation_prompt()` (complete system) are different
    strings; the spec-fidelity rules belong to the second."""
    assert "STATUS VOCABULARIES" in PROMPT
    assert "STATUS VOCABULARIES" in SOURCE
    single = ClassDiagramHandler(None).get_system_prompt()
    assert len(PROMPT) > len(single)


# ----------------------------------------------------------------------
# Padding and invented requirements
#
# Measured 2026-09-17 across 8 live runs. Every run's instructions were
# 54-197 characters; the hotel spec was 4,622. Run 0ceb8611's instructions
# read "...Include user authentication and a responsive design" for a spec
# that never mentions users or login, and the run built both. The prompt
# had asked for auth twice by example ("Devise auth", "JWT"), and the
# class-diagram prompt demanded status fields unconditionally.
# ----------------------------------------------------------------------


def test_attributes_are_not_padded_when_the_user_listed_them():
    low = PROMPT.lower()
    assert "use their list and stop there" in low
    assert "never padded to hit a count" in low


def test_status_fields_are_conditional_not_demanded():
    """The reasoning prompt used to say, flatly, 'include IDs, timestamps,
    status fields' — which is why Room and Bill got statuses the spec has
    no trace of."""
    low = PROMPT.lower()
    assert "status field only if that entity" in low
    assert "include ids, timestamps, status fields." not in low


def test_stub_classes_are_still_discouraged_for_bare_names():
    """Removing the floor outright would make 'make a hotel app' produce
    two-attribute classes."""
    low = PROMPT.lower()
    assert "only named" in low
    assert "3-5+ attributes" in low
