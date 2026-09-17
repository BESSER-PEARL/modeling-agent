"""The run must receive the user's spec, not only a summary of it.

Root cause found on the 2026-09-17 hotel run. ``trigger_smart_generator``
carried ``refined_instructions`` — by its own schema "1-3 short paragraphs"
written by the classifier — and nothing else. The downstream gap analyser
diffs the request against the model to recover requirements the modelling
step lost, so it was diffing against a summary.

What that cost, measured on run 95bf7725:

  * it DID recover "Implement Booking.checkIn() and Booking.checkOut() ...
    user requested 'stay management'" — the phrase survived into the
    summary, and the diff worked;
  * it did NOT see the two status vocabularies ("awaiting payment,
    confirmed, cancelled" / "not arrived, checked in, checked out") or the
    four business rules — those sentences were compressed away;
  * it attributed three JWT tasks to 'user requested "personalized screens
    and navigation"', which is assistant flow wording the summary had
    absorbed. The user never wrote it.
"""
from __future__ import annotations

import pytest

from src.handlers.smart_generation_handler import (
    GenerationClassification,
    build_trigger_smart_generator_payload,
)

SUMMARY = "Build a hotel booking web app. FastAPI, SQLite, React."

SPEC = (
    "A booking has a commercial status: awaiting payment, confirmed, or "
    "cancelled. Separately it has a physical status: not arrived, checked "
    "in, or checked out. The total number of guests must not exceed the "
    "combined capacity of the rooms booked."
)


def _payload(**kwargs):
    return build_trigger_smart_generator_payload(
        GenerationClassification(
            route="smart", refined_instructions=SUMMARY, provider="openai",
            reason="test",
        ),
        **kwargs,
    )


def test_spec_reaches_the_run_verbatim():
    instructions = _payload(original_request=SPEC)["instructions"]
    # The exact sentences the gap analyser has to diff against the model.
    assert "awaiting payment, confirmed, or cancelled" in instructions
    assert "not arrived, checked in, or checked out" in instructions
    assert "combined capacity of the rooms booked" in instructions


def test_summary_is_kept_alongside_it():
    """The summary names the stack; dropping it would lose that."""
    instructions = _payload(original_request=SPEC)["instructions"]
    assert SUMMARY in instructions
    assert instructions.index(SUMMARY) < instructions.index("awaiting payment")


def test_the_spec_is_marked_as_the_authority():
    """Without this the model treats two descriptions as equally valid and
    follows whichever is shorter."""
    instructions = _payload(original_request=SPEC)["instructions"]
    assert "original request, verbatim" in instructions.lower()
    assert "this text wins" in instructions.lower()


def test_no_original_request_leaves_the_payload_unchanged():
    """Short asks ('make a hotel app') never reach the stash, so the old
    single-summary payload must still be produced exactly."""
    assert _payload()["instructions"] == SUMMARY
    assert _payload(original_request="   ")["instructions"] == SUMMARY


def test_a_spec_already_inside_the_summary_is_not_duplicated():
    payload = build_trigger_smart_generator_payload(
        GenerationClassification(
            route="smart", refined_instructions=f"{SUMMARY}\n\n{SPEC}",
            provider="openai", reason="test",
        ),
        original_request=SPEC,
    )
    assert payload["instructions"].count("awaiting payment") == 1


def test_payload_shape_is_untouched():
    payload = _payload(original_request=SPEC)
    assert payload["action"] == "trigger_smart_generator"
    assert payload["provider"] == "openai"
    assert payload["llmModel"]
    assert payload["message"]


def test_non_smart_classification_still_rejected():
    with pytest.raises(ValueError):
        build_trigger_smart_generator_payload(
            GenerationClassification(route="deterministic", reason="x"),
            original_request=SPEC,
        )


# ----------------------------------------------------------------------
# Which text gets stashed
#
# The first version of this fix stashed ``operation_request`` and looked
# correct in every unit test, then captured nothing useful live: the planner
# had already rewritten the 4,622-character spec into one line.
# ----------------------------------------------------------------------

from src.execution.model_operations import original_request_to_stash


class _Req:
    def __init__(self, message):
        self.message = message


FULL_SPEC = (
    "A hotel booking and stay management system. A booking has a commercial "
    "status: awaiting payment, confirmed, or cancelled. Separately it has a "
    "physical status: not arrived, checked in, or checked out. A booking "
    "offers five actions, each of which reports back whether it succeeded."
)

PLANNER_REWRITE = (
    "create a hotel booking and stay management system with persons "
    "(employees and guests), rooms, bookings, billing, and the relationships "
    "between them, including statuses and the actions a booking supports"
)


def test_stashes_the_user_message_not_the_planner_rewrite():
    stashed = original_request_to_stash(_Req(FULL_SPEC), "complete_system", "ClassDiagram")
    assert stashed == FULL_SPEC
    assert "awaiting payment" in stashed
    assert stashed != PLANNER_REWRITE


def test_planner_rewrite_is_not_what_the_rule_reads():
    """Both strings clear the length floor, so length alone cannot tell them
    apart — the rule has to read the right field."""
    assert len(PLANNER_REWRITE) >= 200
    stashed = original_request_to_stash(_Req(FULL_SPEC), "complete_system", "ClassDiagram")
    assert "persons (employees and guests), rooms, bookings" not in stashed


def test_only_class_diagram_creation_stashes():
    assert original_request_to_stash(_Req(FULL_SPEC), "complete_system", "GUINoCodeDiagram") is None
    assert original_request_to_stash(_Req(FULL_SPEC), "modify_model", "ClassDiagram") is None


def test_short_asks_are_not_stashed():
    assert original_request_to_stash(_Req("make a hotel app"), "complete_system", "ClassDiagram") is None
    assert original_request_to_stash(_Req(""), "complete_system", "ClassDiagram") is None


def test_missing_message_attribute_is_tolerated():
    assert original_request_to_stash(object(), "complete_system", "ClassDiagram") is None
