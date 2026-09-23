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
    # Machine framing must not make an accepted boundary-sized spec invalid.
    from agent_config import MAX_USER_MESSAGE_CHARS
    from utilities.message_limits import UserMessageTooLong
    full = SPEC + "x" * (MAX_USER_MESSAGE_CHARS - len(SPEC))
    assert _payload(original_request=full)["instructions"] == full
    with pytest.raises(UserMessageTooLong, match="64,000"):
        _payload(original_request=full + "x")


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
    """Absent original context keeps the single-summary payload unchanged."""
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


def test_concise_specs_are_preserved_but_empty_asks_are_not():
    assert original_request_to_stash(_Req("make a hotel app"), "complete_system", "ClassDiagram") == "make a hotel app"
    assert original_request_to_stash(_Req(""), "complete_system", "ClassDiagram") is None


def test_missing_message_attribute_is_tolerated():
    assert original_request_to_stash(object(), "complete_system", "ClassDiagram") is None


def test_original_context_is_project_scoped_and_direct_requests_are_frozen(monkeypatch):
    from src.handlers import generation_handler as generation
    from session_keys import PENDING_SMART_GEN_ORIGINAL_REQUEST
    from utilities.original_request import remember_original_request, original_request_for_project

    class Session:
        project_id = "hotel"

        def __init__(self):
            self.data = {}

        def get(self, key):
            return self.data.get(key)

        def set(self, key, value):
            self.data[key] = value

    session = Session()
    monkeypatch.setattr(generation, "_active_project_id", lambda s: s.project_id)
    remember_original_request(session, FULL_SPEC, "hotel")
    assert original_request_for_project(session, None) == ""
    assert original_request_for_project(session, "other") == ""
    generation._build_smart_gen_confirmation(session, SUMMARY, "openai", user_message="generate application")
    assert session.get(PENDING_SMART_GEN_ORIGINAL_REQUEST) == FULL_SPEC

    # A new/imported project cannot inherit the previous project's authority.
    session.project_id = "imported-library"
    generation._build_smart_gen_confirmation(session, "Generate this library", "openai", user_message="generate application")
    assert session.get(PENDING_SMART_GEN_ORIGINAL_REQUEST) == ""
    direct = "Generate this imported library. Anonymous users can read, but only owners can edit."
    generation._build_smart_gen_confirmation(session, "Make a library app", "openai", user_message=direct)
    assert session.get(PENDING_SMART_GEN_ORIGINAL_REQUEST) == direct
    assert original_request_for_project(session, "imported-library") == direct
    assert FULL_SPEC not in session.get(PENDING_SMART_GEN_ORIGINAL_REQUEST)

    # A later global update must not mutate the already-confirmable request.
    remember_original_request(session, "An unrelated new request", "imported-library")
    assert session.get(PENDING_SMART_GEN_ORIGINAL_REQUEST) == direct
    assert generation._pending_smart_gen_project_changed(session) is False
    session.project_id = "another-project"
    assert generation._pending_smart_gen_project_changed(session) is True


def test_same_project_create_gui_confirm_keeps_raw_spec_and_followups(monkeypatch):
    from src.handlers import generation_handler as generation
    from session_keys import PENDING_SMART_GEN_ORIGINAL_REQUEST
    from utilities.original_request import remember_original_request, original_request_for_project

    class Session:
        def __init__(self):
            self.data = {}

        def get(self, key):
            return self.data.get(key)

        def set(self, key, value):
            self.data[key] = value

    session = Session()
    monkeypatch.setattr(generation, "_active_project_id", lambda s: "hotel")
    remember_original_request(session, FULL_SPEC, "hotel")
    assert original_request_to_stash(_Req("Generate GUI screens"), "complete_system", "GUINoCodeDiagram") is None
    generation._build_smart_gen_confirmation(session, SUMMARY, "openai", reason_prefix="Model rebuilt and ready.")
    assert session.get(PENDING_SMART_GEN_ORIGINAL_REQUEST) == FULL_SPEC
    change = "Use SQLite, FastAPI and React."
    combined = generation._original_for_smart_generation(session, change)
    assert FULL_SPEC in combined and change in combined
    assert generation._original_for_smart_generation(session, change) == combined
    assert generation._original_for_smart_generation(session, "yes") == combined
    assert generation._original_for_smart_generation(session, "yes!") == combined
    assert original_request_for_project(session, "hotel") == combined
    from agent_config import MAX_USER_MESSAGE_CHARS
    from utilities.message_limits import UserMessageTooLong
    maximum = "x" * MAX_USER_MESSAGE_CHARS
    remember_original_request(session, maximum, "hotel")
    with pytest.raises(UserMessageTooLong, match="combined original specification"):
        generation._original_for_smart_generation(session, "Also enforce phone validation.")
    assert original_request_for_project(session, "hotel") == maximum

    # Exercise the real modeling dispatch after confirmation / mismatch resume,
    # where request.message may have been reconstructed from a planner summary.
    from unittest.mock import MagicMock
    import execution.model_operations as modeling
    from protocol.types import AssistantRequest, WorkspaceContext
    from session_keys import MISMATCH_REGEN_PENDING
    from tests.conftest import FakeSession
    handler = MagicMock()
    handler.generate_complete_system.return_value = {"action": "assistant_message", "message": "captured"}
    factory = MagicMock()
    factory.get_handler.return_value = handler
    monkeypatch.setattr(modeling.ctx, "diagram_factory", factory)
    monkeypatch.setattr(modeling, "reply_progress", lambda *args: None)
    monkeypatch.setattr(modeling, "reply_message", lambda *args: None)
    monkeypatch.setattr(modeling, "resolve_target_model", lambda *args: None)
    monkeypatch.setattr(modeling, "build_workspace_context_block", lambda *args: "")
    for project_id, confirming, expected in (
        ("hotel", True, FULL_SPEC),
        ("hotel", False, FULL_SPEC),
        ("other", True, PLANNER_REWRITE),
        (None, True, PLANNER_REWRITE),
    ):
        resumed = FakeSession()
        remember_original_request(resumed, FULL_SPEC, "hotel")
        if not confirming:
            resumed.set(MISMATCH_REGEN_PENDING, PLANNER_REWRITE)
        request = AssistantRequest(
            message=PLANNER_REWRITE if project_id == "hotel" else "replace",
            context=WorkspaceContext(project_snapshot={"id": project_id, "diagrams": {}}),
        )
        modeling.execute_model_operation(
            resumed, request,
            {"diagramType": "ClassDiagram", "mode": "complete_system", "request": PLANNER_REWRITE},
            "complete_system", _skip_existing_check=confirming,
        )
        call = handler.generate_complete_system.call_args
        assert call.kwargs["raw_request"] == expected
        assert expected in call.args[0]
        if expected != FULL_SPEC:
            assert FULL_SPEC not in call.args[0]


# ----------------------------------------------------------------------
# The summary must not invent requirements
#
# Run 0ceb8611's instructions, in full, for a spec that never mentions
# users or login: "Generate a web app for a hotel booking and stay
# management system with screens for Dashboard, Bookings, Booking Detail,
# Guests, Rooms, and more. Include user authentication and a responsive
# design." The run then built auth. The field description had asked for
# auth twice by example — "Devise auth" and "JWT".
# ----------------------------------------------------------------------


def _refined_instructions_description() -> str:
    return GenerationClassification.model_fields["refined_instructions"].description


def test_field_description_no_longer_uses_auth_as_an_example():
    desc = _refined_instructions_description()
    assert "Devise auth" not in desc
    # JWT may only appear inside the prohibition, never as a sample to copy.
    assert "(JWT, Docker, migrations, tests)" not in desc


def test_field_description_forbids_inventing_the_usual_web_app_extras():
    low = _refined_instructions_description().lower()
    assert "invent nothing" in low
    for banned in ("authentication", "login", "roles", "jwt",
                   "responsive design", "styling", "navigation"):
        assert banned in low, f"{banned!r} is not named as something to not invent"


def test_field_description_asks_to_preserve_concrete_nouns():
    """The other half of the failure: 4,622 characters became 197."""
    low = _refined_instructions_description().lower()
    assert "status values" in low
    assert "named operations" in low
    assert "stated rules" in low


def test_the_summary_is_labelled_as_notes_that_cannot_add_anything():
    """Live (2026-09-23): the summary of a hotel spec that asks for SQLite and
    no login said "PostgreSQL ... login/signup, role-based access ... Docker".
    It stays (it can carry a stack the user accepted with a plain "yes"), but
    as notes the coding model may not add features or override the spec from."""
    instructions = _payload(original_request=SPEC)["instructions"]
    notes = instructions.index("## Assistant's notes")
    assert notes < instructions.index(SUMMARY) < instructions.index("original request, verbatim")
    header = instructions[notes:instructions.index(SUMMARY)].lower()
    assert "may be inaccurate" in header
    assert "never add" in header and "user's request wins" in header
