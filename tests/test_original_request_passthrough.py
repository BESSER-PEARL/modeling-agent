"""The run must receive the user's spec, not only a summary of it.

``trigger_smart_generator`` used to carry only ``refined_instructions`` —
by its own schema "1-3 short paragraphs" written by the classifier. The
downstream gap analyser diffs the request against the model to recover
requirements the modelling step lost, so it was diffing against a summary:
status vocabularies and business rules were compressed away, and assistant
flow wording the summary had absorbed was attributed to the user.
"""
from __future__ import annotations

import pytest

from codegen.smart_generation_handler import (
    GenerationClassification,
    build_trigger_smart_generator_payload,
)
from prompts.classifier import SYSTEM_PROMPT
from prompts.spec_driven import with_later_request
from routing.unified_classifier import UnifiedClassification
from common import pending_dialogs

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
    from common.agent_config import MAX_USER_MESSAGE_CHARS
    from protocol.message_limits import UserMessageTooLong
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
# Stashing ``operation_request`` is not enough: the planner has already
# rewritten a long spec into one line by then.
# ----------------------------------------------------------------------

from execution.model_operations import original_request_to_stash


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
    from codegen import generation_handler as generation
    from common.pending_dialogs import PENDING_SMART_GEN_ORIGINAL_REQUEST
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
    assert pending_dialogs.get(session, PENDING_SMART_GEN_ORIGINAL_REQUEST) == FULL_SPEC

    # A new/imported project cannot inherit the previous project's authority.
    session.project_id = "imported-library"
    generation._build_smart_gen_confirmation(session, "Generate this library", "openai", user_message="generate application")
    assert pending_dialogs.get(session, PENDING_SMART_GEN_ORIGINAL_REQUEST) == ""
    direct = "Generate this imported library. Anonymous users can read, but only owners can edit."
    generation._build_smart_gen_confirmation(session, "Make a library app", "openai", user_message=direct)
    assert pending_dialogs.get(session, PENDING_SMART_GEN_ORIGINAL_REQUEST) == direct
    assert original_request_for_project(session, "imported-library") == direct
    assert FULL_SPEC not in pending_dialogs.get(session, PENDING_SMART_GEN_ORIGINAL_REQUEST)

    # A later global update must not mutate the already-confirmable request.
    remember_original_request(session, "An unrelated new request", "imported-library")
    assert pending_dialogs.get(session, PENDING_SMART_GEN_ORIGINAL_REQUEST) == direct
    assert generation._pending_smart_gen_project_changed(session) is False
    session.project_id = "another-project"
    assert generation._pending_smart_gen_project_changed(session) is True


def test_same_project_create_gui_confirm_keeps_raw_spec_and_followups(monkeypatch):
    from codegen import generation_handler as generation
    from common.pending_dialogs import PENDING_SMART_GEN_ORIGINAL_REQUEST
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
    assert pending_dialogs.get(session, PENDING_SMART_GEN_ORIGINAL_REQUEST) == FULL_SPEC
    change = "Use SQLite, FastAPI and React."
    combined = generation._original_for_smart_generation(session, change)
    assert FULL_SPEC in combined and change in combined
    assert generation._original_for_smart_generation(session, change) == combined
    assert generation._original_for_smart_generation(session, "yes") == combined
    assert generation._original_for_smart_generation(session, "yes!") == combined
    assert original_request_for_project(session, "hotel") == combined
    from common.agent_config import MAX_USER_MESSAGE_CHARS
    from protocol.message_limits import UserMessageTooLong
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
    from common.pending_dialogs import MISMATCH_REGEN_PENDING
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
            pending_dialogs.put(resumed, MISMATCH_REGEN_PENDING, PLANNER_REWRITE)
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
# For a spec that never mentions users or login, the summary read "...
# Include user authentication and a responsive design." and the run built
# auth: the field description had asked for auth by example ("Devise auth",
# "JWT").
# ----------------------------------------------------------------------


def _refined_instructions_description() -> str:
    # The classifier's schema is the one the model reads; GenerationClassification
    # is only built in code (generation_handler.py) and never sent to a model.
    return UnifiedClassification.model_fields["refined_instructions"].description


class _DictSession:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value


_STORED_SPEC = "An order has a status: open, paid or shipped. A customer places many orders."


@pytest.mark.parametrize("command", [
    "generate the web app",
    "Generate the web app.",
    "generate the app",
    "generate my app",
    "generate my web app",
    "generate the web application",
    "generate the webapp",
    "please generate the web app",
    "Generate the web app, please!",
    "can you generate the app?",
    "generate the web app now",
    # Phrasings that were already treated as bare commands.
    "generate web app",
    "generate it",
    "build the app",
    "create the app",
    "generate anyway with my current model",
    # The confirmation chip's sentence.
    "continue with the generation",
    "Continue with the generation.",
])
def test_bare_generate_command_is_not_carried_as_a_requirement(monkeypatch, command):
    """A command that only says "generate it" adds nothing to the user's spec."""
    from codegen import generation_handler as generation
    from utilities.original_request import remember_original_request, original_request_for_project

    session = _DictSession()
    monkeypatch.setattr(generation, "_active_project_id", lambda s: "shop")
    remember_original_request(session, _STORED_SPEC, "shop")

    assert generation._original_for_smart_generation(session, command) == _STORED_SPEC
    assert original_request_for_project(session, "shop") == _STORED_SPEC


@pytest.mark.parametrize("command", [
    "continue with the generation",
    "generate anyway with my current model",
])
def test_confirmation_sentence_is_never_stored_as_the_request(monkeypatch, command):
    from codegen import generation_handler as generation
    from utilities.original_request import original_request_for_project

    session = _DictSession()
    monkeypatch.setattr(generation, "_active_project_id", lambda s: "shop")

    assert generation._original_for_smart_generation(session, command) == ""
    assert original_request_for_project(session, "shop") == ""


@pytest.mark.parametrize("request_text", [
    "generate the web app with a dark theme",
    "generate the app in French",
    "generate the web app and add a login page",
    "build the app for mobile",
    "generate the invoice report",
])
def test_generate_request_with_content_is_carried(monkeypatch, request_text):
    from codegen import generation_handler as generation
    from utilities.original_request import remember_original_request

    session = _DictSession()
    monkeypatch.setattr(generation, "_active_project_id", lambda s: "shop")
    remember_original_request(session, _STORED_SPEC, "shop")

    combined = generation._original_for_smart_generation(session, request_text)
    assert combined.startswith(_STORED_SPEC)
    assert combined.endswith(request_text)
    assert "Subsequent user request, verbatim" in combined


def test_the_routing_rulebook_does_not_use_auth_as_an_example():
    assert "Devise auth" not in SYSTEM_PROMPT
    assert "Add user authentication" not in SYSTEM_PROMPT


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
    """The summary of a hotel spec that asks for SQLite and
    no login said "PostgreSQL ... login/signup, role-based access ... Docker".
    It stays (it can carry a stack the user accepted with a plain "yes"), but
    as notes the coding model may not add features or override the spec from."""
    instructions = _payload(original_request=SPEC)["instructions"]
    notes = instructions.index("## Assistant's notes")
    assert notes < instructions.index(SUMMARY) < instructions.index("original request, verbatim")
    header = instructions[notes:instructions.index(SUMMARY)].lower()
    assert "may be inaccurate" in header
    assert "never add" in header and "user's request wins" in header


# ---------------------------------------------------------------------
# A content-free follow-up ("regenerate with fixes") re-runs the earlier
# request; it is never the request itself.
# ---------------------------------------------------------------------

_FOLLOW_UPS = [
    # The shapes named in the contract.
    "regenerate with fixes.",
    "Regenerate with fixes",
    "rebuild",
    "redo it",
    "retry",
    "try again",
    "run it again",
    "generate again",
    "regenerate the app",
    "regenerate it and fix the errors",
    "rebuild the app and fix the issues",
    "regenerate with fixes please",
    "fix it",
    "fix the errors",
    "fix the remaining issues",
    "fix the remaining blockers",
    "Can you regenerate the application's frontend and the backend?",
    # More of the same shape.
    "please regenerate",
    "please try again",
    "try it again",
    "could you try again?",
    "ok, regenerate it",
    "regenerate the application",
    "re-generate the app",
    "rerun the generation",
    "re-run it",
    "retry the build",
    "redo the generation",
    "rebuild the project",
    "generate it again",
    "generate the app again",
    "build it again",
    "do it again",
    "regenerate both the frontend and the backend",
    "rebuild the backend and the frontend",
    "regenerate the code with the fixes",
    "regenerate and fix the problems",
    "fix all the errors",
    "fix them",
    "please fix the bugs",
    "fix the errors and regenerate",
    "retry with fixes",
    # The whole-app fix and repeat forms.
    "fix the app",
    "fix my app",
    "can you fix the code?",
    "fix it again",
    "try once more",
    "try it one more time",
]

_CONTENT_REQUESTS = [
    # The shapes named in the contract.
    "regenerate with a dark theme",
    "fix the login so users can reset their password",
    "rebuild it with PostgreSQL",
    # More that name what to change.
    "regenerate the app in French",
    "regenerate the backend in Go",
    "rebuild it as a mobile app",
    "regenerate with authentication",
    "try again but use React",
    "fix the date format on the orders page",
    "fix the errors in the checkout page",
    "redo the Book entity",
    "regenerate the app with a search bar",
    "retry with Django instead",
    "rebuild the frontend with Vue",
    "fix the price validation",
    "run it again with Docker",
    "generate again and add a customer list",
    "regenerate with fixes and add pagination",
    "fix it so tasks can be archived",
    "regenerate the app for an online shop",
    "rebuild the project using MySQL",
    "try again with a smaller model",
    "fix the remaining issues in the invoice form",
    "redo it in TypeScript",
    "regenerate the orders module",
    "fix the broken links in the menu",
    "please regenerate the app and include an admin page",
    "rebuild it with two roles: author and reader",
    "create an order tracking app",
    # One part of the app names what to rebuild.
    "regenerate the backend",
    "regenerate the frontend",
    "regenerate the ui",
    "regenerate the api",
    "regenerate the app's backend",
    "fix the frontend",
]


@pytest.mark.parametrize("message", _FOLLOW_UPS)
def test_content_free_follow_up_is_recognised(message):
    from codegen import generation_handler as generation

    assert generation._is_content_free_follow_up(message)


@pytest.mark.parametrize("message", _CONTENT_REQUESTS)
def test_request_with_content_is_not_a_content_free_follow_up(message):
    from codegen import generation_handler as generation

    assert not generation._is_content_free_follow_up(message)


def test_the_bare_generate_matcher_is_unchanged():
    from codegen import generation_handler as generation

    assert generation._BARE_GENERATE_RE.pattern == (
        r"(?:(?:can|could) you )?(?:please )?"
        r"(?:generate|build|create) "
        r"(?:it|(?:(?:the|my|this|a|an) )?(?:web ?)?(?:app|application|code))"
        r"(?: now)?(?: please)?"
    )


_ORDER_SPEC = "An order tracking app: customers place orders; an order is open, paid or shipped."
_LAST_RUN_SPEC = "A task board: tasks have a title, a due date and a done flag."


def _follow_up_session(message, *, last_generation=None, history=(), github_repo=None):
    import uuid

    from memory import get_memory, memory_session_key
    from tests.conftest import FakeSession

    context = {"activeDiagramType": "ClassDiagram", "projectSnapshot": {"id": "proj-1", "diagrams": {}}}
    if last_generation is not None:
        context["lastGeneration"] = last_generation
    if github_repo is not None:
        context["githubRepo"] = github_repo
    session = FakeSession({
        "action": "user_message", "protocolVersion": "2.0", "message": message,
        "context": context, "sessionId": uuid.uuid4().hex,
    })
    memory = get_memory(memory_session_key(session))
    for role, text in history:
        (memory.add_user if role == "user" else memory.add_assistant)(text)
    memory.add_user(message)  # recorded before the generation handler runs
    return session


def test_follow_up_uses_the_remembered_request_first():
    from codegen import generation_handler as generation
    from utilities.original_request import original_request_for_project, remember_original_request

    session = _follow_up_session(
        "regenerate with fixes.", last_generation={"request": _LAST_RUN_SPEC},
        history=[("user", "a recipe app with ingredients and steps")],
    )
    remember_original_request(session, _ORDER_SPEC, "proj-1")

    original = generation._original_for_smart_generation(session, "regenerate with fixes.")

    assert original == with_later_request(_ORDER_SPEC, "regenerate with fixes.")
    assert original_request_for_project(session, "proj-1") == _ORDER_SPEC


@pytest.mark.parametrize("message", ["try again", "fix the errors", "regenerate the app"])
def test_a_content_free_follow_up_is_the_newest_message_but_not_remembered(message):
    """The harness read the whole earlier request as the newest message: a retry
    sent nothing of its own, and a fix sent a note under a heading it does not parse."""
    from codegen import generation_handler as generation
    from utilities.original_request import original_request_for_project, remember_original_request

    session = _follow_up_session(message)
    remember_original_request(session, _ORDER_SPEC, "proj-1")

    assert generation._original_for_smart_generation(session, message) == with_later_request(_ORDER_SPEC, message)
    assert original_request_for_project(session, "proj-1") == _ORDER_SPEC


def test_follow_up_uses_the_editors_last_run_request_next():
    from codegen import generation_handler as generation
    from utilities.original_request import original_request_for_project

    session = _follow_up_session(
        "regenerate with fixes.", last_generation={"request": _LAST_RUN_SPEC, "expired": True},
        history=[("user", "a recipe app with ingredients and steps")],
    )

    original = generation._original_for_smart_generation(session, "regenerate with fixes.")

    assert original.startswith(_LAST_RUN_SPEC)
    assert "recipe" not in original
    assert original_request_for_project(session, "proj-1") == ""


def test_follow_up_never_rebuilds_the_request_from_the_conversation():
    # The chat may hold another project's app, a question, an edit or an app the
    # user started over from; none of it is reliably this project's request.
    from codegen import generation_handler as generation
    from utilities.original_request import original_request_for_project

    session = _follow_up_session("regenerate with fixes.", history=[
        ("user", _ORDER_SPEC),
        ("assistant", "Created the class diagram with Order and Customer."),
        ("user", "why did the run fail?"),
        ("user", "add a delivery date to orders"),
        ("user", "Start over: a task board where tasks have a title and a due date."),
    ])

    assert generation._original_for_smart_generation(session, "regenerate with fixes.") is None
    assert original_request_for_project(session, "proj-1") == ""


def test_follow_up_with_nothing_to_go_on_finds_no_request():
    from codegen import generation_handler as generation
    from utilities.original_request import original_request_for_project

    session = _follow_up_session("regenerate with fixes.", history=[("user", "yes")])

    assert generation._original_for_smart_generation(session, "regenerate with fixes.") is None
    assert original_request_for_project(session, "proj-1") == ""


def test_a_follow_up_with_content_is_still_carried():
    from codegen import generation_handler as generation
    from utilities.original_request import remember_original_request

    session = _follow_up_session("regenerate with a dark theme")
    remember_original_request(session, _ORDER_SPEC, "proj-1")

    original = generation._original_for_smart_generation(session, "regenerate with a dark theme")
    assert original.startswith(_ORDER_SPEC)
    assert original.endswith("regenerate with a dark theme")


@pytest.mark.parametrize("message", ["generate the app", "continue with the generation", ""])
def test_a_command_without_content_runs_the_editors_last_request(message):
    from codegen import generation_handler as generation

    session = _follow_up_session(message, last_generation={"request": _LAST_RUN_SPEC, "expired": True})

    assert generation._original_for_smart_generation(session, message) == _LAST_RUN_SPEC


def test_a_follow_up_with_content_extends_the_editors_last_request():
    from codegen import generation_handler as generation
    from utilities.original_request import original_request_for_project

    message = "regenerate with a dark theme"
    session = _follow_up_session(message, last_generation={"request": _LAST_RUN_SPEC, "expired": True})

    original = generation._original_for_smart_generation(session, message)

    assert original.startswith(_LAST_RUN_SPEC)
    assert original.endswith(message)
    assert "## Subsequent user request, verbatim" in original
    assert original_request_for_project(session, "proj-1") == original


def test_the_remembered_request_wins_over_the_editors_last_request():
    from codegen import generation_handler as generation
    from utilities.original_request import remember_original_request

    session = _follow_up_session("generate the app", last_generation={"request": _LAST_RUN_SPEC})
    remember_original_request(session, _ORDER_SPEC, "proj-1")

    assert generation._original_for_smart_generation(session, "generate the app") == _ORDER_SPEC


def test_a_replacing_request_ignores_the_editors_last_request():
    from codegen import generation_handler as generation

    message = "a recipe app with ingredients and steps"
    session = _follow_up_session(message, last_generation={"request": _LAST_RUN_SPEC})

    assert generation._original_for_smart_generation(session, message, replace=True) == message


class _FollowUpProvider:
    def parse(self, *, messages, schema, temperature, max_tokens):
        from routing.unified_classifier import UnifiedClassification

        return UnifiedClassification(
            intent="generation_intent", generation_route="smart",
            refined_instructions="Regenerate the application.", provider="openai", reason="test",
        )


def _generation_turn(session, message, monkeypatch):
    from codegen import generation_handler as generation
    from protocol.types import AssistantRequest

    monkeypatch.setattr(generation, "_get_llm_provider", lambda: _FollowUpProvider())
    return generation.handle_generation_request(session, AssistantRequest(message=message))


def test_the_run_gets_the_original_request_not_the_follow_up(monkeypatch):
    from utilities.original_request import remember_original_request

    session = _follow_up_session("regenerate with fixes.")
    remember_original_request(session, _ORDER_SPEC, "proj-1")

    asked = _generation_turn(session, "regenerate with fixes.", monkeypatch)
    assert asked["action"] == "assistant_message"
    run = _generation_turn(session, "continue with the generation", monkeypatch)

    assert run["action"] == "trigger_smart_generator"
    original = run["instructions"].split("## The user's original request, verbatim\n\n", 1)[1]
    assert _ORDER_SPEC in original
    assert original.endswith("regenerate with fixes.")


_CONTINUE_SUMMARY = (
    "Continue development of the existing task board app, fixing the issues left by the previous run."
)
_STARTING_POINT = (
    "## Starting point\n\n"
    "The previous result is no longer available, so there is no earlier code to continue or fix. "
    "Build the whole application from the current model and the requests below; where they ask to "
    "continue or fix earlier work, build that work as part of the new application."
)
_CONTINUE_MESSAGE = "can you continue with the development?"


class _ContinueProvider(_FollowUpProvider):
    def __init__(self, summary=_CONTINUE_SUMMARY):
        self.summary = summary

    def parse(self, *, messages, schema, temperature, max_tokens):
        return UnifiedClassification(
            intent="generation_intent", generation_route="smart",
            refined_instructions=self.summary, provider="openai", reason="test",
        )


def _confirm(session):
    from codegen import generation_handler as generation
    from protocol.types import AssistantRequest

    run = generation.handle_generation_request(session, AssistantRequest(message="continue with the generation"))
    assert run["action"] == "trigger_smart_generator"
    return run


def _confirmed_run(last_generation, monkeypatch, github_repo=None, message=_CONTINUE_MESSAGE,
                   summary=_CONTINUE_SUMMARY):
    from codegen import generation_handler as generation
    from protocol.types import AssistantRequest

    session = _follow_up_session(message, last_generation=last_generation, github_repo=github_repo)
    monkeypatch.setattr(generation, "_get_llm_provider", lambda: _ContinueProvider(summary))
    asked = generation.handle_generation_request(session, AssistantRequest(message=message))
    assert asked["action"] == "assistant_message"
    return _confirm(session)


def _assert_rebuilt(instructions, summary=_CONTINUE_SUMMARY):
    # The starting point comes before the user's request, which ends with their newest message.
    notes, requests = instructions.split("## The user's original request, verbatim\n\n", 1)
    assert f"{summary}\n\n" in notes
    assert notes.endswith(f"{_STARTING_POINT}\n\n")
    assert "no longer available" not in requests


def test_an_expired_result_is_rebuilt_not_continued(monkeypatch):
    run = _confirmed_run({"request": _LAST_RUN_SPEC, "expired": True}, monkeypatch)

    _assert_rebuilt(run["instructions"])
    assert _LAST_RUN_SPEC in run["instructions"]


def test_a_live_result_keeps_the_summary(monkeypatch):
    run = _confirmed_run({"request": _LAST_RUN_SPEC}, monkeypatch)

    assert _CONTINUE_SUMMARY in run["instructions"]
    assert "## Starting point" not in run["instructions"]


@pytest.mark.parametrize("has_code", [True, False])
def test_a_linked_repo_is_never_called_gone(monkeypatch, has_code):
    # The editor sends hasCode false when it does not know, and may still continue from the repo.
    run = _confirmed_run(
        {"request": _LAST_RUN_SPEC, "expired": True}, monkeypatch,
        github_repo={"fullName": "acme/tasks", "hasCode": has_code},
    )

    assert _CONTINUE_SUMMARY in run["instructions"]
    assert "## Starting point" not in run["instructions"]


def test_a_rebuilt_model_resume_starts_from_the_model():
    from codegen import generation_handler as generation

    session = _follow_up_session(_CONTINUE_MESSAGE, last_generation={"request": _LAST_RUN_SPEC, "expired": True})
    generation._build_smart_gen_confirmation(
        session, _CONTINUE_SUMMARY, "openai", reason_prefix="Your model is ready.", user_message=_CONTINUE_MESSAGE,
    )

    _assert_rebuilt(_confirm(session)["instructions"])


def test_generate_anyway_after_a_mismatch_starts_from_the_model():
    from codegen import generation_handler as generation
    from protocol.types import AssistantRequest

    session = _follow_up_session(_CONTINUE_MESSAGE, last_generation={"request": _LAST_RUN_SPEC, "expired": True})
    classification = GenerationClassification(
        route="smart", refined_instructions=_CONTINUE_SUMMARY, provider="openai", reason="test",
    )
    generation._build_mismatch_confirmation(session, classification, "a recipe app", user_message=_CONTINUE_MESSAGE)
    run = generation.handle_generation_request(
        session, AssistantRequest(message="generate anyway with my current model"),
    )

    assert run["action"] == "trigger_smart_generator"
    _assert_rebuilt(run["instructions"])


@pytest.mark.parametrize("summary", [
    "Add a dark theme to the existing app. Use PostgreSQL and a Django backend.",
    "Continue development of the existing app by adding PostgreSQL and a Django backend.",
    "- Add a dark theme to the existing app\n- Use PostgreSQL and a Django backend",
])
def test_a_summary_accepted_with_yes_reaches_the_rebuild_whole(monkeypatch, summary):
    # After a plain "yes" the summary is the only place the accepted choices live.
    run = _confirmed_run(
        {"request": _LAST_RUN_SPEC, "expired": True}, monkeypatch, message="yes", summary=summary,
    )

    _assert_rebuilt(run["instructions"], summary)


def test_without_a_request_the_run_starts_from_the_model():
    payload = build_trigger_smart_generator_payload(
        GenerationClassification(route="smart", refined_instructions=_CONTINUE_SUMMARY, provider="openai", reason="t"),
        previous_result_gone=True,
    )

    assert payload["instructions"] == (
        "## Starting point\n\n"
        "The previous result is no longer available, so there is no earlier code to continue or fix. "
        f"Build the whole application from the current model.\n\n{_CONTINUE_SUMMARY}"
    )


def test_nothing_to_go_on_asks_again_and_arms_nothing(monkeypatch):
    from common.pending_dialogs import PENDING_SMART_GEN_INSTRUCTIONS

    session = _follow_up_session("regenerate with fixes.")
    pending_dialogs.put(session, PENDING_SMART_GEN_INSTRUCTIONS, "An older prepared run.")

    reply = _generation_turn(session, "regenerate with fixes.", monkeypatch)

    assert reply == {
        "action": "assistant_message",
        "message": (
            "I no longer have your earlier request here, so I can't tell what to generate again. "
            "Please describe again what the application should do."
        ),
    }
    assert pending_dialogs.get(session, PENDING_SMART_GEN_INSTRUCTIONS) is None
    assert _generation_turn(session, "continue with the generation", monkeypatch)["action"] != (
        "trigger_smart_generator")


@pytest.mark.parametrize("last_generation, github_repo", [
    ({"request": _LAST_RUN_SPEC, "expired": True}, None),
    ({"request": _LAST_RUN_SPEC, "expired": False}, None),
    ({"request": _LAST_RUN_SPEC}, None),
    ({"request": _LAST_RUN_SPEC, "expired": True}, {"fullName": "acme/tasks", "hasCode": False}),
])
def test_a_fix_request_is_sent_in_the_users_words_whatever_the_previous_result(last_generation, github_repo):
    from codegen import generation_handler as generation

    session = _follow_up_session("regenerate with fixes.", last_generation=last_generation, github_repo=github_repo)

    original = generation._original_for_smart_generation(session, "regenerate with fixes.")
    assert original == with_later_request(_LAST_RUN_SPEC, "regenerate with fixes.")


@pytest.mark.parametrize("message", ["regenerate the backend", "regenerate the ui"])
def test_a_part_scoped_redo_keeps_its_scope(message):
    from codegen import generation_handler as generation
    from utilities.original_request import remember_original_request

    session = _follow_up_session(message)
    remember_original_request(session, _ORDER_SPEC, "proj-1")

    original = generation._original_for_smart_generation(session, message)
    assert original.startswith(_ORDER_SPEC)
    assert original.endswith(message)
