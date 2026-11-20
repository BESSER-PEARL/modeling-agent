"""A follow-up reaches the Spec-Driven Agent with the user's newest message
identified exactly, and the assistant's decisions travel as fields.

The harness reads the newest message from the instructions' last
``LATER_REQUEST_MARKER`` section (BESSER
``besser/spec_driven_agent/planning/requirements_ledger.py``). Before, a retry
sent nothing of its own and a fix sent a note under a heading the harness does
not parse, so the whole earlier request was read as the newest message, and a
route named in it became the fix run's target. The contract tests run against
BESSER when it is importable and are skipped otherwise.
"""

import uuid

import pytest

from codegen import generation_handler as generation
from codegen.smart_generation_handler import GenerationClassification
from common.session_keys import UNIFIED_CLASSIFICATION
from prompts import spec_driven
from protocol.types import AssistantRequest
from routing.unified_classifier import UnifiedClassification
from tests.conftest import FakeSession
from utilities.original_request import original_request_for_project, remember_original_request

SPEC = "A library app. GET /books returns 404 when a book is missing. Members borrow books."
EARLIER = "Add a due date to loans."


def _session(message, *, last_generation=None, repo=None):
    context = {"activeDiagramType": "ClassDiagram", "projectSnapshot": {"id": "proj-1", "diagrams": {}}}
    if last_generation is not None:
        context["lastGeneration"] = last_generation
    if repo is not None:
        context["githubRepo"] = repo
    return FakeSession({"action": "user_message", "protocolVersion": "2.0", "message": message,
                        "context": context, "sessionId": uuid.uuid4().hex})


class _Provider:
    def __init__(self, **verdict):
        self.verdict = verdict

    def parse(self, *, messages, schema, temperature, max_tokens):
        return UnifiedClassification(intent="generation_intent", generation_route="smart",
                                     refined_instructions="Update the library app.", provider="openai",
                                     reason="test", **self.verdict)


def _run(message, monkeypatch, *, last_generation=None, previous=SPEC, **verdict):
    """The trigger the assistant sends for ``message`` once the user confirms."""
    session = _session(message, last_generation=last_generation)
    remember_original_request(session, previous, "proj-1")
    provider = _Provider(**verdict)
    monkeypatch.setattr(generation, "_get_llm_provider", lambda: provider)
    session.set(UNIFIED_CLASSIFICATION, provider.parse(messages=[], schema=None, temperature=0, max_tokens=0))
    asked = generation.handle_generation_request(session, AssistantRequest(message=message))
    assert asked["action"] == "assistant_message"
    run = generation.handle_generation_request(session, AssistantRequest(message="continue with the generation"))
    assert run["action"] == "trigger_smart_generator"
    return session, run


@pytest.fixture
def ledger():
    module = pytest.importorskip("besser.spec_driven_agent.planning.requirements_ledger")
    if not hasattr(module, "later_messages"):
        pytest.skip("the installed BESSER predates the shared follow-up headings")
    return module


def test_both_repos_use_the_same_headings(ledger):
    assert spec_driven.VERBATIM_MARKER == ledger.VERBATIM_MARKER
    assert spec_driven.LATER_REQUEST_MARKER == ledger.LATER_REQUEST_MARKER


@pytest.mark.parametrize("message", [
    "add a dark theme",          # plain follow-up
    "try again",                 # retry
    "fix it",                    # fix
    "regenerate the app",        # regenerate
])
@pytest.mark.parametrize("previous", [SPEC, spec_driven.with_later_request(SPEC, EARLIER)])
@pytest.mark.parametrize("last_generation", [{"runId": "r1", "expired": False}, {"request": SPEC, "expired": True}])
def test_the_harness_reads_the_users_newest_message(ledger, monkeypatch, message, previous, last_generation):
    _, run = _run(message, monkeypatch, previous=previous, last_generation=last_generation)

    assert ledger.latest_user_message(run["instructions"]) == message
    assert ledger.later_messages(previous) == ledger.later_messages(run["instructions"])[:-1]


def test_a_fix_request_takes_no_route_from_the_earlier_request(ledger, monkeypatch):
    fix_target = pytest.importorskip("besser.spec_driven_agent.planning.fix_target")
    _, run = _run("fix it", monkeypatch)

    target = fix_target.parse_reported_target(run["instructions"])
    assert "/books" not in (getattr(target, "descriptor", "") or "")


def test_continue_sends_the_earlier_request_unchanged(monkeypatch):
    # A bare command states nothing new: the harness continues earlier work without a call.
    _, run = _run("continue", monkeypatch)

    assert run["instructions"].endswith(f"{spec_driven.VERBATIM_MARKER}\n\n"
                                        "This is the authority. Where the summary above is shorter or "
                                        f"differs, this text wins.\n\n{SPEC}")


@pytest.mark.parametrize("message", ["try again", "can you continue with the development?"])
def test_a_message_that_states_nothing_new_is_sent_but_not_remembered(monkeypatch, message):
    """Retries and continues piled up in the remembered request as permanent
    'takes precedence' requirements of every later run."""
    session, run = _run(message, monkeypatch, states_requirements=False)

    assert run["instructions"].endswith(message)
    assert original_request_for_project(session, "proj-1") == SPEC


@pytest.mark.parametrize("message, verdict, remembered", [
    ("try again", {"states_requirements": False}, False),
    ("can you continue with the development?", {"states_requirements": False}, False),
    ("add a dark theme", {}, True),
    ("continue", {"states_requirements": False}, True),  # the earlier request, sent unchanged
])
def test_the_trigger_says_when_the_newest_message_is_not_part_of_the_request(monkeypatch, message, verdict,
                                                                            remembered):
    """The editor kept a retry the assistant did not remember as part of the project's
    request, and sent it back as the request once the assistant had lost its memory."""
    _, run = _run(message, monkeypatch, **verdict)

    assert run.get("remember_newest", True) is remembered


def test_a_message_that_states_something_is_remembered(monkeypatch):
    session, _ = _run("add a dark theme", monkeypatch)

    assert original_request_for_project(session, "proj-1") == spec_driven.with_later_request(SPEC, "add a dark theme")


# ---------------------------------------------------------------------
# Start over: the user's decision is the run's mode on every path.
# ---------------------------------------------------------------------

_EDITABLE = {"runId": "run-abc", "expired": False, "request": "a library app"}


def test_starting_over_rebuilds_the_app_without_a_linked_repo(monkeypatch):
    """Without a linked repo the trigger carried no mode, and the editor turned it
    into an edit of the stored run."""
    session = _session("scrap the app and build it again from scratch", last_generation=_EDITABLE)
    session.set(UNIFIED_CLASSIFICATION, UnifiedClassification(
        intent="generation_intent", generation_route="smart", refined_instructions="Rebuild the app.",
        start_over=True, reason="test"))

    asked = generation._build_smart_gen_confirmation(
        session, "Rebuild the app.", "openai", user_message="scrap the app and build it again from scratch")

    assert asked["suggestedActions"][0]["label"] == "Rebuild"
    assert "will be dropped" in asked["message"] and "update your existing app" not in asked["message"]
    run = generation.handle_generation_request(session, AssistantRequest(message="continue with the generation"))
    assert run["action"] == "trigger_smart_generator" and run["mode"] == "generate"


def test_starting_over_after_a_domain_mismatch_keeps_the_mode():
    session = _session("start over as a recipe app", last_generation=_EDITABLE)
    session.set(UNIFIED_CLASSIFICATION, UnifiedClassification(
        intent="generation_intent", generation_route="smart", refined_instructions="A recipe app.",
        start_over=True, reason="test"))
    classification = GenerationClassification(
        route="smart", refined_instructions="A recipe app.", provider="openai", reason="test")
    generation._build_mismatch_confirmation(session, classification, "a recipe app",
                                            user_message="start over as a recipe app")

    run = generation.handle_generation_request(
        session, AssistantRequest(message="generate anyway with my current model"))

    assert run["action"] == "trigger_smart_generator" and run["mode"] == "generate"


def _generate_anyway_text(asked):
    return asked["message"].split("**Generate anyway**", 1)[1].split("•", 1)[0]


@pytest.mark.parametrize("start_over, last_generation, repo, dropped", [
    (True, _EDITABLE, None, "earlier runs will be dropped"),
    (True, {"request": "a library app", "expired": False},
     {"fullName": "acme/library", "hasCode": True}, "will replace the code in **acme/library**"),
    (False, _EDITABLE, None, None),
    (True, {"request": "a library app", "expired": True}, None, None),
])
def test_generate_anyway_says_when_it_drops_earlier_changes(start_over, last_generation, repo, dropped):
    """'Generate anyway' fires the stashed start-over run at once: it is the only
    confirmation, so it must say what the rebuild discards."""
    session = _session("start over as a recipe app", last_generation=last_generation, repo=repo)
    session.set(UNIFIED_CLASSIFICATION, UnifiedClassification(
        intent="generation_intent", generation_route="smart", refined_instructions="A recipe app.",
        start_over=start_over, reason="test"))
    classification = GenerationClassification(
        route="smart", refined_instructions="A recipe app.", provider="openai", reason="test")

    asked = generation._build_mismatch_confirmation(session, classification, "a recipe app",
                                                    user_message="start over as a recipe app")

    text = _generate_anyway_text(asked)
    if dropped:
        assert dropped in text
    else:
        assert "dropped" not in text and "replace the code" not in text


def test_a_change_to_the_app_leaves_the_mode_to_the_editor():
    session = _session("add a search bar", last_generation=_EDITABLE)
    session.set(UNIFIED_CLASSIFICATION, UnifiedClassification(
        intent="generation_intent", generation_route="smart", refined_instructions="Add a search bar.",
        reason="test"))

    asked = generation._build_smart_gen_confirmation(session, "Add a search bar.", "openai",
                                                     user_message="add a search bar")

    assert asked["suggestedActions"][0]["label"] == "Fix it"
    run = generation.handle_generation_request(session, AssistantRequest(message="continue with the generation"))
    assert "mode" not in run
