"""Coming back to a project linked to GitHub, days after the last run.

The agent's only evidence of an app was a session clock of 30 minutes, so a later
request read as a first build ("I'll build ...") while the editor edited the repo's
code, and "what did we build?" had no answer.
"""
import json
import time

import pytest

from codegen.generation_handler import _looks_like_fix_request, handle_generation_request
from common.session_keys import LAST_SMART_GEN_AT, LAST_SMART_GEN_PROJECT_ID, LAST_SMART_GEN_SUMMARY
from memory import get_memory, memory_session_key
from protocol.adapters import parse_assistant_request
from protocol.types import AssistantRequest, WorkspaceContext
from routing.generation_signals import linked_repo_recap, remember_repo_recap, repo_holds_app
from tests.conftest import FakeSession

RECAP = "**Recent work**\n- 2026-11-01: `add a search bar`"
REPO = {"fullName": "ana/library", "hasCode": True, "recap": RECAP}


def _session(github_repo=None, message="add auth to it") -> FakeSession:
    context = {"activeDiagramType": "ClassDiagram", "projectSnapshot": {"id": "p1", "diagrams": {}}}
    if github_repo is not None:
        context["githubRepo"] = github_repo
    return FakeSession({"action": "user_message", "protocolVersion": "2.0", "message": message, "context": context})


def _request(session: FakeSession, message: str) -> AssistantRequest:
    return AssistantRequest(message=message, context=WorkspaceContext(github_repo=parse_assistant_request(session).context.github_repo))


def test_the_repo_reaches_the_agent():
    assert parse_assistant_request(_session(REPO)).context.github_repo == {
        "full_name": "ana/library", "has_code": True, "recap": RECAP,
    }


@pytest.mark.parametrize("bad", ["nonsense", 3, {"fullName": 3}, {"hasCode": True}])
def test_a_malformed_repo_is_ignored(bad):
    assert parse_assistant_request(_session(bad)).context.github_repo is None


def test_a_huge_recap_is_cut():
    repo = parse_assistant_request(_session({**REPO, "recap": "x" * 50_000})).context.github_repo
    assert len(repo["recap"]) <= 4_000


def test_a_repo_with_code_means_an_app_exists_days_later():
    session = _session(REPO)
    assert repo_holds_app(session) is True
    assert _looks_like_fix_request(session, "add auth to it", "p1") is True


def test_a_repo_holding_only_the_model_is_still_a_first_build():
    session = _session({**REPO, "hasCode": False})
    assert repo_holds_app(session) is False
    assert _looks_like_fix_request(session, "build the app", "p1") is False


def test_no_repo_keeps_the_session_evidence():
    assert repo_holds_app(_session()) is False
    assert _looks_like_fix_request(_session(), "build the app", "p1") is False


def test_a_question_days_later_is_answered_from_the_recap():
    session = _session(REPO, "what did we generate?")
    result = handle_generation_request(session, _request(session, "what did we generate?"))
    assert result["action"] == "assistant_message"
    assert "add a search bar" in result["message"]


def test_a_fresh_run_summary_wins_over_the_older_recap():
    session = _session(REPO, "what did we generate?")
    session.set(LAST_SMART_GEN_AT, time.time())
    session.set(LAST_SMART_GEN_PROJECT_ID, "p1")
    session.set(LAST_SMART_GEN_SUMMARY, "Finished a minute ago.")
    result = handle_generation_request(session, _request(session, "what did we generate?"))
    assert "Finished a minute ago." in result["message"]


def test_an_old_summary_gives_way_to_the_recap():
    session = _session(REPO, "what did we generate?")
    session.set(LAST_SMART_GEN_AT, time.time() - 3 * 24 * 60 * 60)
    session.set(LAST_SMART_GEN_SUMMARY, "Finished days ago.")
    result = handle_generation_request(session, _request(session, "what did we generate?"))
    assert "add a search bar" in result["message"]


def test_the_recap_enters_the_conversation_once_as_a_record():
    # So a free question ("where did we stop?") reaches the model with it.
    session = _session(REPO)
    remember_repo_recap(session)
    remember_repo_recap(session)
    turns = get_memory(memory_session_key(session)).get_last_n(10)
    recorded = [turn for turn in turns if "add a search bar" in turn["content"]]
    assert len(recorded) == 1
    assert "not instructions" in recorded[0]["content"]
    assert "ana/library" in recorded[0]["content"]


def test_a_new_chat_gets_the_recap_again():
    # New Chat rotates the payload sessionId, which keys memory; the session stays.
    session = _session(REPO)
    session.event.json["sessionId"] = "first-chat"
    remember_repo_recap(session)
    second = _session(REPO)
    second.event.json["sessionId"] = "second-chat"
    session.event = second.event
    session.delete("_parsed_assistant_request")
    session.delete("_parsed_request_event_id")
    remember_repo_recap(session)
    assert memory_session_key(session) == "second-chat"
    assert "add a search bar" in json.dumps(get_memory("second-chat").get_last_n(10))


def test_a_changed_recap_is_recorded_again():
    session = _session(REPO)
    remember_repo_recap(session)
    session.event = _session({**REPO, "recap": "**Recent work**\n- `add payments`"}).event
    session.delete("_parsed_assistant_request")
    session.delete("_parsed_request_event_id")
    remember_repo_recap(session)
    contents = json.dumps(get_memory(memory_session_key(session)).get_last_n(10))
    assert "add a search bar" in contents and "add payments" in contents


def test_an_empty_recap_records_nothing():
    session = _session({**REPO, "recap": ""})
    remember_repo_recap(session)
    assert linked_repo_recap(session) == ""
    assert get_memory(memory_session_key(session)).get_last_n(10) == []
