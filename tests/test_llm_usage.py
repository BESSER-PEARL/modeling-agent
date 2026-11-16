"""llmUsage: the totals of a turn's LLM calls on its terminal replies.

The editor records the assistant's turn as an LLM tool call; without these
totals it named a hardcoded model. Every LLM call records through
TokenTracker.record, every reply leaves through session_helpers._send_to_session:
the turn is summed at the first and stamped at the second.
"""
import contextvars
import json
import threading

import pytest

from app import state_bodies
from conversation import session_helpers
from llm import byok_config, tracking
from llm.byok_config import BYOKConfig
from llm.provider import LLMProvider
from llm.tracking import get_tracker
from llm.tracking.turn_usage import TurnUsage
from protocol.actions import Action, assistant_message, show_trace
from tests.conftest import FakeLLM, FakeSession


def _session(turn_id=None) -> FakeSession:
    payload = {"action": "user_message", "protocolVersion": "2.0", "message": "describe my model",
               "context": {"activeDiagramType": "ClassDiagram"}, "sessionId": "s-usage"}
    if turn_id:
        payload["turnId"] = turn_id
    return FakeSession(payload)


def _frames(session):
    return [json.loads(r) for r in session.replies]


# --- TurnUsage ---------------------------------------------------------------------

def test_no_call_no_usage():
    assert TurnUsage("t").snapshot() is None


def test_totals_are_named_after_the_model_that_used_most():
    usage = TurnUsage("t")
    usage.add(900, 40, model="gpt-6-luna", provider="openai")        # the classifier
    usage.add(3000, 1200, model="gpt-6-sol", provider="openai")      # the answer
    usage.add(200, 60, model="gpt-6-luna", provider="openai")        # a translation
    assert usage.snapshot() == {"model": "gpt-6-sol", "provider": "openai",
                                "inputTokens": 4100, "outputTokens": 1300}


def test_a_tie_keeps_the_first_model():
    usage = TurnUsage("t")
    usage.add(10, 0, model="a", provider="openai")
    usage.add(0, 10, model="b", provider="openai")
    assert usage.snapshot()["model"] == "a"


def test_the_same_event_keeps_its_turn_and_a_new_one_starts_fresh():
    first = tracking.begin_turn(1)
    assert tracking.begin_turn(1) is first
    second = tracking.begin_turn(2)
    assert second is not first and tracking.current_turn() is second


# --- Recording ---------------------------------------------------------------------

def test_every_recorded_call_counts_in_the_open_turn():
    tracking.begin_turn("evt")
    get_tracker().record(prompt_tokens=120, completion_tokens=30, model="gpt-6-luna")
    get_tracker().record_from_usage(type("U", (), {"prompt_tokens": 80, "completion_tokens": 20})(),
                                    model="gpt-6-luna")
    assert tracking.current_turn().snapshot() == {"model": "gpt-6-luna", "provider": "openai",
                                                  "inputTokens": 200, "outputTokens": 50}


def test_without_an_open_turn_nothing_is_kept():
    get_tracker().record(prompt_tokens=120, completion_tokens=30, model="gpt-6-luna")
    assert tracking.current_turn() is None


def test_a_users_key_reports_its_provider():
    tracking.begin_turn("evt")
    token = byok_config.current_byok.set(BYOKConfig(provider="anthropic", api_key="sk-x"))
    try:
        get_tracker().record(prompt_tokens=10, completion_tokens=5, model="claude-sonnet-5-5")
    finally:
        byok_config.current_byok.reset(token)
    assert tracking.current_turn().snapshot()["provider"] == "anthropic"


def test_a_call_through_the_provider_is_counted():
    tracking.begin_turn("evt")
    LLMProvider(FakeLLM("an answer of some length"), "gpt-6-luna").complete("a prompt " * 10)
    snapshot = tracking.current_turn().snapshot()
    assert snapshot["model"] == "gpt-6-luna" and snapshot["inputTokens"] > 0


def test_a_pool_thread_with_the_turns_context_counts_in_it():
    # A parallel plan runs its groups in pool threads that copy the request's context.
    tracking.begin_turn("evt")
    context = contextvars.copy_context()
    worker = threading.Thread(target=context.run, args=(
        get_tracker().record,), kwargs={"prompt_tokens": 7, "completion_tokens": 3, "model": "m"})
    worker.start()
    worker.join()
    assert tracking.current_turn().snapshot()["inputTokens"] == 7


# --- Stamping at the reply choke point ---------------------------------------------

def _with_usage(session):
    session_helpers.begin_turn_usage(session)
    get_tracker().record(prompt_tokens=1000, completion_tokens=250, model="gpt-6-luna")


USAGE = {"model": "gpt-6-luna", "provider": "openai", "inputTokens": 1000, "outputTokens": 250}


def test_a_terminal_reply_carries_the_turns_usage():
    session = _session()
    _with_usage(session)
    session_helpers.reply_message(session, "Your model has one class, Book.")
    assert _frames(session)[0] == {"action": "assistant_message", "message": "Your model has one class, Book.",
                                   "llmUsage": USAGE}


def test_a_stream_stamps_its_done_frame_only():
    session = _session()
    _with_usage(session)
    stream_id = session_helpers.reply_stream_start(session)
    session_helpers.reply_stream_chunk(session, "Hello.", stream_id)
    session_helpers.reply_progress(session, "working")
    session_helpers.reply_stream_done(session, stream_id, "Hello.")
    frames = _frames(session)
    assert [f["action"] for f in frames] == ["stream_start", "stream_chunk", "progress", "stream_done"]
    assert [("llmUsage" in f) for f in frames] == [False, False, False, True]


@pytest.mark.parametrize("payload", [
    show_trace("lineage", "Open", "a1"),
    {"action": Action.INJECT_COMPLETE_SYSTEM, "systemSpec": {}, "diagramType": "ClassDiagram", "message": "Built"},
    {"action": Action.AGENT_ERROR, "code": "x", "message": "failed"},
    {"action": Action.TRIGGER_GENERATOR, "generatorType": "sql", "message": "Generating"},
])
def test_every_terminal_action_is_stamped(payload):
    session = _session()
    _with_usage(session)
    session_helpers.reply_payload(session, payload)
    assert _frames(session)[0]["llmUsage"] == USAGE


def test_a_later_reply_carries_the_totals_so_far():
    session = _session()
    _with_usage(session)
    session_helpers.reply_message(session, "Building it now.")
    get_tracker().record(prompt_tokens=500, completion_tokens=50, model="gpt-6-sol")
    session_helpers.reply_payload(session, assistant_message("Done."))
    first, last = (f["llmUsage"] for f in _frames(session))
    assert first["inputTokens"] == 1000
    assert last == {"model": "gpt-6-luna", "provider": "openai", "inputTokens": 1500, "outputTokens": 300}


def test_no_llm_call_no_field():
    session = _session()
    session_helpers.begin_turn_usage(session)
    session_helpers.reply_message(session, "Hi!")
    assert "llmUsage" not in _frames(session)[0]


def test_no_open_turn_no_field():
    session = _session()
    get_tracker().record(prompt_tokens=10, completion_tokens=5, model="m")
    session_helpers.reply_message(session, "Hi!")
    assert _frames(session)[0] == {"action": "assistant_message", "message": "Hi!"}


def test_the_field_rides_along_with_the_turn_stamps_and_replays_as_sent():
    session = _session(turn_id="t-9")
    _with_usage(session)
    session_helpers.reply_message(session, "Your model has one class.")
    sent = _frames(session)[0]
    assert sent["turnId"] == "t-9" and sent["llmUsage"] == USAGE
    replay = FakeSession({**session.event.json, "action": "replay_last_response", "appliedSeqs": []})
    session_helpers.replay_last_reply(replay, state_bodies.parse_assistant_request(replay))
    assert json.loads(replay.replies[0]) == sent


def test_the_transition_hook_opens_the_turn_before_the_classifier(monkeypatch):
    session = _session()

    def classify(s, request, provider):
        get_tracker().record(prompt_tokens=900, completion_tokens=40, model="gpt-6-luna")
        return type("V", (), {"intent": "describe_model_intent", "generation_route": None,
                              "generator_type": None, "reason": "r"})()

    monkeypatch.setattr(state_bodies, "get_or_classify", classify)
    assert state_bodies._ensure_unified_classification(session) is False
    # The state body of the same event keeps the turn the hook opened.
    session_helpers.begin_turn_usage(session)
    get_tracker().record(prompt_tokens=100, completion_tokens=200, model="gpt-6-sol")
    session_helpers.reply_message(session, "Book has a title.")
    assert _frames(session)[0]["llmUsage"] == {"model": "gpt-6-luna", "provider": "openai",
                                                "inputTokens": 1000, "outputTokens": 240}


def test_the_next_event_starts_from_zero():
    session = _session()
    _with_usage(session)
    session.event = _session().event
    session_helpers.begin_turn_usage(session)
    session_helpers.reply_message(session, "Hi again!")
    assert "llmUsage" not in _frames(session)[0]
