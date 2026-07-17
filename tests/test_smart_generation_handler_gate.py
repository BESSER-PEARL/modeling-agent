"""Tests for the confirm-before-smart gate (B-2).

The Spec-Driven Agent runs on the USER'S OWN API key, so no code
path may emit ``trigger_smart_generator`` without an explicit user
confirmation. The gate stashes the payload (+ timestamp), replies with
run/cancel quick actions, and the confirm handler rejects stale stashes.

``handlers.generation_handler`` imports ``baf.core.session`` at module
level; the stub is installed at test RUN time (autouse fixture) and the
handler is imported lazily so the pre-existing collection errors of
baf-dependent test modules stay untouched.
"""

import sys
import types

import pytest

from tests.conftest import FakeSession


def _ensure_baf_stub():
    if "baf.core.session" in sys.modules:
        return
    baf = sys.modules.get("baf") or types.ModuleType("baf")
    baf_core = types.ModuleType("baf.core")
    baf_session = types.ModuleType("baf.core.session")

    class _StubSession:  # pragma: no cover - placeholder type only
        pass

    baf_session.Session = _StubSession
    baf_core.session = baf_session
    baf.core = baf_core
    sys.modules.setdefault("baf", baf)
    sys.modules["baf.core"] = baf_core
    sys.modules["baf.core.session"] = baf_session


@pytest.fixture(autouse=True)
def _baf_stub():
    _ensure_baf_stub()
    yield


def _gen_handler():
    import handlers.generation_handler as gen_mod
    return gen_mod


def _make_request(message: str):
    from protocol.types import AssistantRequest, WorkspaceContext
    return AssistantRequest(
        message=message,
        context=WorkspaceContext(active_diagram_type="ClassDiagram"),
    )


def _smart_classification():
    from handlers.smart_generation_handler import GenerationClassification
    return GenerationClassification(
        route="smart",
        refined_instructions="Build a Rails 7 API for the Library domain.",
        provider="anthropic",
        reason="user named rails",
    )


def _patch_provider(monkeypatch, decision):
    gen_mod = _gen_handler()

    class _FakeProvider:
        def parse(self, *, messages, schema, temperature, max_tokens):
            return decision

    monkeypatch.setattr(gen_mod, "_get_llm_provider", lambda: _FakeProvider(), raising=False)


# ---------------------------------------------------------------------
# Gate payload shape
# ---------------------------------------------------------------------


def test_smart_route_stashes_and_asks_instead_of_firing(monkeypatch):
    gen_mod = _gen_handler()
    from session_keys import (
        PENDING_SMART_GEN_INSTRUCTIONS,
        PENDING_SMART_GEN_PROVIDER,
        PENDING_SMART_GEN_TIMESTAMP,
    )

    _patch_provider(monkeypatch, _smart_classification())
    session = FakeSession()
    result = gen_mod.handle_generation_request(session, _make_request("build me a rails api"))

    # Never trigger_smart_generator on first contact
    assert result["action"] == "assistant_message"
    assert "API key" in result["message"]
    assert "Rails" in result["message"]  # refined instructions summarized

    actions = result["suggestedActions"]
    assert all(isinstance(a, dict) and {"label", "prompt"} <= set(a) for a in actions)
    prompts = [a["prompt"] for a in actions]
    assert "generate anyway with my current model" in prompts
    # Cancel button removed (product decision) — only Run is offered.
    assert "cancel the generation" not in prompts

    # Stash is set with a numeric timestamp
    assert session.get(PENDING_SMART_GEN_INSTRUCTIONS).startswith("Build a Rails 7 API")
    assert session.get(PENDING_SMART_GEN_PROVIDER) == "anthropic"
    assert isinstance(session.get(PENDING_SMART_GEN_TIMESTAMP), float)


def test_confirm_fires_trigger_and_clears_stash(monkeypatch):
    gen_mod = _gen_handler()
    from session_keys import PENDING_SMART_GEN_INSTRUCTIONS

    _patch_provider(monkeypatch, _smart_classification())
    session = FakeSession()
    gen_mod.handle_generation_request(session, _make_request("build me a rails api"))

    confirm = gen_mod.handle_generation_request(
        session, _make_request("generate anyway with my current model"),
    )
    assert confirm["action"] == "trigger_smart_generator"
    assert "Rails" in confirm["instructions"]
    assert confirm["provider"] == "anthropic"
    assert session.get(PENDING_SMART_GEN_INSTRUCTIONS) is None


@pytest.mark.parametrize("reply", ["yes", "Yes, please!", "run it now"])
def test_natural_confirmation_fires_trigger(monkeypatch, reply):
    gen_mod = _gen_handler()
    from session_keys import PENDING_SMART_GEN_INSTRUCTIONS

    _patch_provider(monkeypatch, _smart_classification())
    session = FakeSession()
    gen_mod.handle_generation_request(session, _make_request("build me a rails api"))

    result = gen_mod.handle_generation_request(session, _make_request(reply))

    assert result["action"] == "trigger_smart_generator"
    assert session.get(PENDING_SMART_GEN_INSTRUCTIONS) is None


def test_cancel_clears_stash_without_firing(monkeypatch):
    gen_mod = _gen_handler()
    from session_keys import PENDING_SMART_GEN_INSTRUCTIONS

    _patch_provider(monkeypatch, _smart_classification())
    session = FakeSession()
    gen_mod.handle_generation_request(session, _make_request("build me a rails api"))

    cancel = gen_mod.handle_generation_request(
        session, _make_request("cancel the generation"),
    )
    assert cancel["action"] == "assistant_message"
    assert "unchanged" in cancel["message"]
    assert session.get(PENDING_SMART_GEN_INSTRUCTIONS) is None


@pytest.mark.parametrize("reply", ["no", "No, thanks!", "cancel"])
def test_natural_cancellation_never_fires(monkeypatch, reply):
    gen_mod = _gen_handler()
    from session_keys import PENDING_SMART_GEN_INSTRUCTIONS

    _patch_provider(monkeypatch, _smart_classification())
    session = FakeSession()
    gen_mod.handle_generation_request(session, _make_request("build me a rails api"))

    result = gen_mod.handle_generation_request(session, _make_request(reply))

    assert result["action"] == "assistant_message"
    assert "cancel" in result["message"].lower()
    assert session.get(PENDING_SMART_GEN_INSTRUCTIONS) is None


@pytest.mark.parametrize(
    "reply",
    ["yes, but cancel", "no, actually run it", "do not cancel the generation"],
)
def test_mixed_or_qualified_confirmation_is_not_interpreted(reply):
    gen_mod = _gen_handler()

    assert gen_mod._smart_gen_confirmation_decision(reply) is None


def test_pending_smart_confirmation_routes_to_generation_handler():
    gen_mod = _gen_handler()
    from session_keys import PENDING_SMART_GEN_INSTRUCTIONS

    session = FakeSession()
    session.set(PENDING_SMART_GEN_INSTRUCTIONS, "Build a Rails API.")

    assert gen_mod.should_route_to_generation(session, _make_request("yes")) is True


# ---------------------------------------------------------------------
# 30-minute expiry
# ---------------------------------------------------------------------


def test_confirm_rejects_stash_older_than_30_minutes(monkeypatch):
    gen_mod = _gen_handler()
    from session_keys import (
        PENDING_SMART_GEN_INSTRUCTIONS,
        PENDING_SMART_GEN_TIMESTAMP,
    )

    _patch_provider(monkeypatch, _smart_classification())
    session = FakeSession()
    gen_mod.handle_generation_request(session, _make_request("build me a rails api"))

    # Mock time: jump 30 minutes + 1 second past the stash timestamp.
    stashed_ts = session.get(PENDING_SMART_GEN_TIMESTAMP)
    monkeypatch.setattr(
        gen_mod.time, "time", lambda: stashed_ts + gen_mod._SMART_GEN_STASH_TTL_SECONDS + 1,
    )

    result = gen_mod.handle_generation_request(
        session, _make_request("generate anyway with my current model"),
    )
    assert result["action"] == "assistant_message"
    assert "expired" in result["message"].lower()
    # Stale stash is cleared so it can never hijack a later flow
    assert session.get(PENDING_SMART_GEN_INSTRUCTIONS) is None


def test_confirm_with_legacy_stash_without_timestamp_is_rejected(monkeypatch):
    """Stashes written before the timestamp key existed are of unknown
    age — they must not fire."""
    gen_mod = _gen_handler()
    from session_keys import PENDING_SMART_GEN_INSTRUCTIONS

    session = FakeSession()
    session.set(PENDING_SMART_GEN_INSTRUCTIONS, "Build something old.")

    result = gen_mod.handle_generation_request(
        session, _make_request("generate anyway with my current model"),
    )
    assert result["action"] == "assistant_message"
    assert "expired" in result["message"].lower()
    assert session.get(PENDING_SMART_GEN_INSTRUCTIONS) is None


def test_stash_freshness_helper():
    gen_mod = _gen_handler()
    import time as _time

    assert gen_mod._smart_gen_stash_is_fresh(_time.time()) is True
    assert gen_mod._smart_gen_stash_is_fresh(
        _time.time() - gen_mod._SMART_GEN_STASH_TTL_SECONDS - 1
    ) is False
    assert gen_mod._smart_gen_stash_is_fresh(None) is False
    assert gen_mod._smart_gen_stash_is_fresh("yesterday") is False
    assert gen_mod._smart_gen_stash_is_fresh(True) is False
