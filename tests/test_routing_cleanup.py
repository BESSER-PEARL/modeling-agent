"""Behavioral pins for the intent-routing cleanup (architecture review).

Covers the composition-level behaviors the per-guard unit tests never did:

  * Confirmation pivots: a long "no, just add an email attribute…" typed at
    the replace/keep prompt is a NEW REQUEST, not a keep answer (the old
    KEEP_KEYWORDS contained 'no' and 'add', so the pivot was swallowed and
    the user's instruction discarded). A short "no" still answers keep.
  * Config-flow suppression: while a REAL generator config flow is pending
    (e.g. Django project-name Q&A), answer-shaped verdicts (decline /
    fallback / generation) stay in generation_state instead of routing away
    and orphaning the flow; a clear pivot to modify/create still routes.
  * Structured smart-gen recency: a completed smart generation sets
    LAST_SMART_GEN_AT, and the classifier renders a structured
    'RECENT SMART GENERATION' line from it (no reply-copy grepping).
  * Registry sync: the two _DETERMINISTIC_GENERATOR_TYPES Literals (unified
    classifier vs dispatch schema) must stay identical.
"""
import time
import types
from dataclasses import replace as _dc_replace  # noqa: F401
from typing import get_args

from unittest.mock import patch, MagicMock

import confirmation
from confirmation import handle_pending_system_confirmation
from session_helpers import json_intent_matches
from session_keys import (
    LAST_SMART_GEN_AT,
    PENDING_COMPLETE_SYSTEM,
    PENDING_GENERATOR_TYPE,
    UNIFIED_CLASSIFICATION,
)
from unified_classifier import UnifiedClassification, _build_user_block
from protocol.types import AssistantRequest, WorkspaceContext

from tests.conftest import FakeSession


def _uc(intent):
    return UnifiedClassification(intent=intent, reason="stub")


def _req(message):
    return AssistantRequest(
        message=message,
        context=WorkspaceContext(active_diagram_type="ClassDiagram"),
    )


# ---------------------------------------------------------------------
# Confirmation pivots (KEEP_KEYWORDS two-tier fix)
# ---------------------------------------------------------------------

_PENDING = {
    "operation": {},
    "default_mode": "complete_system",
    "operation_request": "create a shop system",
    "can_add_tab": False,
}


class TestConfirmationPivots:
    def _run(self, message):
        session = FakeSession()
        session.set(PENDING_COMPLETE_SYSTEM, dict(_PENDING))
        executed = {}
        with patch.object(confirmation, "parse_assistant_request",
                          return_value=_req(message)), \
             patch.object(confirmation, "reply_message"), \
             patch.object(confirmation, "reply_payload"), \
             patch.object(confirmation, "execute_model_operation",
                          side_effect=lambda **kw: executed.update(kw) or None):
            handled = handle_pending_system_confirmation(session)
        return handled, executed, session

    def test_long_pivot_with_no_and_add_is_a_new_request(self):
        handled, executed, session = self._run(
            "no, just add an email attribute to Customer")
        assert handled is False              # falls through to normal routing
        assert executed == {}                # stored create NOT executed
        assert not session.get(PENDING_COMPLETE_SYSTEM)  # pending cleared

    def test_short_no_still_answers_keep(self):
        handled, executed, session = self._run("no")
        assert handled is True
        assert executed.get("_replace_existing") is False  # keep-mode

    def test_explicit_keep_negation_still_keeps(self):
        # Longer than the weak-tier gate, but contains the explicit 'keep'.
        handled, executed, session = self._run(
            "no, keep my model, do not replace it")
        assert handled is True
        assert executed.get("_replace_existing") is False


# ---------------------------------------------------------------------
# Config-flow suppression in json_intent_matches
# ---------------------------------------------------------------------

class TestConfigFlowSuppression:
    def _session(self, pending, cached_intent):
        session = FakeSession()
        if pending is not None:
            session.set(PENDING_GENERATOR_TYPE, pending)
        if cached_intent is not None:
            session.set(UNIFIED_CLASSIFICATION, _uc(cached_intent))
        return session

    def test_answer_shaped_decline_stays_in_generation_state(self):
        """'no docker' mid-config classifies as decline — must NOT route away."""
        session = self._session("django", "decline_intent")
        assert json_intent_matches(
            session, {"intent_name": "decline_intent"}) is False

    def test_undecided_stays_in_generation_state(self):
        session = self._session("django", None)
        assert json_intent_matches(
            session, {"intent_name": "modify_model_intent"}) is False

    def test_clear_pivot_to_modify_still_routes(self):
        session = self._session("django", "modify_model_intent")
        assert json_intent_matches(
            session, {"intent_name": "modify_model_intent"}) is True

    def test_awaiting_selection_always_suppresses(self):
        session = self._session("_awaiting_selection", "modify_model_intent")
        assert json_intent_matches(
            session, {"intent_name": "modify_model_intent"}) is False

    def test_no_pending_flow_matches_normally(self):
        session = self._session(None, "modify_model_intent")
        assert json_intent_matches(
            session, {"intent_name": "modify_model_intent"}) is True
        assert json_intent_matches(
            session, {"intent_name": "create_complete_system_intent"}) is False


# ---------------------------------------------------------------------
# Structured smart-gen recency signal
# ---------------------------------------------------------------------

class TestSmartGenRecencySignal:
    def test_result_handler_sets_timestamp(self):
        from handlers.generation_handler import _handle_frontend_event
        session = FakeSession()
        request = types.SimpleNamespace(
            action="frontend_event", message="",
            raw_payload={
                "eventType": "generator_result", "ok": True,
                "metadata": {"smart": True, "costUsd": 0.1},
            },
        )
        _handle_frontend_event(request, session)
        ts = session.get(LAST_SMART_GEN_AT)
        assert isinstance(ts, float) and (time.time() - ts) < 5

    def test_user_block_renders_structured_line(self):
        block = _build_user_block(_req("add auth to it"), None,
                                  recent_smart_gen=True)
        assert "RECENT SMART GENERATION" in block

    def test_user_block_omits_line_without_signal(self):
        block = _build_user_block(_req("add auth to it"), None)
        assert "RECENT SMART GENERATION" not in block


# ---------------------------------------------------------------------
# Generator-registry sync
# ---------------------------------------------------------------------

class TestGeneratorRegistrySync:
    def test_deterministic_generator_literals_identical(self):
        from unified_classifier import (
            _DETERMINISTIC_GENERATOR_TYPES as unified_lit,
        )
        from handlers.smart_generation_handler import (
            _DETERMINISTIC_GENERATOR_TYPES as dispatch_lit,
        )
        assert set(get_args(unified_lit)) == set(get_args(dispatch_lit)), (
            "the classifier's generator list and the dispatch schema's list "
            "have drifted — update both (they carry 'must stay in sync' "
            "comments) or extract a shared registry"
        )


# ---------------------------------------------------------------------
# Adversarial-review fixes (post-cleanup)
# ---------------------------------------------------------------------

class _CountingProvider:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def parse(self, *, messages, schema, temperature, max_tokens):
        self.calls += 1
        return self.result


class TestSyntheticSubRequestClassification:
    """A compound "create X and generate Y" plan dispatches a synthetic
    'generate django' sub-request that SHARES the original message's cached
    create verdict. Adapting that verdict routed the generation back into
    modeling (recursion); the cache must only be trusted for GENERATION
    verdicts — anything else re-classifies the sub-request's own text."""

    def test_non_generation_cache_triggers_fresh_classification(self, monkeypatch):
        import handlers.generation_handler as gen_mod
        from handlers.generation_handler import (
            _get_classification_from_cache_or_classify,
        )
        session = FakeSession()
        session.set(UNIFIED_CLASSIFICATION, _uc("create_complete_system_intent"))
        provider = _CountingProvider(UnifiedClassification(
            intent="generation_intent", generation_route="deterministic",
            generator_type="django", reason="fresh sub-request verdict",
        ))
        monkeypatch.setattr(gen_mod, "_get_llm_provider", lambda: provider,
                            raising=False)
        result = _get_classification_from_cache_or_classify(
            session, _req("generate django"))
        assert provider.calls == 1            # fresh classify, not the cache
        assert result.route == "deterministic"
        assert result.generator_type == "django"

    def test_generation_cache_is_used_without_llm_call(self, monkeypatch):
        import handlers.generation_handler as gen_mod
        from handlers.generation_handler import (
            _get_classification_from_cache_or_classify,
        )
        session = FakeSession()
        session.set(UNIFIED_CLASSIFICATION, UnifiedClassification(
            intent="generation_intent", generation_route="deterministic",
            generator_type="sql", reason="cached"))
        provider = _CountingProvider(None)
        monkeypatch.setattr(gen_mod, "_get_llm_provider", lambda: provider,
                            raising=False)
        result = _get_classification_from_cache_or_classify(
            session, _req("generate sql"))
        assert provider.calls == 0
        assert result.generator_type == "sql"


class TestConfigFlowCancel:
    """Declining / cancelling mid config-collection must EXIT the flow —
    previously the reply looped the prompt and the third attempt auto-filled
    defaults and generated the very thing the user was refusing."""

    def _run(self, message, cached_intent=None):
        from handlers.generation_handler import handle_generation_request
        from session_keys import CONFIG_PROMPT_ATTEMPTS
        session = FakeSession()
        session.set(PENDING_GENERATOR_TYPE, "django")
        session.set(CONFIG_PROMPT_ATTEMPTS, 2)
        if cached_intent:
            session.set(UNIFIED_CLASSIFICATION, _uc(cached_intent))
        result = handle_generation_request(session, _req(message))
        return result, session

    def test_explicit_cancel_exits_the_flow(self):
        result, session = self._run("cancel")
        assert "cancelled" in result["message"].lower()
        assert not session.get(PENDING_GENERATOR_TYPE)

    def test_decline_verdict_exits_the_flow(self):
        result, session = self._run("nah forget this whole thing",
                                    cached_intent="decline_intent")
        assert "cancelled" in result["message"].lower()
        assert not session.get(PENDING_GENERATOR_TYPE)

    def test_config_answer_continues_the_flow(self):
        # A real config answer must NOT be treated as a cancel. (Needs a class
        # model in context or the prerequisite check ends the flow first.)
        from handlers.generation_handler import handle_generation_request
        from session_keys import CONFIG_PROMPT_ATTEMPTS
        model = {"elements": {"c1": {"type": "Class", "name": "Book"}},
                 "relationships": {}}
        request = AssistantRequest(
            message="project_name=shop",
            context=WorkspaceContext(
                active_diagram_type="ClassDiagram", active_model=model,
                project_snapshot={"name": "P",
                                  "diagrams": {"ClassDiagram": [{"model": model}]}},
            ),
        )
        session = FakeSession()
        session.set(PENDING_GENERATOR_TYPE, "django")
        session.set(CONFIG_PROMPT_ATTEMPTS, 0)
        result = handle_generation_request(session, request)
        # Continues the flow: either re-prompts for the missing fields
        # (pending kept) or, once satisfied, triggers the generator.
        assert session.get(PENDING_GENERATOR_TYPE) or (
            result.get("action") == "trigger_generator")
        assert "cancelled" not in (result.get("message") or "").lower()


class TestPendingStashInterjections:
    """Interjections at the smart-gen confirmation ("Do you want to
    continue?") must not silently kill the prepared run with a wrong reply."""

    def _session_with_stash(self, cached_intent):
        import time as _t
        from session_keys import (
            PENDING_SMART_GEN_INSTRUCTIONS,
            PENDING_SMART_GEN_PROVIDER,
            PENDING_SMART_GEN_TIMESTAMP,
        )
        session = FakeSession()
        session.set(PENDING_SMART_GEN_INSTRUCTIONS, "build a shop app")
        session.set(PENDING_SMART_GEN_PROVIDER, "anthropic")
        session.set(PENDING_SMART_GEN_TIMESTAMP, _t.time())
        session.set(UNIFIED_CLASSIFICATION, _uc(cached_intent))
        return session

    def test_out_of_scope_interjection_answers_and_keeps_stash(self):
        from handlers.generation_handler import handle_generation_request
        from reply_copy import OUT_OF_SCOPE_REDIRECT
        from session_keys import PENDING_SMART_GEN_INSTRUCTIONS
        session = self._session_with_stash("out_of_scope_intent")
        result = handle_generation_request(session, _req("draw me a cat"))
        assert result["message"] == OUT_OF_SCOPE_REDIRECT
        assert session.get(PENDING_SMART_GEN_INSTRUCTIONS)  # stash survives

    def test_decline_at_confirmation_cancels_cleanly(self):
        from handlers.generation_handler import handle_generation_request
        from session_keys import PENDING_SMART_GEN_INSTRUCTIONS
        session = self._session_with_stash("decline_intent")
        result = handle_generation_request(session, _req("nah, I am good"))
        assert "cancelled" in result["message"].lower()
        assert not session.get(PENDING_SMART_GEN_INSTRUCTIONS)


class TestErrorFallbackMarker:
    """_safe_fallback tags ERROR fallbacks so the generation adapter can show
    the resilient generator MENU on outages while a DELIBERATE none-fit
    verdict gets the clarify reply."""

    def test_safe_fallback_is_tagged(self):
        from unified_classifier import classify_message
        result = classify_message(_req("generate django"), llm_provider=None)
        assert result.intent == "fallback_intent"
        assert result.reason.startswith("[classifier-error]")

    def test_adapter_menu_on_error_but_other_on_deliberate(self):
        from handlers.generation_handler import _classification_to_legacy
        error = UnifiedClassification(
            intent="fallback_intent", reason="[classifier-error] LLM down")
        assert _classification_to_legacy(error).route == "deterministic"
        deliberate = UnifiedClassification(
            intent="fallback_intent", reason="none of the intents fit")
        assert _classification_to_legacy(deliberate).route == "other"


class TestOutageNetPins:
    """The classifier-outage net in _modeling_state_body exists and is scoped
    to bare phrases / explicit artifact asks (never real requests)."""

    def test_phrase_set_and_pattern(self):
        from state_bodies import _OUTAGE_DECLINE_PHRASES, _OUTAGE_ARTIFACT_RE
        assert "nothing" in _OUTAGE_DECLINE_PHRASES
        assert "never mind" in _OUTAGE_DECLINE_PHRASES
        assert _OUTAGE_ARTIFACT_RE.search("generate a picture of a cat")
        assert _OUTAGE_ARTIFACT_RE.search("tell me a joke")
        assert not _OUTAGE_ARTIFACT_RE.search("create a photo-sharing app")
        assert not _OUTAGE_ARTIFACT_RE.search("a library with books and loans")
