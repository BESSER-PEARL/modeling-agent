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
