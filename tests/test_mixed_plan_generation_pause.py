"""Mixed "design X and generate Y" plans pause before generating — for EVERY
generator type, not just web_app.

Live bug: "design a hospital system and generate the Pydantic models" injected
the model with a message asking "review or continue with generating?" and then
immediately ran the generation anyway, self-answering its own question.

The fix generalizes the web-app pause: execution/planning.py strips the
generation op from a mixed plan and stashes it in the existing pending-
generation state (PENDING_GENERATOR_TYPE / PENDING_GENERATOR_CONFIG) marked
with PLAN_GENERATION_CONFIRM_FLAG; handle_generation_request then fires it only
on an affirmative answer. Direct generation requests (no modeling step) keep
their current immediate-dispatch behavior — covered here and in
test_webapp_generation_gate.py.
"""

import types
from unittest.mock import MagicMock, patch

import execution.planning as planning
from handlers.generation_handler import handle_generation_request
from protocol.types import AssistantRequest, WorkspaceContext
from session_keys import (
    PENDING_GENERATOR_CONFIG,
    PENDING_GENERATOR_TYPE,
    PLAN_GENERATION_CONFIRM_FLAG,
    UNIFIED_CLASSIFICATION,
)

from tests.conftest import FakeSession


_CLASS_MODEL = {
    "elements": {
        "class-1": {"type": "Class", "name": "Doctor"},
    },
    "relationships": {},
}


def _make_request(message: str) -> AssistantRequest:
    return AssistantRequest(
        action="user_message",
        message=message,
        context=WorkspaceContext(
            active_diagram_type="ClassDiagram",
            active_model=_CLASS_MODEL,
            project_snapshot={
                "name": "Hospital",
                "diagrams": {"ClassDiagram": [{"model": _CLASS_MODEL}]},
            },
        ),
    )


def _arm_paused_generation(session, generator_type="pydantic", config=None):
    """Stash a plan-paused generation exactly as execution/planning.py does."""
    session.set(PENDING_GENERATOR_TYPE, generator_type)
    session.set(
        PENDING_GENERATOR_CONFIG,
        {**(config or {}), PLAN_GENERATION_CONFIRM_FLAG: True},
    )


# ---------------------------------------------------------------------------
# Planning side: the mixed plan strips + stashes; failures disarm the stash
# ---------------------------------------------------------------------------

def _run_plan(plan, model_op_result="ClassDiagram", model_op_raises=False):
    gen = MagicMock(return_value={"action": "trigger_generator"})
    prompt = MagicMock()
    session = FakeSession()
    if model_op_raises:
        model_op = MagicMock(side_effect=RuntimeError("boom"))
    else:
        model_op = MagicMock(return_value=model_op_result)
    with patch.object(planning, "plan_assistant_operations", return_value=plan), \
         patch.object(planning, "execute_model_operation", model_op), \
         patch.object(planning, "reply_payload"), \
         patch.object(planning, "reply_message"), \
         patch.object(planning, "emit_webapp_generate_prompt", prompt), \
         patch.object(planning, "handle_generation_request", gen), \
         patch.object(planning, "_report_progress"), \
         patch.object(planning, "build_request_for_target", side_effect=lambda r, t: r):
        planning.execute_planned_operations(
            session, MagicMock(), "complete_system", "create_complete_system_intent",
        )
    return gen, prompt, session


class TestMixedPlanPause:
    def test_mixed_plan_pauses_pydantic_generation(self):
        """The live repro shape: model step + pydantic generation step."""
        plan = [
            {"type": "model", "diagramType": "ClassDiagram",
             "mode": "complete_system", "request": "design a hospital system"},
            {"type": "generation", "generatorType": "pydantic", "config": {}},
        ]
        gen, prompt, session = _run_plan(plan)
        assert not gen.called, "generation must wait for the user's answer"
        assert not prompt.called
        assert session.get(PENDING_GENERATOR_TYPE) == "pydantic"
        stash = session.get(PENDING_GENERATOR_CONFIG)
        assert stash.get(PLAN_GENERATION_CONFIRM_FLAG) is True

    def test_paused_stash_preserves_planned_config(self):
        plan = [
            {"type": "model", "diagramType": "ClassDiagram",
             "mode": "complete_system", "request": "design a store"},
            {"type": "generation", "generatorType": "sql",
             "config": {"dialect": "postgresql"}},
        ]
        gen, _prompt, session = _run_plan(plan)
        assert not gen.called
        stash = session.get(PENDING_GENERATOR_CONFIG)
        assert stash.get("dialect") == "postgresql"
        assert stash.get(PLAN_GENERATION_CONFIRM_FLAG) is True

    def test_generation_only_plan_dispatches_immediately(self):
        """Direct generation (no modeling step) keeps its current behavior."""
        plan = [{"type": "generation", "generatorType": "pydantic", "config": {}}]
        gen, _prompt, session = _run_plan(plan)
        assert gen.called
        assert session.get(PENDING_GENERATOR_TYPE) is None

    def test_model_op_failure_clears_paused_generation(self):
        """A broken build must never leave a generator armed to fire later."""
        plan = [
            {"type": "model", "diagramType": "ClassDiagram",
             "mode": "complete_system", "request": "design a hospital system"},
            {"type": "generation", "generatorType": "pydantic", "config": {}},
        ]
        gen, _prompt, session = _run_plan(plan, model_op_raises=True)
        assert not gen.called
        assert session.get(PENDING_GENERATOR_TYPE) is None
        assert session.get(PENDING_GENERATOR_CONFIG) is None


# ---------------------------------------------------------------------------
# Consumption side: only an affirmative fires the stashed generator
# ---------------------------------------------------------------------------

class TestPausedGenerationConfirmation:
    def _cache_intent(self, session, intent, flow_action=None, flow_answer=None):
        session.set(UNIFIED_CLASSIFICATION, types.SimpleNamespace(
            intent=intent,
            generation_route=None,
            generator_type=None,
            refined_instructions=None,
            provider="anthropic",
            reason="cached",
            domain_mismatch=False,
            suggested_new_domain=None,
            pending_flow_action=flow_action,
            pending_flow_answer=flow_answer,
        ))

    def test_yes_fires_the_paused_generator(self):
        session = FakeSession()
        _arm_paused_generation(session, "pydantic")
        result = handle_generation_request(session, _make_request("yes"))
        assert result["action"] == "trigger_generator"
        assert result["generatorType"] == "pydantic"
        # The internal marker must never leak into the trigger config.
        assert PLAN_GENERATION_CONFIRM_FLAG not in result.get("config", {})
        assert session.get(PENDING_GENERATOR_TYPE) is None

    def test_ok_fires_the_paused_generator(self):
        session = FakeSession()
        _arm_paused_generation(session, "pydantic")
        result = handle_generation_request(session, _make_request("ok"))
        assert result["action"] == "trigger_generator"
        assert result["generatorType"] == "pydantic"

    def test_bare_generate_fires_the_paused_generator(self):
        session = FakeSession()
        _arm_paused_generation(session, "pydantic")
        result = handle_generation_request(session, _make_request("generate"))
        assert result["action"] == "trigger_generator"
        assert result["generatorType"] == "pydantic"

    def test_classifier_confirm_verdict_fires_the_paused_generator(self):
        session = FakeSession()
        _arm_paused_generation(session, "pydantic")
        self._cache_intent(session, "generation_intent",
                           flow_action="answer", flow_answer="confirm")
        result = handle_generation_request(
            session, _make_request("please go ahead with it"))
        assert result["action"] == "trigger_generator"
        assert result["generatorType"] == "pydantic"

    def test_no_cancels_the_paused_generator(self):
        session = FakeSession()
        _arm_paused_generation(session, "pydantic")
        result = handle_generation_request(session, _make_request("no"))
        assert result["action"] == "assistant_message"
        assert "won't run code generation" in result["message"]
        assert session.get(PENDING_GENERATOR_TYPE) is None

    def test_decline_verdict_cancels_the_paused_generator(self):
        session = FakeSession()
        _arm_paused_generation(session, "pydantic")
        self._cache_intent(session, "decline_intent")
        result = handle_generation_request(
            session, _make_request("rather not, thanks"))
        assert result["action"] == "assistant_message"
        assert session.get(PENDING_GENERATOR_TYPE) is None

    def test_unrelated_message_abandons_the_pause(self, monkeypatch):
        """A non-answer clears the stash and is routed on its own merits —
        it must NOT fire the paused generator."""
        import handlers.generation_handler as gen_mod
        from unified_classifier import UnifiedClassification

        class _Provider:
            model_name = "test-model"

            def parse(self, **_kwargs):
                return UnifiedClassification(
                    intent="out_of_scope_intent",
                    reason="not generation",
                )

        monkeypatch.setattr(
            gen_mod, "_get_llm_provider", lambda: _Provider(), raising=False)
        session = FakeSession()
        _arm_paused_generation(session, "pydantic")
        self._cache_intent(session, "modeling_help_intent", flow_action="answer")
        result = handle_generation_request(
            session, _make_request("what is an association class?"))
        assert session.get(PENDING_GENERATOR_TYPE) is None
        assert result is None or result.get("action") != "trigger_generator"

    def test_confirm_with_missing_required_config_prompts_for_it(self):
        """Confirming a generator that still needs config drops into the
        normal config-collection flow (marker stripped, no auto-dispatch)."""
        session = FakeSession()
        _arm_paused_generation(session, "sql")  # sql requires a dialect
        result = handle_generation_request(session, _make_request("yes"))
        assert result["action"] == "assistant_message"
        assert "dialect" in result["message"].lower()
        # Pending state persists for the config answer — without the marker.
        assert session.get(PENDING_GENERATOR_TYPE) == "sql"
        stored = session.get(PENDING_GENERATOR_CONFIG)
        assert PLAN_GENERATION_CONFIRM_FLAG not in (stored or {})
