"""Tests for the unified message classifier."""

import pytest

from protocol.types import AssistantRequest, WorkspaceContext
from session_keys import (
    UNIFIED_CLASSIFICATION,
    UNIFIED_CLASSIFICATION_EVENT_ID,
)
from unified_classifier import (
    _SYSTEM_PROMPT,
    UnifiedClassification,
    classify_message,
    get_or_classify,
)

from tests.conftest import FakeSession


def _make_request(message: str) -> AssistantRequest:
    return AssistantRequest(
        message=message,
        context=WorkspaceContext(
            active_diagram_type="ClassDiagram",
            project_snapshot={"name": "LibraryProject", "diagrams": {}},
            diagram_summaries=[
                {"type": "ClassDiagram", "title": "Library", "elementCount": 4},
            ],
        ),
    )


class _FakeProvider:
    def __init__(self, decision, raise_on_parse=False):
        self.decision = decision
        self.raise_on_parse = raise_on_parse
        self.calls = 0

    def parse(self, *, messages, schema, temperature, max_tokens):
        self.calls += 1
        if self.raise_on_parse:
            raise RuntimeError("simulated failure")
        return self.decision


class _FakeEvent:
    def __init__(self, event_id=None):
        if event_id is not None:
            self.id = event_id


class _CacheableSession(FakeSession):
    """Extends FakeSession with an ``event`` attribute so
    ``get_or_classify`` can use a stable event id as cache key."""

    def __init__(self, event_id="evt-1"):
        super().__init__()
        self.event = _FakeEvent(event_id)


class TestClassifyMessage:
    def test_returns_fallback_when_provider_none(self):
        result = classify_message(_make_request("generate django"), llm_provider=None)
        assert isinstance(result, UnifiedClassification)
        assert result.intent == "fallback_intent"

    def test_returns_fallback_on_empty_message(self):
        result = classify_message(_make_request(""), llm_provider=_FakeProvider(None))
        assert result.intent == "fallback_intent"

    def test_returns_fallback_on_llm_exception(self):
        provider = _FakeProvider(None, raise_on_parse=True)
        result = classify_message(
            _make_request("build me a rails api"), llm_provider=provider,
        )
        assert result.intent == "fallback_intent"

    def test_returns_llm_verdict_verbatim_for_generation(self):
        expected = UnifiedClassification(
            intent="generation_intent",
            generation_route="smart",
            refined_instructions="Build a Rails 7 app.",
            provider="anthropic",
            reason="user named rails",
        )
        provider = _FakeProvider(expected)
        result = classify_message(
            _make_request("build me a rails api"), llm_provider=provider,
        )
        assert result.intent == "generation_intent"
        assert result.generation_route == "smart"
        assert result.refined_instructions.startswith("Build a Rails 7")

    def test_smart_without_instructions_demoted_to_deterministic_unknown(self):
        bad = UnifiedClassification(
            intent="generation_intent",
            generation_route="smart",
            refined_instructions="",
            reason="bug",
        )
        result = classify_message(
            _make_request("do something"), llm_provider=_FakeProvider(bad),
        )
        assert result.intent == "generation_intent"
        assert result.generation_route == "deterministic"
        assert result.generator_type is None

    def test_generation_intent_without_route_demoted_to_fallback(self):
        bad = UnifiedClassification(
            intent="generation_intent",
            generation_route=None,
            reason="bug",
        )
        result = classify_message(
            _make_request("do something"), llm_provider=_FakeProvider(bad),
        )
        assert result.intent == "fallback_intent"

    def test_every_intent_name_accepted_by_schema(self):
        intents = [
            "hello_intent", "create_complete_system_intent",
            "modify_model_intent", "modeling_help_intent",
            "describe_model_intent", "uml_spec_intent",
            "generation_intent", "workflow_intent", "fallback_intent",
        ]
        for name in intents:
            cls = UnifiedClassification(
                intent=name,
                generation_route="smart" if name == "generation_intent" else None,
                refined_instructions="x" if name == "generation_intent" else None,
                reason="test",
            )
            assert cls.intent == name


class TestGetOrClassifyCache:
    def test_caches_result_by_event_id(self):
        session = _CacheableSession(event_id="evt-42")
        request = _make_request("build me a rails api")
        provider = _FakeProvider(UnifiedClassification(
            intent="generation_intent",
            generation_route="smart",
            refined_instructions="Build Rails.",
            reason="rails",
        ))

        first = get_or_classify(session, request, provider)
        second = get_or_classify(session, request, provider)

        assert provider.calls == 1
        assert first is second
        assert session.get(UNIFIED_CLASSIFICATION) is first
        assert session.get(UNIFIED_CLASSIFICATION_EVENT_ID) == "evt-42"

    def test_different_event_id_triggers_new_classification(self):
        session = _CacheableSession(event_id="evt-1")
        request = _make_request("build me a rails api")
        provider = _FakeProvider(UnifiedClassification(
            intent="generation_intent",
            generation_route="smart",
            refined_instructions="Build Rails.",
            reason="rails",
        ))

        get_or_classify(session, request, provider)
        session.event = _FakeEvent("evt-2")
        get_or_classify(session, request, provider)

        assert provider.calls == 2

    def test_no_event_id_safe(self):
        class _NoEventSession(FakeSession):
            event = None

        session = _NoEventSession()
        provider = _FakeProvider(UnifiedClassification(
            intent="hello_intent", reason="greeting",
        ))
        result = get_or_classify(session, _make_request("hello"), provider)
        assert result.intent == "hello_intent"


class TestSchemaContracts:
    """Pin the exact literal values so TypeScript frontend and backend stay in sync."""

    def test_generation_route_values(self):
        for v in ("smart", "deterministic", "modeling", "other"):
            cls = UnifiedClassification(
                intent="generation_intent",
                generation_route=v,
                refined_instructions="x" if v == "smart" else None,
                reason="test",
            )
            assert cls.generation_route == v

    def test_deterministic_generator_types(self):
        for v in (
            "django", "backend", "web_app", "sql", "sqlalchemy",
            "python", "java", "pydantic", "jsonschema", "smartdata",
            "agent", "qiskit", "rest_api", "rdf", "export", "deploy",
        ):
            cls = UnifiedClassification(
                intent="generation_intent",
                generation_route="deterministic",
                generator_type=v,
                reason="test",
            )
            assert cls.generator_type == v

    def test_target_diagram_types(self):
        for v in (
            "ClassDiagram", "ObjectDiagram", "StateMachineDiagram",
            "AgentDiagram", "GUINoCodeDiagram", "QuantumCircuitDiagram",
        ):
            cls = UnifiedClassification(
                intent="create_complete_system_intent",
                target_diagram_type=v,
                reason="test",
            )
            assert cls.target_diagram_type == v


class _RecordingProvider:
    """Fake provider that records the messages it was handed and returns a
    canned decision — lets us assert on the prompt / user block the
    classifier actually builds without a real LLM call."""

    def __init__(self, decision):
        self.decision = decision
        self.messages = None

    def parse(self, *, messages, schema, temperature, max_tokens):
        self.messages = messages
        return self.decision


class TestSmartGenFollowUpRouting:
    """Guardrails for the smart-gen feature-follow-up routing fix.

    After a smart / Vibe-Driven generation, a follow-up like 'add a
    authentification system to it' must route to generation_intent
    (smart), not modify_model_intent (which used to add a class to the
    diagram). These are deterministic checks on the prompt content and on
    a mocked-classification pass-through (the routing decision itself is
    LLM-based and not unit-testable here).
    """

    def test_system_prompt_documents_smart_gen_followup(self):
        prompt = _SYSTEM_PROMPT
        # The dedicated rule block exists...
        assert "SMART-GEN FOLLOW-UP" in prompt
        # ...names the recency cues the classifier sees in history...
        assert "Vibe-Driven Generator" in prompt
        assert "[smart-generation outcome]" in prompt
        # ...pins the intended routing target...
        assert "generation_route='smart'" in prompt
        assert "reuse_for_generation" in prompt
        # ...and calls out the exact production bug (UserAccountSystem class).
        assert "UserAccountSystem" in prompt

    def test_prompt_keeps_genuine_model_edits_on_modify(self):
        # The discriminator examples for real model edits must survive so
        # 'add a Payment class' etc. stay on modify_model_intent.
        prompt = _SYSTEM_PROMPT
        assert "add a Payment class" in prompt
        assert "stays modify_model_intent EVEN right" in prompt

    def test_recent_smart_gen_history_reaches_classifier(self):
        history = [
            {"role": "user", "content": "build me a webapp from my model"},
            {"role": "assistant",
             "content": "[smart-generation outcome] Smart generation "
                        "finished successfully."},
        ]
        decision = UnifiedClassification(
            intent="generation_intent",
            generation_route="smart",
            model_disposition="reuse_for_generation",
            refined_instructions="Add user authentication to the app.",
            reason="feature follow-up after smart gen",
        )
        provider = _RecordingProvider(decision)
        classify_message(
            _make_request("add a authentification system to it"),
            llm_provider=provider,
            history=history,
        )
        # System prompt is our authoritative rule set.
        assert provider.messages[0]["content"] == _SYSTEM_PROMPT
        user_block = provider.messages[1]["content"]
        # The smart-gen outcome + follow-up message are both visible to the
        # classifier so it can apply the SMART-GEN FOLLOW-UP rule.
        assert "Smart generation finished successfully" in user_block
        assert "add a authentification system to it" in user_block

    def test_smart_gen_followup_classification_passes_through(self):
        # A well-formed smart follow-up classification must survive
        # _post_validate verbatim (smart route + instructions present).
        decision = UnifiedClassification(
            intent="generation_intent",
            generation_route="smart",
            model_disposition="reuse_for_generation",
            refined_instructions="Add user authentication: login/signup.",
            reason="add auth to generated app",
        )
        result = classify_message(
            _make_request("add authentication to it"),
            llm_provider=_RecordingProvider(decision),
        )
        assert result.intent == "generation_intent"
        assert result.generation_route == "smart"
        assert result.model_disposition == "reuse_for_generation"
        assert result.refined_instructions.startswith("Add user authentication")
