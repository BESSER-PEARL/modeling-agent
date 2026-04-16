"""Tests for the unified message classifier."""

import pytest

from protocol.types import AssistantRequest, WorkspaceContext
from session_keys import (
    UNIFIED_CLASSIFICATION,
    UNIFIED_CLASSIFICATION_EVENT_ID,
)
from unified_classifier import (
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
