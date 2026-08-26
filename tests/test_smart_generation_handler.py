"""Tests for generation sub-routing + the smart-gen payload.

The legacy generation-only classifier (``classify_generation_request``
and its second, drifted prompt) is retired: generation sub-routing now
comes from the UNIFIED classifier's single per-message call, adapted via
``generation_handler._classification_to_legacy``. These tests pin:

  * handle_generation_request dispatches on the adapted ``route`` field
    (fakes stub the unified classifier's structured output).
  * The dispatch never crashes when no LLM provider is available.
  * build_trigger_smart_generator_payload requires ``route='smart'``.
"""

import pytest

from handlers.generation_handler import (
    handle_generation_request,
    should_route_to_generation,
)
from handlers.smart_generation_handler import (
    GenerationClassification,
    build_trigger_smart_generator_payload,
)
from unified_classifier import UnifiedClassification
from protocol.types import AssistantRequest, WorkspaceContext
from session_keys import PENDING_GENERATOR_TYPE

from tests.conftest import FakeSession


_CLASS_MODEL = {
    "elements": {"class-1": {"type": "Class", "name": "Book"}},
    "relationships": {},
}


def _make_request(message: str) -> AssistantRequest:
    return AssistantRequest(
        message=message,
        context=WorkspaceContext(
            active_diagram_type="ClassDiagram",
            active_model=_CLASS_MODEL,
            project_snapshot={
                "name": "LibraryProject",
                "diagrams": {"ClassDiagram": [{"model": _CLASS_MODEL}]},
            },
            diagram_summaries=[
                {"type": "ClassDiagram", "title": "Library", "elementCount": 4},
            ],
        ),
    )


class _FakeProvider:
    """Minimal LLMProvider stub matching the ``.parse(...)`` signature."""

    def __init__(self, decision, raise_on_parse=False):
        self.decision = decision
        self.raise_on_parse = raise_on_parse
        self.calls = []

    def parse(self, *, messages, schema, temperature, max_tokens):
        self.calls.append({
            "messages": messages,
            "schema": schema.__name__,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        if self.raise_on_parse:
            raise RuntimeError("simulated provider failure")
        return self.decision


def _wrap(decision: GenerationClassification) -> UnifiedClassification:
    """Adapt a legacy-shaped stub decision into the unified classifier's
    structured output (what a real ``provider.parse`` now returns)."""
    return UnifiedClassification(
        intent="generation_intent",
        generation_route=decision.route,
        generator_type=decision.generator_type,
        refined_instructions=decision.refined_instructions,
        provider=decision.provider,
        reason=decision.reason,
    )


# ---------------------------------------------------------------------
# Classification fallback safety (cache empty, no provider)
# ---------------------------------------------------------------------


class TestClassificationFallback:
    def test_no_provider_returns_safe_route(self, monkeypatch):
        """With an empty cache and no LLM provider, the sub-router must
        return a safe classification (menu fallback), never crash."""
        import handlers.generation_handler as gen_mod
        from handlers.generation_handler import _get_classification_from_cache_or_classify
        monkeypatch.setattr(gen_mod, "_get_llm_provider", None, raising=False)
        session = FakeSession()
        result = _get_classification_from_cache_or_classify(
            session, _make_request("generate django"))
        assert result.route in ("deterministic", "smart", "modeling", "other")

    def test_provider_exception_returns_safe_route(self, monkeypatch):
        import handlers.generation_handler as gen_mod
        from handlers.generation_handler import _get_classification_from_cache_or_classify
        fake = _FakeProvider(None, raise_on_parse=True)
        monkeypatch.setattr(gen_mod, "_get_llm_provider", lambda: fake, raising=False)
        session = FakeSession()
        result = _get_classification_from_cache_or_classify(
            session, _make_request("build me a rails api"))
        assert result.route in ("deterministic", "smart", "modeling", "other")


# ---------------------------------------------------------------------
# build_trigger_smart_generator_payload
# ---------------------------------------------------------------------


class TestBuildPayload:
    def test_payload_shape(self):
        cls = GenerationClassification(
            route="smart",
            refined_instructions="Build a FastAPI backend with JWT + PostgreSQL",
            provider="anthropic",
            reason="matched fastapi stack",
        )
        payload = build_trigger_smart_generator_payload(cls)
        assert payload["action"] == "trigger_smart_generator"
        assert payload["provider"] == "anthropic"
        assert payload["llmModel"] == "claude-sonnet-4-6"
        assert payload["instructions"].startswith("Build a FastAPI")
        assert "message" in payload

    def test_payload_openai_provider_uses_openai_default_model(self):
        cls = GenerationClassification(
            route="smart",
            refined_instructions="…",
            provider="openai",
            reason="…",
        )
        payload = build_trigger_smart_generator_payload(cls)
        assert payload["provider"] == "openai"
        assert payload["llmModel"] == "gpt-4o"

    def test_payload_has_no_api_key_fields(self):
        cls = GenerationClassification(
            route="smart", refined_instructions="…", reason="…",
        )
        payload = build_trigger_smart_generator_payload(cls)
        assert "api_key" not in payload
        assert "apiKey" not in payload

    def test_raises_on_non_smart_classification(self):
        cls = GenerationClassification(
            route="deterministic", generator_type="django", reason="…",
        )
        with pytest.raises(ValueError):
            build_trigger_smart_generator_payload(cls)

    def test_raises_on_empty_instructions(self):
        cls = GenerationClassification(
            route="smart", refined_instructions="   ", reason="…",
        )
        with pytest.raises(ValueError):
            build_trigger_smart_generator_payload(cls)


# ---------------------------------------------------------------------
# handle_generation_request — integration with the classifier
# ---------------------------------------------------------------------


class TestHandleGenerationRequest:
    def _patch_provider(self, monkeypatch, decision):
        """Patch ``_get_llm_provider`` to return a fake provider whose
        ``parse`` yields the unified classifier's structured output."""
        import handlers.generation_handler as gen_mod
        fake = _FakeProvider(_wrap(decision))
        monkeypatch.setattr(gen_mod, "_get_llm_provider", lambda: fake, raising=False)
        return fake

    def test_smart_route_asks_for_confirmation_then_fires_on_confirm(self, monkeypatch):
        """The smart route never auto-fires (it spends the user's own API
        key — B-2): it stashes the payload and asks; the trigger is only
        emitted after the explicit confirm phrase."""
        self._patch_provider(monkeypatch, GenerationClassification(
            route="smart",
            refined_instructions="Build a Rails 7 API for the Library domain.",
            provider="anthropic",
            reason="user named rails",
        ))
        session = FakeSession()
        request = _make_request("build me a rails api")
        result = handle_generation_request(session, request)
        # Gate: assistant_message with run/cancel quick actions, stash set.
        assert result["action"] == "assistant_message"
        assert "API key" in result["message"]
        prompts = [a["prompt"] for a in result["suggestedActions"]]
        assert "generate anyway with my current model" in prompts
        # Cancel button removed (product decision) — only Run is offered;
        # the user can still cancel by typing.
        assert "cancel the generation" not in prompts
        from session_keys import PENDING_SMART_GEN_INSTRUCTIONS
        assert "Rails" in session.get(PENDING_SMART_GEN_INSTRUCTIONS)

        # Confirm: now (and only now) the trigger payload is emitted.
        confirm = handle_generation_request(
            session, _make_request("generate anyway with my current model"),
        )
        assert confirm["action"] == "trigger_smart_generator"
        assert "Rails" in confirm["instructions"]

    def test_deterministic_with_type_falls_through_to_config_flow(self, monkeypatch):
        self._patch_provider(monkeypatch, GenerationClassification(
            route="deterministic",
            generator_type="pydantic",
            reason="user said pydantic",
        ))
        session = FakeSession()
        request = _make_request("generate pydantic classes")
        result = handle_generation_request(session, request)
        assert result["action"] == "trigger_generator"
        assert result["generatorType"] == "pydantic"

    def test_deterministic_without_type_shows_menu(self, monkeypatch):
        self._patch_provider(monkeypatch, GenerationClassification(
            route="deterministic",
            generator_type=None,
            reason="unclear which generator",
        ))
        session = FakeSession()
        request = _make_request("generate stuff")
        result = handle_generation_request(session, request)
        assert result["action"] == "assistant_message"
        assert "available options" in result["message"].lower()

    def test_modeling_route_builds_the_model(self, monkeypatch):
        """A ``route='modeling'`` verdict now BUILDS the model inline instead
        of bouncing with a 'rephrase' message (returns None, reply sent)."""
        self._patch_provider(monkeypatch, GenerationClassification(
            route="modeling",
            reason="user asked for a class diagram",
        ))
        import execution
        called = {}
        monkeypatch.setattr(
            execution, "execute_planned_operations",
            lambda **kw: called.update(kw) or None,
            raising=False,
        )
        session = FakeSession()
        request = _make_request("generate a class diagram for a library")
        result = handle_generation_request(session, request)
        assert result is None
        assert called.get("matched_intent") == "create_complete_system_intent"

    def test_other_route_returns_helpful_message(self, monkeypatch):
        self._patch_provider(monkeypatch, GenerationClassification(
            route="other",
            reason="small talk",
        ))
        session = FakeSession()
        request = _make_request("hello how are you")
        result = handle_generation_request(session, request)
        assert result["action"] == "assistant_message"

    def test_pending_generator_bypasses_classifier(self, monkeypatch):
        """When a config prompt is in progress, skip the classifier — the
        next message is config input, not a new intent."""
        fake = self._patch_provider(monkeypatch, GenerationClassification(
            route="smart",  # would route to smart-gen if called
            refined_instructions="X",
            reason="smart",
        ))
        session = FakeSession()
        session.set(PENDING_GENERATOR_TYPE, "django")
        request = _make_request("my_project")  # user is answering "what project name?"
        result = handle_generation_request(session, request)
        assert result["action"] != "trigger_smart_generator"
        # Classifier must NOT have been called for the pending-flow message.
        assert len(fake.calls) == 0


# ---------------------------------------------------------------------
# should_route_to_generation — the simplified gatekeeper
# ---------------------------------------------------------------------


class TestShouldRouteToGeneration:
    """The gatekeeper is now stateless (no text heuristics, no LLM).
    It only checks: frontend_event OR pending_generator."""

    def test_frontend_event_always_routes(self):
        session = FakeSession()
        request = AssistantRequest(action="frontend_event", message="", raw_payload={})
        assert should_route_to_generation(session, request) is True

    def test_pending_generator_routes(self):
        session = FakeSession()
        session.set(PENDING_GENERATOR_TYPE, "django")
        request = _make_request("my_project")
        assert should_route_to_generation(session, request) is True

    def test_plain_generation_request_no_longer_routes_here(self):
        """The gatekeeper no longer runs text heuristics. A message
        like 'generate django' goes through BAF's intent classifier
        and its ``json_intent_matches`` transition — NOT through
        this gatekeeper."""
        session = FakeSession()
        request = _make_request("generate django")
        assert should_route_to_generation(session, request) is False

    def test_smart_gen_phrase_no_longer_routes_here(self):
        """Same as above — smart-gen detection also moved to the
        classifier inside ``generation_state``."""
        session = FakeSession()
        request = _make_request("build me a full-stack fastapi backend")
        assert should_route_to_generation(session, request) is False
