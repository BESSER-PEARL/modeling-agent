"""Tests for the unified LLM-based generation classifier.

The legacy keyword-based helpers (``should_route_to_smart_gen``,
``refine_instructions``, ``_SMART_GEN_COMPLEX_PHRASES``,
``_COMPOUND_INTENT_RE``, ``_UNSUPPORTED_LANGUAGE_RE``,
``_DETERMINISTIC_ONLY_HEADS``) were deleted in favour of a single LLM
call via :func:`classify_generation_request`. These tests pin the new
contract:

  * classify_generation_request returns a GenerationClassification
    regardless of LLM availability (falls back safely).
  * handle_generation_request dispatches on the classification's
    ``route`` field.
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
    classify_generation_request,
)
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


# ---------------------------------------------------------------------
# classify_generation_request
# ---------------------------------------------------------------------


class TestClassifyGenerationRequest:
    def test_returns_safe_fallback_when_provider_none(self):
        request = _make_request("generate django")
        result = classify_generation_request(request, llm_provider=None)
        assert isinstance(result, GenerationClassification)
        assert result.route == "deterministic"
        assert result.generator_type is None

    def test_returns_safe_fallback_on_llm_exception(self):
        request = _make_request("build me a rails api")
        provider = _FakeProvider(None, raise_on_parse=True)
        result = classify_generation_request(request, llm_provider=provider)
        assert result.route == "deterministic"
        assert result.generator_type is None
        assert "failed" in result.reason.lower() or "unavailable" in result.reason.lower()

    def test_smart_route_returned_verbatim(self):
        request = _make_request("build me a rails api")
        expected = GenerationClassification(
            route="smart",
            refined_instructions="Build a Rails 7 API for the Library domain…",
            provider="anthropic",
            reason="user named Rails which is not a BESSER built-in",
        )
        provider = _FakeProvider(expected)
        result = classify_generation_request(request, llm_provider=provider)
        assert result.route == "smart"
        assert result.refined_instructions.startswith("Build a Rails 7 API")

    def test_deterministic_route_returned_verbatim(self):
        request = _make_request("generate django")
        expected = GenerationClassification(
            route="deterministic",
            generator_type="django",
            reason="user explicitly said django",
        )
        provider = _FakeProvider(expected)
        result = classify_generation_request(request, llm_provider=provider)
        assert result.route == "deterministic"
        assert result.generator_type == "django"

    @pytest.mark.parametrize("generator_type", ["rest_api", "rdf"])
    def test_legacy_schema_accepts_deterministic_api_and_semantic_generators(
        self, generator_type,
    ):
        classification = GenerationClassification(
            route="deterministic",
            generator_type=generator_type,
            reason="BESSER built-in",
        )

        assert classification.generator_type == generator_type

    def test_smart_with_empty_instructions_demoted_to_deterministic_unknown(self):
        """If the LLM returns ``route='smart'`` but forgets to write
        instructions, treat it as a bug in the classifier and fall
        back to 'deterministic-unknown' so the caller shows the menu
        instead of raising downstream."""
        request = _make_request("do something")
        bad = GenerationClassification(
            route="smart",
            refined_instructions="",  # empty — invalid
            reason="classifier bug",
        )
        provider = _FakeProvider(bad)
        result = classify_generation_request(request, llm_provider=provider)
        assert result.route == "deterministic"
        assert result.generator_type is None

    def test_workspace_context_included_in_user_block(self):
        request = _make_request("generate code")
        expected = GenerationClassification(
            route="other", reason="chat"
        )
        provider = _FakeProvider(expected)
        classify_generation_request(request, llm_provider=provider)
        user_msg = provider.calls[0]["messages"][1]["content"]
        assert "ClassDiagram" in user_msg
        assert "Library" in user_msg

    def test_system_prompt_explains_extras_route_to_smart(self):
        """The classifier prompt MUST explain that a BESSER built-in +
        any extra feature (auth, JWT, Docker, migrations, …) routes to
        smart — not deterministic. This is the rule that catches cases
        like 'web app with authentication' where the deterministic
        web_app template can't produce the auth layer on its own.

        We pin this by asserting the prompt text contains the relevant
        guidance. If someone trims the prompt and loses this section,
        the LLM will start routing 'web_app with auth' to the
        deterministic path and the user gets scaffolding without auth.
        """
        from handlers.smart_generation_handler import _CLASSIFIER_SYSTEM_PROMPT
        prompt = _CLASSIFIER_SYSTEM_PROMPT.lower()
        # Must mention that deterministic = scaffolding only
        assert "scaffolding" in prompt or "baseline" in prompt
        # Must call out extras as smart triggers
        for feature in ["auth", "jwt", "docker", "migration"]:
            assert feature in prompt, (
                f"system prompt doesn't mention '{feature}' as a smart trigger"
            )
        # Must have an explicit "web_app with auth → smart" example
        assert "web app with authentication" in prompt or "web_app with oauth" in prompt


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
        """Patch ``_get_llm_provider`` to return a fake provider."""
        import handlers.generation_handler as gen_mod
        fake = _FakeProvider(decision)
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

    def test_modeling_route_redirects_to_modeling(self, monkeypatch):
        self._patch_provider(monkeypatch, GenerationClassification(
            route="modeling",
            reason="user asked for a class diagram",
        ))
        session = FakeSession()
        request = _make_request("generate a class diagram for a library")
        result = handle_generation_request(session, request)
        assert result["action"] == "assistant_message"
        assert "create a diagram" in result["message"].lower()

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
