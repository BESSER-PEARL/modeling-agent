"""Tests for the generation handler (handlers/generation_handler.py)."""

import pytest
from handlers.generation_handler import (
    detect_generator_type,
    parse_inline_generator_config,
    _required_missing,
    _normalize_defaults,
    _build_config_prompt,
    _looks_like_mixed_modeling_and_generation,
    handle_generation_request,
    should_route_to_generation,
    GENERATOR_KEYWORDS,
)
from handlers.smart_generation_handler import GenerationClassification
from protocol.types import AssistantRequest, WorkspaceContext

from tests.conftest import FakeSession


def _make_request(message: str, action: str = "user_message") -> AssistantRequest:
    return AssistantRequest(
        action=action,
        message=message,
        context=WorkspaceContext(
            project_snapshot={"name": "TestProject", "diagrams": {}},
        ),
    )


class _FakeLLMProvider:
    """Minimal provider stub that returns a fixed classification."""

    def __init__(self, decision: GenerationClassification):
        self.decision = decision

    def parse(self, *, messages, schema, temperature, max_tokens):
        return self.decision


def _patch_classifier(monkeypatch, decision: GenerationClassification):
    """Make ``_get_llm_provider()`` return a stub that yields ``decision``."""
    import handlers.generation_handler as gen_mod
    monkeypatch.setattr(
        gen_mod, "_get_llm_provider",
        lambda: _FakeLLMProvider(decision),
        raising=False,
    )


# ---------------------------------------------------------------------------
# detect_generator_type
# ---------------------------------------------------------------------------

class TestDetectGeneratorType:
    def test_detects_django(self):
        assert detect_generator_type("generate django code") == "django"

    def test_detects_sql(self):
        assert detect_generator_type("generate sql schema") == "sql"

    def test_sqlalchemy_before_sql(self):
        """sqlalchemy keywords must be matched before plain sql."""
        assert detect_generator_type("generate sqlalchemy models") == "sqlalchemy"

    def test_web_app(self):
        assert detect_generator_type("generate a web app") == "web_app"

    def test_qiskit(self):
        assert detect_generator_type("generate qiskit code") == "qiskit"

    def test_none_for_unrelated(self):
        assert detect_generator_type("create a User class") is None

    def test_empty_string(self):
        assert detect_generator_type("") is None

    def test_backend(self):
        assert detect_generator_type("generate a full backend") == "backend"

    def test_smartdata(self):
        assert detect_generator_type("generate smart data output") == "smartdata"


# ---------------------------------------------------------------------------
# parse_inline_generator_config
# ---------------------------------------------------------------------------

class TestParseInlineConfig:
    def test_django_project_name(self):
        request = _make_request("generate django project_name=hotel_app")
        config = parse_inline_generator_config("django", request.message, request)
        assert config["project_name"] == "hotel_app"

    def test_sql_dialect(self):
        request = _make_request("generate sql with postgresql")
        config = parse_inline_generator_config("sql", request.message, request)
        assert config["dialect"] == "postgresql"

    def test_sqlalchemy_dbms(self):
        request = _make_request("generate sqlalchemy with mysql")
        config = parse_inline_generator_config("sqlalchemy", request.message, request)
        assert config["dbms"] == "mysql"

    def test_jsonschema_mode(self):
        request = _make_request("generate json schema smart")
        config = parse_inline_generator_config("jsonschema", request.message, request)
        assert config["mode"] == "smart_data"

    def test_qiskit_backend_and_shots(self):
        request = _make_request("generate qiskit backend=aer_simulator shots=2048")
        config = parse_inline_generator_config("qiskit", request.message, request)
        assert config["backend"] == "aer_simulator"
        assert config["shots"] == 2048

    def test_backend_framework(self):
        request = _make_request("generate backend with fastapi")
        config = parse_inline_generator_config("backend", request.message, request)
        assert config["framework"] == "fastapi"

    def test_smartdata_format(self):
        request = _make_request("generate smart data in rdf")
        config = parse_inline_generator_config("smartdata", request.message, request)
        assert config["output_format"] == "rdf"


# ---------------------------------------------------------------------------
# _required_missing
# ---------------------------------------------------------------------------

class TestRequiredMissing:
    def test_all_present(self):
        assert _required_missing("django", {
            "project_name": "p", "app_name": "a", "containerization": False
        }) == []

    def test_some_missing(self):
        missing = _required_missing("django", {"project_name": "p"})
        assert "app_name" in missing
        assert "containerization" in missing

    def test_no_required_fields(self):
        assert _required_missing("python", {}) == []

    def test_backend_no_required_fields(self):
        assert _required_missing("backend", {}) == []

    def test_smartdata_no_required_fields(self):
        assert _required_missing("smartdata", {}) == []


# ---------------------------------------------------------------------------
# _normalize_defaults
# ---------------------------------------------------------------------------

class TestNormalizeDefaults:
    def test_django_defaults(self):
        request = _make_request("")
        config = _normalize_defaults("django", request, {})
        assert config["project_name"] == "testproject"  # sanitized from TestProject
        assert "app_name" in config
        assert config["containerization"] is False

    def test_sql_defaults(self):
        request = _make_request("")
        config = _normalize_defaults("sql", request, {})
        assert config["dialect"] == "sqlite"

    def test_sqlalchemy_defaults(self):
        request = _make_request("")
        config = _normalize_defaults("sqlalchemy", request, {})
        assert config["dbms"] == "sqlite"

    def test_qiskit_defaults(self):
        request = _make_request("")
        config = _normalize_defaults("qiskit", request, {})
        assert config["backend"] == "aer_simulator"
        assert config["shots"] == 1024

    def test_backend_defaults(self):
        request = _make_request("")
        config = _normalize_defaults("backend", request, {})
        assert config["framework"] == "django"

    def test_smartdata_defaults(self):
        request = _make_request("")
        config = _normalize_defaults("smartdata", request, {})
        assert config["output_format"] == "json"


# ---------------------------------------------------------------------------
# _build_config_prompt
# ---------------------------------------------------------------------------

class TestBuildConfigPrompt:
    def test_django_prompt(self):
        prompt = _build_config_prompt("django", ["project_name"])
        assert "project_name" in prompt

    def test_backend_prompt(self):
        prompt = _build_config_prompt("backend", [])
        assert "framework" in prompt.lower() or "backend" in prompt.lower()

    def test_smartdata_prompt(self):
        prompt = _build_config_prompt("smartdata", [])
        assert "format" in prompt.lower() or "smartdata" in prompt.lower()


# ---------------------------------------------------------------------------
# _looks_like_mixed_modeling_and_generation
# ---------------------------------------------------------------------------

class TestLooksMixedModelingAndGeneration:
    def test_pure_generation(self):
        assert not _looks_like_mixed_modeling_and_generation("generate django code")

    def test_mixed(self):
        assert _looks_like_mixed_modeling_and_generation(
            "create a class diagram for a library and then generate django code"
        )

    def test_pure_modeling(self):
        assert not _looks_like_mixed_modeling_and_generation("create a class diagram")


# ---------------------------------------------------------------------------
# should_route_to_generation
# ---------------------------------------------------------------------------

class TestShouldRouteToGeneration:
    def test_frontend_event(self):
        request = _make_request("", action="frontend_event")
        session = FakeSession()
        assert should_route_to_generation(session, request) is True

    def test_pending_generator(self):
        request = _make_request("sqlite")
        session = FakeSession()
        session.set("pending_generator_type", "sql")
        assert should_route_to_generation(session, request) is True

    def test_generator_keyword_no_longer_routes_here(self):
        """The gatekeeper no longer runs text heuristics. BAF's intent
        classifier handles routing to ``generation_state`` for
        conversational messages — this gatekeeper only fires for
        frontend_event callbacks and mid-flow pending-generator state.
        """
        request = _make_request("generate django code")
        session = FakeSession()
        assert should_route_to_generation(session, request) is False

    def test_no_generation(self):
        request = _make_request("create a User class")
        session = FakeSession()
        assert should_route_to_generation(session, request) is False


# ---------------------------------------------------------------------------
# handle_generation_request (integration)
# ---------------------------------------------------------------------------

class TestHandleGenerationRequest:
    def test_trigger_with_defaults(self, monkeypatch):
        _patch_classifier(monkeypatch, GenerationClassification(
            route="deterministic", generator_type="python",
            reason="user said python classes",
        ))
        request = _make_request("generate python classes")
        session = FakeSession()
        result = handle_generation_request(session, request)
        assert result["action"] == "trigger_generator"
        assert result["generatorType"] == "python"

    def test_asks_for_missing_fields(self):
        request = _make_request("generate qiskit code")
        request.context.project_snapshot = None
        session = FakeSession()
        result = handle_generation_request(session, request)
        # Without a project snapshot the handler returns an assistant message
        # prompting the user (no model to generate from).
        assert result["action"] == "assistant_message"

    def test_unknown_generator(self):
        request = _make_request("do something")
        session = FakeSession()
        result = handle_generation_request(session, request)
        assert result["action"] == "assistant_message"
        msg = result["message"].lower()
        assert "available" in msg or "options" in msg or "supported" in msg or "tell me" in msg

    def test_frontend_event_result(self):
        request = _make_request("", action="frontend_event")
        request.raw_payload = {
            "eventType": "generator_result",
            "ok": True,
            "message": "Done!",
        }
        session = FakeSession()
        result = handle_generation_request(session, request)
        assert result["action"] == "assistant_message"
        assert "Done!" in result["message"]

    def test_backend_generator_trigger(self, monkeypatch):
        _patch_classifier(monkeypatch, GenerationClassification(
            route="deterministic", generator_type="backend",
            reason="user said backend",
        ))
        request = _make_request("generate a full backend")
        session = FakeSession()
        result = handle_generation_request(session, request)
        assert result["action"] == "trigger_generator"
        assert result["generatorType"] == "backend"
        assert result["config"]["framework"] == "django"

    def test_smartdata_generator_trigger(self, monkeypatch):
        _patch_classifier(monkeypatch, GenerationClassification(
            route="deterministic", generator_type="smartdata",
            reason="user said smart data",
        ))
        request = _make_request("generate smart data output")
        session = FakeSession()
        result = handle_generation_request(session, request)
        assert result["action"] == "trigger_generator"
        assert result["generatorType"] == "smartdata"
        assert result["config"]["output_format"] == "json"


# ---------------------------------------------------------------------------
# detect_generator_type — word boundary fixes
# ---------------------------------------------------------------------------

class TestDetectGeneratorTypeBoundary:
    def test_sql_does_not_match_sqlalchemy(self):
        """'sql' keyword with boundary check must not match inside 'sqlalchemy'."""
        # sqlalchemy should be detected as sqlalchemy, not sql
        assert detect_generator_type("generate sqlalchemy") == "sqlalchemy"

    def test_sql_standalone(self):
        assert detect_generator_type("generate sql") == "sql"

    def test_backend_standalone(self):
        assert detect_generator_type("generate backend code") == "backend"

    def test_full_backend(self):
        assert detect_generator_type("generate a full backend") == "backend"


# ---------------------------------------------------------------------------
# should_route_to_generation — diagram creation guard
# ---------------------------------------------------------------------------

class TestShouldRouteToGenerationDiagramGuard:
    def test_generate_class_diagram_not_routed(self):
        """'generate a class diagram' must NOT route to generation."""
        request = _make_request("generate a class diagram")
        session = FakeSession()
        assert should_route_to_generation(session, request) is False

    def test_generate_state_machine_not_routed(self):
        request = _make_request("generate a state machine for orders")
        session = FakeSession()
        assert should_route_to_generation(session, request) is False

    def test_generate_django_no_longer_routes_via_gatekeeper(self):
        """Removed in the LLM-classifier refactor: BAF's intent
        classifier routes this message to ``generation_state`` via the
        ``json_intent_matches`` transition, not via this gatekeeper."""
        request = _make_request("generate django code")
        session = FakeSession()
        assert should_route_to_generation(session, request) is False


# ---------------------------------------------------------------------------
# handle_generation_request — safety net for misrouted modeling requests
# ---------------------------------------------------------------------------

class TestHandleGenerationRequestSafetyNet:
    def test_modeling_request_redirected(self, monkeypatch):
        """If BAF misclassifies 'create a web app for hotel booking' as
        generation, the LLM classifier inside ``generation_state``
        should catch it and return ``route='modeling'`` — then the
        handler redirects instead of triggering web_app generator."""
        _patch_classifier(monkeypatch, GenerationClassification(
            route="modeling",
            reason="user wants a new diagram, not code",
        ))
        request = _make_request("create a web app for hotel booking")
        session = FakeSession()
        result = handle_generation_request(session, request)
        assert result["action"] == "assistant_message"
        assert "design" in result["message"].lower() or "create" in result["message"].lower()

    def test_diagram_creation_redirected(self, monkeypatch):
        """'generate a class diagram' landing in generation handler
        must be redirected via the LLM classifier's ``modeling`` route."""
        _patch_classifier(monkeypatch, GenerationClassification(
            route="modeling",
            reason="user wants a class diagram",
        ))
        request = _make_request("generate a class diagram for a library")
        session = FakeSession()
        result = handle_generation_request(session, request)
        assert result["action"] == "assistant_message"
        assert "diagram" in result["message"].lower() or "create" in result["message"].lower()

    def test_genuine_generation_not_blocked(self, monkeypatch):
        """Genuine 'generate python classes' still triggers the
        deterministic python generator via the LLM classifier."""
        _patch_classifier(monkeypatch, GenerationClassification(
            route="deterministic", generator_type="python",
            reason="user said python classes",
        ))
        request = _make_request("generate python classes")
        session = FakeSession()
        result = handle_generation_request(session, request)
        assert result["action"] == "trigger_generator"
        assert result["generatorType"] == "python"
