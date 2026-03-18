"""Tests for the diagram handler factory and base handler utilities."""

import json
import pytest
from diagram_handlers.registry.factory import DiagramHandlerFactory
from diagram_handlers.registry.metadata import get_diagram_type_info, DIAGRAM_TYPE_METADATA
from diagram_handlers.core.base_handler import (
    validate_spec,
    BaseDiagramHandler,
    LLMPredictionError,
    SINGLE_CLASS_REQUIRED,
    SINGLE_CLASS_OPTIONAL,
    SYSTEM_CLASS_REQUIRED,
    MODIFICATION_REQUIRED,
    MODIFICATION_INNER_REQUIRED,
)
from tests.conftest import FakeLLM


# ---------------------------------------------------------------------------
# validate_spec
# ---------------------------------------------------------------------------

class TestValidateSpec:
    def test_valid_spec(self):
        spec = {"className": "User", "attributes": []}
        errors = validate_spec(spec, SINGLE_CLASS_REQUIRED, SINGLE_CLASS_OPTIONAL)
        assert errors == []

    def test_missing_required(self):
        errors = validate_spec({}, SINGLE_CLASS_REQUIRED)
        assert len(errors) == 1
        assert "className" in errors[0]

    def test_wrong_type(self):
        errors = validate_spec({"className": 42}, SINGLE_CLASS_REQUIRED)
        assert len(errors) == 1
        assert "str" in errors[0]

    def test_non_dict_input(self):
        errors = validate_spec("not a dict", SINGLE_CLASS_REQUIRED)  # type: ignore[arg-type]
        assert len(errors) == 1

    def test_optional_wrong_type(self):
        spec = {"className": "Foo", "attributes": "not a list"}
        errors = validate_spec(spec, SINGLE_CLASS_REQUIRED, SINGLE_CLASS_OPTIONAL)
        assert any("attributes" in e for e in errors)

    def test_modification_spec(self):
        spec = {
            "modification": {
                "action": "add_attribute",
                "target": {"elementName": "User"},
                "changes": {"name": "email"},
            }
        }
        errors = validate_spec(spec, MODIFICATION_REQUIRED)
        assert errors == []
        inner_errors = validate_spec(spec["modification"], MODIFICATION_INNER_REQUIRED)
        assert inner_errors == []


# ---------------------------------------------------------------------------
# DiagramHandlerFactory
# ---------------------------------------------------------------------------

class TestDiagramHandlerFactory:
    def test_all_supported_types(self):
        factory = DiagramHandlerFactory(FakeLLM())
        supported = factory.get_supported_types()
        expected = {
            "ClassDiagram", "ObjectDiagram", "StateMachineDiagram",
            "AgentDiagram", "GUINoCodeDiagram", "QuantumCircuitDiagram",
        }
        assert set(supported) == expected

    def test_is_supported(self):
        factory = DiagramHandlerFactory(FakeLLM())
        assert factory.is_supported("ClassDiagram")
        assert not factory.is_supported("FooDiagram")

    def test_get_handler_returns_correct_type(self):
        factory = DiagramHandlerFactory(FakeLLM())
        handler = factory.get_handler("ClassDiagram")
        assert handler is not None
        assert handler.get_diagram_type() == "ClassDiagram"

    def test_get_handler_unknown(self):
        factory = DiagramHandlerFactory(FakeLLM())
        assert factory.get_handler("NonExistent") is None


# ---------------------------------------------------------------------------
# get_diagram_type_info
# ---------------------------------------------------------------------------

class TestGetDiagramTypeInfo:
    def test_known_type(self):
        info = get_diagram_type_info("StateMachineDiagram")
        assert info["name"] == "State Machine Diagram"
        assert "state" in info["keywords"]

    def test_unknown_falls_back_to_class(self):
        info = get_diagram_type_info("FooDiagram")
        assert info["name"] == "Class Diagram"

    def test_all_metadata_entries_have_required_keys(self):
        for dt, meta in DIAGRAM_TYPE_METADATA.items():
            assert "name" in meta, f"{dt} missing 'name'"
            assert "description" in meta, f"{dt} missing 'description'"
            assert "keywords" in meta, f"{dt} missing 'keywords'"


# ---------------------------------------------------------------------------
# BaseDiagramHandler utilities
# ---------------------------------------------------------------------------

class TestBaseDiagramHandlerUtils:
    def _handler(self, response: str = '{"className": "Test"}') -> BaseDiagramHandler:
        """Create a concrete handler via the factory (ClassDiagram)."""
        factory = DiagramHandlerFactory(FakeLLM(response))
        return factory.get_handler("ClassDiagram")

    def test_clean_json_response_strips_fences(self):
        h = self._handler()
        raw = '```json\n{"a": 1}\n```'
        assert h.clean_json_response(raw) == '{"a": 1}'

    def test_clean_json_response_skips_prose(self):
        h = self._handler()
        raw = 'Here is your JSON: {"a": 1}'
        assert h.clean_json_response(raw) == '{"a": 1}'

    def test_parse_json_safely(self):
        h = self._handler()
        assert h.parse_json_safely('{"x": 1}') == {"x": 1}
        assert h.parse_json_safely("not json") is None

    def test_parse_and_validate_success(self):
        h = self._handler()
        result = h.parse_and_validate(
            '{"className": "Foo", "attributes": []}',
            SINGLE_CLASS_REQUIRED,
            SINGLE_CLASS_OPTIONAL,
        )
        assert result["className"] == "Foo"

    def test_parse_and_validate_raises_on_invalid(self):
        h = self._handler()
        with pytest.raises(ValueError, match="missing"):
            h.parse_and_validate("{}", SINGLE_CLASS_REQUIRED)

    def test_predict_with_retry_success(self):
        h = self._handler('{"ok": true}')
        result = h.predict_with_retry("test prompt")
        assert result == '{"ok": true}'

    def test_predict_with_retry_empty_then_success(self):
        llm = FakeLLM()
        llm.responses = ["", '{"ok": true}']
        factory = DiagramHandlerFactory(llm)
        h = factory.get_handler("ClassDiagram")
        result = h.predict_with_retry("test prompt", max_retries=1)
        assert result == '{"ok": true}'

    def test_predict_with_retry_all_fail(self):
        llm = FakeLLM()
        llm.responses = ["", ""]
        factory = DiagramHandlerFactory(llm)
        h = factory.get_handler("ClassDiagram")
        with pytest.raises(LLMPredictionError):
            h.predict_with_retry("test prompt", max_retries=1)

    def test_generate_uuid(self):
        h = self._handler()
        uid = h.generate_uuid()
        assert len(uid) == 36  # UUID format
        assert uid.count("-") == 4
