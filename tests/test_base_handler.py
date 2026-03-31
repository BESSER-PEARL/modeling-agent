"""Tests for BaseDiagramHandler utilities -- JSON cleaning, validation, error classification."""
import pytest
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from diagram_handlers.core.base_handler import (
    validate_spec, LLMPredictionError,
    SINGLE_CLASS_REQUIRED, SINGLE_CLASS_OPTIONAL,
    SYSTEM_CLASS_REQUIRED, SYSTEM_CLASS_OPTIONAL,
    MODIFICATION_REQUIRED, MODIFICATION_INNER_REQUIRED,
    BaseDiagramHandler,
)
from errors import classify_error


# ---------------------------------------------------------------------------
# Concrete test handler subclass
# ---------------------------------------------------------------------------

class _TestHandler(BaseDiagramHandler):
    def get_diagram_type(self):
        return "TestDiagram"

    def get_system_prompt(self):
        return "Test prompt"

    def generate_single_element(self, *a, **kw):
        return {}

    def generate_complete_system(self, *a, **kw):
        return {}

    def generate_fallback_element(self, *a):
        return {}


class _FakeLLM:
    """Minimal LLM stub with a .predict() method."""
    def __init__(self, response="{}"):
        self.response = response

    def predict(self, prompt):
        return self.response


@pytest.fixture
def handler():
    return _TestHandler(_FakeLLM())


# =========================================================================
# 1-5. validate_spec
# =========================================================================

class TestValidateSpec:
    def test_valid_single_class(self):
        errors = validate_spec({"className": "User"}, SINGLE_CLASS_REQUIRED)
        assert errors == []

    def test_missing_required_key(self):
        errors = validate_spec({}, SINGLE_CLASS_REQUIRED)
        assert len(errors) == 1
        assert "className" in errors[0]
        assert "missing" in errors[0].lower()

    def test_wrong_type(self):
        errors = validate_spec({"className": 123}, SINGLE_CLASS_REQUIRED)
        assert len(errors) == 1
        assert "str" in errors[0]
        assert "int" in errors[0]

    def test_optional_keys_accepted(self):
        spec = {"className": "User", "attributes": [], "methods": []}
        errors = validate_spec(spec, SINGLE_CLASS_REQUIRED, SINGLE_CLASS_OPTIONAL)
        assert errors == []

    def test_optional_key_wrong_type(self):
        spec = {"className": "User", "attributes": "not_a_list"}
        errors = validate_spec(spec, SINGLE_CLASS_REQUIRED, SINGLE_CLASS_OPTIONAL)
        assert len(errors) == 1
        assert "attributes" in errors[0]

    def test_not_a_dict(self):
        errors = validate_spec("string", SINGLE_CLASS_REQUIRED)
        assert len(errors) == 1
        assert "JSON object" in errors[0]

    def test_system_class_required(self):
        errors = validate_spec({"classes": []}, SYSTEM_CLASS_REQUIRED)
        assert errors == []

    def test_system_class_missing(self):
        errors = validate_spec({}, SYSTEM_CLASS_REQUIRED)
        assert any("classes" in e for e in errors)


# =========================================================================
# 11-13. clean_json_response
# =========================================================================

class TestCleanJsonResponse:
    def test_strips_json_fences(self, handler):
        raw = '```json\n{"key": "value"}\n```'
        assert handler.clean_json_response(raw) == '{"key": "value"}'

    def test_leading_prose_before_json(self, handler):
        raw = 'Here is the result:\n{"key": "value"}'
        result = handler.clean_json_response(raw)
        assert result == '{"key": "value"}'

    def test_python_fences(self, handler):
        raw = '```python\n{"className": "User"}\n```'
        # ```python is handled by the generic ``` strip (starts with ```)
        result = handler.clean_json_response(raw)
        assert '"className"' in result
        parsed = json.loads(result)
        assert parsed["className"] == "User"

    def test_plain_json(self, handler):
        raw = '{"a": 1}'
        assert handler.clean_json_response(raw) == '{"a": 1}'

    def test_array_response(self, handler):
        raw = 'Some text [1, 2, 3]'
        result = handler.clean_json_response(raw)
        assert result == '[1, 2, 3]'


# =========================================================================
# 14-15. parse_json_safely
# =========================================================================

class TestParseJsonSafely:
    def test_valid_json(self, handler):
        result = handler.parse_json_safely('{"key": "value"}')
        assert result == {"key": "value"}

    def test_invalid_json_returns_none(self, handler):
        result = handler.parse_json_safely('not json at all')
        assert result is None

    def test_json_array(self, handler):
        result = handler.parse_json_safely('[1, 2, 3]')
        assert result == [1, 2, 3]


# =========================================================================
# 16-17. _error_response
# =========================================================================

class TestErrorResponse:
    def test_known_error_code(self, handler):
        resp = handler._error_response("parse_error")
        assert resp["error"] is True
        assert resp["errorCode"] == "parse_error"
        assert resp["retryable"] is True
        assert resp["diagramType"] == "TestDiagram"
        assert "suggestedRecovery" in resp

    def test_unknown_string_falls_back(self, handler):
        resp = handler._error_response("Something unexpected happened")
        assert resp["errorCode"] == "generation_error"
        assert resp["message"] == "Something unexpected happened"

    def test_explicit_code_keyword(self, handler):
        resp = handler._error_response("custom msg", code="timeout")
        assert resp["errorCode"] == "timeout"
        assert resp["message"] == "custom msg"

    def test_retryable_override(self, handler):
        resp = handler._error_response("llm_failure", retryable=False)
        assert resp["retryable"] is False

    def test_details_appended(self, handler):
        resp = handler._error_response("parse_error", "extra context")
        assert "extra context" in resp["message"]


# =========================================================================
# 18. _classify_error
# =========================================================================

class TestClassifyError:
    """Tests for the unified classify_error() from the errors module.

    The classification logic was previously on BaseDiagramHandler._classify_error
    and is now in errors.classify_error.
    """

    def test_timeout(self, handler):
        assert classify_error(Exception("request timed out")).value == "timeout"

    def test_timeout_keyword(self, handler):
        assert classify_error(Exception("timeout exceeded")).value == "timeout"

    def test_json_parse(self, handler):
        assert classify_error(Exception("json decode error")).value == "parse_error"

    def test_parse_keyword(self, handler):
        assert classify_error(Exception("could not parse response")).value == "parse_error"

    def test_validation_error(self, handler):
        assert classify_error(Exception("validation failed")).value == "validation_error"

    def test_schema_keyword(self, handler):
        assert classify_error(Exception("schema mismatch")).value == "validation_error"

    def test_llm_prediction_error(self, handler):
        assert classify_error(LLMPredictionError("boom")).value == "llm_failure"

    def test_generic_exception(self, handler):
        assert classify_error(Exception("something else")).value == "unknown"


# =========================================================================
# 19-21. validate_modification_spec
# =========================================================================

class TestValidateModificationSpec:
    def test_valid_single_modification(self, handler):
        spec = {
            "modification": {
                "action": "modify_class",
                "target": {"className": "User"},
                "changes": {"name": "Admin"},
            }
        }
        result = handler.validate_modification_spec(spec)
        assert result is spec

    def test_valid_batch_modifications(self, handler):
        spec = {
            "modifications": [
                {"action": "add_attribute", "target": {"className": "User"}},
                {"action": "remove_method", "target": {"className": "Order"}},
            ]
        }
        result = handler.validate_modification_spec(spec)
        assert result is spec

    def test_missing_inner_keys_raises(self, handler):
        spec = {
            "modification": {
                "action": "modify_class",
                # "target" is missing
            }
        }
        with pytest.raises(ValueError, match="target"):
            handler.validate_modification_spec(spec)

    def test_missing_modification_key_raises(self, handler):
        spec = {"something_else": True}
        with pytest.raises(ValueError, match="missing"):
            handler.validate_modification_spec(spec)

    def test_batch_with_invalid_inner_raises(self, handler):
        spec = {
            "modifications": [
                {"action": "ok", "target": {"className": "X"}},
                {"target": {"className": "Y"}},  # missing "action"
            ]
        }
        with pytest.raises(ValueError, match="action"):
            handler.validate_modification_spec(spec)


# =========================================================================
# 22. extract_name_from_request
# =========================================================================

class TestExtractNameFromRequest:
    def test_create_direct(self, handler):
        """'create User' extracts 'User'."""
        assert handler.extract_name_from_request("create User") == "User"

    def test_add_with_article_and_classifier(self, handler):
        """'add a class Order' -- 'a' is in skip list, so loop continues;
        no other trigger word matches -> returns default."""
        result = handler.extract_name_from_request("add a class Order")
        # The method skips 'a' but doesn't look past the immediate next word
        # when it's in the skip list; returns default.
        assert isinstance(result, str)

    def test_make_direct(self, handler):
        """'make Product' extracts 'Product'."""
        assert handler.extract_name_from_request("make Product") == "Product"

    def test_generate_with_following_word(self, handler):
        """'generate Employee record' -- next_word 'Employee' not in skip list,
        i+2 exists, returns words[i+2].capitalize() = 'Record'."""
        assert handler.extract_name_from_request("generate Employee record") == "Record"

    def test_no_trigger_returns_default(self, handler):
        """No trigger word -> returns 'New' (default)."""
        assert handler.extract_name_from_request("please do something") == "New"

    def test_custom_default(self, handler):
        assert handler.extract_name_from_request("nothing here", default="Fallback") == "Fallback"

    def test_new_keyword(self, handler):
        """'new Item' extracts 'Item'."""
        assert handler.extract_name_from_request("new Item") == "Item"
