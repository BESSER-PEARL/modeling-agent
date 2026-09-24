"""Tolerant parsing of LLM JSON: a stray regex backslash must not discard a model.

Live (2026-09-23, Qwen3 8B on the hotel spec): two of three complete
class-diagram answers were rejected with "Invalid JSON: invalid escape at line
193 column 61". The OCL rules held the email / phone regexes written with single
backslashes (``\\.``, ``\\+``), which JSON does not allow inside strings. The
agent then fell back to an empty six-class skeleton. OpenAI keys are immune
(strict structured output); every JSON-mode provider (Anthropic, Mistral,
Nebius, the free tier) was exposed.

The repair runs only after ``json.loads`` has failed, so every answer that
parses today is untouched.
"""
import json
import os
import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utilities.json_repair import loads_tolerant, repair_invalid_escapes  # noqa: E402

EMAIL_OCL = r"context Person inv validEmail: self.email.matches('^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')"
PHONE_OCL = r"context Person inv validPhone: self.phone.matches('^\+?[0-9]{7,15}$')"


def _broken(value):
    """Serialise ``value`` the way a model that forgets to double regex
    backslashes would: json.dumps, then undo the escaping of backslashes."""
    return json.dumps(value).replace("\\\\", "\\")


# ---------------------------------------------------------------------------
# The live failure
# ---------------------------------------------------------------------------

def test_single_backslash_regexes_are_recovered_exactly():
    answer = {"name": "HotelBookingSystem", "ocl": [EMAIL_OCL, PHONE_OCL]}
    broken = _broken(answer)
    with pytest.raises(json.JSONDecodeError):
        json.loads(broken)

    assert loads_tolerant(broken) == answer


@pytest.mark.parametrize("escape", ["\\.", "\\+", "\\d", "\\w", "\\s", "\\(", "\\[", "\\$", "\\-", "\\e"])
def test_each_regex_escape_is_kept_literally(escape):
    assert loads_tolerant('{"re": "a' + escape + 'b"}') == {"re": "a" + escape + "b"}


# ---------------------------------------------------------------------------
# Valid JSON is never changed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    '{"a": "line\\nbreak", "b": "tab\\t", "c": "quote \\" inside"}',
    '{"path": "C:\\\\Users\\\\x", "u": "caf\\u00e9", "slash": "a\\/b"}',
    '{"re": "\\\\d+\\\\.\\\\d+"}',          # correctly doubled regex
    '["\\b\\f\\r", 1, 2.5, true, null, {"nested": ["x"]}]',
    '"just a string"', '42', '[]', '{}',
])
def test_valid_json_is_untouched(text):
    assert repair_invalid_escapes(text) == text
    assert loads_tolerant(text) == json.loads(text)


@settings(max_examples=2000, deadline=None)
@given(st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text(),
    lambda children: st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=8), children, max_size=4),
    max_leaves=20,
))
def test_any_valid_json_parses_exactly_as_json_loads(value):
    text = json.dumps(value)
    assert repair_invalid_escapes(text) == text
    assert loads_tolerant(text) == json.loads(text)


# A backslash followed by a character that is never a valid JSON escape.
_regexish = st.text(alphabet=st.sampled_from(list("abcXYZ019 .+*?()[]{}^$|-_@%")), max_size=12).map(
    lambda s: "".join("\\" + ch if ch in ".+*?()[]{}^$|-" else ch for ch in s))


@settings(max_examples=2000, deadline=None)
@given(st.dictionaries(st.text(alphabet="abcxyz", min_size=1, max_size=6), _regexish, min_size=1, max_size=5))
def test_any_single_backslash_regex_payload_is_recovered_exactly(value):
    assert loads_tolerant(_broken(value)) == value


# ---------------------------------------------------------------------------
# What must still fail
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("garbage", [
    "I will create a focused class specification for the Person class...",
    "{\n    private String id; // Unique identifier\n    private String firstName;\n}",
    '{"name": "x", }',                       # trailing comma: not an escape problem
    '{"name": "unterminated}',
    '{"a": 1} {"b": 2}',                      # extra data
])
def test_garbage_still_raises(garbage):
    with pytest.raises(json.JSONDecodeError):
        loads_tolerant(garbage)


def test_a_backslash_outside_any_string_is_not_touched():
    text = '{"a": 1, \\. "b": 2}'
    assert repair_invalid_escapes(text) == text
    with pytest.raises(json.JSONDecodeError):
        loads_tolerant(text)


def test_the_original_error_is_raised_when_repair_does_not_help():
    text = '{"re": "\\.", }'
    with pytest.raises(json.JSONDecodeError) as caught:
        loads_tolerant(text)
    with pytest.raises(json.JSONDecodeError) as original:
        json.loads(text)
    assert str(caught.value) == str(original.value)


# ---------------------------------------------------------------------------
# End to end through the handler's JSON-mode structured path
# ---------------------------------------------------------------------------

HOTEL_ANSWER = {
    "name": "HotelBookingSystem",
    "classes": [
        {"n": "Person", "a": ["personId: int!", "firstName: str", "email: str", "phone: str"], "m": [], "k": "abstract"},
        {"n": "Guest", "a": [], "m": [], "k": ""},
        {"n": "Room", "a": ["roomNumber: int!", "capacity: int", "price: float"], "m": [], "k": ""},
        {"n": "Booking", "a": ["bookingNumber: int!", "arrival: date", "departure: date"],
         "m": ["cancel() -> bool"], "k": ""},
    ],
    "rels": [
        {"f": "Guest", "t": "Person", "k": "inher", "how_many_TARGET_for_one_SOURCE": "1",
         "how_many_SOURCE_for_one_TARGET": "1", "l": "person"},
        {"f": "Booking", "t": "Room", "k": "assoc", "how_many_TARGET_for_one_SOURCE": "1..*",
         "how_many_SOURCE_for_one_TARGET": "0..*", "l": "rooms", "ls": "bookings"},
    ],
    "ocl": [EMAIL_OCL, PHONE_OCL],
}


class _JsonModeLLM:
    """No OpenAI .parse() client, so predict_structured takes the JSON-mode
    fallback, like an Anthropic / Mistral / Nebius / free-tier key."""
    name = "fake-json-mode"

    def __init__(self, text):
        self.text = text

    def predict(self, prompt, **_):
        return self.text


def test_a_complete_model_survives_single_backslash_regexes():
    from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler
    from schemas.compact_class_diagram import CompactSystemClassSpec

    handler = ClassDiagramHandler(_JsonModeLLM(_broken(HOTEL_ANSWER)))
    spec = handler.predict_structured("hotel", CompactSystemClassSpec, max_retries=0)

    assert [c.n for c in spec.classes] == ["Person", "Guest", "Room", "Booking"]
    assert spec.ocl == [EMAIL_OCL, PHONE_OCL]


def test_the_classifier_json_path_survives_it_too(monkeypatch):
    from types import SimpleNamespace
    from pydantic import BaseModel

    import byok
    from llm.provider import LLMProvider

    class Rule(BaseModel):
        ocl: str

    stub = SimpleNamespace(provider="anthropic", openai_client=None,
                           predict_raw=lambda prompt, **kw: _broken({"ocl": EMAIL_OCL}))
    token = byok.set_current("anthropic", "sk-user", None)
    monkeypatch.setattr(byok, "get_active_client", lambda: stub)
    try:
        result = LLMProvider(object(), model_name="gpt-4o-mini").parse(
            [{"role": "user", "content": "x"}], schema=Rule)
    finally:
        byok.reset_current(token)
    assert result.ocl == EMAIL_OCL


def test_the_per_class_fallback_parser_survives_it_too():
    from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler

    handler = ClassDiagramHandler(_JsonModeLLM(""))
    assert handler.parse_json_safely(_broken({"ocl": [PHONE_OCL]})) == {"ocl": [PHONE_OCL]}


# ---------------------------------------------------------------------------
# Stringified lists/objects (Sonnet 5 sent them in 16 of 30 fresh tool calls)
# ---------------------------------------------------------------------------

def _stringified(answer, *keys):
    """``answer`` with the named top-level fields sent as JSON strings."""
    return json.dumps({k: json.dumps(v) if k in keys else v for k, v in answer.items()})


@pytest.mark.parametrize("keys", [("classes",), ("rels",), ("ocl",), ("classes", "rels", "ocl")])
def test_a_model_with_stringified_lists_survives_json_mode(keys):
    from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler
    from schemas.compact_class_diagram import CompactSystemClassSpec

    handler = ClassDiagramHandler(_JsonModeLLM(_stringified(HOTEL_ANSWER, *keys)))
    spec = handler.predict_structured("hotel", CompactSystemClassSpec, max_retries=0)

    assert [c.n for c in spec.classes] == ["Person", "Guest", "Room", "Booking"]
    assert spec.classes[0].a == HOTEL_ANSWER["classes"][0]["a"]
    assert spec.ocl == [EMAIL_OCL, PHONE_OCL]


def test_stringified_values_are_decoded_at_every_depth_and_when_rewrapped():
    from schemas.compact_class_diagram import CompactSystemClassSpec
    from utilities.json_repair import validate_llm_json

    classes = [dict(c, a=json.dumps(c["a"])) for c in HOTEL_ANSWER["classes"]]
    answer = dict(HOTEL_ANSWER, classes=json.dumps({"classes": classes}))
    spec = validate_llm_json(CompactSystemClassSpec, json.dumps(answer))
    assert spec.classes[2].a == ["roomNumber: int!", "capacity: int", "price: float"]


def test_a_string_field_that_looks_like_json_is_never_decoded():
    from schemas.compact_class_diagram import CompactSystemClassSpec
    from utilities.json_repair import coerce_to_model

    answer = dict(HOTEL_ANSWER, name='["NotAList"]', ocl=['["context x inv: true"]'],
                  classes=[dict(HOTEL_ANSWER["classes"][0], n='{"a": 1}', a=['["x: int"]'])])
    assert coerce_to_model(answer, CompactSystemClassSpec) == answer


@pytest.mark.parametrize("value", ["not json", '"a string"', '{"other": []}', "[1, 2"])
def test_a_string_that_is_not_the_declared_shape_still_fails_validation(value):
    from pydantic import ValidationError
    from schemas.compact_class_diagram import CompactSystemClassSpec
    from utilities.json_repair import validate_llm_json

    with pytest.raises(ValidationError):
        validate_llm_json(CompactSystemClassSpec, json.dumps(dict(HOTEL_ANSWER, classes=value)))


def test_the_classifier_json_path_decodes_stringified_lists(monkeypatch):
    from types import SimpleNamespace
    from typing import List, Optional
    from pydantic import BaseModel

    import byok
    from llm.provider import LLMProvider

    class Rules(BaseModel):
        ocl: Optional[List[str]] = None

    stub = SimpleNamespace(provider="anthropic", openai_client=None,
                           predict_raw=lambda prompt, **kw: json.dumps({"ocl": json.dumps([EMAIL_OCL])}))
    token = byok.set_current("anthropic", "sk-user", None)
    monkeypatch.setattr(byok, "get_active_client", lambda: stub)
    try:
        result = LLMProvider(object(), model_name="gpt-4o-mini").parse(
            [{"role": "user", "content": "x"}], schema=Rules)
    finally:
        byok.reset_current(token)
    assert result.ocl == [EMAIL_OCL]
