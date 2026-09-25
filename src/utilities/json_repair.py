"""Tolerant JSON parsing for LLM answers.

Models writing a regex into a JSON string often forget to double its
backslashes (``\\.`` instead of ``\\\\.``), which JSON rejects as an invalid
escape, which would discard the whole answer over one character. A failed
parse is retried once with those backslashes doubled, i.e. taken literally, as
the model meant them. Text that already parses is never modified.

Models answering in JSON mode (Anthropic, Mistral, Nebius, Qwen) also send a
list or object field as a JSON string now and then; ``coerce_to_model``
decodes those, guided by the Pydantic field types.
"""

from __future__ import annotations

import json
import types
import typing
from typing import Any

from pydantic import BaseModel

_VALID_ESCAPES = set('"\\/bfnrt')
_HEX = set("0123456789abcdefABCDEF")


def repair_invalid_escapes(text: str) -> str:
    """Double every backslash inside a JSON string that does not start a valid
    escape. Backslashes outside strings and valid escapes are left alone."""
    out: list[str] = []
    in_string = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if not in_string:
            if ch == '"':
                in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == '"':
            in_string = False
            out.append(ch)
            i += 1
        elif ch == "\\":
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt in _VALID_ESCAPES and nxt:
                out.append(text[i:i + 2])
                i += 2
            elif nxt == "u" and len(text) >= i + 6 and all(c in _HEX for c in text[i + 2:i + 6]):
                out.append(text[i:i + 6])
                i += 6
            else:
                out.append("\\\\")
                i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def loads_tolerant(text: str) -> Any:
    """``json.loads``, retried once with invalid escapes repaired.

    Raises the ORIGINAL error when the repair does not produce valid JSON, so
    callers see the same failure as before.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as original:
        repaired = repair_invalid_escapes(text)
        if repaired == text:
            raise
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            raise original from None


def _options(annotation: Any) -> list[Any]:
    """The alternatives of ``Optional``/``Union``/``Annotated``, flattened."""
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _options(typing.get_args(annotation)[0])
    if origin in (typing.Union, types.UnionType):
        return [o for arg in typing.get_args(annotation) for o in _options(arg)]
    return [annotation]


def _container(option: Any) -> type | None:
    """``list``/``dict`` when ``option`` only accepts that JSON shape, else None."""
    origin = typing.get_origin(option) or option
    if origin in (list, tuple, set):
        return list
    if origin is dict or (isinstance(origin, type) and issubclass(origin, BaseModel)):
        return dict
    return None


def _coerce(value: Any, annotation: Any, key: str | None) -> Any:
    options = [o for o in _options(annotation) if o is not type(None)]
    shapes = [_container(o) for o in options]
    # Decode only where no alternative would accept the string itself.
    if isinstance(value, str) and shapes and all(shapes):
        try:
            decoded = loads_tolerant(value)
        except ValueError:
            decoded = None
        if (list in shapes and isinstance(decoded, dict) and len(decoded) == 1
                and isinstance(decoded.get(key), list)):
            decoded = decoded[key]
        if any(isinstance(decoded, shape) for shape in shapes):
            value = decoded
    for option in options:
        origin = typing.get_origin(option) or option
        if isinstance(value, dict) and isinstance(origin, type) and issubclass(origin, BaseModel):
            return coerce_to_model(value, origin)
        if isinstance(value, list) and origin in (list, tuple, set) and typing.get_args(option):
            item = typing.get_args(option)[0]
            return [_coerce(v, item, None) for v in value]
    return value


def coerce_to_model(data: Any, model: type[BaseModel]) -> Any:
    """``data`` with stringified lists/objects decoded wherever ``model``'s
    fields require one, at every depth; everything else is left unchanged."""
    if not isinstance(data, dict):
        return data
    return {
        key: _coerce(value, model.model_fields[key].annotation, key) if key in model.model_fields else value
        for key, value in data.items()
    }


def validate_llm_json(schema: type[BaseModel], text: str) -> BaseModel:
    """Validate a JSON-mode answer against ``schema``: strictly first, then
    with invalid escapes repaired and stringified fields decoded."""
    try:
        return schema.model_validate_json(text)
    except ValueError:
        return schema.model_validate(coerce_to_model(loads_tolerant(text), schema))
