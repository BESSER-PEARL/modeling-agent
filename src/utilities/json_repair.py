"""Tolerant JSON parsing for LLM answers.

Models writing a regex into a JSON string often forget to double its
backslashes (``\\.`` instead of ``\\\\.``), which JSON rejects as an invalid
escape. The whole answer was then discarded over one character. Here a failed
parse is retried once with those backslashes doubled, i.e. taken literally, as
the model meant them. Text that already parses is never modified.
"""

from __future__ import annotations

import json
from typing import Any

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
