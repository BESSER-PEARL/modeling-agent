"""Regression tests for the modification-prompt caching refactor.

These guard the invariants that make OpenAI's automatic prompt caching
hit on the class-diagram modification path:

* The system message is byte-identical across calls (any per-request
  content lives in the *user* message instead).
* The class-diagram system prompt is large enough to clear the
  ~1024-token cache threshold.
* Every action declared in a handler's system prompt has a matching
  Pydantic schema entry, so adding an action without updating both
  surfaces fails CI.
"""

import re
import sys
from pathlib import Path

import pytest

# Make src/ importable when running pytest from the repo root.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diagram_handlers.types.class_diagram_handler import (  # noqa: E402
    MODIFY_SYSTEM_PROMPT_CLASS,
    ClassDiagramHandler,
)
from diagram_handlers.types.agent_diagram_handler import (  # noqa: E402
    MODIFY_SYSTEM_PROMPT_AGENT,
)
from diagram_handlers.types.object_diagram_handler import (  # noqa: E402
    MODIFY_SYSTEM_PROMPT_OBJECT,
)
from diagram_handlers.types.state_machine_handler import (  # noqa: E402
    MODIFY_SYSTEM_PROMPT_STATE_MACHINE,
)
from schemas import ClassModification  # noqa: E402


# ---------------------------------------------------------------------------
# Prefix stability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "system_prompt",
    [
        MODIFY_SYSTEM_PROMPT_CLASS,
        MODIFY_SYSTEM_PROMPT_AGENT,
        MODIFY_SYSTEM_PROMPT_OBJECT,
        MODIFY_SYSTEM_PROMPT_STATE_MACHINE,
    ],
    ids=["class", "agent", "object", "state_machine"],
)
def test_system_prompt_is_module_level_constant(system_prompt):
    """The system prompt must be a static string built at import time, not
    rebuilt from per-request inputs. Any non-deterministic content (model
    summary, user request, timestamps, ids) belongs in the user message.
    """
    assert isinstance(system_prompt, str)
    assert system_prompt.strip(), "system prompt is empty"
    # Sentinels for things that would silently break cache hits.
    forbidden = [
        # Anything that looks like a UUID hex run; conservative regex.
        (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}"), "uuid-shaped string"),
        # ISO-ish timestamps.
        (re.compile(r"\b20\d{2}-[01]\d-[0-3]\d[T ][0-2]\d:[0-5]\d"), "timestamp"),
    ]
    for pat, label in forbidden:
        assert not pat.search(system_prompt), (
            f"system prompt contains a {label}; it must stay byte-identical "
            f"across calls for OpenAI prompt caching to hit"
        )


def test_class_modification_prompt_clears_cache_threshold():
    """OpenAI automatic prompt caching only kicks in for prefixes >=1024
    tokens. The class-diagram modification path is the hot one — keep its
    system prompt above the threshold so subsequent turns hit the cache.
    """
    # ~4 chars per token is the standard OpenAI rule of thumb.
    approx_tokens = len(MODIFY_SYSTEM_PROMPT_CLASS) // 4
    assert approx_tokens >= 1024, (
        f"class-diagram system prompt is only ~{approx_tokens} tokens; "
        f"OpenAI's prompt cache requires a >=1024-token prefix to activate"
    )


# ---------------------------------------------------------------------------
# Schema/prompt action coverage
# ---------------------------------------------------------------------------


def _action_literals_from_schema():
    """Extract the union of action literals from ClassModification.action.

    Pydantic stores ``Literal["add_class", ...]`` as the type annotation;
    walk it via ``typing.get_args`` of the field's annotation.
    """
    import typing
    annotation = ClassModification.model_fields["action"].annotation
    return set(typing.get_args(annotation))


def test_class_prompt_mentions_every_schema_action():
    """Every action listed in the class-diagram Pydantic schema must be
    referenced somewhere in the system prompt. If the schema gains a new
    action but the prompt isn't updated (or vice-versa), the LLM either
    refuses to emit the action or emits an action the validator rejects.
    """
    schema_actions = _action_literals_from_schema()
    missing = [a for a in schema_actions if a not in MODIFY_SYSTEM_PROMPT_CLASS]
    assert not missing, (
        f"actions {missing} are declared in ClassModification.action but "
        f"never mentioned in MODIFY_SYSTEM_PROMPT_CLASS — the LLM has no "
        f"way to know about them"
    )


# ---------------------------------------------------------------------------
# generate_modification call shape
# ---------------------------------------------------------------------------


def test_class_handler_passes_system_prompt_separately(monkeypatch):
    """class_diagram_handler used to concatenate system+user into a single
    full_prompt and pass "" as system_prompt. After the refactor, the
    system message must be MODIFY_SYSTEM_PROMPT_CLASS and the user
    message must NOT contain it (otherwise the prefix doesn't dedupe
    across calls and caching never hits).
    """
    captured = {}

    def fake_execute(self, user_prompt, system_prompt, schema_cls, **kwargs):
        captured["user"] = user_prompt
        captured["system"] = system_prompt
        return {"modifications": []}

    monkeypatch.setattr(ClassDiagramHandler, "_execute_modification", fake_execute)

    handler = ClassDiagramHandler.__new__(ClassDiagramHandler)
    handler.generate_modification("rename User to Customer", current_model=None)

    assert captured["system"] == MODIFY_SYSTEM_PROMPT_CLASS
    # The user message carries the request and (when present) the model
    # summary, never the system prompt itself.
    assert "rename User to Customer" in captured["user"]
    assert MODIFY_SYSTEM_PROMPT_CLASS not in captured["user"], (
        "user message contains the system prompt — the cache prefix won't "
        "dedupe across calls"
    )


def test_class_handler_system_prompt_stable_across_model_summaries(monkeypatch):
    """Two calls with different current_model payloads must produce the
    same system message — only the user message may differ.
    """
    captured: list = []

    def fake_execute(self, user_prompt, system_prompt, schema_cls, **kwargs):
        captured.append({"user": user_prompt, "system": system_prompt})
        return {"modifications": []}

    monkeypatch.setattr(ClassDiagramHandler, "_execute_modification", fake_execute)

    handler = ClassDiagramHandler.__new__(ClassDiagramHandler)

    model_a = {
        "elements": {"c1": {"type": "Class", "name": "User", "attributes": []}},
        "relationships": {},
    }
    model_b = {
        "elements": {
            "c2": {"type": "Class", "name": "Order", "attributes": []},
            "c3": {"type": "Class", "name": "Product", "attributes": []},
        },
        "relationships": {},
    }

    handler.generate_modification("add an attribute", current_model=model_a)
    handler.generate_modification("add an attribute", current_model=model_b)

    assert len(captured) == 2
    assert captured[0]["system"] == captured[1]["system"]
    # User messages should differ since the model summary differs.
    assert captured[0]["user"] != captured[1]["user"]
