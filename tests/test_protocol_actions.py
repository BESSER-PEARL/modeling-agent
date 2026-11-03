"""protocol.actions: the wire action names and the envelope builders.

The editor parses these JSON frames, so a builder must produce exactly the
dict literal it replaced: same keys, same key order, same values.
"""

import ast
import json
from pathlib import Path

import pytest

from common.errors import ErrorCode, build_error_response
from protocol.actions import (
    Action,
    assistant_message,
    inject_complete_system,
    inject_element,
    model_reply,
)

SRC = Path(__file__).resolve().parent.parent / "src"


def _same_bytes(built, literal):
    assert built == literal
    assert json.dumps(built) == json.dumps(literal)
    assert list(built) == list(literal)


def test_action_is_its_wire_string():
    assert Action.INJECT_COMPLETE_SYSTEM == "inject_complete_system"
    assert f"{Action.MODIFY_MODEL}" == str(Action.MODIFY_MODEL) == "modify_model"
    assert json.dumps({"action": Action.ASSISTANT_MESSAGE}) == '{"action": "assistant_message"}'
    assert Action.STREAM_DONE in {"stream_done"}


def test_action_covers_the_documented_wire_protocol():
    assert {a.value for a in Action} == {
        "user_message", "user_voice", "user_set_variable", "frontend_event", "replay_last_response",
        "inject_element", "inject_complete_system", "modify_model", "assistant_message", "agent_error",
        "create_diagram_tab", "trigger_generator", "trigger_smart_generator", "trigger_github_import",
        "trigger_export", "trigger_deploy", "auto_generate_gui",
        "progress", "stream_start", "stream_chunk", "stream_done",
        "agent_reply_str",
    }


def test_assistant_message_builder():
    _same_bytes(assistant_message("Hi"), {"action": "assistant_message", "message": "Hi"})
    actions = [{"label": "Confirm", "prompt": "confirm"}]
    _same_bytes(
        assistant_message("Sure?", suggestedActions=actions),
        {"action": "assistant_message", "message": "Sure?", "suggestedActions": actions},
    )
    _same_bytes(
        assistant_message("boom", isError=True),
        {"action": "assistant_message", "message": "boom", "isError": True},
    )


def test_inject_element_builder():
    element = {"className": "Book", "position": {"x": 0, "y": 0}}
    _same_bytes(
        inject_element(element, "ClassDiagram", "Added **Book**."),
        {"action": "inject_element", "element": element, "diagramType": "ClassDiagram",
         "message": "Added **Book**."},
    )


def test_inject_complete_system_builder():
    spec = {"systemName": "Library", "classes": [], "relationships": []}
    _same_bytes(
        inject_complete_system(spec, "ClassDiagram", "Built it."),
        {"action": "inject_complete_system", "systemSpec": spec, "diagramType": "ClassDiagram",
         "message": "Built it."},
    )
    _same_bytes(
        inject_complete_system(spec, "ClassDiagram", "Built it.", replaceExisting=True),
        {"action": "inject_complete_system", "systemSpec": spec, "diagramType": "ClassDiagram",
         "message": "Built it.", "replaceExisting": True},
    )


@pytest.mark.parametrize("action", [Action.INJECT_ELEMENT, Action.INJECT_COMPLETE_SYSTEM, Action.MODIFY_MODEL])
def test_model_reply_builder(action):
    model = {"cols": [["H"]], "qubitCount": 1}
    _same_bytes(
        model_reply(action, "QuantumCircuitDiagram", model, "Done."),
        {"action": action.value, "diagramType": "QuantumCircuitDiagram", "model": model, "message": "Done."},
    )


def test_error_envelope_uses_the_wire_action():
    # common sits below protocol, so errors.py keeps its one literal; pin it here.
    assert build_error_response(ErrorCode.LLM_FAILURE)["action"] == Action.ASSISTANT_MESSAGE


# --- no wire-action literal outside protocol/ --------------------------------

# "modify_model" is also a state, intent and operation-mode name, so it is only
# a wire literal where it sits next to an "action" key, field or keyword.
_DUAL_USE = {"modify_model"}
# common is below protocol in the import layers and cannot import Action.
_ALLOWED = {"common/errors.py"}


def _is_action_expr(node) -> bool:
    """x["action"], x.get("action", ...), x.action, getattr(x, "action", ...)."""
    if isinstance(node, ast.Attribute):
        return node.attr == "action"
    if isinstance(node, ast.Subscript):
        return isinstance(node.slice, ast.Constant) and node.slice.value == "action"
    if isinstance(node, ast.Call):
        args = node.args
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            return bool(args) and isinstance(args[0], ast.Constant) and args[0].value == "action"
        if isinstance(node.func, ast.Name) and node.func.id == "getattr":
            return len(args) > 1 and isinstance(args[1], ast.Constant) and args[1].value == "action"
    return False


def _constants(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node
    elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        for elt in node.elts:
            yield from _constants(elt)


def _wire_literals(tree):
    """Yield (lineno, value) of every wire-action string literal."""
    wire = {a.value for a in Action}
    docstrings = {
        id(n.body[0].value) for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)
    }
    in_action_context = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "action":
                    in_action_context.update(id(c) for c in _constants(value))
        elif isinstance(node, ast.Compare):
            sides = [node.left, *node.comparators]
            if any(_is_action_expr(s) for s in sides):
                for side in sides:
                    in_action_context.update(id(c) for c in _constants(side))
        elif isinstance(node, ast.keyword) and node.arg == "action":
            in_action_context.update(id(c) for c in _constants(node.value))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "action":
            if node.value is not None:
                in_action_context.update(id(c) for c in _constants(node.value))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and node.value in wire) or id(node) in docstrings:
            continue
        if node.value in _DUAL_USE and id(node) not in in_action_context:
            continue
        yield node.lineno, node.value


def test_wire_actions_are_spelled_only_in_protocol():
    found = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel.startswith("protocol/") or rel in _ALLOWED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found += [f"{rel}:{line}: {value!r}" for line, value in _wire_literals(tree)]
    assert not found, "use protocol.actions.Action instead of:\n" + "\n".join(found)


def test_the_literal_scan_finds_literals():
    tree = ast.parse(
        'reply({"action": "modify_model"})\n'
        'if payload.get("action") in ("inject_element", "x"): pass\n'
        'request.action == "frontend_event"\n'
        'telemetry(session, "assistant_message")\n'
        'mode = "modify_model"\n'
    )
    assert sorted(_wire_literals(tree)) == [
        (1, "modify_model"), (2, "inject_element"), (3, "frontend_event"), (4, "assistant_message"),
    ]
