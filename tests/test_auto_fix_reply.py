"""An editor repair request that changes nothing says the diagram is unchanged.

After a model fails backend validation the editor sends one ``[auto-fix]``
request listing the issues (BESSER-PEARL/BESSER#744). Live on
editor.besser-pearl.org (2026-11-09) the agent answered it with "I couldn't
process that modification. Please try again or rephrase your request." (1 of
2 runs) and "I couldn't find a **name wall?** field to remove." (2 of 8): the
model wrote junk into ``target.attributeName`` (3 of 8 timed runs), and two
runs took ~60 s instead of ~4 s, the signature of a run-on generation. The
user never wrote that message, so "rephrase your request" is wrong and junk
must not be quoted. The editor's own status line lists the issues still open
(BESSER-Web-Modeling-Editor#276), so the reply does not repeat them.

The fake model below returns the two failing shapes: an output the parser
rejects (a run-on cut off at the token limit) and a garbled member name.
"""
import copy
import json

import pytest

import execution.model_operations as mo
from common.errors import ModelRefusal
from conversation.reply_copy import AUTO_FIX_UNFIXED
from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler
from protocol.types import AssistantRequest, WorkspaceContext
from tests.conftest import FakeLLM, FakeSession

ISSUE = "Class 'Brand' defines attribute 'name' which already exists in a parent class."
AUTO_FIX = (
    "[auto-fix] The last change left the diagram with validation errors. "
    f"Fix exactly these, changing only what is necessary:\n- {ISSUE}"
)

# Brand (name, aliases) inherits from Product (price, name): the probe's model.
MODEL = {
    "version": "3.0.0", "type": "ClassDiagram",
    "elements": {
        "c1": {"id": "c1", "name": "Brand", "type": "Class", "owner": None,
               "attributes": ["a1", "a2"], "methods": []},
        "a1": {"id": "a1", "name": "name: str", "type": "ClassAttribute", "owner": "c1"},
        "a2": {"id": "a2", "name": "aliases: str", "type": "ClassAttribute", "owner": "c1"},
        "c2": {"id": "c2", "name": "Product", "type": "Class", "owner": None,
               "attributes": ["a3", "a5"], "methods": []},
        "a3": {"id": "a3", "name": "price: float", "type": "ClassAttribute", "owner": "c2"},
        "a5": {"id": "a5", "name": "name: str", "type": "ClassAttribute", "owner": "c2"},
    },
    "relationships": {
        "g1": {"id": "g1", "type": "ClassInheritance",
               "source": {"element": "c1"}, "target": {"element": "c2"}},
    },
}

RUN_ON = (
    '{"modifications": [{"action": "remove_element", "target": {"className": "Brand", '
    '"attributeName": "name head exam exam exam exam exam exam exam exam exam exam'
)
GARBLED = json.dumps({"modifications": [
    {"action": "remove_element", "target": {"className": "Brand", "attributeName": "name wall?"}},
]})
FIXED = json.dumps({"modifications": [
    {"action": "remove_element", "target": {"className": "Brand", "attributeName": "name"}},
]})


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)


def _send(monkeypatch, message: str, llm_output: str) -> dict:
    handler = ClassDiagramHandler(FakeLLM(llm_output))
    factory = type("Factory", (), {"get_handler": lambda self, _t: handler})()
    monkeypatch.setattr(mo.ctx, "diagram_factory", factory)
    monkeypatch.setattr(mo, "reply_progress", lambda *a, **k: None)
    monkeypatch.setattr(mo, "resolve_target_model", lambda *a, **k: copy.deepcopy(MODEL))
    monkeypatch.setattr(mo, "build_workspace_context_block", lambda *a, **k: "")
    sent = []
    monkeypatch.setattr(mo, "reply_payload", lambda _s, payload: sent.append(payload))
    monkeypatch.setattr(mo, "reply_message", lambda _s, text: sent.append({"action": "assistant_message",
                                                                          "message": text}))
    request = AssistantRequest(message=message, context=WorkspaceContext(active_diagram_type="ClassDiagram"))
    operation = {"diagramType": "ClassDiagram", "mode": "modify_model", "request": message}
    mo.execute_model_operation(FakeSession(), request, operation, "modify_model", _skip_existing_check=True)
    assert sent, "no reply was sent"
    return sent[-1]


def test_a_failed_repair_says_the_diagram_is_unchanged(monkeypatch):
    reply = _send(monkeypatch, AUTO_FIX, RUN_ON)
    assert reply == {"action": "assistant_message", "message": AUTO_FIX_UNFIXED}
    # The editor lists the open issues in its own status line; not twice.
    assert ISSUE not in reply["message"]


def test_a_repair_naming_a_junk_member_asks_without_quoting_it(monkeypatch):
    reply = _send(monkeypatch, AUTO_FIX, GARBLED)
    assert reply["message"] == "I couldn't tell which attribute of **Brand** to remove. Which one did you mean?"


@pytest.mark.parametrize("where", ["escapes the handler", "inside the handler"])
def test_a_repair_that_crashes_is_not_asked_to_rephrase(monkeypatch, where):
    def _crash(*_a, **_k):
        raise RuntimeError("handler bug")
    target = "generate_modification" if where == "escapes the handler" else "_build_modification"
    monkeypatch.setattr(ClassDiagramHandler, target, _crash)
    reply = _send(monkeypatch, AUTO_FIX, FIXED)
    assert reply == {"action": "assistant_message", "message": AUTO_FIX_UNFIXED}


def test_a_crash_on_a_user_request_applies_nothing(monkeypatch):
    """The crash fallback used to ship a placeholder ``modify_class Unknown``
    op, which the editor then rejected with a second failure."""
    monkeypatch.setattr(ClassDiagramHandler, "_build_modification",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("handler bug")))
    reply = _send(monkeypatch, "remove the duplicate name from Brand", FIXED)
    assert reply["action"] == "assistant_message"
    assert "modification" not in reply
    assert "rephrase" in reply["message"]


def test_a_refused_repair_keeps_the_reason(monkeypatch):
    def _refuse(*_a, **_k):
        raise ModelRefusal("openai", "violence")
    monkeypatch.setattr(ClassDiagramHandler, "predict_structured", _refuse)
    reply = _send(monkeypatch, AUTO_FIX, FIXED)
    assert reply["message"].startswith("The AI model declined this request")


def test_a_duplicate_end_repair_keeps_its_guidance(monkeypatch):
    """The deterministic duplicate-end repair explains what to rename; the
    editor's status line only repeats the validation error."""
    request = (
        "[auto-fix] The last change left the diagram with validation errors. "
        "Fix exactly these, changing only what is necessary:\n"
        "- The class 'Brand' cannot have two association ends with the same name: 'items'"
    )
    reply = _send(monkeypatch, request, FIXED)
    assert reply["action"] == "assistant_message"
    assert reply["message"].startswith("I couldn't apply this fix automatically.")
    assert "'items'" in reply["message"]


def test_a_successful_repair_is_applied(monkeypatch):
    reply = _send(monkeypatch, AUTO_FIX, FIXED)
    assert reply["action"] == "modify_model"
    assert reply["message"] == "Removed attribute **name** from **Brand**."


def test_a_user_request_keeps_its_own_failure_reply(monkeypatch):
    reply = _send(monkeypatch, "remove the duplicate name from Brand", RUN_ON)
    assert "rephrase" in reply["message"].lower()
