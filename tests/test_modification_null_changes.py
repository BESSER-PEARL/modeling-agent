"""A modify_* operation the LLM returns with ``"changes": null`` changes nothing.

It is dropped like other malformed operations instead of failing the whole
request; the rest of the batch is applied and the reply says a part was skipped.
"""

import pytest

from diagram_handlers.types.agent_diagram_handler import AgentDiagramHandler
from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler
from diagram_handlers.types.state_machine_handler import StateMachineHandler
from schemas import AgentModificationResponse, ClassModificationResponse, StateMachineModificationResponse


def _book_author_model():
    return {
        "elements": {
            "c-book": {"id": "c-book", "name": "Book", "type": "Class", "owner": None,
                       "attributes": ["a-title"], "methods": []},
            "a-title": {"id": "a-title", "name": "title", "type": "ClassAttribute",
                        "owner": "c-book", "attributeType": "str", "visibility": "public"},
            "c-author": {"id": "c-author", "name": "Author", "type": "Class", "owner": None,
                         "attributes": [], "methods": []},
        },
        "relationships": {},
    }


def _run(handler, schema, llm_output, model, *, fail_on_crash=True, request="tidy this up"):
    parsed = schema.model_validate(llm_output)
    handler.predict_structured = lambda *a, **k: parsed

    def _crashed(exc, *a, **k):
        pytest.fail(f"modification crashed: {exc!r}")

    if fail_on_crash:
        handler._modification_crash_reply = _crashed
    return handler.generate_modification(request, model)


def test_class_modify_with_null_changes_does_not_fail_the_request():
    result = _run(ClassDiagramHandler(llm=None), ClassModificationResponse, {"modifications": [
        {"action": "modify_class", "target": {"className": "Book"}, "changes": None},
    ]}, _book_author_model())

    assert result["action"] == "assistant_message"
    assert "concrete change" in result["message"]


def test_class_null_changes_op_is_skipped_and_the_rest_applied():
    result = _run(ClassDiagramHandler(llm=None), ClassModificationResponse, {"modifications": [
        {"action": "modify_class", "target": {"className": "Book"}, "changes": None},
        {"action": "add_attribute", "target": {"className": "Author"},
         "changes": {"name": "birthYear", "type": "int"}},
    ]}, _book_author_model())

    assert result["action"] == "modify_model"
    mods = result.get("modifications") or [result["modification"]]
    assert [m["action"] for m in mods] == ["add_attribute"]
    assert "skipped 1 part" in result["message"]


def test_remove_with_null_changes_is_kept():
    result = _run(ClassDiagramHandler(llm=None), ClassModificationResponse, {"modifications": [
        {"action": "remove_element", "target": {"className": "Author"}, "changes": None},
    ]}, _book_author_model())

    assert result["action"] == "modify_model"
    assert result["modification"]["action"] == "remove_element"


def test_other_diagram_types_drop_null_change_modifies_too():
    model = {"elements": {"s1": {"id": "s1", "name": "Idle", "type": "State"}}, "relationships": {}}
    result = _run(StateMachineHandler(None), StateMachineModificationResponse, {"modifications": [
        {"action": "modify_state", "target": {"stateName": "Idle"}, "changes": None},
    ]}, model)

    assert result["action"] == "assistant_message"
    assert "concrete change" in result["message"]


def test_class_null_changes_on_a_missing_member_says_it_was_not_found():
    result = _run(ClassDiagramHandler(llm=None), ClassModificationResponse, {"modifications": [
        {"action": "modify_attribute", "target": {"className": "Book", "attributeName": "isbn"},
         "changes": None},
    ]}, _book_author_model())

    assert result["action"] == "assistant_message"
    assert "couldn't find" in result["message"]
    assert "isbn" in result["message"]


def test_agent_null_changes_on_a_missing_state_keeps_the_agent_reply():
    model = {"elements": {"s1": {"id": "s1", "name": "greeting", "type": "AgentState"}},
             "relationships": {}}
    result = _run(AgentDiagramHandler(None), AgentModificationResponse, {"modifications": [
        {"action": "modify_state", "target": {"stateName": "farewell"}, "changes": None},
    ]}, model, fail_on_crash=False,
        request="rename the farewell state to goodbye")

    assert result["action"] == "assistant_message"
    assert "couldn't find the element you referred to" in result["message"]
