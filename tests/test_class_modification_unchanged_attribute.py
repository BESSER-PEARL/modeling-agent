"""An attribute edit that changes nothing is not reported as applied.

Live evidence (BESSER #832, experimental, 2026-11-24): ``Habit.frequency`` was
``float``; "update frequency to number" produced ``modify_attribute`` with
``changes.type = "number"``, which the editor stores as ``float``. The stored
model was identical before and after, no trace entry was written, and the reply
still said "Updated attribute **frequency** in **Habit**."
"""

import pytest

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler
from schemas import ClassModificationResponse


@pytest.fixture
def handler():
    return ClassDiagramHandler(llm=None)


def _habit_model(frequency_type="float"):
    return {
        "elements": {
            "c-habit": {"id": "c-habit", "name": "Habit", "type": "Class", "owner": None},
            "a-freq": {"id": "a-freq", "name": "frequency", "type": "ClassAttribute",
                       "owner": "c-habit", "attributeType": frequency_type, "visibility": "private"},
            "a-streak": {"id": "a-streak", "name": "streakCount", "type": "ClassAttribute",
                         "owner": "c-habit", "attributeType": "int", "visibility": "public"},
            "c-reminder": {"id": "c-reminder", "name": "Reminder", "type": "Class", "owner": None},
            "a-rfreq": {"id": "a-rfreq", "name": "frequency", "type": "ClassAttribute",
                        "owner": "c-reminder", "attributeType": "Frequency", "visibility": "public"},
        },
        "relationships": {},
    }


def _modify(handler, ops, model, request="update frequency to number"):
    parsed = ClassModificationResponse.model_validate({"modifications": ops})
    handler.predict_structured = lambda *a, **k: parsed
    return handler.generate_modification(request, model)


def _retype(type_name, attribute="frequency", cls="Habit"):
    return {"action": "modify_attribute", "target": {"className": cls, "attributeName": attribute},
            "changes": {"type": type_name}}


def _actions(result):
    mods = result.get("modifications") or ([result["modification"]] if result.get("modification") else [])
    return [m["action"] for m in mods]


def test_retyping_to_the_current_type_changes_nothing_and_says_so(handler):
    result = _modify(handler, [_retype("number")], _habit_model("float"))

    assert result["action"] == "assistant_message"
    message = result["message"]
    assert "Updated attribute" not in message
    assert "**frequency** in **Habit** is already `float`" in message
    # "number" is not a BESSER type: the user may have wanted whole numbers.
    assert "`number` is stored as `float`" in message and "`int`" in message


def test_the_same_type_under_its_own_name_gets_no_alias_hint(handler):
    result = _modify(handler, [_retype("float")], _habit_model("float"))

    assert result["action"] == "assistant_message"
    assert "is already `float`" in result["message"]
    assert "stored as" not in result["message"]


def test_a_real_retype_is_still_applied(handler):
    result = _modify(handler, [_retype("number")], _habit_model("str"))

    assert _actions(result) == ["modify_attribute"]
    assert "Updated attribute **frequency** in **Habit**" in result["message"]


def test_only_the_named_class_is_compared(handler):
    """Reminder.frequency is an enumeration: retyping it to float is a change."""
    result = _modify(handler, [_retype("float", cls="Reminder")], _habit_model("float"))

    assert _actions(result) == ["modify_attribute"]


def test_an_op_with_one_real_change_is_kept(handler):
    op = _retype("float")
    op["changes"]["visibility"] = "public"  # stored: private

    result = _modify(handler, [op], _habit_model("float"))

    assert _actions(result) == ["modify_attribute"]


def test_unchanged_op_is_dropped_from_a_batch_and_noted(handler):
    result = _modify(handler, [_retype("number"), _retype("float", attribute="streakCount")],
                     _habit_model("float"))

    assert _actions(result) == ["modify_attribute"]
    message = result["message"]
    assert "streakCount" in message.split("\n\n")[0]
    assert "frequency" not in message.split("\n\n")[0]
    assert "**frequency** in **Habit** is already `float`" in message


def test_an_attribute_without_a_stored_type_is_not_judged(handler):
    model = _habit_model("float")
    del model["elements"]["a-freq"]["attributeType"]

    result = _modify(handler, [_retype("float")], model)

    assert _actions(result) == ["modify_attribute"]
