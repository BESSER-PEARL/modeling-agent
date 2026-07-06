"""Tests for missing-target detection in class-diagram modifications.

Repro: a model has no ``priority`` field; the user says "Remove priority and add
difficulty (easy, medium, hard)". The agent used to add the Difficulty enum but
SILENTLY drop the "remove priority" with zero feedback (the class-diagram modify
path had no missing-target detection, unlike BPMN's ``elementFound`` signal).

The fix adds a deterministic validation pass in ``generate_modification``: every
removal/modify op that names an existing element is resolved against the current
model; unresolved targets are dropped and a plain-language "couldn't find …"
note is appended. Additions are never validated.
"""

import pytest

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler


@pytest.fixture
def handler():
    return ClassDiagramHandler(llm=None)


def _model_with_task_title():
    """Apollon editor-format model: a Task class whose only attribute is
    ``title`` (attributes are stored as separate ``ClassAttribute`` elements
    linked to the class by ``owner``). There is no ``priority`` field."""
    return {
        "elements": {
            "task-1": {"id": "task-1", "name": "Task", "type": "Class", "owner": None},
            "attr-1": {"id": "attr-1", "name": "title", "type": "ClassAttribute", "owner": "task-1"},
        },
        "relationships": {},
    }


def _remaining_actions(spec):
    if isinstance(spec.get("modifications"), list):
        return [m["action"] for m in spec["modifications"]]
    if isinstance(spec.get("modification"), dict):
        return [spec["modification"]["action"]]
    return []


# ---------------------------------------------------------------------------
# Model index / helper-level validation
# ---------------------------------------------------------------------------

class TestModelIndex:
    def test_parses_apollon_attributes(self, handler):
        classes, attrs, methods = handler._build_model_index(_model_with_task_title())
        assert "task" in classes
        assert "title" in attrs
        assert methods == set()

    def test_parses_inline_attributes(self, handler):
        model = {"elements": {"c1": {"name": "Task", "type": "Class",
                                     "attributes": [{"name": "title"}]}}}
        _classes, attrs, _methods = handler._build_model_index(model)
        assert "title" in attrs


class TestDropPhantomTargetOps:
    def test_phantom_attribute_removal_dropped_with_note(self, handler):
        spec = {"action": "modify_model", "modification": {
            "action": "remove_element",
            "target": {"className": "Task", "attributeName": "priority"}}}
        notes = handler._drop_phantom_target_ops(spec, _model_with_task_title())
        assert len(notes) == 1
        assert "priority" in notes[0].lower()
        assert "couldn't find" in notes[0].lower()
        # The phantom op is gone (spec has no ops left).
        assert not spec.get("modification") and not spec.get("modifications")

    def test_existing_attribute_removal_kept(self, handler):
        spec = {"action": "modify_model", "modification": {
            "action": "remove_element",
            "target": {"className": "Task", "attributeName": "title"}}}
        notes = handler._drop_phantom_target_ops(spec, _model_with_task_title())
        assert notes == []
        assert _remaining_actions(spec) == ["remove_element"]

    def test_legit_add_survives_phantom_drop(self, handler):
        spec = {"action": "modify_model", "modifications": [
            {"action": "remove_element",
             "target": {"className": "Task", "attributeName": "priority"}},
            {"action": "add_class", "target": {"className": None},
             "changes": {"className": "Difficulty", "isEnumeration": True,
                         "attributes": [{"name": "easy"}, {"name": "medium"}, {"name": "hard"}]}},
        ]}
        notes = handler._drop_phantom_target_ops(spec, _model_with_task_title())
        assert any("priority" in n.lower() for n in notes)
        assert _remaining_actions(spec) == ["add_class"]

    def test_additions_are_never_validated(self, handler):
        # add_attribute naming a not-yet-existing field must not be dropped.
        spec = {"action": "modify_model", "modification": {
            "action": "add_attribute",
            "target": {"className": "Task", "attributeName": "dueDate"},
            "changes": {"type": "Date"}}}
        notes = handler._drop_phantom_target_ops(spec, _model_with_task_title())
        assert notes == []
        assert _remaining_actions(spec) == ["add_attribute"]

    def test_phantom_class_removal_dropped(self, handler):
        spec = {"action": "modify_model", "modification": {
            "action": "remove_element", "target": {"className": "Ghost"}}}
        notes = handler._drop_phantom_target_ops(spec, _model_with_task_title())
        assert len(notes) == 1
        assert "ghost" in notes[0].lower()

    def test_no_model_means_no_dropping(self, handler):
        spec = {"action": "modify_model", "modification": {
            "action": "remove_element",
            "target": {"className": "Task", "attributeName": "priority"}}}
        notes = handler._drop_phantom_target_ops(spec, {"elements": {}})
        assert notes == []
        assert _remaining_actions(spec) == ["remove_element"]


# ---------------------------------------------------------------------------
# generate_modification integration (LLM call stubbed via _execute_modification)
# ---------------------------------------------------------------------------

class TestGenerateModificationIntegration:
    def test_remove_phantom_add_real_reports_and_keeps_add(self, handler, monkeypatch):
        """(a) phantom removal dropped, (b) reply names the missing field,
        (c) the legitimate add in the same request still succeeds."""
        canned = {"action": "modify_model", "modifications": [
            {"action": "remove_element",
             "target": {"className": "Task", "attributeName": "priority"}},
            {"action": "add_class", "target": {"className": None},
             "changes": {"className": "Difficulty", "isEnumeration": True,
                         "attributes": [{"name": "easy"}, {"name": "medium"}, {"name": "hard"}]}},
        ], "message": "Applied 2 changes."}
        monkeypatch.setattr(handler, "_execute_modification", lambda *a, **k: canned)

        result = handler.generate_modification(
            "Remove priority and add difficulty (easy, medium, hard)",
            current_model=_model_with_task_title(),
        )
        # (a) phantom removal dropped; (c) add survives.
        assert _remaining_actions(result) == ["add_class"]
        # (b) the reply message names the missing field with a not-found note.
        assert "priority" in result["message"].lower()
        assert "couldn't find" in result["message"].lower()
        # The Difficulty enum is still mentioned (the add succeeded and is described).
        assert "difficulty" in result["message"].lower()

    def test_only_phantom_removal_yields_plain_message(self, handler, monkeypatch):
        """When every op is a phantom target, surface a plain not-found message
        instead of shipping an empty modification."""
        canned = {"action": "modify_model", "modification": {
            "action": "remove_element",
            "target": {"className": "Task", "attributeName": "priority"}},
            "message": "Removed priority."}
        monkeypatch.setattr(handler, "_execute_modification", lambda *a, **k: canned)

        result = handler.generate_modification(
            "remove priority", current_model=_model_with_task_title(),
        )
        assert result["action"] == "assistant_message"
        assert "priority" in result["message"].lower()
        assert "couldn't find" in result["message"].lower()
