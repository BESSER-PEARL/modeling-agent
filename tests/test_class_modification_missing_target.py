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

    def test_reply_lists_only_the_changes_that_remain(self, handler):
        """A modify op on a missing member that carries changes is dropped;
        the reply must not also report it as applied."""
        from schemas import ClassModificationResponse
        model = {
            "elements": {
                "c-book": {"id": "c-book", "name": "Book", "type": "Class", "owner": None},
                "a-title": {"id": "a-title", "name": "title", "type": "ClassAttribute",
                            "owner": "c-book"},
                "c-author": {"id": "c-author", "name": "Author", "type": "Class", "owner": None},
            },
            "relationships": {},
        }
        parsed = ClassModificationResponse.model_validate({"modifications": [
            {"action": "modify_attribute", "target": {"className": "Book", "attributeName": "isbn"},
             "changes": {"isOptional": True}},
            {"action": "add_attribute", "target": {"className": "Author"},
             "changes": {"name": "birthYear", "type": "int"}},
        ]})
        handler.predict_structured = lambda *a, **k: parsed

        result = handler.generate_modification("make isbn optional and give authors a birth year", model)

        assert _remaining_actions(result) == ["add_attribute"]
        message = result["message"]
        assert "birthYear" in message
        assert "couldn't find a **isbn** field" in message
        assert "Applied 2 changes" not in message
        assert "isbn" not in message.split("\n\n")[0]


def _library_with_abstract_publication():
    return {
        "elements": {
            "c-pub": {"id": "c-pub", "name": "Publication", "type": "AbstractClass", "owner": None},
            "a-title": {"id": "a-title", "name": "title", "type": "ClassAttribute", "owner": "c-pub"},
            "c-author": {"id": "c-author", "name": "Author", "type": "Class", "owner": None},
        },
        "relationships": {
            "r1": {"id": "r1", "type": "ClassBidirectional",
                   "source": {"element": "c-pub"}, "target": {"element": "c-author"}},
        },
    }


def _modify(handler, ops, model, request="update the diagram"):
    from schemas import ClassModificationResponse
    parsed = ClassModificationResponse.model_validate({"modifications": ops})
    handler.predict_structured = lambda *a, **k: parsed
    return handler.generate_modification(request, model)


class TestExistingClassReferences:
    def test_add_class_of_an_existing_class_adds_only_its_new_members(self, handler):
        result = _modify(handler, [
            {"action": "add_class", "target": {},
             "changes": {"className": "Publication", "attributes": [
                 {"name": "title", "type": "str"}, {"name": "isbn", "type": "str"}]}},
            {"action": "add_relationship",
             "target": {"sourceClass": "Author", "targetClass": "Publication"},
             "changes": {"relationshipType": "Association", "name": "reviewed"}},
        ], _library_with_abstract_publication())

        mods = result["modifications"]
        assert [m["action"] for m in mods][:2] == ["add_attribute", "add_relationship"]
        assert mods[0]["target"] == {"className": "Publication"}
        assert mods[0]["changes"]["name"] == "isbn"
        assert "Added the **Publication** class" not in result["message"]
        assert "isbn" in result["message"]

    def test_add_class_of_an_existing_class_with_nothing_new_changes_nothing(self, handler):
        result = _modify(handler, [
            {"action": "add_class", "target": {},
             "changes": {"className": "publication", "attributes": [{"name": "title"}]}},
        ], _library_with_abstract_publication())

        assert result["action"] == "assistant_message"
        assert "**Publication** is already in the diagram" in result["message"]

    def _with_book(self):
        model = _library_with_abstract_publication()
        model["elements"]["c-book"] = {"id": "c-book", "name": "Book", "type": "Class", "owner": None}
        model["elements"]["a-isbn"] = {"id": "a-isbn", "name": "isbn", "type": "ClassAttribute",
                                       "owner": "c-book"}
        model["relationships"]["r2"] = {"id": "r2", "type": "ClassInheritance",
                                        "source": {"element": "c-book"}, "target": {"element": "c-pub"}}
        return model

    def test_re_added_class_gets_no_surrogate_id_and_no_inherited_attribute(self, handler):
        result = _modify(handler, [
            {"action": "add_class", "target": {},
             "changes": {"className": "Book", "attributes": [
                 {"name": "id", "type": "int"}, {"name": "isbn"}, {"name": "title"},
                 {"name": "pages", "type": "int"}]}},
        ], self._with_book(), request="add pages to books")

        mods = result.get("modifications") or [result["modification"]]
        assert [(m["action"], m["target"], m["changes"]["name"]) for m in mods] == [
            ("add_attribute", {"className": "Book"}, "pages")]

    def test_re_added_class_with_nothing_new_is_mentioned_in_a_batch(self, handler):
        result = _modify(handler, [
            {"action": "add_class", "target": {},
             "changes": {"className": "Author", "isAbstract": True, "attributes": []}},
            {"action": "add_attribute", "target": {"className": "Publication"},
             "changes": {"name": "year", "type": "int"}},
        ], self._with_book())

        assert _remaining_actions(result) == ["add_attribute"]
        assert "**Author** is already in the diagram with those members." in result["message"]
        assert "**Author** already exists and is not abstract" in result["message"]
        assert "nothing changed" not in result["message"]

    def test_add_class_after_removing_that_class_is_kept(self, handler):
        result = _modify(handler, [
            {"action": "remove_element", "target": {"className": "Author"}},
            {"action": "add_class", "target": {},
             "changes": {"className": "Author", "attributes": [{"name": "name"}]}},
        ], _library_with_abstract_publication())

        assert [m["action"] for m in result["modifications"]] == ["remove_element", "add_class"]

    def test_add_class_after_removing_that_class_by_id_is_kept(self, handler):
        result = _modify(handler, [
            {"action": "remove_element", "target": {"className": "c-author"}},
            {"action": "add_class", "target": {},
             "changes": {"className": "Author", "attributes": [{"name": "name"}]}},
        ], _library_with_abstract_publication())

        assert [m["action"] for m in result["modifications"]] == ["remove_element", "add_class"]
        assert result["modifications"][0]["target"]["className"] == "Author"

    def test_add_class_of_the_name_a_rename_gives_adds_to_the_renamed_class(self, handler):
        result = _modify(handler, [
            {"action": "modify_class", "target": {"className": "Author"}, "changes": {"name": "Writer"}},
            {"action": "add_class", "target": {},
             "changes": {"className": "Writer", "attributes": [{"name": "bio"}]}},
        ], _library_with_abstract_publication())

        mods = result["modifications"]
        assert [(m["action"], m["target"].get("className")) for m in mods] == [
            ("modify_class", "Author"), ("add_attribute", "Writer")]
        assert mods[1]["changes"]["name"] == "bio"
        assert "Added the **Writer** class" not in result["message"]

    def test_add_class_before_the_rename_that_gives_its_name_is_kept(self, handler):
        result = _modify(handler, [
            {"action": "add_class", "target": {},
             "changes": {"className": "Writer", "attributes": [{"name": "bio"}]}},
            {"action": "modify_class", "target": {"className": "Author"}, "changes": {"name": "Writer"}},
        ], _library_with_abstract_publication())

        assert [m["action"] for m in result["modifications"]] == ["add_class", "modify_class"]

    def test_a_class_named_by_its_element_id_is_resolved(self, handler):
        result = _modify(handler, [
            {"action": "remove_element",
             "target": {"sourceClass": "c-pub", "targetClass": "Author"}},
        ], _library_with_abstract_publication(), request="unlink Publication from Author")

        assert result["action"] == "modify_model"
        assert result["modification"]["target"]["sourceClass"] == "Publication"
        assert "c-pub" not in result["message"]


def _shop_with_identifiers():
    return {
        "elements": {
            "c-cust": {"id": "c-cust", "name": "Customer", "type": "Class", "owner": None},
            "a-email": {"id": "a-email", "name": "email", "type": "ClassAttribute",
                        "owner": "c-cust", "isExternalId": True},
            "a-nick": {"id": "a-nick", "name": "nickname", "type": "ClassAttribute", "owner": "c-cust"},
            "c-order": {"id": "c-order", "name": "Order", "type": "Class", "owner": None},
            "a-code": {"id": "a-code", "name": "code", "type": "ClassAttribute",
                       "owner": "c-order", "isId": True},
        },
        "relationships": {},
    }


class TestIdentifierStaysRequired:
    """BUML rejects an optional identifier; the editor's repair then deleted
    the attribute. The modification keeps it required and says why."""

    @pytest.mark.parametrize("cls, attr, shown", [
        ("Customer", "email", "**email** in **Customer**"),
        ("Order", "code", "**code** in **Order**"),
        # the note names the attribute and class as the diagram has them
        ("customer", "email: str", "**email** in **Customer**"),
    ])
    def test_making_an_identifier_optional_changes_nothing(self, handler, cls, attr, shown):
        result = _modify(handler, [
            {"action": "modify_attribute", "target": {"className": cls, "attributeName": attr},
             "changes": {"isOptional": True}},
        ], _shop_with_identifiers(), request=f"make {attr} optional")

        assert result["action"] == "assistant_message"
        assert result["message"].count("I kept") == 1
        assert f"I kept {shown} required" in result["message"]
        assert "identifier mark in the attribute's settings" in result["message"]

    def test_other_changes_to_the_identifier_and_the_batch_still_apply(self, handler):
        result = _modify(handler, [
            {"action": "modify_attribute", "target": {"className": "Customer", "attributeName": "email"},
             "changes": {"isOptional": True, "type": "text"}},
            {"action": "modify_attribute", "target": {"className": "Customer", "attributeName": "nickname"},
             "changes": {"isOptional": True}},
        ], _shop_with_identifiers())

        email, nickname = result["modifications"]
        assert "isOptional" not in email["changes"] and email["changes"]["type"] == "text"
        assert nickname["changes"]["isOptional"] is True
        assert "I kept **email** in **Customer** required" in result["message"]


class TestEveryClassKindIsIndexed:
    """An edit on an abstract class, interface or enumeration is not a missing target."""

    def _model(self):
        return {
            "elements": {
                "c-pub": {"id": "c-pub", "name": "Publication", "type": "AbstractClass"},
                "c-lend": {"id": "c-lend", "name": "Lendable", "type": "Interface"},
                "m-lend": {"id": "m-lend", "name": "+lend(days: int): bool", "type": "ClassMethod",
                           "owner": "c-lend"},
                "c-genre": {"id": "c-genre", "name": "Genre", "type": "Enumeration"},
                "l-poem": {"id": "l-poem", "name": "POETRY", "type": "ClassAttribute", "owner": "c-genre"},
            },
            "relationships": {},
        }

    @pytest.mark.parametrize("mod", [
        {"action": "modify_class", "target": {"className": "Lendable"}, "changes": {"name": "Borrowable"}},
        {"action": "remove_element", "target": {"className": "Lendable", "methodName": "lend"}},
        {"action": "remove_element", "target": {"className": "Lendable"}},
        {"action": "modify_class", "target": {"className": "Publication"}, "changes": {"name": "Work"}},
        {"action": "remove_element", "target": {"className": "Genre", "attributeName": "POETRY"}},
    ])
    def test_edit_is_kept(self, handler, mod):
        spec = {"action": "modify_model", "modification": mod}
        assert handler._drop_phantom_target_ops(spec, self._model()) == []
        assert spec["modification"] is mod
