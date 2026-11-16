"""No surrogate ``id`` attribute on generated classes.

The generators give every class its own primary key. A modeled ``id`` that the
spec cannot mark as the identifier is reported by the validator on each class
("Attribute 'id' is not the identifier"), rejected by Django and breaks the
inheritance mapping on a subclass. The prompts no longer ask for one, and
``ClassDiagramHandler._strip_surrogate_ids`` removes one the model adds
anyway, unless the request itself asks for an id. Identifiers the user names
(``isbn``, ``sku``) are ordinary attributes and stay.
"""

import pytest

import diagram_handlers.types.class_diagram_handler as handler_mod
from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler
from prompts import class_diagram as class_prompts
from schemas import SingleClassSpec
from schemas.compact_class_diagram import (
    CompactClassSpec,
    CompactRelationshipSpec,
    CompactSystemClassSpec,
)


@pytest.fixture
def handler():
    return ClassDiagramHandler(None)


def _cls(name, attributes, enum=False):
    return {"className": name, "isEnumeration": enum,
            "attributes": [a if isinstance(a, dict) else {"name": a, "type": "str"}
                           for a in attributes]}


def _names(classes, class_name):
    cls = next(c for c in classes if c["className"] == class_name)
    return [a["name"] for a in cls["attributes"]]


class TestStripSurrogateIds:
    def test_id_is_removed_from_every_class(self, handler):
        classes = [_cls("Book", ["id", "title"]), _cls("Author", ["id"]),
                   _cls("Loan", ["returnBy"])]
        handler._strip_surrogate_ids(classes, "a library with books, authors and loans")
        assert _names(classes, "Book") == ["title"]
        assert _names(classes, "Author") == []
        assert _names(classes, "Loan") == ["returnBy"]

    @pytest.mark.parametrize("name", ["id", "ID", "Id"])
    def test_any_casing_of_id(self, handler, name):
        classes = [_cls("Book", [name, "title"])]
        handler._strip_surrogate_ids(classes, "a library")
        assert _names(classes, "Book") == ["title"]

    def test_id_flagged_as_external_identifier_is_still_the_surrogate(self, handler):
        classes = [_cls("Book", [{"name": "id", "type": "str", "isExternalId": True}, "title"])]
        handler._strip_surrogate_ids(classes, "a library")
        assert _names(classes, "Book") == ["title"]

    def test_named_identifiers_stay(self, handler):
        classes = [_cls("Book", [{"name": "isbn", "type": "str", "isExternalId": True},
                                 "sku", "orderId", "identity", "id"])]
        handler._strip_surrogate_ids(classes, "books are identified by their ISBN")
        assert _names(classes, "Book") == ["isbn", "sku", "orderId", "identity"]
        assert classes[0]["attributes"][0]["isExternalId"] is True

    def test_enumeration_literal_named_id_stays(self, handler):
        classes = [_cls("DocumentKind", ["ID", "PASSPORT"], enum=True)]
        handler._strip_surrogate_ids(classes, "travel documents")
        assert _names(classes, "DocumentKind") == ["ID", "PASSPORT"]

    @pytest.mark.parametrize("request_text", [
        "each product has an id, a name and a price",
        "Products have an ID and a price",
        "every order gets a unique identifier",
        "store the ids of the products",
    ])
    def test_id_the_request_asks_for_is_kept(self, handler, request_text):
        classes = [_cls("Product", ["id", "name"])]
        handler._strip_surrogate_ids(classes, request_text)
        assert _names(classes, "Product") == ["id", "name"]

    @pytest.mark.parametrize("request_text", [
        "a video library where paid members rate valid entries",
        "every room is identified by its room number",
        "", None,
    ])
    def test_other_words_are_not_a_request_for_an_id(self, handler, request_text):
        classes = [_cls("Product", ["id", "name"])]
        handler._strip_surrogate_ids(classes, request_text)
        assert _names(classes, "Product") == ["name"]

    @pytest.mark.parametrize("classes", [None, [], [None, "x", {"className": "A"}]])
    def test_degenerate_input_is_a_noop(self, handler, classes):
        handler._strip_surrogate_ids(classes, "anything")


def _compact():
    return CompactSystemClassSpec(
        name="Library",
        classes=[
            CompactClassSpec(n="Book", a=["id: str", "isbn: str!", "title: str"], m=[], k=""),
            CompactClassSpec(n="Member", a=["id: int", "name: str"], m=[], k=""),
        ],
        rels=[CompactRelationshipSpec(
            f="Book", t="Member", k="assoc", how_many_TARGET_for_one_SOURCE="0..1",
            how_many_SOURCE_for_one_TARGET="0..*", l="borrower", ls="loans")],
        ocl=[],
    )


class TestGeneratedSystem:
    def _generate(self, monkeypatch, user_request, **kwargs):
        monkeypatch.setattr(handler_mod, "COMPACT_SPEC_ENABLED", True, raising=True)
        handler = ClassDiagramHandler(None)
        monkeypatch.setattr(handler, "predict_two_pass_structured", lambda **_: _compact())
        return handler.generate_complete_system(user_request, **kwargs)["systemSpec"]["classes"]

    def test_generated_classes_carry_no_id(self, monkeypatch):
        classes = self._generate(monkeypatch, "a library; each book is known by its ISBN")
        assert _names(classes, "Book") == ["isbn", "title"]
        assert _names(classes, "Member") == ["name"]

    def test_the_users_own_message_decides_not_the_added_context(self, monkeypatch):
        classes = self._generate(
            monkeypatch, "Context: session id 42\n\na library",
            raw_request="a library")
        assert _names(classes, "Member") == ["name"]

    def test_requested_id_is_kept(self, monkeypatch):
        classes = self._generate(monkeypatch, "a library; every member has an id and a name")
        assert _names(classes, "Member") == ["id", "name"]


def test_single_class_carries_no_id(handler, monkeypatch):
    monkeypatch.setattr(handler, "predict_structured", lambda *a, **k: SingleClassSpec(
        className="Product",
        attributes=[{"name": "id", "type": "str"}, {"name": "price", "type": "float"}],
    ))
    result = handler.generate_single_element("create a Product class")
    assert [a["name"] for a in result["element"]["attributes"]] == ["price"]


class TestAddClassModification:
    """A class added to an existing diagram gets no surrogate id either."""

    def _modify(self, handler, monkeypatch, request_text):
        from schemas import ClassModificationResponse
        reply = ClassModificationResponse(modifications=[
            {"action": "add_class", "target": {},
             "changes": {"className": "Award", "attributes": [
                 {"name": "id", "type": "str"}, {"name": "edition", "type": "int"}]}},
            {"action": "add_class", "target": {},
             "changes": {"className": "DocumentKind", "isEnumeration": True,
                         "attributes": [{"name": "ID"}, {"name": "PASSPORT"}]}},
            {"action": "add_attribute", "target": {"className": "Book"},
             "changes": {"name": "id", "type": "str"}},
        ])
        monkeypatch.setattr(handler, "predict_structured", lambda *a, **k: reply)
        model = {"elements": {"c1": {"id": "c1", "name": "Book", "type": "Class",
                                     "attributes": [], "methods": []}},
                 "relationships": {}}
        result = handler.generate_modification(request_text, current_model=model)
        return {m["changes"].get("className") or m["action"]:
                [a["name"] for a in m["changes"].get("attributes") or []]
                for m in result["modifications"]}

    def test_new_class_carries_no_id(self, handler, monkeypatch):
        added = self._modify(handler, monkeypatch, "also store awards and document kinds")
        assert added["Award"] == ["edition"]
        assert added["DocumentKind"] == ["ID", "PASSPORT"]
        # an attribute added to an existing class is the user's request
        assert "add_attribute" in added

    def test_requested_id_is_kept(self, handler, monkeypatch):
        added = self._modify(handler, monkeypatch, "add an Award class with an id and an edition")
        assert added["Award"] == ["id", "edition"]


class TestStarterClasses:
    def test_fallback_class_has_no_id(self, handler):
        spec = handler.generate_fallback_element("create a Product class")["element"]
        assert "id" not in [a["name"] for a in spec["attributes"]]
        assert spec["attributes"]

    def test_fallback_system_has_no_id(self, handler):
        classes = handler.generate_fallback_system()["systemSpec"]["classes"]
        assert all(a["name"] != "id" for c in classes for a in c["attributes"])
        assert all(c["attributes"] for c in classes)


class TestPrompts:
    def test_single_class_examples_do_not_show_an_id(self):
        assert "→ id," not in class_prompts.SYSTEM_PROMPT
        assert "surrogate id" in class_prompts.SYSTEM_PROMPT

    def test_generation_prompt_does_not_ask_for_ids(self):
        prompt = class_prompts.SYSTEM_GENERATION_PROMPT
        assert "plus an id" not in prompt
        assert "include IDs" not in prompt
        assert "NO SURROGATE ID" in prompt
