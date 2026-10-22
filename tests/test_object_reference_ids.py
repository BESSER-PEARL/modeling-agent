"""Object-diagram classId / attributeId come from the reference class diagram.

The LLM is shown the reference ids and asked to copy them, but it can omit
or mis-copy them. A class or attribute matched by name in the reference
diagram must win over whatever id the LLM returned.
"""

import copy

from diagram_handlers.types.object_diagram_handler import ObjectDiagramHandler
from schemas.object_diagram import ObjectModificationResponse, SingleObjectSpec

REFERENCE = {
    "type": "ClassDiagram",
    "elements": {
        "cls-book": {"id": "cls-book", "type": "Class", "name": "Book", "attributes": ["att-title", "att-pages"], "methods": []},
        "att-title": {"id": "att-title", "type": "ClassAttribute", "name": "+ title: str", "attributeType": "str", "owner": "cls-book"},
        "att-pages": {"id": "att-pages", "type": "ClassAttribute", "name": "pages", "attributeType": "int", "owner": "cls-book"},
        "cls-author": {"id": "cls-author", "type": "Class", "name": "Author", "attributes": ["att-name"], "methods": []},
        "att-name": {"id": "att-name", "type": "ClassAttribute", "name": "name", "attributeType": "str", "owner": "cls-author"},
    },
    "relationships": {},
}


def _book_attributes():
    return [
        # Mis-copied: another class's attribute id, and a truncated id.
        {"name": "title", "value": "Dune", "attributeId": "att-name"},
        {"name": "pages", "value": "412", "attributeId": "att-pag"},
    ]


def test_add_object_takes_catalog_ids_over_mis_copied_ones(monkeypatch):
    response = ObjectModificationResponse.model_validate({
        "modifications": [{
            "action": "add_object",
            "target": {"objectName": "dune"},
            "changes": {"className": "book", "classId": "cls-author", "attributes": _book_attributes()},
        }],
    })
    monkeypatch.setattr(ObjectDiagramHandler, "predict_structured", lambda self, *a, **k: response)

    result = ObjectDiagramHandler(None).generate_modification(
        "add a book dune", current_model=None, reference_diagram=REFERENCE,
    )

    mods = result.get("modifications") or [result["modification"]]
    changes = mods[0]["changes"]
    assert changes["className"] == "Book"
    assert changes["classId"] == "cls-book"
    assert {a["name"]: a["attributeId"] for a in changes["attributes"]} == {
        "title": "att-title",
        "pages": "att-pages",
    }


def test_single_object_takes_catalog_ids_over_mis_copied_ones(monkeypatch):
    spec = SingleObjectSpec.model_validate({
        "objectName": "dune",
        "className": "Book",
        "classId": "cls-author",
        "attributes": _book_attributes(),
    })
    monkeypatch.setattr(ObjectDiagramHandler, "predict_structured", lambda self, *a, **k: spec)

    result = ObjectDiagramHandler(None).generate_single_element(
        "add a book dune", existing_model=None, reference_diagram=REFERENCE,
    )

    element = result["element"]
    assert element["classId"] == "cls-book"
    assert {a["name"]: (a["attributeId"], a["type"]) for a in element["attributes"]} == {
        "title": ("att-title", "str"),
        "pages": ("att-pages", "int"),
    }


def test_unknown_class_keeps_the_ids_the_llm_sent(monkeypatch):
    spec = SingleObjectSpec.model_validate({
        "objectName": "x1",
        "className": "Ghost",
        "classId": "cls-ghost",
        "attributes": [{"name": "label", "value": "a", "attributeId": "att-label"}],
    })
    monkeypatch.setattr(ObjectDiagramHandler, "predict_structured", lambda self, *a, **k: spec)

    element = ObjectDiagramHandler(None).generate_single_element(
        "add a ghost", existing_model=None, reference_diagram=REFERENCE,
    )["element"]

    assert element["classId"] == "cls-ghost"
    assert element["attributes"][0]["attributeId"] == "att-label"


def test_single_object_matches_class_and_attribute_names_case_insensitively(monkeypatch):
    spec = SingleObjectSpec.model_validate({
        "objectName": "dune",
        "className": "book",
        "classId": "cls-author",
        "attributes": [{"name": "Title", "value": "Dune", "attributeId": "att-name"}],
    })
    monkeypatch.setattr(ObjectDiagramHandler, "predict_structured", lambda self, *a, **k: spec)

    element = ObjectDiagramHandler(None).generate_single_element(
        "add a book dune", existing_model=None, reference_diagram=REFERENCE,
    )["element"]

    assert (element["className"], element["classId"]) == ("Book", "cls-book")
    assert element["attributes"][0]["attributeId"] == "att-title"


def test_both_paths_resolve_a_duplicated_class_name_to_the_same_id(monkeypatch):
    reference = copy.deepcopy(REFERENCE)
    reference["elements"]["cls-book2"] = {"id": "cls-book2", "type": "Class", "name": "Book", "attributes": [], "methods": []}
    single = SingleObjectSpec.model_validate({"objectName": "b", "className": "Book", "classId": "cls-book", "attributes": []})
    added = ObjectModificationResponse.model_validate({
        "modifications": [{
            "action": "add_object",
            "target": {"objectName": "b"},
            "changes": {"className": "Book", "classId": "cls-book", "attributes": []},
        }],
    })
    handler = ObjectDiagramHandler(None)

    monkeypatch.setattr(ObjectDiagramHandler, "predict_structured", lambda self, *a, **k: single)
    single_id = handler.generate_single_element("add b", existing_model=None, reference_diagram=reference)["element"]["classId"]
    monkeypatch.setattr(ObjectDiagramHandler, "predict_structured", lambda self, *a, **k: added)
    result = handler.generate_modification("add b", current_model=None, reference_diagram=reference)
    mods = result.get("modifications") or [result["modification"]]

    assert single_id == mods[0]["changes"]["classId"]
