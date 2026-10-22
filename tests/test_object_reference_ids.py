"""Object-diagram classId / attributeId come from the reference class diagram.

The LLM is shown the reference ids and asked to copy them, but it can omit
or mis-copy them. A class or attribute matched by name in the reference
diagram must win over whatever id the LLM returned.
"""

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
