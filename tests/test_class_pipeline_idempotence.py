"""The class-diagram guards settle in one pass.

Feeding the output of ``generate_complete_system`` back through it must
change nothing. A second pass that still changes something means a guard
undoes or redoes another one: an end named ``keywords_1`` by the uniqueness
guard used to come back as ``keywords1``, because the label splitter
re-coerced a name that was already valid.
"""

import copy

import pytest

import diagram_handlers.types.class_diagram_handler as handler_mod
from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler
from schemas.class_diagram import SystemClassSpec


def _cls(name, attributes=(), **flags):
    """Attributes as 'name', 'name:type' or a dict."""
    attrs = []
    for a in attributes:
        if isinstance(a, dict):
            attrs.append(a)
        else:
            attr_name, _, type_ = a.partition(":")
            attrs.append({"name": attr_name, "type": type_ or "str"})
    return {"className": name, "attributes": attrs, "methods": [], **flags}


def _enum(name, literals):
    return {"className": name, "isEnumeration": True, "methods": [],
            "attributes": [{"name": literal} for literal in literals]}


def _rel(source, target, name=None, source_role=None, type_="Association", sm="0..*", tm="1"):
    return {"type": type_, "source": source, "target": target, "sourceMultiplicity": sm,
            "targetMultiplicity": tm, "name": name, "sourceRole": source_role}


def _spec(classes, relationships=(), constraints=()):
    return {"systemName": "Sample", "classes": list(classes),
            "relationships": list(relationships), "constraints": list(constraints)}


MODELS = {
    "end named like an attribute": (
        "articles with an author name",
        _spec([_cls("Article", ["title", "author"]), _cls("Writer", ["name"])],
              [_rel("Article", "Writer", "author", "articles")])),
    "text attribute restating a to-many end": (
        "articles with keywords",
        _spec([_cls("Article", ["title", "keywords:List[str]"]), _cls("Keyword", ["word"])],
              [_rel("Article", "Keyword", "keywords", "articles", tm="0..*")])),
    "end named like an inherited attribute": (
        "posts and articles with keywords",
        _spec([_cls("Post", ["title", "keywords"]), _cls("Article", ["pages:int"]),
               _cls("Keyword", ["word"])],
              [_rel("Article", "Post", type_="Inheritance"),
               _rel("Article", "Keyword", "keywords", "articles", tm="0..*")])),
    "two unnamed links between one pair": (
        "doctors work in a department and may head one",
        _spec([_cls("Doctor", ["name"]), _cls("Department", ["name"])],
              [_rel("Doctor", "Department"), _rel("Doctor", "Department", sm="0..1")])),
    "label naming both ends": (
        "bookings and their contact person",
        _spec([_cls("Booking", ["day:date"]), _cls("Person", ["name"])],
              [_rel("Booking", "Person", "contact / bookingsAsContact")])),
    "id attributes of linked and unlinked classes": (
        "orders of customers, shipped from a depot",
        _spec([_cls("Order", ["total:float", "customerId", "depotId",
                              {"name": "couponId", "type": "str", "isOptional": True}]),
               _cls("Customer", ["name"]), _cls("Depot", ["town"]), _cls("Coupon", ["code"])],
              [_rel("Order", "Customer", "customer", "orders")])),
    "hierarchy with its own id": (
        "engineers are people",
        _spec([_cls("Person", ["personId", "name"]), _cls("Engineer", ["personId", "grade"])],
              [_rel("Engineer", "Person", type_="Inheritance")])),
    "class-typed and collection attributes": (
        "carts with items, delivered to an address",
        _spec([_cls("Cart", ["items:List[Item]", "address:Address", "flags:List[Flag]"]),
               _cls("Item", ["name"]), _cls("Address", ["street"]), _enum("Flag", ["gift", "fragile"])],
              [_rel("Item", "Cart", "cart", "items", type_="Composition")])),
    "enumeration used by an attribute and by a relationship": (
        "exams with a grade and a level",
        _spec([_cls("Exam", ["subject", "grade:Grade"]), _enum("Grade", ["A", "B", "C"]),
               _enum("Level", ["BASIC", "ADVANCED"])],
              [_rel("Exam", "Level", "level")])),
    "surrogate ids": (
        "parcels kept in depots",
        _spec([_cls("Parcel", ["id", "weight:float", "leftOn:date"]), _cls("Depot", ["id", "town"])],
              [_rel("Parcel", "Depot", "depot", "parcels")])),
    "an id a rule reads": (
        "parcels; parcel numbers are positive",
        _spec([_cls("Parcel", ["id:int", "weight:float"])],
              constraints=[{"context": "Parcel", "name": "positive",
                            "expression": "context Parcel inv positive: self.id > 0"}])),
    "mandatory cycle": (
        "a booking covers at least one room",
        _spec([_cls("Booking", ["day:date"]), _cls("BookedRoom", ["price:float"])],
              [_rel("BookedRoom", "Booking", "booking", "rooms", sm="1..*", tm="1")])),
}


def _build(spec, request_text, monkeypatch):
    monkeypatch.setattr(handler_mod, "COMPACT_SPEC_ENABLED", False, raising=True)
    handler = ClassDiagramHandler(None)
    parsed = SystemClassSpec.model_validate(spec)
    monkeypatch.setattr(handler, "predict_two_pass_structured", lambda **kwargs: parsed)
    result = copy.deepcopy(handler.generate_complete_system(request_text)["systemSpec"])
    for cls in result["classes"]:
        cls.pop("position", None)
    return result


@pytest.mark.parametrize("name", sorted(MODELS))
def test_a_second_pass_changes_nothing(name, monkeypatch):
    request_text, spec = MODELS[name]
    first = _build(copy.deepcopy(spec), request_text, monkeypatch)
    second = _build(copy.deepcopy(first), request_text, monkeypatch)
    assert second == first


def test_the_first_pass_did_something(monkeypatch):
    """The models above are not already clean: the guards do rename and
    remove on the first pass."""
    request_text, spec = MODELS["end named like an attribute"]
    first = _build(copy.deepcopy(spec), request_text, monkeypatch)
    assert first["relationships"][0]["name"] == "author_1"
    request_text, spec = MODELS["text attribute restating a to-many end"]
    first = _build(copy.deepcopy(spec), request_text, monkeypatch)
    assert [a["name"] for a in first["classes"][0]["attributes"]] == ["title"]
    request_text, spec = MODELS["id attributes of linked and unlinked classes"]
    first = _build(copy.deepcopy(spec), request_text, monkeypatch)
    assert [a["name"] for a in first["classes"][0]["attributes"]] == [
        "total", "depotId", "couponId"]
    assert len(first["relationships"]) == 1
