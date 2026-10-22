"""extract_class / merge_classes / promote_attribute on the modify path (#35).

The structured-output schema only has ``action`` / ``target`` / ``changes``,
but the expanders read top-level fields (``sourceClass``, ``classes``,
``targetName``, ``newClass``) that pydantic never produces. Every refactor
therefore became a placeholder class (``NewClass`` / ``MergedClass`` /
``PromotedClass``) and an empty remove target.

The LLM outputs below are the ones gpt-6-luna returned live (2026-10-22) for
the three requests against ``_model()``.
"""

import pytest

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler
from schemas import ClassModificationResponse


def _attr(eid, owner, name, t="str"):
    return {"id": eid, "name": name, "type": "ClassAttribute", "owner": owner,
            "attributeType": t, "visibility": "public"}


def _model():
    """Editor-format model: members are separate elements owned by the class."""
    return {
        "elements": {
            "c-cust": {"id": "c-cust", "name": "Customer", "type": "Class", "owner": None,
                       "attributes": ["a1", "a2", "a3", "a4", "a5"], "methods": []},
            "a1": _attr("a1", "c-cust", "name"),
            "a2": _attr("a2", "c-cust", "street"),
            "a3": _attr("a3", "c-cust", "city"),
            "a4": _attr("a4", "c-cust", "zip"),
            "a5": _attr("a5", "c-cust", "phone"),
            "c-client": {"id": "c-client", "name": "Client", "type": "Class", "owner": None,
                         "attributes": ["b1", "b2"], "methods": ["m1"]},
            "b1": _attr("b1", "c-client", "name"),
            "b2": _attr("b2", "c-client", "loyaltyPoints", "int"),
            "m1": {"id": "m1", "name": "+ redeem(points: int): bool", "type": "ClassMethod",
                   "owner": "c-client"},
            "c-order": {"id": "c-order", "name": "Order", "type": "Class", "owner": None,
                        "attributes": ["o1", "o2"], "methods": []},
            "o1": _attr("o1", "c-order", "total", "float"),
            "o2": _attr("o2", "c-order", "status"),
        },
        "relationships": {
            "r1": {"id": "r1", "type": "ClassBidirectional", "name": "places",
                   "source": {"element": "c-client", "multiplicity": "1", "role": "client"},
                   "target": {"element": "c-order", "multiplicity": "0..*", "role": "orders"}},
        },
    }


def _run(llm_output, model=None):
    handler = ClassDiagramHandler(llm=None)
    parsed = ClassModificationResponse.model_validate(llm_output)
    handler.predict_structured = lambda *a, **k: parsed
    spec = handler.generate_modification("refactor", model or _model())
    assert spec["action"] == "modify_model", spec
    return spec.get("modifications") or [spec["modification"]]


def _of(mods, action):
    return [m for m in mods if m["action"] == action]


def _no_placeholders(mods):
    text = repr(mods)
    for placeholder in ("NewClass", "MergedClass", "PromotedClass"):
        assert placeholder not in text


def _pairs(mods):
    return [frozenset((m["target"]["sourceClass"], m["target"]["targetClass"]))
            for m in _of(mods, "add_relationship")]


EXTRACT = {"modifications": [{
    "action": "extract_class",
    "target": {"className": "Customer", "attributeName": "street, city, zip"},
    "changes": {"className": "Address", "attributes": [
        {"name": "street", "type": "str"},
        {"name": "city", "type": "str"},
        {"name": "zip", "type": "str"},
    ]},
}]}

MERGE = {"modifications": [{
    "action": "merge_classes",
    "target": {"className": "Client"},
    "changes": {"name": "Customer"},
}]}

PROMOTE = {"modifications": [
    {"action": "promote_attribute",
     "target": {"className": "Customer", "attributeName": "phone"},
     "changes": {"className": "Phone"}},
    {"action": "add_relationship",
     "target": {"sourceClass": "Customer", "targetClass": "Phone"},
     "changes": {"relationshipType": "Association", "sourceMultiplicity": "1",
                 "targetMultiplicity": "1", "roleName": "phone"}},
]}


def test_extract_class_moves_the_named_attributes():
    mods = _run(EXTRACT)
    _no_placeholders(mods)
    [add] = _of(mods, "add_class")
    assert add["changes"]["className"] == "Address"
    assert [(a["name"], a["type"]) for a in add["changes"]["attributes"]] == [
        ("street", "str"), ("city", "str"), ("zip", "str")]
    removed = {(m["target"]["className"], m["target"]["attributeName"])
               for m in _of(mods, "remove_element")}
    assert removed == {("Customer", "street"), ("Customer", "city"), ("Customer", "zip")}
    assert _pairs(mods) == [frozenset(("Customer", "Address"))]


def test_merge_into_existing_class_keeps_every_member_and_link():
    mods = _run(MERGE)
    _no_placeholders(mods)
    # The editor's addClass skips an existing name, so the survivor must be
    # extended in place, never re-added.
    assert _of(mods, "add_class") == []
    added = [(m["target"]["className"], m["changes"]["name"], m["changes"]["type"])
             for m in _of(mods, "add_attribute")]
    assert added == [("Customer", "loyaltyPoints", "int")]  # "name" already exists
    [method] = _of(mods, "add_method")
    assert method["target"]["className"] == "Customer"
    assert method["changes"]["name"] == "redeem"
    assert method["changes"]["returnType"] == "bool"
    assert method["changes"]["parameters"] == [{"name": "points", "type": "int"}]
    [rel] = _of(mods, "add_relationship")
    assert (rel["target"]["sourceClass"], rel["target"]["targetClass"]) == ("Customer", "Order")
    assert (rel["changes"]["sourceMultiplicity"], rel["changes"]["targetMultiplicity"]) == ("1", "0..*")
    [remove] = _of(mods, "remove_element")
    assert remove["target"] == {"className": "Client"}
    # The absorbed class is removed only after its members were carried over.
    assert mods[-1]["action"] == "remove_element"


def test_merge_two_classes_into_a_new_class():
    mods = _run({"modifications": [
        {"action": "merge_classes", "target": {"className": "Customer"},
         "changes": {"name": "Party"}},
        {"action": "merge_classes", "target": {"className": "Client"},
         "changes": {"name": "Party"}},
    ]})
    [add] = _of(mods, "add_class")
    assert add["changes"]["className"] == "Party"
    assert [a["name"] for a in add["changes"]["attributes"]] == [
        "name", "street", "city", "zip", "phone"]
    # The second merge extends the class the first one created.
    assert [m["changes"]["name"] for m in _of(mods, "add_attribute")] == ["loyaltyPoints"]
    assert {m["target"]["className"] for m in _of(mods, "remove_element")} == {"Customer", "Client"}
    assert _pairs(mods) == [frozenset(("Party", "Order"))]


def test_promote_attribute_moves_the_attribute_into_the_new_class():
    mods = _run(PROMOTE)
    _no_placeholders(mods)
    [add] = _of(mods, "add_class")
    assert add["changes"]["className"] == "Phone"
    assert [(a["name"], a["type"]) for a in add["changes"]["attributes"]] == [("phone", "str")]
    [remove] = _of(mods, "remove_element")
    assert remove["target"] == {"className": "Customer", "attributeName": "phone"}
    # The LLM already linked the two classes: no second association.
    assert _pairs(mods) == [frozenset(("Customer", "Phone"))]


@pytest.mark.parametrize("llm_output", [
    {"modifications": [{"action": "merge_classes", "target": {"className": "Client"}}]},
    {"modifications": [{"action": "extract_class", "target": {"className": "Customer"},
                        "changes": {"attributes": [{"name": "street"}]}}]},
])
def test_incomplete_refactor_is_not_applied_with_a_placeholder(llm_output):
    handler = ClassDiagramHandler(llm=None)
    parsed = ClassModificationResponse.model_validate(llm_output)
    handler.predict_structured = lambda *a, **k: parsed
    spec = handler.generate_modification("refactor", _model())
    assert "modifications" not in spec and "modification" not in spec
    assert spec["action"] == "assistant_message"


# split_class / add_enum had the same defect (``sourceClass`` / ``newClasses``
# / ``enumName`` / ``values``). gpt-6-luna emits split_class as one op per new
# class, shaped like extract_class (live, 2026-10-22). It never chose add_enum
# in five live enum requests (it uses add_class + isEnumeration), so ENUM is
# the schema-valid shape matching that add_class form.

SPLIT = {"modifications": [
    {"action": "split_class", "target": {"className": "Customer"},
     "changes": {"className": "ContactInfo", "attributes": [
         {"name": "name", "type": "String"}, {"name": "phone", "type": "String"}]}},
    {"action": "split_class", "target": {"className": "Customer"},
     "changes": {"className": "PostalAddress", "attributes": [
         {"name": "street", "type": "String"}, {"name": "city", "type": "String"},
         {"name": "zip", "type": "String"}]}},
]}

ENUM = {"modifications": [{
    "action": "add_enum",
    "target": {"className": "Order", "attributeName": "status"},
    "changes": {"className": "OrderStatus", "attributes": [
        {"name": "PENDING"}, {"name": "PAID"}, {"name": "SHIPPED"}]},
}]}


def test_split_class_moves_each_part_out_of_the_source():
    mods = _run(SPLIT)
    added = {m["changes"]["className"]: [a["name"] for a in m["changes"]["attributes"]]
             for m in _of(mods, "add_class")}
    assert added == {"ContactInfo": ["name", "phone"],
                     "PostalAddress": ["street", "city", "zip"]}
    removed = {m["target"]["attributeName"] for m in _of(mods, "remove_element")}
    assert removed == {"name", "phone", "street", "city", "zip"}
    assert sorted(map(sorted, _pairs(mods))) == [
        ["ContactInfo", "Customer"], ["Customer", "PostalAddress"]]


def test_split_class_into_subclasses_points_inheritance_at_the_source():
    split = {"modifications": [dict(SPLIT["modifications"][0])]}
    split["modifications"][0]["changes"] = dict(
        split["modifications"][0]["changes"], relationshipType="Inheritance")
    mods = _run(split)
    [rel] = _of(mods, "add_relationship")
    assert (rel["target"]["sourceClass"], rel["target"]["targetClass"]) == ("ContactInfo", "Customer")


def test_add_enum_creates_the_literals_and_retypes_the_attribute():
    mods = _run(ENUM)
    [add] = _of(mods, "add_class")
    assert add["changes"]["className"] == "OrderStatus"
    assert add["changes"]["isEnumeration"] is True
    assert [a["name"] for a in add["changes"]["attributes"]] == ["PENDING", "PAID", "SHIPPED"]
    [retype] = _of(mods, "modify_attribute")
    assert retype["target"] == {"className": "Order", "attributeName": "status"}
    assert retype["changes"]["type"] == "OrderStatus"
    assert "NewEnum" not in repr(mods)
