"""Foreign-key attributes in a generated class diagram.

A link between two classes is an association. An attribute holding the id of
the other class (``Order.customerId``) restates it: beside the association the
generated backend defines the foreign key twice, and without the association
the link is not in the model at all.
``ClassDiagramHandler._rewrite_foreign_key_attributes`` removes such an
attribute when the owner is linked to the class or role it names, and turns
it into an association when it is not. Identifiers that name no class of the
model are left alone.
"""

import copy

import pytest

import diagram_handlers.types.class_diagram_handler as handler_mod
from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler
from schemas.compact_class_diagram import (
    CompactClassSpec,
    CompactRelationshipSpec,
    CompactSystemClassSpec,
)


@pytest.fixture
def handler():
    return ClassDiagramHandler(None)


def _cls(name, attributes=(), enum=False):
    return {"className": name, "isEnumeration": enum, "attributes": list(attributes)}


def _attr(name, type_="str", **flags):
    return {"name": name, "type": type_, **flags}


def _assoc(source, target, name=None, source_role=None, type_="Association", **extra):
    return {"type": type_, "source": source, "target": target,
            "sourceMultiplicity": "0..*", "targetMultiplicity": "1",
            "name": name, "sourceRole": source_role, **extra}


def _spec(classes, relationships=()):
    return {"systemName": "T", "classes": list(classes),
            "relationships": list(relationships)}


def _names(spec, class_name):
    cls = next(c for c in spec["classes"] if c["className"] == class_name)
    return [a["name"] for a in cls["attributes"]]


class TestLinkedClass:
    @pytest.mark.parametrize("attribute", ["customerId", "customer_id", "customerID", "CustomerId"])
    def test_id_of_a_linked_class_is_removed(self, handler, attribute):
        rels = [_assoc("Order", "Customer", "placedBy", "orders")]
        spec = _spec([_cls("Order", [_attr("total", "float"), _attr(attribute)]),
                      _cls("Customer", [_attr("name")])], copy.deepcopy(rels))
        handler._rewrite_foreign_key_attributes(spec)
        assert _names(spec, "Order") == ["total"]
        assert spec["relationships"] == rels

    def test_integer_id_is_removed_too(self, handler):
        spec = _spec([_cls("Order", [_attr("customerId", "int")]), _cls("Customer")],
                     [_assoc("Order", "Customer")])
        handler._rewrite_foreign_key_attributes(spec)
        assert _names(spec, "Order") == []

    def test_link_stated_from_the_other_class_counts(self, handler):
        rels = [_assoc("Chapter", "Book", "book", "chapters", type_="Composition")]
        spec = _spec([_cls("Book", [_attr("chapterId")]), _cls("Chapter")], copy.deepcopy(rels))
        handler._rewrite_foreign_key_attributes(spec)
        assert _names(spec, "Book") == []
        assert spec["relationships"] == rels

    def test_id_named_after_an_end_role_is_removed(self, handler):
        rels = [_assoc("Transfer", "Account", "payer", "outgoing"),
                _assoc("Transfer", "Account", "payee", "incoming")]
        spec = _spec([_cls("Transfer", [_attr("amount", "float"), _attr("payerId"),
                                        _attr("payeeId")]),
                      _cls("Account")], copy.deepcopy(rels))
        handler._rewrite_foreign_key_attributes(spec)
        assert _names(spec, "Transfer") == ["amount"]
        assert spec["relationships"] == rels

    def test_role_on_the_source_end_counts(self, handler):
        # Club is the target: it reaches Player through sourceRole "captain".
        spec = _spec([_cls("Club", [_attr("captainId")]), _cls("Player")],
                     [_assoc("Player", "Club", "captainOf", "captain")])
        handler._rewrite_foreign_key_attributes(spec)
        assert _names(spec, "Club") == []

    def test_link_inherited_from_a_parent_counts(self, handler):
        rels = [_assoc("Person", "Company", "employer", "staff"),
                {"type": "Inheritance", "source": "Engineer", "target": "Person"}]
        spec = _spec([_cls("Person"), _cls("Engineer", [_attr("companyId")]), _cls("Company")],
                     copy.deepcopy(rels))
        handler._rewrite_foreign_key_attributes(spec)
        assert _names(spec, "Engineer") == []
        assert spec["relationships"] == rels

    def test_association_class_is_linked_to_both_ends(self, handler):
        rels = [_assoc("Employee", "Project", "projects", "staff",
                       associationClass="Assignment")]
        spec = _spec([_cls("Employee"), _cls("Project"),
                      _cls("Assignment", [_attr("hours", "float"), _attr("employeeId"),
                                          _attr("projectId")])],
                     copy.deepcopy(rels))
        handler._rewrite_foreign_key_attributes(spec)
        assert _names(spec, "Assignment") == ["hours"]
        assert spec["relationships"] == rels

    def test_last_word_of_a_linked_class_name_is_enough(self, handler):
        spec = _spec([_cls("Ticket", [_attr("agentId")]), _cls("SupportAgent")],
                     [_assoc("Ticket", "SupportAgent", "handledBy", "tickets")])
        handler._rewrite_foreign_key_attributes(spec)
        assert _names(spec, "Ticket") == []

    def test_last_word_shared_by_two_linked_classes_is_kept(self, handler):
        spec = _spec([_cls("Invoice", [_attr("orderId")]), _cls("SalesOrder"),
                      _cls("PurchaseOrder")],
                     [_assoc("Invoice", "SalesOrder", "sale"),
                      _assoc("Invoice", "PurchaseOrder", "purchase")])
        handler._rewrite_foreign_key_attributes(spec)
        assert _names(spec, "Invoice") == ["orderId"]

    def test_name_ending_inside_a_word_is_kept(self, handler):
        spec = _spec([_cls("Product", [_attr("groupId")]), _cls("Subgroup")],
                     [_assoc("Product", "Subgroup", "family")])
        handler._rewrite_foreign_key_attributes(spec)
        assert _names(spec, "Product") == ["groupId"]


class TestUnlinkedClass:
    def test_id_of_an_unlinked_class_becomes_an_association(self, handler):
        spec = _spec([_cls("Order", [_attr("total", "float"), _attr("customerId")]),
                      _cls("Customer")])
        handler._rewrite_foreign_key_attributes(spec)
        assert _names(spec, "Order") == ["total"]
        assert spec["relationships"] == [{
            "type": "Association", "source": "Order", "target": "Customer",
            "sourceMultiplicity": "0..*", "targetMultiplicity": "1", "name": "customer",
        }]

    def test_optional_id_becomes_an_optional_end(self, handler):
        spec = _spec([_cls("Task", [_attr("project_id", isOptional=True)]), _cls("Project")])
        handler._rewrite_foreign_key_attributes(spec)
        assert spec["relationships"][0]["targetMultiplicity"] == "0..1"
        assert spec["relationships"][0]["name"] == "project"

    @pytest.mark.parametrize("attribute", ["sales_order_id", "SalesOrderId"])
    def test_the_new_end_is_named_in_camel_case(self, handler, attribute):
        spec = _spec([_cls("Invoice", [_attr(attribute)]), _cls("SalesOrder")])
        handler._rewrite_foreign_key_attributes(spec)
        assert [(r["target"], r["name"]) for r in spec["relationships"]] == [
            ("SalesOrder", "salesOrder")]

    def test_a_link_to_another_class_does_not_count(self, handler):
        spec = _spec([_cls("Order", [_attr("customerId")]), _cls("Customer"), _cls("Product")],
                     [_assoc("Order", "Product", "products")])
        handler._rewrite_foreign_key_attributes(spec)
        assert _names(spec, "Order") == []
        assert [(r["source"], r["target"]) for r in spec["relationships"]] == [
            ("Order", "Product"), ("Order", "Customer")]


class TestLeftAlone:
    @pytest.mark.parametrize("attribute", [
        _attr("id"), _attr("isbn"), _attr("orderNumber"), _attr("externalId"),
        _attr("trackingId"), _attr("paid", "bool"), _attr("android"), _attr("valid"),
        _attr("customerId", isExternalId=True), _attr("customerId", "date"),
        _attr("customerId", None), _attr("customerIds"), _attr("legacyCustomerId"),
    ])
    def test_other_attributes_are_untouched(self, handler, attribute):
        spec = _spec([_cls("Order", [attribute]), _cls("Customer")],
                     [_assoc("Order", "Customer", "customer", "orders")])
        before = copy.deepcopy(spec)
        handler._rewrite_foreign_key_attributes(spec)
        assert spec == before

    def test_a_class_keeps_the_id_named_after_itself(self, handler):
        spec = _spec([_cls("Order", [_attr("orderId")]), _cls("Customer")])
        before = copy.deepcopy(spec)
        handler._rewrite_foreign_key_attributes(spec)
        assert spec == before

    def test_own_id_is_kept_beside_a_self_association(self, handler):
        spec = _spec([_cls("Folder", [_attr("folderId")])],
                     [_assoc("Folder", "Folder", "parent", "children")])
        before = copy.deepcopy(spec)
        handler._rewrite_foreign_key_attributes(spec)
        assert spec == before

    def test_an_enumeration_is_not_a_link_target(self, handler):
        spec = _spec([_cls("Order", [_attr("stageId")]),
                      _cls("Stage", [{"name": "OPEN"}], enum=True)])
        before = copy.deepcopy(spec)
        handler._rewrite_foreign_key_attributes(spec)
        assert spec == before


def test_the_generation_prompt_forbids_id_attributes_for_links():
    prompt = ClassDiagramHandler(None)._get_system_generation_prompt()
    assert "A LINK IS NEVER AN ID ATTRIBUTE" in prompt
    assert "customerId" in prompt


def test_generated_system_links_classes_by_association_only(monkeypatch):
    """Through generate_complete_system: one id beside its association, one
    with no association at all."""
    compact = CompactSystemClassSpec(
        name="Shop",
        classes=[
            CompactClassSpec(n="Order", m=[], k="",
                             a=["placedOn: date", "customerId: str", "couponId: str?"]),
            CompactClassSpec(n="Customer", a=["name: str"], m=[], k=""),
            CompactClassSpec(n="Coupon", a=["code: str!"], m=[], k=""),
        ],
        rels=[CompactRelationshipSpec(
            f="Order", t="Customer", k="assoc", how_many_TARGET_for_one_SOURCE="1",
            how_many_SOURCE_for_one_TARGET="0..*", l="customer", ls="orders")],
        ocl=[],
    )
    monkeypatch.setattr(handler_mod, "COMPACT_SPEC_ENABLED", True, raising=True)
    handler = ClassDiagramHandler(None)
    monkeypatch.setattr(handler, "predict_two_pass_structured", lambda **kwargs: compact)

    spec = handler.generate_complete_system("a shop")["systemSpec"]

    assert _names(spec, "Order") == ["placedOn"]
    assert [(r["source"], r["target"], r["name"], r["targetMultiplicity"])
            for r in spec["relationships"]] == [
        ("Order", "Customer", "customer", "1"),
        ("Order", "Coupon", "coupon", "0..1"),
    ]
