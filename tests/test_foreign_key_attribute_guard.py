"""Foreign-key attributes in a generated class diagram.

A link between two classes is an association. An attribute holding the id of
the other class (``Order.customerId``) beside that association restates it: the
generated backend then defines the foreign key twice and asks for the same
reference twice. ``ClassDiagramHandler._rewrite_foreign_key_attributes``
removes such an attribute only when exactly one to-one link stores the same
reference. Any other id attribute stays, and the guard never adds a link:
keeping a redundant attribute is harmless, losing a link is not.
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
        rels = [_assoc("Customer", "Order", "orders", "customer",
                       sourceMultiplicity="1", targetMultiplicity="0..*")]
        spec = _spec([_cls("Order", [_attr("customerId")]), _cls("Customer")], copy.deepcopy(rels))
        handler._rewrite_foreign_key_attributes(spec)
        assert _names(spec, "Order") == []
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
                     [_assoc("Player", "Club", "captainOf", "captain", sourceMultiplicity="1",
                             targetMultiplicity="0..1")])
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

    def test_an_optional_id_beside_an_optional_link_is_removed(self, handler):
        spec = _spec([_cls("Task", [_attr("projectId", isOptional=True)]), _cls("Project")],
                     [_assoc("Task", "Project", "project", "tasks", targetMultiplicity="0..1")])
        handler._rewrite_foreign_key_attributes(spec)
        assert _names(spec, "Task") == []

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


class TestAnotherLinkToTheSameClass:
    """The one link to the class the id names may be a different link:
    removing the id would lose the link it stands for."""

    def test_an_optional_link_with_a_role_of_its_own(self, handler):
        spec = _spec([_cls("Order", [_attr("total", "float"), _attr("customerId")]),
                      _cls("Customer")],
                     [_assoc("Order", "Customer", "referredBy", "referrals",
                             targetMultiplicity="0..1")])
        before = copy.deepcopy(spec)
        handler._rewrite_foreign_key_attributes(spec)
        assert spec == before

    def test_a_link_the_other_class_states_for_another_role(self, handler):
        # a department is headed by one employee; the id is the employee's department
        spec = _spec([_cls("Employee", [_attr("departmentId")]), _cls("Department")],
                     [_assoc("Department", "Employee", "head", "headOf",
                             sourceMultiplicity="0..1", targetMultiplicity="1")])
        before = copy.deepcopy(spec)
        handler._rewrite_foreign_key_attributes(spec)
        assert spec == before

    def test_a_single_reference_beside_a_to_many_link(self, handler):
        # the main category of a product that belongs to several categories
        spec = _spec([_cls("Product", [_attr("categoryId")]), _cls("Category")],
                     [_assoc("Product", "Category", "categories", "products",
                             targetMultiplicity="0..*")])
        before = copy.deepcopy(spec)
        handler._rewrite_foreign_key_attributes(spec)
        assert spec == before

    def test_two_links_to_the_class_none_named_by_the_id(self, handler):
        spec = _spec([_cls("Order", [_attr("addressId")]), _cls("Address")],
                     [_assoc("Order", "Address", "billingAddress"),
                      _assoc("Order", "Address", "shippingAddress")])
        before = copy.deepcopy(spec)
        handler._rewrite_foreign_key_attributes(spec)
        assert spec == before

    def test_two_ends_with_the_role_the_id_names(self, handler):
        spec = _spec([_cls("Match", [_attr("hostId")]), _cls("Team"), _cls("Club")],
                     [_assoc("Match", "Team", "host"), _assoc("Club", "Match", "hostedMatches", "host",
                                                             sourceMultiplicity="1")])
        before = copy.deepcopy(spec)
        handler._rewrite_foreign_key_attributes(spec)
        assert spec == before

    def test_the_last_word_of_a_linked_class_name_is_not_enough(self, handler):
        spec = _spec([_cls("Ticket", [_attr("agentId")]), _cls("SupportAgent")],
                     [_assoc("Ticket", "SupportAgent", "handledBy", "tickets")])
        before = copy.deepcopy(spec)
        handler._rewrite_foreign_key_attributes(spec)
        assert spec == before

    def test_an_account_is_not_the_linked_bank_account(self, handler):
        # the sign-in account of an employee who also holds bank accounts
        spec = _spec([_cls("Employee", [_attr("name"), _attr("accountId")]),
                      _cls("BankAccount", [_attr("iban")])],
                     [_assoc("BankAccount", "Employee", "holder", "bankAccounts")])
        before = copy.deepcopy(spec)
        handler._rewrite_foreign_key_attributes(spec)
        assert spec == before


class TestNoLinkIsAdded:
    """An id naming a class the owner is not linked to is left as it is: it
    may be a plain value, or name another thing that shares the class name."""

    def test_id_of_an_unlinked_class_stays_an_attribute(self, handler):
        spec = _spec([_cls("Order", [_attr("total", "float"), _attr("customerId")]),
                      _cls("Customer")])
        before = copy.deepcopy(spec)
        handler._rewrite_foreign_key_attributes(spec)
        assert spec == before

    def test_an_id_the_request_keeps_as_plain_text(self, handler):
        spec = _spec([_cls("Ticket", [_attr("title"), _attr("deviceId")]),
                      _cls("Device", [_attr("serial")])])
        before = copy.deepcopy(spec)
        handler._rewrite_foreign_key_attributes(
            spec, "each ticket stores the device id as plain text; devices are listed separately")
        assert spec == before

    def test_an_id_naming_an_unrelated_class_of_the_same_name(self, handler):
        # a browser session beside the sessions of a conference programme
        spec = _spec([_cls("Session", [_attr("title")]), _cls("Room"), _cls("User"),
                      _cls("Login", [_attr("at", "datetime"), _attr("sessionId")])],
                     [_assoc("Session", "Room", "room"), _assoc("Login", "User", "user", "logins")])
        before = copy.deepcopy(spec)
        handler._rewrite_foreign_key_attributes(
            spec, "a conference app with sessions and rooms; log every login of a user")
        assert spec == before

    def test_a_link_to_another_class_does_not_count(self, handler):
        spec = _spec([_cls("Order", [_attr("customerId")]), _cls("Customer"), _cls("Product")],
                     [_assoc("Order", "Product", "products")])
        before = copy.deepcopy(spec)
        handler._rewrite_foreign_key_attributes(spec)
        assert spec == before


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


def _inherits(child, parent):
    return {"type": "Inheritance", "source": child, "target": parent}


def _associations(spec):
    return [(r["source"], r["target"]) for r in spec["relationships"]
            if r.get("type") != "Inheritance"]


class TestOwnHierarchy:
    """``<x>Id`` naming the class itself or a class of its generalization chain
    is that hierarchy's identifier: a link from a class to its own ancestor
    makes the generated backend pass the same key twice."""

    def test_id_of_the_parent_is_not_a_link_to_the_parent(self, handler):
        spec = _spec(
            [_cls("Person", [_attr("personId"), _attr("name")]),
             _cls("Engineer", [_attr("personId"), _attr("grade")])],
            [_inherits("Engineer", "Person")],
        )
        before = copy.deepcopy(spec)
        handler._rewrite_foreign_key_attributes(spec)
        assert spec == before

    def test_id_of_a_grandparent_or_of_a_subclass_is_left_alone(self, handler):
        spec = _spec(
            [_cls("Asset", [_attr("truckId")]), _cls("Vehicle"),
             _cls("Truck", [_attr("assetId"), _attr("vehicleId")])],
            [_inherits("Vehicle", "Asset"), _inherits("Truck", "Vehicle")],
        )
        before = copy.deepcopy(spec)
        handler._rewrite_foreign_key_attributes(spec)
        assert spec == before

    def test_id_of_an_unlinked_sibling_is_left_alone(self, handler):
        spec = _spec(
            [_cls("Person"), _cls("Engineer", [_attr("managerId")]), _cls("Manager")],
            [_inherits("Engineer", "Person"), _inherits("Manager", "Person")],
        )
        before = copy.deepcopy(spec)
        handler._rewrite_foreign_key_attributes(spec)
        assert spec == before

    def test_generated_model_links_no_class_to_its_own_ancestor(self, monkeypatch):
        compact = CompactSystemClassSpec(
            name="Staff",
            classes=[
                CompactClassSpec(n="Person", a=["personId: str", "name: str"], m=[], k=""),
                CompactClassSpec(n="Engineer", a=["personId: str", "grade: str"], m=[], k=""),
            ],
            rels=[CompactRelationshipSpec(
                f="Engineer", t="Person", k="inher", how_many_TARGET_for_one_SOURCE="",
                how_many_SOURCE_for_one_TARGET="", l="")],
            ocl=[],
        )
        monkeypatch.setattr(handler_mod, "COMPACT_SPEC_ENABLED", True, raising=True)
        handler = ClassDiagramHandler(None)
        monkeypatch.setattr(handler, "predict_two_pass_structured", lambda **kwargs: compact)

        spec = handler.generate_complete_system("engineers are people")["systemSpec"]

        assert _associations(spec) == []
        # the subclass copy is the shadow guard's to remove, as before
        assert _names(spec, "Person") == ["personId", "name"]
        assert _names(spec, "Engineer") == ["grade"]


class TestInheritedLinks:
    def test_a_link_only_a_subclass_has_does_not_count_for_the_parent(self, handler):
        spec = _spec(
            [_cls("Person", [_attr("companyId")]), _cls("Engineer"), _cls("Company")],
            [_inherits("Engineer", "Person"), _assoc("Engineer", "Company", "employer", "staff")],
        )
        before = copy.deepcopy(spec)
        handler._rewrite_foreign_key_attributes(spec)
        assert spec == before

    def test_hierarchy_direction_is_settled_before_links_are_counted(self, monkeypatch):
        """The model states the hierarchy backwards (one class the subclass of
        two). Once it is turned round the subclass inherits the parent's link,
        so its id attribute goes and no second link is added."""
        def rel(f, t, k, name="", source_role=""):
            many = "" if k == "inher" else "0..*"
            return CompactRelationshipSpec(
                f=f, t=t, k=k, how_many_TARGET_for_one_SOURCE="" if k == "inher" else "1",
                how_many_SOURCE_for_one_TARGET=many, l=name, ls=source_role)

        compact = CompactSystemClassSpec(
            name="Staff",
            classes=[
                CompactClassSpec(n="Person", a=["name: str"], m=[], k=""),
                CompactClassSpec(n="Engineer", a=["grade: str", "companyId: str"], m=[], k=""),
                CompactClassSpec(n="Manager", a=["level: int"], m=[], k=""),
                CompactClassSpec(n="Company", a=["name: str"], m=[], k=""),
            ],
            rels=[rel("Person", "Engineer", "inher"), rel("Person", "Manager", "inher"),
                  rel("Person", "Company", "assoc", "employer", "staff")],
            ocl=[],
        )
        monkeypatch.setattr(handler_mod, "COMPACT_SPEC_ENABLED", True, raising=True)
        handler = ClassDiagramHandler(None)
        monkeypatch.setattr(handler, "predict_two_pass_structured", lambda **kwargs: compact)

        spec = handler.generate_complete_system("a staff directory")["systemSpec"]

        assert _names(spec, "Engineer") == ["grade"]
        assert _associations(spec) == [("Person", "Company")]


class TestTheRequestDecides:
    def test_an_id_the_request_names_as_written_is_the_users_field(self, handler):
        spec = _spec(
            [_cls("Scan", [_attr("ticketId")]), _cls("Ticket"), _cls("Hall")],
            [_assoc("Ticket", "Hall", "hall")],
        )
        before = copy.deepcopy(spec)
        handler._rewrite_foreign_key_attributes(
            spec, "tickets for a hall; each scan stores the printed ticketId")
        assert spec == before

    @pytest.mark.parametrize("written", ["customerId", "customer_id", "CUSTOMER_ID"])
    def test_named_as_written_also_beside_a_link(self, handler, written):
        spec = _spec([_cls("Order", [_attr("customerId")]), _cls("Customer")],
                     [_assoc("Order", "Customer", "customer", "orders")])
        handler._rewrite_foreign_key_attributes(spec, f"orders keep a {written} field")
        assert _names(spec, "Order") == ["customerId"]


class TestMembersInUse:
    def test_an_id_a_constraint_reads_is_kept(self, handler):
        spec = _spec([_cls("Order", [_attr("customerId")]), _cls("Customer")],
                     [_assoc("Order", "Customer", "customer", "orders")])
        spec["constraints"] = [{"context": "Order", "name": "known",
                                "expression": "context Order inv known: self.customerId <> ''"}]
        handler._rewrite_foreign_key_attributes(spec, "orders must name their customer")
        assert _names(spec, "Order") == ["customerId"]

    @pytest.mark.parametrize("expression", [
        # navigates an end Order does not have
        "context Order inv sameBuyer: self.lines->forAll(l | l.buyer = self.customerId)",
        # a class used as a collection
        "context Order inv known: Customer->select(c | c.name = self.customerId)->size() = 1",
    ])
    def test_an_id_only_a_rule_that_does_not_fit_reads_is_removed(self, handler, expression):
        spec = _spec([_cls("Order", [_attr("total", "float"), _attr("customerId")]),
                      _cls("Customer", [_attr("name")])],
                     [_assoc("Order", "Customer", "customer", "orders")])
        spec["constraints"] = [{"context": "Order", "name": "r", "expression": expression}]
        handler._rewrite_foreign_key_attributes(spec, "orders")
        assert _names(spec, "Order") == ["total"]

    def test_a_rule_reading_ends_inherited_and_ocl_operations_fits(self, handler):
        spec = _spec([_cls("Item", [_attr("label")]), _cls("Order", [_attr("customerId")]),
                      _cls("Customer"), _cls("Line", [_attr("qty", "int")])],
                     [_assoc("Order", "Customer", "customer", "orders"),
                      _assoc("Line", "Order", "order", "lines"),
                      {"type": "Inheritance", "source": "Order", "target": "Item"}])
        spec["constraints"] = [{"context": "Order", "name": "r", "expression":
                                "context Order inv r: self.oclIsKindOf(Item) and self.label <> 'x.y' "
                                "and self.lines->size() > 0 and self.customer.orders->notEmpty() "
                                "and self.customerId <> '' and Customer.allInstances()->size() > 0"}]
        assert handler._mismatched_constraints(spec) == []
        handler._rewrite_foreign_key_attributes(spec, "orders")
        assert _names(spec, "Order") == ["customerId"]

    def test_an_id_a_method_body_reads_is_kept(self, handler):
        spec = _spec([_cls("Order", [_attr("customerId")]), _cls("Customer")])
        spec["classes"][0]["methods"] = [{
            "name": "tagline", "returnType": "str", "parameters": [],
            "implementationType": "code",
            "code": "def tagline(self):\n    return 'order of ' + self.customerId",
        }]
        handler._rewrite_foreign_key_attributes(spec, "orders")
        assert _names(spec, "Order") == ["customerId"]
        assert spec["relationships"] == []


def test_the_generation_prompt_forbids_id_attributes_for_links():
    prompt = ClassDiagramHandler(None)._get_system_generation_prompt()
    assert "A LINK IS NEVER AN ID ATTRIBUTE" in prompt
    assert "customerId" in prompt


def _generate(monkeypatch, classes, rels, request_text="a shop"):
    compact = CompactSystemClassSpec(name="Shop", classes=classes, rels=rels, ocl=[])
    monkeypatch.setattr(handler_mod, "COMPACT_SPEC_ENABLED", True, raising=True)
    handler = ClassDiagramHandler(None)
    monkeypatch.setattr(handler, "predict_two_pass_structured", lambda **kwargs: compact)
    return handler.generate_complete_system(request_text)


def _order_to_customer(role, how_many):
    return CompactRelationshipSpec(
        f="Order", t="Customer", k="assoc", how_many_TARGET_for_one_SOURCE=how_many,
        how_many_SOURCE_for_one_TARGET="0..*", l=role, ls="orders")


def test_generated_system_drops_the_id_its_association_stores(monkeypatch):
    result = _generate(monkeypatch, [
        CompactClassSpec(n="Order", a=["placedOn: date", "customerId: str"], m=[], k=""),
        CompactClassSpec(n="Customer", a=["name: str"], m=[], k=""),
    ], [_order_to_customer("customer", "1")])

    spec = result["systemSpec"]
    assert _names(spec, "Order") == ["placedOn"]
    assert [(r["source"], r["target"], r["name"], r["targetMultiplicity"])
            for r in spec["relationships"]] == [("Order", "Customer", "customer", "1")]


def test_generated_system_keeps_ids_of_other_links_and_adds_none(monkeypatch):
    result = _generate(monkeypatch, [
        CompactClassSpec(n="Order", m=[], k="",
                         a=["placedOn: date", "customerId: str", "couponId: str?"]),
        CompactClassSpec(n="Customer", a=["name: str"], m=[], k=""),
        CompactClassSpec(n="Coupon", a=["code: str!"], m=[], k=""),
    ], [_order_to_customer("referredBy", "0..1")],
        "orders placed by a customer, possibly referred by another customer")

    spec = result["systemSpec"]
    assert _names(spec, "Order") == ["placedOn", "customerId", "couponId"]
    assert [(r["source"], r["target"], r["name"]) for r in spec["relationships"]] == [
        ("Order", "Customer", "referredBy")]
    assert "association to" not in result["message"]
