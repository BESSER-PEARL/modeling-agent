"""Tests for the class-typed-attribute guard.

An attribute whose type names another CLASS (e.g. PointOfInterest.location :
Location) crashes the deterministic FastAPI/SQLAlchemy/Pydantic generators and
forces the expensive LLM-from-scratch fallback. The guard rewrites such an
attribute into an Association owner->target (role = attribute name); a
PascalCase type naming no class/enum in the spec is coerced to String.
"""

import pytest

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler


@pytest.fixture
def handler():
    return ClassDiagramHandler(llm=None)


def _spec(classes, relationships=None):
    return {"systemName": "T", "classes": classes, "relationships": relationships or []}


class TestClassTypedAttributeGuard:
    def test_class_typed_attribute_becomes_association(self, handler):
        spec = _spec([
            {"className": "PointOfInterest", "isEnumeration": False,
             "attributes": [{"name": "name", "type": "String"},
                            {"name": "location", "type": "Location"}]},
            {"className": "Location", "isEnumeration": False,
             "attributes": [{"name": "lat", "type": "float"}]},
        ])
        handler._rewrite_class_typed_attributes(spec)
        poi = next(c for c in spec["classes"] if c["className"] == "PointOfInterest")
        assert [a["name"] for a in poi["attributes"]] == ["name"]  # primitive kept, class-typed dropped
        assert any(
            r["type"] == "Association" and r["source"] == "PointOfInterest"
            and r["target"] == "Location" and r["name"] == "location"
            for r in spec["relationships"]
        )

    def test_distinct_roles_for_multiple_links_to_same_class(self, handler):
        spec = _spec([
            {"className": "Route", "isEnumeration": False,
             "attributes": [{"name": "startLocation", "type": "Location"},
                            {"name": "endLocation", "type": "Location"}]},
            {"className": "Location", "isEnumeration": False, "attributes": []},
        ])
        handler._rewrite_class_typed_attributes(spec)
        route = next(c for c in spec["classes"] if c["className"] == "Route")
        assert route["attributes"] == []
        roles = sorted(r["name"] for r in spec["relationships"] if r["target"] == "Location")
        assert roles == ["endLocation", "startLocation"]

    def test_enum_typed_attribute_is_kept(self, handler):
        spec = _spec([
            {"className": "Task", "isEnumeration": False,
             "attributes": [{"name": "status", "type": "TaskStatus"}]},
            {"className": "TaskStatus", "isEnumeration": True,
             "attributes": [{"name": "OPEN"}, {"name": "DONE"}]},
        ])
        handler._rewrite_class_typed_attributes(spec)
        task = next(c for c in spec["classes"] if c["className"] == "Task")
        assert task["attributes"] == [{"name": "status", "type": "TaskStatus"}]
        assert spec["relationships"] == []

    def test_primitive_attributes_untouched(self, handler):
        spec = _spec([
            {"className": "Book", "isEnumeration": False,
             "attributes": [{"name": "title", "type": "String"},
                            {"name": "pages", "type": "int"}]},
        ])
        handler._rewrite_class_typed_attributes(spec)
        assert len(spec["classes"][0]["attributes"]) == 2
        assert spec["relationships"] == []

    def test_unknown_pascalcase_type_coerced_to_string(self, handler):
        spec = _spec([
            {"className": "Order", "isEnumeration": False,
             "attributes": [{"name": "total", "type": "Money"}]},  # Money not defined
        ])
        handler._rewrite_class_typed_attributes(spec)
        assert spec["classes"][0]["attributes"][0]["type"] == "String"
        assert spec["relationships"] == []

    def test_optional_class_attr_uses_0_1_multiplicity(self, handler):
        spec = _spec([
            {"className": "User", "isEnumeration": False,
             "attributes": [{"name": "avatar", "type": "Image", "isOptional": True}]},
            {"className": "Image", "isEnumeration": False, "attributes": []},
        ])
        handler._rewrite_class_typed_attributes(spec)
        rel = next(r for r in spec["relationships"] if r["target"] == "Image")
        assert rel["targetMultiplicity"] == "0..1"


def _cls(name, attributes=()):
    return {"className": name, "isEnumeration": False, "attributes": list(attributes)}


def _between(spec, a, b):
    return [r for r in spec["relationships"] if {r["source"], r["target"]} == {a, b}]


class TestExistingLinkIsNotDuplicated:
    """A class-typed attribute restating a relationship the classes already
    have adds no second relationship; a different role still does."""

    def test_composition_with_the_same_role_is_not_duplicated(self, handler):
        spec = _spec(
            [_cls("OrderLine", [{"name": "quantity", "type": "int"},
                                {"name": "order", "type": "Order"}]),
             _cls("Order")],
            [{"type": "Composition", "source": "OrderLine", "target": "Order",
              "sourceMultiplicity": "0..*", "targetMultiplicity": "1",
              "name": "order", "sourceRole": "lines"}],
        )
        handler._rewrite_class_typed_attributes(spec)
        assert [a["name"] for a in spec["classes"][0]["attributes"]] == ["quantity"]
        assert [r["type"] for r in _between(spec, "OrderLine", "Order")] == ["Composition"]

    def test_role_on_the_source_end_is_recognised(self, handler):
        # Order is the whole (target); it reaches its lines through sourceRole.
        spec = _spec(
            [_cls("Order", [{"name": "lines", "type": "OrderLine"}]), _cls("OrderLine")],
            [{"type": "Composition", "source": "OrderLine", "target": "Order",
              "sourceMultiplicity": "0..*", "targetMultiplicity": "1",
              "name": "order", "sourceRole": "lines"}],
        )
        handler._rewrite_class_typed_attributes(spec)
        assert spec["classes"][0]["attributes"] == []
        assert len(_between(spec, "Order", "OrderLine")) == 1

    def test_unnamed_end_is_the_link_named_after_the_class(self, handler):
        spec = _spec(
            [_cls("Chapter", [{"name": "book", "type": "Book"}]), _cls("Book")],
            [{"type": "Aggregation", "source": "Chapter", "target": "Book",
              "sourceMultiplicity": "0..*", "targetMultiplicity": "1"}],
        )
        handler._rewrite_class_typed_attributes(spec)
        assert spec["classes"][0]["attributes"] == []
        assert [r["type"] for r in _between(spec, "Chapter", "Book")] == ["Aggregation"]

    def test_plural_and_singular_of_the_class_name_are_one_link(self, handler):
        spec = _spec(
            [_cls("Book", [{"name": "chapter", "type": "Chapter"}]), _cls("Chapter")],
            [{"type": "Composition", "source": "Chapter", "target": "Book",
              "sourceMultiplicity": "0..*", "targetMultiplicity": "1",
              "name": "book", "sourceRole": "chapters"}],
        )
        handler._rewrite_class_typed_attributes(spec)
        assert len(_between(spec, "Chapter", "Book")) == 1

    def test_an_end_with_its_own_role_is_another_link(self, handler):
        """``customer`` beside an existing ``referredBy``: the purchaser is not the
        referrer, so the attribute still becomes its own association."""
        spec = _spec(
            [_cls("Order", [{"name": "customer", "type": "Customer"}]), _cls("Customer")],
            [{"type": "Association", "source": "Order", "target": "Customer",
              "sourceMultiplicity": "0..*", "targetMultiplicity": "0..1",
              "name": "referredBy"}],
        )
        handler._rewrite_class_typed_attributes(spec)
        roles = sorted(r["name"] for r in _between(spec, "Order", "Customer"))
        assert roles == ["customer", "referredBy"]

    def test_an_attribute_a_rule_reads_keeps_an_end_of_its_name(self, handler):
        spec = _spec(
            [_cls("Book", [{"name": "chapter", "type": "Chapter"}]), _cls("Chapter")],
            [{"type": "Composition", "source": "Chapter", "target": "Book",
              "sourceMultiplicity": "0..*", "targetMultiplicity": "1",
              "name": "book", "sourceRole": "chapters"}],
        )
        spec["constraints"] = [{"context": "Book", "name": "opens",
                                "expression": "context Book inv opens: self.chapter.title <> ''"}]
        handler._rewrite_class_typed_attributes(spec)
        assert "chapter" in [r["name"] for r in _between(spec, "Book", "Chapter")]

    def test_a_second_role_between_the_same_classes_survives(self, handler):
        spec = _spec(
            [_cls("Order", [{"name": "shippingAddress", "type": "Address"}]),
             _cls("Address")],
            [{"type": "Association", "source": "Order", "target": "Address",
              "sourceMultiplicity": "0..*", "targetMultiplicity": "1",
              "name": "billingAddress"}],
        )
        handler._rewrite_class_typed_attributes(spec)
        roles = sorted(r["name"] for r in _between(spec, "Order", "Address"))
        assert roles == ["billingAddress", "shippingAddress"]

    def test_two_roles_beside_a_composition_both_survive(self, handler):
        spec = _spec(
            [_cls("Order", [{"name": "billingAddress", "type": "Address"},
                            {"name": "shippingAddress", "type": "Address"}]),
             _cls("Address")],
            [{"type": "Composition", "source": "Address", "target": "Order",
              "sourceMultiplicity": "0..*", "targetMultiplicity": "1",
              "name": "order", "sourceRole": "addressBook"}],
        )
        handler._rewrite_class_typed_attributes(spec)
        assert spec["classes"][0]["attributes"] == []
        roles = sorted(r["name"] for r in _between(spec, "Order", "Address"))
        assert roles == ["billingAddress", "order", "shippingAddress"]

    def test_inheritance_between_the_classes_is_not_a_link(self, handler):
        spec = _spec(
            [_cls("Manager", [{"name": "employee", "type": "Employee"}]), _cls("Employee")],
            [{"type": "Inheritance", "source": "Manager", "target": "Employee"}],
        )
        handler._rewrite_class_typed_attributes(spec)
        assert [r["type"] for r in _between(spec, "Manager", "Employee")] == [
            "Inheritance", "Association"]


class TestCollectionOfAClass:
    """``List[Item]`` is a to-many link to Item, in every spelling."""

    @pytest.mark.parametrize("type_", [
        "List[Item]", "list[Item]", "List<Item>", "Set[Item]", "Item[]", "Array<Item>",
    ])
    def test_collection_of_a_class_becomes_a_to_many_association(self, handler, type_):
        spec = _spec([_cls("Cart", [{"name": "items", "type": type_}]), _cls("Item")])
        handler._rewrite_class_typed_attributes(spec)
        assert spec["classes"][0]["attributes"] == []
        assert spec["relationships"] == [{
            "type": "Association", "source": "Cart", "target": "Item",
            "sourceMultiplicity": "0..*", "targetMultiplicity": "0..*", "name": "items",
            "sourceRole": None, "associationClass": None,
        }]

    def test_collection_restating_an_existing_link_adds_nothing(self, handler):
        spec = _spec(
            [_cls("Cart", [{"name": "items", "type": "List[Item]"}]), _cls("Item")],
            [{"type": "Composition", "source": "Item", "target": "Cart",
              "sourceMultiplicity": "0..*", "targetMultiplicity": "1",
              "name": "cart", "sourceRole": "items"}],
        )
        handler._rewrite_class_typed_attributes(spec)
        assert spec["classes"][0]["attributes"] == []
        assert [r["type"] for r in spec["relationships"]] == ["Composition"]

    @pytest.mark.parametrize("type_", ["List[str]", "List[Unknown]", "Item[5]"])
    def test_other_bracketed_types_are_still_coerced_to_string(self, handler, type_):
        spec = _spec([_cls("Cart", [{"name": "labels", "type": type_}]), _cls("Item")])
        handler._rewrite_class_typed_attributes(spec)
        assert spec["classes"][0]["attributes"] == [{"name": "labels", "type": "String"}]
        assert spec["relationships"] == []

    def test_collection_of_an_enumeration_is_not_a_link(self, handler):
        spec = _spec([
            _cls("Cart", [{"name": "flags", "type": "List[Flag]"}]),
            {"className": "Flag", "isEnumeration": True, "attributes": [{"name": "GIFT"}]},
        ])
        handler._rewrite_class_typed_attributes(spec)
        assert spec["classes"][0]["attributes"] == [{"name": "flags", "type": "String"}]
        assert spec["relationships"] == []
