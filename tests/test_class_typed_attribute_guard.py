"""Tests for the class-typed-attribute guard.

An attribute whose type names another CLASS (e.g. PointOfInterest.location :
Location) crashes the deterministic FastAPI/SQLAlchemy/Pydantic generators and
forces the expensive LLM-from-scratch fallback. The guard rewrites such an
attribute into an Association owner->target (role = attribute name); a
PascalCase type naming no class/enum in the spec is coerced to String.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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
