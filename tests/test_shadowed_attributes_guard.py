"""Shadowed-attribute guard: a subclass must not redefine an attribute an
ancestor already declares (BUML validates attribute shadowing). The LLM stamps
id/createdAt/updatedAt on every class — including subclasses of a base class
that already has them — producing a wall of validation warnings (live case:
Person <- Doctor/Patient/Staff each re-declaring id/createdAt/updatedAt).
The guard strips the shadowed copies deterministically; method overrides are
deliberately untouched (overriding is legitimate OO).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler  # noqa: E402


def _attr(name):
    return {"name": name, "type": "str"}


def _cls(name, attrs, methods=None):
    return {"className": name,
            "attributes": [_attr(a) for a in attrs],
            "methods": methods or []}


def _inherit(child, parent):
    return {"type": "Inheritance", "source": child, "target": parent}


def _run(spec):
    ClassDiagramHandler(None)._strip_shadowed_attributes(spec)
    return {c["className"]: [a["name"] for a in c["attributes"]]
            for c in spec["classes"]}


class TestShadowedAttributeGuard:
    def test_live_case_person_hierarchy(self):
        spec = {
            "classes": [
                _cls("Person", ["id", "createdAt", "updatedAt", "name"]),
                _cls("Doctor", ["id", "createdAt", "updatedAt", "specialty"]),
                _cls("Patient", ["id", "createdAt", "insuranceNo"]),
                _cls("Staff", ["updatedAt", "role"]),
            ],
            "relationships": [
                _inherit("Doctor", "Person"),
                _inherit("Patient", "Person"),
                _inherit("Staff", "Person"),
            ],
        }
        out = _run(spec)
        assert out["Person"] == ["id", "createdAt", "updatedAt", "name"]
        assert out["Doctor"] == ["specialty"]
        assert out["Patient"] == ["insuranceNo"]
        assert out["Staff"] == ["role"]

    def test_transitive_chain(self):
        spec = {
            "classes": [
                _cls("A", ["id"]),
                _cls("B", ["bOnly"]),
                _cls("C", ["id", "cOnly"]),  # shadows A.id via B
            ],
            "relationships": [_inherit("B", "A"), _inherit("C", "B")],
        }
        out = _run(spec)
        assert out["C"] == ["cOnly"]

    def test_no_inheritance_untouched(self):
        spec = {
            "classes": [_cls("Book", ["id", "title"]), _cls("Author", ["id"])],
            "relationships": [
                {"type": "Association", "source": "Book", "target": "Author"},
            ],
        }
        out = _run(spec)
        assert out["Book"] == ["id", "title"]
        assert out["Author"] == ["id"]

    def test_non_shadowed_names_survive_and_methods_untouched(self):
        spec = {
            "classes": [
                _cls("Person", ["id"], methods=[{"name": "describe"}]),
                _cls("Doctor", ["id", "specialty"],
                     methods=[{"name": "describe"}]),  # legitimate override
            ],
            "relationships": [_inherit("Doctor", "Person")],
        }
        ClassDiagramHandler(None)._strip_shadowed_attributes(spec)
        doctor = spec["classes"][1]
        assert [a["name"] for a in doctor["attributes"]] == ["specialty"]
        assert [m["name"] for m in doctor["methods"]] == ["describe"]

    def test_inheritance_cycle_does_not_hang(self):
        spec = {
            "classes": [_cls("A", ["id", "x"]), _cls("B", ["id", "y"])],
            "relationships": [_inherit("A", "B"), _inherit("B", "A")],
        }
        out = _run(spec)  # must terminate; each sees the other as ancestor
        assert out["A"] == ["x"]
        assert out["B"] == ["y"]


class TestSanitizeMemberTypes:
    """Spec-level backstop: decorated types normalized on BOTH schema paths."""

    def _spec(self):
        return {
            "classes": [{
                "className": "Book",
                "attributes": [
                    {"name": "title", "type": "str"},
                    {"name": "subtitle", "type": "str?", "isOptional": False},
                ],
                "methods": [{
                    "name": "updateDetails", "returnType": "Book?",
                    "parameters": [
                        {"name": "title", "type": "str"},
                        {"name": "description", "type": "str?"},
                    ],
                }],
            }],
            "relationships": [],
        }

    def test_normalizes_attributes_params_and_returns(self):
        spec = self._spec()
        ClassDiagramHandler(None)._sanitize_member_types(spec)
        book = spec["classes"][0]
        subtitle = book["attributes"][1]
        assert (subtitle["type"], subtitle["isOptional"]) == ("str", True)
        method = book["methods"][0]
        assert method["returnType"] == "Book"
        assert [p["type"] for p in method["parameters"]] == ["str", "str"]

    def test_clean_spec_untouched(self):
        spec = self._spec()
        spec["classes"][0]["attributes"].pop()  # drop the decorated ones
        spec["classes"][0]["methods"] = []
        before = [dict(a) for a in spec["classes"][0]["attributes"]]
        ClassDiagramHandler(None)._sanitize_member_types(spec)
        assert spec["classes"][0]["attributes"] == before
