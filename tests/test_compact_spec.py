"""Compact structured-output spec: string decoding + expansion + handler wiring.

The compact schema (schemas/compact_class_diagram.py) makes complete-system
generation ~2.4x faster by letting the LLM emit dense strings instead of one
JSON object per member. These tests pin the deterministic expansion — every
decoration of the encoding, every tolerant fallback — and that the handler
swaps schemas correctly (with the BESSER_AGENT_COMPACT_SPEC kill switch).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schemas.compact_class_diagram import (  # noqa: E402
    COMPACT_ENCODING_RULES,
    CompactClassSpec,
    CompactRelationshipSpec,
    CompactSystemClassSpec,
    _parse_attribute,
    _parse_method,
    expand_compact_spec,
)
from schemas import SystemClassSpec  # noqa: E402


# ---------------------------------------------------------------------------
# Attribute string decoding
# ---------------------------------------------------------------------------

class TestAttributeDecoding:
    def test_plain(self):
        a = _parse_attribute("price: float", False)
        assert (a.name, a.type, a.visibility) == ("price", "float", "public")
        assert not a.isDerived and not a.isOptional and a.defaultValue is None

    def test_full_decorations(self):
        a = _parse_attribute("- /total: float = 0.0 ?", False)
        assert a.name == "total"
        assert a.type == "float"
        assert a.visibility == "private"
        assert a.isDerived is True
        assert a.isOptional is True
        assert a.defaultValue == "0.0"

    @pytest.mark.parametrize("prefix,vis", [
        ("+", "public"), ("-", "private"), ("#", "protected"), ("~", "package"),
    ])
    def test_visibilities(self, prefix, vis):
        assert _parse_attribute(f"{prefix}x: int", False).visibility == vis

    def test_enum_literal_bare_name(self):
        a = _parse_attribute("AVAILABLE", True)
        assert a.name == "AVAILABLE"
        assert a.type is None

    def test_enum_class_strips_type_even_if_given(self):
        assert _parse_attribute("SOLD: str", True).type is None

    def test_optional_marker_glued_to_attribute_type(self):
        a = _parse_attribute("description: str?", False)
        assert (a.type, a.isOptional) == ("str", True)

    def test_default_value_with_spaces(self):
        a = _parse_attribute("status: str = in progress", False)
        assert a.defaultValue == "in progress"

    def test_junk_never_raises(self):
        for junk in ["", "   ", ":", "?", "= 5", "::::"]:
            a = _parse_attribute(junk, False)
            assert a.name  # some sane fallback name


# ---------------------------------------------------------------------------
# Method string decoding
# ---------------------------------------------------------------------------

class TestMethodDecoding:
    def test_void_no_params(self):
        m = _parse_method("close()")
        assert (m.name, m.returnType, m.parameters) == ("close", "void", [])

    def test_params_and_return(self):
        m = _parse_method("decreasePrice(percent: float) -> float")
        assert m.name == "decreasePrice"
        assert m.returnType == "float"
        assert len(m.parameters) == 1
        assert (m.parameters[0].name, m.parameters[0].type) == ("percent", "float")

    def test_multiple_params(self):
        m = _parse_method("book(room: Room, nights: int) -> Booking")
        assert [p.name for p in m.parameters] == ["room", "nights"]
        assert [p.type for p in m.parameters] == ["Room", "int"]

    def test_untyped_param_defaults(self):
        m = _parse_method("notify(message)")
        assert m.parameters[0].type == "String"

    def test_abstract_and_visibility(self):
        m = _parse_method("# calculate() -> float {abstract}")
        assert m.visibility == "protected"
        assert m.isAbstract is True
        assert m.returnType == "float"

    def test_junk_never_raises(self):
        for junk in ["", "()", "->", "((", "-> int"]:
            m = _parse_method(junk)
            assert m.name  # sane fallback

    def test_optional_marker_on_param_type_stripped(self):
        # Live case: 'str?' is not a BUML type — the '?' marker is defined
        # for attribute entries only, but the model glued it to a parameter.
        m = _parse_method("updateDetails(title: str, description: str?): any")
        assert [p.type for p in m.parameters] == ["str", "str"]
        assert m.returnType == "any"  # colon-style return is honored too

    def test_optional_marker_on_return_type_stripped(self):
        assert _parse_method("find() -> Book?").returnType == "Book"

    def test_unterminated_paren_still_parses_params(self):
        m = _parse_method("foo(x: int")
        assert [p.type for p in m.parameters] == ["int"]


# ---------------------------------------------------------------------------
# Full expansion
# ---------------------------------------------------------------------------

def _compact_sample() -> CompactSystemClassSpec:
    return CompactSystemClassSpec(
        name="LibrarySystem",
        classes=[
            CompactClassSpec(n="Book", a=["title: str", "price: float"],
                             m=["decreasePrice(percent: float)"], k=""),
            CompactClassSpec(n="Media", a=["id: str"], m=[], k="abstract"),
            CompactClassSpec(n="BookStatus", a=["AVAILABLE", "LOANED"],
                             m=[], k="enum"),
        ],
        rels=[
            CompactRelationshipSpec(f="Book", t="Media", k="inher",
                                    sm="", tm="", l=""),
            CompactRelationshipSpec(f="Library", t="Book", k="comp",
                                    sm="1", tm="*", l="catalog"),
        ],
        ocl=["context Book inv positivePrice: self.price > 0", "", "not an invariant"],
    )


class TestExpansion:
    def test_expands_to_canonical_spec(self):
        spec = expand_compact_spec(_compact_sample())
        assert isinstance(spec, SystemClassSpec)
        assert spec.systemName == "LibrarySystem"
        names = [c.className for c in spec.classes]
        assert names == ["Book", "Media", "BookStatus"]

    def test_class_kinds(self):
        spec = expand_compact_spec(_compact_sample())
        book, media, status = spec.classes
        assert not book.isAbstract and not book.isEnumeration
        assert media.isAbstract
        assert status.isEnumeration
        # Enum literals: bare names, no type, and no methods on enums.
        assert [a.name for a in status.attributes] == ["AVAILABLE", "LOANED"]
        assert all(a.type is None for a in status.attributes)
        assert status.methods == []

    def test_relationships_kinds_and_multiplicity_normalization(self):
        spec = expand_compact_spec(_compact_sample())
        inher, comp = spec.relationships
        assert inher.type == "Inheritance"
        assert (inher.source, inher.target) == ("Book", "Media")
        assert comp.type == "Composition"
        assert comp.name == "catalog"
        # '*' goes through the canonical validator -> '0..*'
        assert comp.targetMultiplicity == "0..*"

    def test_ocl_parsed_and_junk_dropped(self):
        spec = expand_compact_spec(_compact_sample())
        assert len(spec.constraints) == 1
        c = spec.constraints[0]
        assert c.context == "Book"
        assert c.name == "positivePrice"
        assert c.expression.startswith("context Book inv")

    def test_dump_shape_matches_downstream_expectations(self):
        # The guards in generate_complete_system operate on this dict shape.
        d = expand_compact_spec(_compact_sample()).model_dump()
        assert set(d.keys()) == {"systemName", "classes", "relationships", "constraints"}
        assert d["classes"][0]["attributes"][0]["name"] == "title"


# ---------------------------------------------------------------------------
# Handler wiring
# ---------------------------------------------------------------------------

class TestHandlerWiring:
    def _run(self, monkeypatch, enabled):
        import diagram_handlers.types.class_diagram_handler as mod
        monkeypatch.setattr(mod, "COMPACT_SPEC_ENABLED", enabled, raising=True)
        handler = mod.ClassDiagramHandler(None)
        captured = {}

        def fake_two_pass(**kwargs):
            captured.update(kwargs)
            if kwargs["response_schema"] is CompactSystemClassSpec:
                return _compact_sample()
            return expand_compact_spec(_compact_sample())

        monkeypatch.setattr(handler, "predict_two_pass_structured", fake_two_pass)
        result = handler.generate_complete_system("create a library system")
        return captured, result

    def test_compact_enabled_swaps_schema_and_expands(self, monkeypatch):
        captured, result = self._run(monkeypatch, enabled=True)
        assert captured["response_schema"] is CompactSystemClassSpec
        assert COMPACT_ENCODING_RULES.strip()[:20] in captured["system_prompt"]
        spec = result["systemSpec"]
        assert result["action"] == "inject_complete_system"
        assert [c["className"] for c in spec["classes"]] == ["Book", "Media", "BookStatus"]
        # Downstream guards + layout ran on the expanded spec.
        assert "relationships" in spec

    def test_kill_switch_restores_canonical_schema(self, monkeypatch):
        captured, result = self._run(monkeypatch, enabled=False)
        assert captured["response_schema"] is SystemClassSpec
        assert COMPACT_ENCODING_RULES.strip()[:20] not in captured["system_prompt"]
        assert result["action"] == "inject_complete_system"
