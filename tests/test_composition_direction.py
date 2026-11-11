"""The whole of a composition is its TARGET end (BESSER-PEARL/BESSER#746).

The editor draws the filled diamond at a composition's target end, and
BESSER's JSON -> BUML conversion marks the target end ``is_composite``: the
target is the whole. The agent stated compositions the other way round
(``Brand -> BrandURL`` for "a Brand has many BrandURLs"), so the diamond landed
on BrandURL. Asked about it afterwards, the agent read the diagram summary
``Brand -> BrandURL (composition) [1..0..*]`` as "each BrandURL belongs to one
Brand", the opposite of what the canvas showed.
"""
import pytest

from diagram_handlers.core.layout_engine import _sugiyama_build_graph
from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler
from ingestion.file_conversion_handler import _validate_class_diagram_spec
from prompts import class_diagram as class_prompts
from prompts import file_conversion as file_prompts
from schemas.class_diagram import ClassModification, RelationshipSpec
from schemas.compact_class_diagram import COMPACT_ENCODING_RULES, CompactRelationshipSpec
from utilities.model_context import detailed_model_summary
from utilities.whole_part import is_reversed_composition, upper_bound, whole_part_kind


@pytest.fixture
def handler():
    return ClassDiagramHandler(None)


def _editor_model(source: str, target: str, source_mult: str, target_mult: str,
                  rel_type: str = "ClassComposition", name: str = "urls"):
    """A two-class editor model with one relationship ``source -> target``."""
    def cls(name):
        return {"id": f"c_{name}", "name": name, "type": "Class", "owner": None,
                "attributes": [], "methods": []}
    return {
        "type": "ClassDiagram",
        "elements": {"c_Brand": cls("Brand"), "c_BrandURL": cls("BrandURL")},
        "relationships": {
            "r1": {
                "id": "r1", "type": rel_type, "name": name,
                "source": {"element": f"c_{source}", "multiplicity": source_mult},
                "target": {"element": f"c_{target}", "multiplicity": target_mult},
            },
        },
    }


# ---------------------------------------------------------------------------
# Reading: the summary the LLM sees names the whole
# ---------------------------------------------------------------------------

class TestSummaryNamesTheWhole:
    def test_the_canvas_of_the_issue_reads_as_brandurl_owning_brand(self):
        # As drawn in the issue: the diamond (target) at BrandURL.
        summary = detailed_model_summary(
            _editor_model("Brand", "BrandURL", "1", "0..*"), "ClassDiagram")
        assert "BrandURL is the whole, Brand its part" in summary
        assert "diamond at BrandURL" in summary

    def test_a_correct_composition_reads_as_brand_owning_brandurl(self):
        summary = detailed_model_summary(
            _editor_model("BrandURL", "Brand", "0..*", "1"), "ClassDiagram")
        assert "Brand is the whole, BrandURL its part" in summary

    def test_aggregation_names_the_whole_too(self):
        summary = detailed_model_summary(
            _editor_model("BrandURL", "Brand", "0..*", "1", rel_type="ClassAggregation"),
            "ClassDiagram")
        assert "(aggregation: Brand is the whole, BrandURL its part" in summary

    def test_multiplicities_are_tagged_per_end(self):
        # The old "[1..0..*]" read as one range.
        summary = detailed_model_summary(
            _editor_model("Brand", "BrandURL", "1", "0..*", rel_type="ClassBidirectional"),
            "ClassDiagram")
        assert "[Brand 1, BrandURL 0..*]" in summary
        assert "1..0..*" not in summary
        assert "(association)" in summary


# ---------------------------------------------------------------------------
# Writing: a composition stated whole -> part is turned round
# ---------------------------------------------------------------------------

class TestSystemSpecGuard:
    def test_swaps_the_composition_of_the_issue(self, handler):
        spec = {"classes": [{"className": "Brand"}, {"className": "BrandURL"}],
                "relationships": [{
                    "type": "Composition", "source": "Brand", "target": "BrandURL",
                    "sourceMultiplicity": "1", "targetMultiplicity": "0..*",
                    "name": "urls", "sourceRole": "brand",
                }]}
        handler._fix_reversed_compositions(spec)
        assert spec["relationships"] == [{
            "type": "Composition", "source": "BrandURL", "target": "Brand",
            "sourceMultiplicity": "0..*", "targetMultiplicity": "1",
            "name": "brand", "sourceRole": "urls",
        }]

    def test_a_missing_source_role_leaves_the_new_target_unnamed(self, handler):
        rel = {"type": "Composition", "source": "Order", "target": "OrderLine",
               "sourceMultiplicity": "1", "targetMultiplicity": "1..*", "name": "lines"}
        handler._fix_reversed_compositions({"relationships": [rel]})
        assert (rel["source"], rel["target"]) == ("OrderLine", "Order")
        assert rel["sourceRole"] == "lines"
        assert "name" not in rel

    @pytest.mark.parametrize("rel", [
        # Already part -> whole.
        {"type": "Composition", "source": "BrandURL", "target": "Brand",
         "sourceMultiplicity": "0..*", "targetMultiplicity": "1"},
        # Both ends single-valued: no evidence either way.
        {"type": "Composition", "source": "Car", "target": "Engine",
         "sourceMultiplicity": "1", "targetMultiplicity": "1"},
        # A shared aggregate may legitimately be many.
        {"type": "Aggregation", "source": "Team", "target": "Player",
         "sourceMultiplicity": "1", "targetMultiplicity": "0..*"},
        {"type": "Association", "source": "Brand", "target": "BrandURL",
         "sourceMultiplicity": "1", "targetMultiplicity": "0..*"},
    ])
    def test_leaves_everything_else_alone(self, handler, rel):
        before = dict(rel)
        handler._fix_reversed_compositions({"relationships": [rel]})
        assert rel == before


class TestModificationGuard:
    def test_swaps_a_new_composition_stated_whole_to_part(self, handler):
        spec = {"modification": {
            "action": "add_relationship",
            "target": {"sourceClass": "Brand", "targetClass": "BrandURL"},
            "changes": {"relationshipType": "Composition", "sourceMultiplicity": "1",
                        "targetMultiplicity": "0..*", "name": "urls"},
        }}
        handler._fix_reversed_composition_mods(spec)
        mod = spec["modification"]
        assert mod["target"] == {"sourceClass": "BrandURL", "targetClass": "Brand"}
        assert mod["changes"]["sourceMultiplicity"] == "0..*"
        assert mod["changes"]["targetMultiplicity"] == "1"
        # "urls" named the part end; it must not land on the whole.
        assert "name" not in mod["changes"]

    def test_a_missing_multiplicity_is_no_evidence(self, handler):
        # Stated part -> whole as the prompt asks: kept, and the whole end made
        # single-valued instead of the editor's default "*".
        spec = {"modifications": [{
            "action": "add_relationship",
            "target": {"sourceClass": "OrderLine", "targetClass": "Order"},
            "changes": {"relationshipType": "Composition"},
        }]}
        handler._fix_reversed_composition_mods(spec)
        mod = spec["modifications"][0]
        assert mod["target"] == {"sourceClass": "OrderLine", "targetClass": "Order"}
        assert mod["changes"]["targetMultiplicity"] == "1"
        assert "sourceMultiplicity" not in mod["changes"]

    def test_a_missing_source_multiplicity_is_the_editor_default(self, handler):
        # Source defaults to "1" on the canvas, so a many-valued whole end is reversed.
        mod = {"action": "add_relationship",
               "target": {"sourceClass": "Brand", "targetClass": "BrandURL"},
               "changes": {"relationshipType": "Composition", "targetMultiplicity": "0..*"}}
        handler._fix_reversed_composition_mods({"modifications": [mod]})
        assert mod["target"] == {"sourceClass": "BrandURL", "targetClass": "Brand"}
        assert mod["changes"]["sourceMultiplicity"] == "0..*"
        assert mod["changes"]["targetMultiplicity"] == "1"


class TestModificationPipeline:
    """The guard sees only what the LLM wrote, never what a refactoring synthesised."""

    def test_extract_class_is_not_turned_round_a_second_time(self, handler):
        spec = {"modification": {
            "action": "extract_class",
            "target": {"className": "User"},
            "changes": {"className": "Address", "attributes": [{"name": "street"}],
                        "relationshipType": "Composition",
                        "sourceMultiplicity": "0..*", "targetMultiplicity": "1"},
        }}
        spec = handler._prepare_modification_spec(spec, None)
        rel = next(m for m in spec["modifications"] if m["action"] == "add_relationship")
        assert rel["target"] == {"sourceClass": "Address", "targetClass": "User"}
        assert rel["changes"]["name"] == "user"

    def test_an_llm_composition_is_still_guarded(self, handler):
        spec = {"modifications": [{
            "action": "add_relationship",
            "target": {"sourceClass": "Brand", "targetClass": "BrandURL"},
            "changes": {"relationshipType": "Composition", "sourceMultiplicity": "1",
                        "targetMultiplicity": "0..*"},
        }]}
        spec = handler._prepare_modification_spec(spec, None)
        assert spec["modifications"][0]["target"]["targetClass"] == "Brand"

    def test_an_existing_canvas_composition_is_never_turned_round(self, handler):
        # A diagram drawn before 8.1.1 with its diamond on the many-valued end:
        # editing that relationship keeps the user's ends; only a NEW
        # add_relationship is guarded.
        old_canvas = _editor_model("Brand", "BrandURL", "1", "0..*")
        mod = {"action": "modify_relationship",
               "target": {"sourceClass": "Brand", "targetClass": "BrandURL"},
               "changes": {"relationshipType": "Composition", "sourceMultiplicity": "1",
                           "targetMultiplicity": "1..*"}}
        spec = handler._prepare_modification_spec({"modifications": [mod]}, old_canvas)
        out = spec["modifications"][0]
        assert out["action"] == "modify_relationship"
        assert out["target"] == {"sourceClass": "Brand", "targetClass": "BrandURL"}
        assert (out["changes"]["sourceMultiplicity"], out["changes"]["targetMultiplicity"]) == ("1", "1..*")
        assert old_canvas["relationships"]["r1"]["source"]["element"] == "c_Brand"

    def test_keeps_a_part_to_whole_composition(self, handler):
        mod = {"action": "add_relationship",
               "target": {"sourceClass": "BrandURL", "targetClass": "Brand"},
               "changes": {"relationshipType": "Composition", "sourceMultiplicity": "0..*",
                           "targetMultiplicity": "1", "name": "brand"}}
        handler._fix_reversed_composition_mods({"modifications": [mod]})
        assert mod["target"]["targetClass"] == "Brand"
        assert mod["changes"]["name"] == "brand"


class TestRefactorings:
    def test_extract_class_makes_the_new_class_the_part(self, handler):
        primitives, _ = handler._expand_extract_class({
            "action": "extract_class",
            "target": {"className": "User"},
            "changes": {"className": "Address",
                        "attributes": [{"name": "street"}, {"name": "city"}]},
        })
        rel = next(p for p in primitives if p["action"] == "add_relationship")
        assert rel["changes"]["relationshipType"] == "Composition"
        assert rel["target"] == {"sourceClass": "Address", "targetClass": "User"}
        assert rel["changes"]["name"] == "user"

    def test_promote_attribute_makes_the_new_class_the_part(self, handler):
        # The editor's promoteAttribute links the two with a ClassComposition
        # whose whole (target) is the original class; the agent must match.
        primitives, _ = handler._expand_promote_attribute({
            "action": "promote_attribute",
            "target": {"className": "User", "attributeName": "address"},
            "changes": {"className": "Address"},
        })
        rel = next(p for p in primitives if p["action"] == "add_relationship")
        assert rel["changes"]["relationshipType"] == "Composition"
        assert rel["target"] == {"sourceClass": "Address", "targetClass": "User"}
        assert rel["changes"]["name"] == "user"

    def test_extract_class_swaps_given_multiplicities_with_the_ends(self, handler):
        rel = handler._refactor_relationship(
            "User", "Address", {"sourceMultiplicity": "1", "targetMultiplicity": "0..*"},
            "Composition")
        assert rel["target"] == {"sourceClass": "Address", "targetClass": "User"}
        assert rel["changes"]["sourceMultiplicity"] == "0..*"
        assert rel["changes"]["targetMultiplicity"] == "1"

    def test_an_extracted_association_still_starts_at_the_original_class(self, handler):
        rel = handler._refactor_relationship("User", "Address", {}, "Association")
        assert rel["target"] == {"sourceClass": "User", "targetClass": "Address"}
        assert rel["changes"]["name"] == "hasAddress"


class TestFileConversion:
    def test_a_reversed_composition_is_turned_round(self):
        result = _validate_class_diagram_spec({
            "classes": [{"className": "Brand"}, {"className": "BrandURL"}],
            "relationships": [{"type": "composition", "source": "Brand", "target": "BrandURL",
                               "sourceMultiplicity": "1", "targetMultiplicity": "*"}],
        }, "model.puml", "PlantUML")
        rel = result["systemSpec"]["relationships"][0]
        assert (rel["source"], rel["target"]) == ("BrandURL", "Brand")


# ---------------------------------------------------------------------------
# Layout and prompts follow the same convention
# ---------------------------------------------------------------------------

def test_layout_places_the_whole_above_its_part():
    classes = {"Brand": {}, "BrandURL": {}}
    _succ, pred, _parents, owner_of, _adj, edges = _sugiyama_build_graph(
        classes, [{"type": "Composition", "source": "BrandURL", "target": "Brand"}])
    assert owner_of == {"BrandURL": "Brand"}
    assert pred["BrandURL"] == {"Brand"}
    assert edges == [("Brand", "BrandURL", 1)]


def test_every_prompt_states_the_direction():
    rule_text = "the SOURCE is the PART and the TARGET is the WHOLE"
    assert rule_text in class_prompts.SYSTEM_GENERATION_PROMPT
    assert rule_text in class_prompts.CLASS_KEY_RULES_BLOCK
    assert "f is the PART and t the WHOLE" in COMPACT_ENCODING_RULES
    assert "PART" in CompactRelationshipSpec.model_fields["f"].description
    assert "WHOLE" in RelationshipSpec.model_fields["target"].description
    assert "targetClass the WHOLE" in ClassModification.model_fields["action"].description
    assert "source is the PART and target the WHOLE" in file_prompts.CLASS_DIAGRAM_SCHEMA


@pytest.mark.parametrize("value, expected", [
    ("1", 1), ("0..1", 1), ("0..*", float("inf")), ("*", float("inf")),
    ("1..*", float("inf")), ("2..5", 5), ("", None), (None, None), ("several", None),
])
def test_upper_bound(value, expected):
    assert upper_bound(value) == expected


def test_helpers():
    assert whole_part_kind("ClassComposition") == "composition"
    assert whole_part_kind("Aggregation") == "aggregation"
    assert whole_part_kind("Association") is None
    assert is_reversed_composition("1", "0..*")
    assert not is_reversed_composition("0..*", "1")
    assert not is_reversed_composition("1", "1")
