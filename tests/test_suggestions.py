"""Tests for the contextual suggestion engine."""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from suggestions import get_suggested_actions, format_suggestions_as_text


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _assert_valid_actions(actions):
    """Every action must be a dict with exactly 'label' and 'prompt' keys."""
    assert isinstance(actions, list)
    for action in actions:
        assert isinstance(action, dict)
        assert set(action.keys()) == {"label", "prompt"}
        assert isinstance(action["label"], str)
        assert isinstance(action["prompt"], str)


# ------------------------------------------------------------------
# 1. ClassDiagram complete_system
# ------------------------------------------------------------------

class TestClassDiagramCompleteSytem:
    def test_returns_non_empty_list(self):
        actions = get_suggested_actions("ClassDiagram", "complete_system")
        assert len(actions) > 0
        _assert_valid_actions(actions)

    def test_includes_generation_and_gui(self):
        actions = get_suggested_actions("ClassDiagram", "complete_system")
        labels = [a["label"] for a in actions]
        assert any("Generate" in l or "generate" in l for l in labels)


# ------------------------------------------------------------------
# 2. ClassDiagram single_element returns different suggestions
# ------------------------------------------------------------------

class TestClassDiagramSingleElement:
    def test_differs_from_complete_system(self):
        complete = get_suggested_actions("ClassDiagram", "complete_system")
        single = get_suggested_actions("ClassDiagram", "single_element")
        assert single != complete

    def test_returns_non_empty(self):
        actions = get_suggested_actions("ClassDiagram", "single_element")
        assert len(actions) > 0
        _assert_valid_actions(actions)


# ------------------------------------------------------------------
# 3. ClassDiagram modify_model
# ------------------------------------------------------------------

class TestClassDiagramModifyModel:
    def test_returns_modify_specific(self):
        actions = get_suggested_actions("ClassDiagram", "modify_model")
        labels = [a["label"] for a in actions]
        assert any("Describe" in l or "change" in l.lower() or "Make" in l for l in labels)
        _assert_valid_actions(actions)


# ------------------------------------------------------------------
# 4. StateMachineDiagram
# ------------------------------------------------------------------

class TestStateMachineDiagram:
    def test_returns_state_machine_suggestions(self):
        actions = get_suggested_actions("StateMachineDiagram", "complete_system")
        assert len(actions) > 0
        _assert_valid_actions(actions)
        labels = [a["label"] for a in actions]
        assert any("state" in l.lower() for l in labels)


# ------------------------------------------------------------------
# 5. GUINoCodeDiagram with ClassDiagram available
# ------------------------------------------------------------------

class TestGUIWithClassDiagram:
    def test_returns_gui_suggestions_with_class(self):
        actions = get_suggested_actions(
            "GUINoCodeDiagram", "complete_system",
            available_diagrams=["ClassDiagram"],
        )
        assert len(actions) > 0
        _assert_valid_actions(actions)
        labels = [a["label"] for a in actions]
        # Should suggest web app / react generation, not "create backend"
        assert any("web" in l.lower() or "react" in l.lower() or "Generate" in l for l in labels)


# ------------------------------------------------------------------
# 6. GUINoCodeDiagram without ClassDiagram
# ------------------------------------------------------------------

class TestGUIWithoutClassDiagram:
    def test_returns_different_suggestions(self):
        with_class = get_suggested_actions(
            "GUINoCodeDiagram", "complete_system",
            available_diagrams=["ClassDiagram"],
        )
        without_class = get_suggested_actions(
            "GUINoCodeDiagram", "complete_system",
            available_diagrams=[],
        )
        assert with_class != without_class

    def test_suggests_create_backend_model(self):
        actions = get_suggested_actions(
            "GUINoCodeDiagram", "complete_system",
            available_diagrams=[],
        )
        labels = [a["label"] for a in actions]
        assert any("backend" in l.lower() or "class" in l.lower() for l in labels)


# ------------------------------------------------------------------
# 7. AgentDiagram
# ------------------------------------------------------------------

class TestAgentDiagram:
    def test_returns_agent_suggestions(self):
        actions = get_suggested_actions("AgentDiagram", "complete_system")
        assert len(actions) > 0
        _assert_valid_actions(actions)
        labels = [a["label"] for a in actions]
        assert any("agent" in l.lower() for l in labels)


# ------------------------------------------------------------------
# 8. ObjectDiagram
# ------------------------------------------------------------------

class TestObjectDiagram:
    def test_returns_object_diagram_suggestions(self):
        actions = get_suggested_actions("ObjectDiagram", "complete_system")
        assert len(actions) > 0
        _assert_valid_actions(actions)
        labels = [a["label"] for a in actions]
        assert any("object" in l.lower() for l in labels)


# ------------------------------------------------------------------
# 9. QuantumCircuitDiagram
# ------------------------------------------------------------------

class TestQuantumCircuitDiagram:
    def test_returns_quantum_suggestions(self):
        actions = get_suggested_actions("QuantumCircuitDiagram", "complete_system")
        assert len(actions) > 0
        _assert_valid_actions(actions)
        labels = [a["label"] for a in actions]
        assert any("qiskit" in l.lower() or "circuit" in l.lower() or "gate" in l.lower() for l in labels)


# ------------------------------------------------------------------
# 10. Generation mode
# ------------------------------------------------------------------

class TestGenerationMode:
    def test_returns_post_generation_suggestions(self):
        actions = get_suggested_actions(
            "ClassDiagram", "generation", generator_type="python",
        )
        assert len(actions) > 0
        _assert_valid_actions(actions)
        labels = [a["label"] for a in actions]
        # Should offer to generate another format or modify
        assert any("another" in l.lower() or "modify" in l.lower() for l in labels)

    def test_generation_mode_via_generator_type_only(self):
        """Even without explicit operation_mode='generation', setting generator_type triggers post-gen."""
        actions = get_suggested_actions(
            "ClassDiagram", "complete_system", generator_type="python",
        )
        assert len(actions) > 0
        labels = [a["label"] for a in actions]
        assert any("another" in l.lower() or "modify" in l.lower() for l in labels)


# ------------------------------------------------------------------
# 11. Empty diagram_type
# ------------------------------------------------------------------

class TestEmptyDiagramType:
    def test_returns_empty_list(self):
        actions = get_suggested_actions("", "complete_system")
        assert actions == []

    def test_none_diagram_type_with_no_generator(self):
        """None diagram_type and no generator_type should yield empty list."""
        actions = get_suggested_actions("", "", generator_type=None)
        assert actions == []


# ------------------------------------------------------------------
# 12. Unknown diagram type returns fallback
# ------------------------------------------------------------------

class TestUnknownDiagramType:
    def test_returns_fallback_suggestions(self):
        actions = get_suggested_actions("SomeUnknownDiagram", "complete_system")
        assert len(actions) > 0
        _assert_valid_actions(actions)
        labels = [a["label"] for a in actions]
        # Fallback includes generic generation / describe
        assert any("generate" in l.lower() or "describe" in l.lower() for l in labels)


# ------------------------------------------------------------------
# 13. Context-aware with model_summary and element_names
# ------------------------------------------------------------------

class TestContextAwareWithElementNames:
    def test_personalized_suggestions_reference_class_names(self):
        summary = {
            "element_names": ["Order", "Customer"],
            "element_count": 2,
            "relationship_count": 1,
        }
        actions = get_suggested_actions(
            "ClassDiagram", "complete_system", model_summary=summary,
        )
        assert len(actions) > 0
        _assert_valid_actions(actions)
        # At least one suggestion should reference an actual class name
        all_text = " ".join(a["label"] + " " + a["prompt"] for a in actions)
        assert "Order" in all_text


# ------------------------------------------------------------------
# 14. Context-aware: no relationships suggests adding them
# ------------------------------------------------------------------

class TestContextAwareNoRelationships:
    def test_suggests_adding_relationships(self):
        summary = {
            "element_names": ["Order", "Customer"],
            "element_count": 2,
            "relationship_count": 0,
        }
        actions = get_suggested_actions(
            "ClassDiagram", "complete_system", model_summary=summary,
        )
        all_text = " ".join(a["label"] + " " + a["prompt"] for a in actions)
        assert "relationship" in all_text.lower()


# ------------------------------------------------------------------
# 15. format_suggestions_as_text
# ------------------------------------------------------------------

class TestFormatSuggestionsAsText:
    def test_formats_into_markdown(self):
        actions = [
            {"label": "Generate Python code", "prompt": "generate python"},
            {"label": "Modify the model", "prompt": ""},
        ]
        text = format_suggestions_as_text(actions)
        assert "**What's next?**" in text
        assert "Generate Python code" in text
        assert "generate python" in text
        assert "Modify the model" in text


# ------------------------------------------------------------------
# 16. format_suggestions_as_text with empty list
# ------------------------------------------------------------------

class TestFormatSuggestionsEmpty:
    def test_returns_empty_string(self):
        text = format_suggestions_as_text([])
        assert text == ""


# ------------------------------------------------------------------
# 17. Suggestion structure validation
# ------------------------------------------------------------------

class TestSuggestionStructure:
    @pytest.mark.parametrize("diagram_type,mode", [
        ("ClassDiagram", "complete_system"),
        ("ClassDiagram", "single_element"),
        ("ClassDiagram", "modify_model"),
        ("StateMachineDiagram", "complete_system"),
        ("GUINoCodeDiagram", "complete_system"),
        ("AgentDiagram", "complete_system"),
        ("ObjectDiagram", "complete_system"),
        ("QuantumCircuitDiagram", "complete_system"),
    ])
    def test_each_action_has_label_and_prompt(self, diagram_type, mode):
        actions = get_suggested_actions(diagram_type, mode)
        _assert_valid_actions(actions)


# ------------------------------------------------------------------
# 18. Suggestion limit: no more than 4
# ------------------------------------------------------------------

class TestSuggestionLimit:
    @pytest.mark.parametrize("diagram_type,mode", [
        ("ClassDiagram", "complete_system"),
        ("ClassDiagram", "single_element"),
        ("ClassDiagram", "modify_model"),
        ("StateMachineDiagram", "complete_system"),
        ("GUINoCodeDiagram", "complete_system"),
        ("AgentDiagram", "complete_system"),
        ("ObjectDiagram", "complete_system"),
        ("QuantumCircuitDiagram", "complete_system"),
        ("UnknownType", "complete_system"),
    ])
    def test_no_more_than_four_suggestions(self, diagram_type, mode):
        actions = get_suggested_actions(diagram_type, mode)
        assert len(actions) <= 4

    def test_limit_with_context_aware(self):
        summary = {
            "element_names": ["A", "B", "C"],
            "element_count": 3,
            "relationship_count": 0,
        }
        actions = get_suggested_actions(
            "ClassDiagram", "complete_system", model_summary=summary,
        )
        assert len(actions) <= 4

    def test_limit_with_generation(self):
        actions = get_suggested_actions(
            "ClassDiagram", "generation", generator_type="python",
        )
        assert len(actions) <= 4
