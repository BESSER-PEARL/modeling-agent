"""Tests for model helper utilities.

The ``to_int`` / ``extract_element_position`` / ``is_primary_layout_element`` /
``build_layout_anchor_lines`` helpers used to live in
``utilities.layout_helpers`` and were tested here. That module was removed
during a refactor and the helpers are not in use anywhere in src/ today
— so the tests for them are gone too. The remaining tests cover
``model_context`` and ``model_resolution``, which are still in use.
"""

import pytest
from utilities.model_context import (
    compact_model_summary,
    detailed_model_summary,
    is_diagram_nontrivial,
)
from utilities.model_resolution import (
    resolve_target_model,
    resolve_object_reference_diagram,
    count_reference_classes,
)
from utilities.model_resolution import resolve_class_diagram as _resolve_class_diagram
from protocol.types import AssistantRequest, WorkspaceContext

from tests.conftest import MINIMAL_CLASS_MODEL, EMPTY_CLASS_MODEL


# ---------------------------------------------------------------------------
# compact_model_summary
# ---------------------------------------------------------------------------

class TestCompactModelSummary:
    def test_class_diagram(self):
        summary = compact_model_summary(MINIMAL_CLASS_MODEL, "ClassDiagram")
        assert "1 class(es)" in summary
        assert "0 relationship" in summary

    def test_empty_model(self):
        summary = compact_model_summary(EMPTY_CLASS_MODEL, "ClassDiagram")
        assert "0 element" in summary

    def test_non_dict(self):
        summary = compact_model_summary(None, "ClassDiagram")
        assert "no structured model" in summary

    def test_gui_diagram(self):
        model = {"pages": [{"id": "p1"}, {"id": "p2"}]}
        summary = compact_model_summary(model, "GUINoCodeDiagram")
        assert "2 page" in summary

    def test_quantum_diagram(self):
        model = {"cols": [{"gates": []}, {"gates": []}]}
        summary = compact_model_summary(model, "QuantumCircuitDiagram")
        assert "2 circuit" in summary


# ---------------------------------------------------------------------------
# resolve_target_model
# ---------------------------------------------------------------------------

class TestResolveTargetModel:
    def test_active_diagram_match(self):
        request = AssistantRequest(
            diagram_type="ClassDiagram",
            current_model=MINIMAL_CLASS_MODEL,
            context=WorkspaceContext(active_diagram_type="ClassDiagram"),
        )
        result = resolve_target_model(request, "ClassDiagram")
        assert result is MINIMAL_CLASS_MODEL

    def test_from_snapshot(self):
        snapshot_model = {"elements": {}, "relationships": {}}
        request = AssistantRequest(
            diagram_type="ClassDiagram",
            context=WorkspaceContext(
                active_diagram_type="ClassDiagram",
                project_snapshot={
                    "diagrams": {
                        "ObjectDiagram": {"model": snapshot_model},
                    }
                },
            ),
        )
        result = resolve_target_model(request, "ObjectDiagram")
        assert result is snapshot_model

    def test_fallback_to_current_model(self):
        # Reconciled to feature behavior (commit b5ff3ec): current_model is
        # only used as a last-resort fallback when the active diagram type
        # MATCHES the requested target type. Returning current_model for a
        # mismatched target used to hallucinate diagrams that don't exist
        # (e.g. a brand-new AgentDiagram request resolving to the active
        # ClassDiagram's model), so the active type must equal the target
        # here for the fallback to fire.
        request = AssistantRequest(
            current_model=MINIMAL_CLASS_MODEL,
            context=WorkspaceContext(active_diagram_type="ClassDiagram"),
        )
        result = resolve_target_model(request, "ClassDiagram")
        assert result is MINIMAL_CLASS_MODEL

    def test_fallback_withheld_on_type_mismatch(self):
        # Companion case for the guard added in b5ff3ec: when the active
        # diagram type does NOT match the requested target, current_model
        # must NOT be used as a fallback.
        request = AssistantRequest(
            current_model=MINIMAL_CLASS_MODEL,
            context=WorkspaceContext(active_diagram_type="ObjectDiagram"),
        )
        result = resolve_target_model(request, "ClassDiagram")
        assert result is None

    def test_no_model_available(self):
        request = AssistantRequest(context=WorkspaceContext())
        assert resolve_target_model(request, "ClassDiagram") is None


# ---------------------------------------------------------------------------
# resolve_object_reference_diagram / count_reference_classes
# ---------------------------------------------------------------------------

class TestResolveObjectReference:
    def test_from_snapshot_class_diagram(self):
        class_model = {"elements": {"cls-1": {"type": "Class", "name": "User"}}, "relationships": {}}
        request = AssistantRequest(
            context=WorkspaceContext(
                active_diagram_type="ObjectDiagram",
                project_snapshot={
                    "diagrams": {
                        "ClassDiagram": {"model": class_model},
                    }
                },
            ),
        )
        result = resolve_object_reference_diagram(request, None)
        assert result is not None

    def test_count_reference_classes(self):
        ref_diagram = {"elements": {
            "cls-1": {"type": "Class", "name": "User"},
            "cls-2": {"type": "Class", "name": "Book"},
            "attr-1": {"type": "ClassAttribute", "name": "name"},
        }, "relationships": {}}
        count = count_reference_classes(ref_diagram)
        assert count == 2  # Only Class type elements

    def test_count_none(self):
        assert count_reference_classes(None) == 0

    def test_count_empty(self):
        assert count_reference_classes({}) == 0


# ---------------------------------------------------------------------------
# _resolve_class_diagram (from modeling_agent.py)
# ---------------------------------------------------------------------------

class TestResolveClassDiagram:
    def test_from_project_snapshot(self):
        request = AssistantRequest(
            context=WorkspaceContext(
                active_diagram_type="GUINoCodeDiagram",
                project_snapshot={
                    "diagrams": {
                        "ClassDiagram": {"model": MINIMAL_CLASS_MODEL},
                    }
                },
            ),
        )
        result = _resolve_class_diagram(request)
        assert result is not None
        assert "elements" in result

    def test_from_active_class_diagram(self):
        request = AssistantRequest(
            current_model=MINIMAL_CLASS_MODEL,
            context=WorkspaceContext(
                active_diagram_type="ClassDiagram",
            ),
        )
        result = _resolve_class_diagram(request)
        assert result is not None
        assert "elements" in result

    def test_no_class_diagram_available(self):
        request = AssistantRequest(
            context=WorkspaceContext(
                active_diagram_type="GUINoCodeDiagram",
                project_snapshot={"diagrams": {}},
            ),
        )
        assert _resolve_class_diagram(request) is None

    def test_empty_snapshot(self):
        request = AssistantRequest(
            context=WorkspaceContext(active_diagram_type="GUINoCodeDiagram"),
        )
        assert _resolve_class_diagram(request) is None

    def test_snapshot_without_model(self):
        request = AssistantRequest(
            context=WorkspaceContext(
                active_diagram_type="GUINoCodeDiagram",
                project_snapshot={
                    "diagrams": {
                        "ClassDiagram": {"title": "Class Diagram"},
                    }
                },
            ),
        )
        assert _resolve_class_diagram(request) is None


# ---------------------------------------------------------------------------
# detailed_model_summary – ClassDiagram
# ---------------------------------------------------------------------------

CLASS_MODEL_FULL = {
    "elements": {
        "c1": {"type": "Class", "name": "User", "attributes": ["a1", "a2"], "methods": ["m1"]},
        "a1": {"type": "ClassAttribute", "name": "+name: str", "owner": "c1", "attributeType": "str"},
        "a2": {"type": "ClassAttribute", "name": "-age: int", "owner": "c1", "attributeType": "int"},
        "m1": {"type": "ClassMethod", "name": "+login()", "owner": "c1"},
        "c2": {"type": "Class", "name": "Order", "attributes": [], "methods": []},
    },
    "relationships": {
        "r1": {
            "type": "Association",
            "name": "places",
            "source": {"element": "c1", "multiplicity": "1"},
            "target": {"element": "c2", "multiplicity": "*"},
        }
    },
}


class TestDetailedModelSummaryClassDiagram:
    def test_includes_class_names(self):
        result = detailed_model_summary(CLASS_MODEL_FULL, "ClassDiagram")
        assert "User" in result
        assert "Order" in result

    def test_includes_attributes(self):
        result = detailed_model_summary(CLASS_MODEL_FULL, "ClassDiagram")
        assert "name" in result
        assert "age" in result

    def test_includes_attribute_types(self):
        result = detailed_model_summary(CLASS_MODEL_FULL, "ClassDiagram")
        assert "str" in result
        assert "int" in result

    def test_includes_methods(self):
        result = detailed_model_summary(CLASS_MODEL_FULL, "ClassDiagram")
        assert "login" in result

    def test_includes_relationships(self):
        # Reconciled to feature behavior (commit 2cfcee9): relationship types
        # are now rendered as human-readable lowercase labels (e.g.
        # "association", "composition") via _REL_LABEL, not the raw
        # "Association" element-type string.
        result = detailed_model_summary(CLASS_MODEL_FULL, "ClassDiagram")
        assert "(association)" in result
        assert "User" in result and "Order" in result

    def test_includes_multiplicities(self):
        result = detailed_model_summary(CLASS_MODEL_FULL, "ClassDiagram")
        assert "1..*" in result

    def test_includes_relationship_name(self):
        result = detailed_model_summary(CLASS_MODEL_FULL, "ClassDiagram")
        assert "places" in result

    def test_prefix_current_class_diagram(self):
        result = detailed_model_summary(CLASS_MODEL_FULL, "ClassDiagram")
        assert result.startswith("Current class diagram:")

    def test_empty_elements(self):
        # Reconciled to feature behavior (commit 2cfcee9): the explicit
        # "Classes (N):" count header is now always emitted (even for N=0)
        # so factual queries ("how many classes?") have a stated number to
        # read — this means an empty model no longer falls back to the
        # compact summary (which is what previously carried the
        # "ClassDiagram" substring this test used to check for).
        result = detailed_model_summary({"elements": {}, "relationships": {}}, "ClassDiagram")
        assert "Classes (0)" in result

    def test_non_dict_input(self):
        result = detailed_model_summary(None, "ClassDiagram")
        assert "no model data" in result

    def test_class_without_attrs_or_methods(self):
        model = {
            "elements": {"c1": {"type": "Class", "name": "EmptyClass"}},
            "relationships": {},
        }
        result = detailed_model_summary(model, "ClassDiagram")
        assert "EmptyClass" in result
        assert "attributes" not in result  # No attr section for empty


# ---------------------------------------------------------------------------
# detailed_model_summary – StateMachineDiagram
# ---------------------------------------------------------------------------

STATE_MODEL_FULL = {
    "elements": {
        "s1": {"type": "StateInitialNode", "name": "Init"},
        "s2": {"type": "State", "name": "Idle", "entryAction": "reset", "exitAction": "", "doActivity": ""},
        "s3": {"type": "State", "name": "Processing", "entryAction": "", "exitAction": "cleanup", "doActivity": "compute"},
        "s4": {"type": "StateFinalNode", "name": "End"},
    },
    "relationships": {
        "t1": {
            "name": "start",
            "source": {"element": "s1"},
            "target": {"element": "s2"},
        },
        "t2": {
            "name": "process",
            "guard": "isValid",
            "effect": "log",
            "source": {"element": "s2"},
            "target": {"element": "s3"},
        },
        "t3": {
            "name": "finish",
            "source": {"element": "s3"},
            "target": {"element": "s4"},
        },
    },
}


class TestDetailedModelSummaryStateMachine:
    def test_includes_state_names(self):
        result = detailed_model_summary(STATE_MODEL_FULL, "StateMachineDiagram")
        assert "Idle" in result
        assert "Processing" in result

    def test_includes_state_types(self):
        # Reconciled to feature behavior (commit 83800f5): StateInitialNode /
        # StateFinalNode are pseudostates, not real states, and are now
        # deliberately EXCLUDED from the state list/count (this fixed an
        # off-by-one where the initial node was counted as an extra state).
        # They're instead surfaced as a separate, non-counted pseudostate
        # summary line — the literal type names no longer appear verbatim.
        result = detailed_model_summary(STATE_MODEL_FULL, "StateMachineDiagram")
        assert "States (2)" in result  # Idle, Processing only — pseudostates excluded
        assert "Pseudostates (not counted as states): 1 initial pseudostate(s), 1 final pseudostate(s)" in result

    def test_includes_entry_action(self):
        result = detailed_model_summary(STATE_MODEL_FULL, "StateMachineDiagram")
        assert "entry=reset" in result

    def test_includes_exit_action(self):
        result = detailed_model_summary(STATE_MODEL_FULL, "StateMachineDiagram")
        assert "exit=cleanup" in result

    def test_includes_do_activity(self):
        result = detailed_model_summary(STATE_MODEL_FULL, "StateMachineDiagram")
        assert "do=compute" in result

    def test_includes_transitions(self):
        result = detailed_model_summary(STATE_MODEL_FULL, "StateMachineDiagram")
        assert "Idle -> Processing" in result

    def test_includes_trigger(self):
        result = detailed_model_summary(STATE_MODEL_FULL, "StateMachineDiagram")
        assert "process" in result

    def test_includes_guard(self):
        result = detailed_model_summary(STATE_MODEL_FULL, "StateMachineDiagram")
        assert "[isValid]" in result

    def test_includes_effect(self):
        result = detailed_model_summary(STATE_MODEL_FULL, "StateMachineDiagram")
        assert "/log" in result

    def test_prefix_current_state_machine(self):
        result = detailed_model_summary(STATE_MODEL_FULL, "StateMachineDiagram")
        assert result.startswith("Current state machine:")

    def test_empty_state_machine(self):
        # Reconciled to feature behavior (commit 83800f5): mirrors the
        # class-diagram change — the "States (N):" count header is always
        # emitted (even for N=0), so an empty model no longer falls back to
        # the compact summary that used to contain "StateMachineDiagram".
        result = detailed_model_summary({"elements": {}, "relationships": {}}, "StateMachineDiagram")
        assert "States (0)" in result


# ---------------------------------------------------------------------------
# detailed_model_summary – ObjectDiagram
# ---------------------------------------------------------------------------

OBJECT_MODEL_FULL = {
    "elements": {
        "o1": {
            "type": "Object",
            "name": "book1",
            "className": "Book",
            "attributes": ["oa1", "oa2"],
        },
        "oa1": {"type": "ObjectAttribute", "name": "title", "value": "1984", "owner": "o1"},
        "oa2": {"type": "ObjectAttribute", "name": "pages", "value": "328", "owner": "o1"},
        "o2": {
            "type": "Object",
            "name": "author1",
            "className": "Author",
            "attributes": [],
        },
    },
    "relationships": {},
}


class TestDetailedModelSummaryObjectDiagram:
    def test_includes_object_names(self):
        result = detailed_model_summary(OBJECT_MODEL_FULL, "ObjectDiagram")
        assert "book1" in result
        assert "author1" in result

    def test_includes_class_reference(self):
        result = detailed_model_summary(OBJECT_MODEL_FULL, "ObjectDiagram")
        assert ": Book" in result
        assert ": Author" in result

    def test_includes_attribute_values(self):
        result = detailed_model_summary(OBJECT_MODEL_FULL, "ObjectDiagram")
        assert "title=1984" in result
        assert "pages=328" in result

    def test_prefix_current_object_diagram(self):
        result = detailed_model_summary(OBJECT_MODEL_FULL, "ObjectDiagram")
        assert result.startswith("Current object diagram:")

    def test_empty_object_diagram(self):
        result = detailed_model_summary({"elements": {}, "relationships": {}}, "ObjectDiagram")
        assert "ObjectDiagram" in result


# ---------------------------------------------------------------------------
# detailed_model_summary – GUINoCodeDiagram
# ---------------------------------------------------------------------------

GUI_MODEL = {
    "pages": [
        {
            "name": "Home",
            "frames": [
                {
                    "component": {
                        "components": [
                            {"type": "hero-section"},
                            {"type": "feature-section"},
                        ]
                    }
                }
            ],
        },
        {"name": "About", "frames": [{"component": {"components": []}}]},
    ]
}


class TestDetailedModelSummaryGUI:
    def test_includes_page_names(self):
        result = detailed_model_summary(GUI_MODEL, "GUINoCodeDiagram")
        assert "Home" in result
        assert "About" in result

    def test_includes_section_counts(self):
        result = detailed_model_summary(GUI_MODEL, "GUINoCodeDiagram")
        assert "2 section(s)" in result
        assert "0 section(s)" in result

    def test_prefix_current_gui_model(self):
        result = detailed_model_summary(GUI_MODEL, "GUINoCodeDiagram")
        assert result.startswith("Current GUI model:")


# ---------------------------------------------------------------------------
# detailed_model_summary – AgentDiagram
# ---------------------------------------------------------------------------

AGENT_MODEL = {
    "elements": {
        "s1": {"type": "AgentState", "name": "Greeting"},
        "s2": {"type": "AgentState", "name": "Farewell"},
        "i1": {"type": "AgentIntent", "name": "say_hello"},
        "i2": {"type": "AgentIntent", "name": "say_bye"},
    },
    # NOTE: relationship reconciled to the real element schema — the source
    # code (_summarize_agent_diagram) only recognizes "AgentStateTransition"
    # / "AgentStateTransitionInit" relationship types, and expects
    # source/target as {"element": <id>} dicts (matching CLASS_MODEL_FULL /
    # STATE_MODEL_FULL above). The original "AgentTransition" type with bare
    # string source/target never matched, so no transition line was ever
    # produced by this fixture.
    "relationships": {
        "t1": {"type": "AgentStateTransition", "source": {"element": "s1"}, "target": {"element": "s2"}},
    },
}


class TestDetailedModelSummaryAgent:
    def test_prefix_current_agent_diagram(self):
        result = detailed_model_summary(AGENT_MODEL, "AgentDiagram")
        assert result.startswith("Current agent diagram:")

    def test_includes_states(self):
        result = detailed_model_summary(AGENT_MODEL, "AgentDiagram")
        assert "Greeting" in result
        assert "Farewell" in result

    def test_includes_intents(self):
        result = detailed_model_summary(AGENT_MODEL, "AgentDiagram")
        assert "say_hello" in result
        assert "say_bye" in result

    def test_includes_transitions(self):
        # Reconciled to feature behavior: _summarize_agent_diagram renders
        # transitions with an ASCII "->" (like the class/state-machine
        # summaries), not a unicode "→" arrow.
        result = detailed_model_summary(AGENT_MODEL, "AgentDiagram")
        assert "Greeting -> Farewell" in result

    def test_empty_agent_model(self):
        model = {"elements": {}, "relationships": {}}
        result = detailed_model_summary(model, "AgentDiagram")
        # Falls back to compact summary
        assert "AgentDiagram" in result


# ---------------------------------------------------------------------------
# detailed_model_summary – QuantumCircuitDiagram
# ---------------------------------------------------------------------------

# Minimal quantum circuit fixture: 2 qubits, 3 columns — H then CNOT
QUANTUM_MODEL = {
    "qubitCount": 2,
    "cols": [
        ["H", 1],       # Col 0: H on q0
        ["*", "X"],      # Col 1: CNOT (control q0, target q1)
        [1, "Measure"],  # Col 2: Measure on q1
    ],
    "gates": [],
    "gateMetadata": {},
    "classicalBitCount": 0,
    "version": "1.0.0",
}


class TestDetailedModelSummaryQuantum:
    def test_prefix_current_quantum_circuit(self):
        result = detailed_model_summary(QUANTUM_MODEL, "QuantumCircuitDiagram")
        assert result.startswith("Current quantum circuit:")

    def test_qubit_and_column_counts(self):
        result = detailed_model_summary(QUANTUM_MODEL, "QuantumCircuitDiagram")
        assert "Qubits: 2" in result
        assert "Columns (time steps): 3" in result

    def test_hadamard_gate(self):
        result = detailed_model_summary(QUANTUM_MODEL, "QuantumCircuitDiagram")
        assert "q0: H (Hadamard)" in result

    def test_cnot_gate(self):
        result = detailed_model_summary(QUANTUM_MODEL, "QuantumCircuitDiagram")
        # Control dot maps to ● (control) and target to X (Pauli-X/NOT)
        assert "q0: \u25cf (control)" in result
        assert "q1: X (Pauli-X/NOT)" in result

    def test_measure_gate(self):
        result = detailed_model_summary(QUANTUM_MODEL, "QuantumCircuitDiagram")
        assert "q1: MEASURE" in result

    def test_empty_circuit(self):
        model = {"qubitCount": 3, "cols": []}
        result = detailed_model_summary(model, "QuantumCircuitDiagram")
        # Even with no columns the header line is produced
        assert "Qubits: 3" in result
        assert "Columns (time steps): 0" in result

    def test_identity_wires_omitted(self):
        """Wires with value 1 (identity) should not appear in output."""
        model = {"qubitCount": 2, "cols": [[1, "X"]]}
        result = detailed_model_summary(model, "QuantumCircuitDiagram")
        assert "q0:" not in result  # q0 is identity
        assert "q1: X (Pauli-X/NOT)" in result

    def test_qubit_count_inferred_from_cols(self):
        """When qubitCount is missing/0, infer from max column length."""
        model = {"cols": [["H", "X", "Y"]]}
        result = detailed_model_summary(model, "QuantumCircuitDiagram")
        assert "Qubits: 3" in result

    def test_truncation_of_large_circuits(self):
        """Circuits with >30 columns should be truncated."""
        model = {"qubitCount": 1, "cols": [["H"]] * 35}
        result = detailed_model_summary(model, "QuantumCircuitDiagram")
        assert "5 more column(s)" in result

    def test_special_gate_symbols(self):
        """Quirk symbols map to human-readable names."""
        model = {"qubitCount": 1, "cols": [["Z^1/2"], ["Z^-1/4"], ["Swap"]]}
        result = detailed_model_summary(model, "QuantumCircuitDiagram")
        assert "S" in result    # Z^1/2 -> S
        assert "T†" in result   # Z^-1/4 -> T†
        assert "SWAP" in result


# ---------------------------------------------------------------------------
# detailed_model_summary – Unknown / fallback
# ---------------------------------------------------------------------------

class TestDetailedModelSummaryFallback:
    def test_unknown_diagram_type_falls_back(self):
        model = {"elements": {"e1": {}}, "relationships": {}}
        result = detailed_model_summary(model, "UnknownDiagram")
        assert "UnknownDiagram" in result

    def test_none_model(self):
        assert "no model data" in detailed_model_summary(None, "ClassDiagram")

    def test_string_model(self):
        assert "no model data" in detailed_model_summary("not a dict", "ClassDiagram")


# ---------------------------------------------------------------------------
# is_diagram_nontrivial — used by describe-model to filter empty/seed diagrams
# ---------------------------------------------------------------------------

class TestIsDiagramNontrivial:
    # ClassDiagram
    def test_class_diagram_with_named_class_is_nontrivial(self):
        model = {
            "elements": {
                "c1": {"type": "Class", "name": "User"},
            },
            "relationships": {},
        }
        assert is_diagram_nontrivial(model, "ClassDiagram") is True

    def test_class_diagram_empty_is_trivial(self):
        assert is_diagram_nontrivial(
            {"elements": {}, "relationships": {}}, "ClassDiagram"
        ) is False

    def test_class_diagram_unnamed_class_is_trivial(self):
        model = {
            "elements": {"c1": {"type": "Class", "name": "  "}},
            "relationships": {},
        }
        assert is_diagram_nontrivial(model, "ClassDiagram") is False

    # ObjectDiagram
    def test_object_diagram_with_object_is_nontrivial(self):
        model = {"elements": {"o1": {"type": "Object", "name": "alice"}}}
        assert is_diagram_nontrivial(model, "ObjectDiagram") is True

    def test_object_diagram_empty_is_trivial(self):
        assert is_diagram_nontrivial({"elements": {}}, "ObjectDiagram") is False

    # StateMachineDiagram
    def test_state_machine_with_state_is_nontrivial(self):
        model = {"elements": {"s1": {"type": "State", "name": "Idle"}}}
        assert is_diagram_nontrivial(model, "StateMachineDiagram") is True

    def test_state_machine_only_initial_is_trivial(self):
        # Only an initial-node marker without a real state = seed content.
        model = {"elements": {"i1": {"type": "StateInitialNode"}}}
        assert is_diagram_nontrivial(model, "StateMachineDiagram") is False

    def test_state_machine_empty_is_trivial(self):
        assert is_diagram_nontrivial(
            {"elements": {}, "relationships": {}}, "StateMachineDiagram"
        ) is False

    # AgentDiagram
    def test_agent_diagram_with_state_is_nontrivial(self):
        model = {"elements": {"a1": {"type": "AgentState", "name": "Greet"}}}
        assert is_diagram_nontrivial(model, "AgentDiagram") is True

    def test_agent_diagram_empty_is_trivial(self):
        assert is_diagram_nontrivial({"elements": {}}, "AgentDiagram") is False

    # GUINoCodeDiagram
    def test_gui_with_pages_is_nontrivial(self):
        model = {"pages": [{"name": "Home"}]}
        assert is_diagram_nontrivial(model, "GUINoCodeDiagram") is True

    def test_gui_without_pages_is_trivial(self):
        assert is_diagram_nontrivial({"pages": []}, "GUINoCodeDiagram") is False
        assert is_diagram_nontrivial({}, "GUINoCodeDiagram") is False

    # QuantumCircuitDiagram
    def test_quantum_circuit_no_gates_is_trivial(self):
        model = {"qubitCount": 16, "cols": []}
        assert is_diagram_nontrivial(model, "QuantumCircuitDiagram") is False

    def test_quantum_circuit_three_random_gates_is_trivial(self):
        # Reproduces the seed-content case described in the bug:
        # 16 qubits, 3 single-qubit gates, no entanglement, no measurement.
        model = {
            "qubitCount": 16,
            "cols": [
                ["H"] + [1] * 15,
                ["X"] + [1] * 15,
                ["Y"] + [1] * 15,
            ],
        }
        assert is_diagram_nontrivial(model, "QuantumCircuitDiagram") is False

    def test_quantum_circuit_with_cnot_is_nontrivial(self):
        # A control + target in the same column = real entanglement.
        model = {
            "qubitCount": 2,
            "cols": [
                ["H", 1],
                ["•", "X"],
            ],
        }
        assert is_diagram_nontrivial(model, "QuantumCircuitDiagram") is True

    def test_quantum_circuit_with_measurement_is_nontrivial(self):
        model = {
            "qubitCount": 1,
            "cols": [["H"], ["Measure"]],
        }
        assert is_diagram_nontrivial(model, "QuantumCircuitDiagram") is True

    def test_quantum_circuit_dense_single_qubit_is_nontrivial(self):
        # >3 gates total, even single-qubit only, counts as deliberate.
        model = {
            "qubitCount": 1,
            "cols": [["H"], ["X"], ["Y"], ["Z"], ["S"]],
        }
        assert is_diagram_nontrivial(model, "QuantumCircuitDiagram") is True

    # Misc
    def test_non_dict_is_trivial(self):
        assert is_diagram_nontrivial(None, "ClassDiagram") is False
        assert is_diagram_nontrivial("nope", "ClassDiagram") is False
        assert is_diagram_nontrivial({}, "ClassDiagram") is False

    def test_unknown_diagram_type_is_permissive(self):
        # Unknown types fall through to "True if non-empty dict".
        assert is_diagram_nontrivial({"foo": "bar"}, "WeirdDiagram") is True
        assert is_diagram_nontrivial({}, "WeirdDiagram") is False
