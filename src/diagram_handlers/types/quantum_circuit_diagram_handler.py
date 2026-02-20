"""
Quantum Circuit Diagram Handler
Handles generation of QuantumCircuitDiagram models.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..core.base_handler import BaseDiagramHandler
from utilities.model_helpers import detailed_model_summary

logger = logging.getLogger(__name__)

DEFAULT_QUBITS = 5
MAX_QUBITS = 12

GATE_SYMBOLS = {
    "H": "H",
    "X": "X",
    "Y": "Y",
    "Z": "Z",
    "S": "Z^1/2",
    "S_DAG": "Z^-1/2",
    "T": "Z^1/4",
    "T_DAG": "Z^-1/4",
    "MEASURE": "Measure",
    "SWAP": "Swap",
    "QFT": "QFT",
    "QFT_DAG": "QFT_dag",
    "PROB": "Chance",
    "AMPLITUDE": "Amps",
}


def _to_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def _normalize_qubit_count(value: Any, fallback: int = DEFAULT_QUBITS) -> int:
    parsed = _to_int(value, fallback)
    if parsed < 1:
        return fallback
    return min(parsed, MAX_QUBITS)


def _default_quantum_model(qubit_count: int = DEFAULT_QUBITS) -> Dict[str, Any]:
    return {
        "cols": [],
        "gates": [],
        "gateMetadata": {},
        "classicalBitCount": 0,
        "version": "1.0.0",
        "qubitCount": qubit_count,
    }


def _normalize_quantum_model(candidate: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(candidate, dict):
        return _default_quantum_model()

    model = copy.deepcopy(candidate)
    cols = model.get("cols")
    if not isinstance(cols, list):
        cols = []
    model["cols"] = cols
    model["gates"] = model.get("gates") if isinstance(model.get("gates"), list) else []
    model["gateMetadata"] = model.get("gateMetadata") if isinstance(model.get("gateMetadata"), dict) else {}
    model["classicalBitCount"] = _to_int(model.get("classicalBitCount"), 0)
    model["version"] = model.get("version") if isinstance(model.get("version"), str) else "1.0.0"

    inferred_qubits = 0
    for col in cols:
        if isinstance(col, list):
            inferred_qubits = max(inferred_qubits, len(col))
    model["qubitCount"] = _normalize_qubit_count(model.get("qubitCount"), fallback=max(inferred_qubits, DEFAULT_QUBITS))

    for index, col in enumerate(cols):
        if not isinstance(col, list):
            cols[index] = [1] * model["qubitCount"]
            continue
        if len(col) < model["qubitCount"]:
            cols[index] = col + [1] * (model["qubitCount"] - len(col))

    return model


def _normalize_gate_name(value: Any) -> str:
    if not isinstance(value, str):
        return "H"

    normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
    if normalized in {"CX", "CNOT"}:
        return "CNOT"
    if normalized in {"CZ", "CONTROLLED_Z"}:
        return "CZ"
    if normalized in {"CY", "CONTROLLED_Y"}:
        return "CY"
    if normalized in {"SWAP_PAIR", "SWAP2"}:
        return "SWAP_PAIR"
    return normalized


def _ensure_column(cols: List[List[Any]], index: int, qubit_count: int) -> None:
    while len(cols) <= index:
        cols.append([1] * qubit_count)


def _place_symbol(col: List[Any], row: int, symbol: Any) -> None:
    if row < 0 or row >= len(col):
        return
    col[row] = symbol


def _operation_to_placements(operation: Dict[str, Any]) -> Tuple[Optional[int], List[Tuple[int, Any]]]:
    gate_name = _normalize_gate_name(operation.get("gate"))
    explicit_column = operation.get("column")
    column = _to_int(explicit_column, -1) if explicit_column is not None else None

    if gate_name in {"CNOT", "CZ", "CY"}:
        control_row = _to_int(operation.get("controlRow"), 0)
        target_row = _to_int(operation.get("targetRow"), max(control_row + 1, 1))
        target_symbol = "X" if gate_name == "CNOT" else "Z" if gate_name == "CZ" else "Y"
        return column, [(control_row, "*"), (target_row, target_symbol)]

    if gate_name == "SWAP_PAIR":
        row = _to_int(operation.get("row"), 0)
        target_row = _to_int(operation.get("targetRow"), row + 1)
        return column, [(row, "Swap"), (target_row, "Swap")]

    row = _to_int(operation.get("row"), 0)
    symbol = GATE_SYMBOLS.get(gate_name, gate_name if gate_name else "H")
    return column, [(row, symbol)]


class QuantumCircuitDiagramHandler(BaseDiagramHandler):
    """Handler for Quantum circuit diagram generation."""

    def get_diagram_type(self) -> str:
        return "QuantumCircuitDiagram"

    def get_system_prompt(self) -> str:
        return """You are a quantum circuit modeling assistant.

Return ONLY JSON with this shape:
{
  "operation": {
    "gate": "H|X|Y|Z|S|T|MEASURE|CNOT|CZ|CY|SWAP|QFT",
    "row": 0,
    "column": 0,
    "controlRow": 0,
    "targetRow": 1
  }
}

Rules:
1. Use controlRow/targetRow only for controlled gates.
2. Keep row and column indexes zero-based.
3. Return JSON only."""

    def _apply_operations(
        self,
        model: Dict[str, Any],
        operations: List[Dict[str, Any]],
        append: bool,
        qubit_count_hint: Optional[int] = None,
    ) -> Dict[str, Any]:
        normalized = _normalize_quantum_model(model)
        existing_cols = normalized.get("cols", []) if append else []

        max_row = normalized.get("qubitCount", DEFAULT_QUBITS) - 1
        for op in operations:
            _, placements = _operation_to_placements(op)
            for row, _ in placements:
                max_row = max(max_row, row)

        qubit_count = _normalize_qubit_count(qubit_count_hint, fallback=max_row + 1)
        qubit_count = max(qubit_count, max_row + 1)

        cols: List[List[Any]] = []
        for col in existing_cols:
            if isinstance(col, list):
                normalized_col = col[:qubit_count] + ([1] * max(0, qubit_count - len(col)))
                cols.append(normalized_col)

        next_free_column = len(cols)

        for op in operations:
            column, placements = _operation_to_placements(op)
            target_column = next_free_column if column is None or column < 0 else column
            _ensure_column(cols, target_column, qubit_count)
            column_values = cols[target_column]
            for row, symbol in placements:
                _place_symbol(column_values, row, symbol)
            if column is None or column < 0:
                next_free_column += 1

        normalized["cols"] = cols
        normalized["qubitCount"] = qubit_count
        return normalized

    def generate_single_element(self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        prompt = self.get_system_prompt()

        try:
            response = self.predict_with_retry(f"{prompt}\n\nUser Request: {user_request}")
            spec = self.parse_json_safely(self.clean_json_response(response or ""))
            if not isinstance(spec, dict) or not isinstance(spec.get("operation"), dict):
                raise ValueError("Invalid operation spec")

            operation = spec.get("operation")
            model = self._apply_operations(_default_quantum_model(), [operation], append=False)
            return {
                "action": "inject_element",
                "diagramType": self.get_diagram_type(),
                "model": model,
                "message": "Added one quantum gate operation.",
            }
        except Exception:
            logger.error("[QuantumCircuit] generate_single_element FAILED", exc_info=True)
            return self.generate_fallback_element(user_request)

    def generate_complete_system(self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        prompt = """You are a quantum circuit design assistant.

Return ONLY JSON with this shape:
{
  "qubitCount": 3,
  "operations": [
    {"column": 0, "row": 0, "gate": "H"},
    {"column": 1, "gate": "CNOT", "controlRow": 0, "targetRow": 1},
    {"column": 2, "row": 0, "gate": "MEASURE"}
  ]
}

Rules:
1. Keep circuit concise and coherent.
2. Use zero-based row/column indexes.
3. Return JSON only."""

        try:
            response = self.predict_with_retry(f"{prompt}\n\nUser Request: {user_request}")
            spec = self.parse_json_safely(self.clean_json_response(response or ""))
            if not isinstance(spec, dict):
                raise ValueError("Invalid circuit spec")

            operations = spec.get("operations") if isinstance(spec.get("operations"), list) else []
            typed_operations = [op for op in operations if isinstance(op, dict)]
            if not typed_operations:
                raise ValueError("No operations generated")

            model = self._apply_operations(
                _default_quantum_model(),
                typed_operations,
                append=False,
                qubit_count_hint=spec.get("qubitCount"),
            )
            return {
                "action": "inject_complete_system",
                "diagramType": self.get_diagram_type(),
                "model": model,
                "message": f"Created quantum circuit with {len(model.get('cols', []))} column(s).",
            }
        except Exception:
            logger.error("[QuantumCircuit] generate_complete_system FAILED", exc_info=True)
            return self.generate_fallback_system()

    def generate_modification(self, user_request: str, current_model: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        base_model = _normalize_quantum_model(current_model)

        # Build context from current circuit using centralized helper
        context_block = ''
        if current_model and isinstance(current_model, dict):
            summary = detailed_model_summary(current_model, 'QuantumCircuitDiagram')
            if summary:
                context_block = f"\n\n{summary}"

        prompt = """You are a quantum circuit assistant.

Return ONLY JSON with this shape:
{
  "mode": "append|replace",
  "qubitCount": 3,
  "operations": [
    {"column": 0, "row": 0, "gate": "H"},
    {"gate": "CNOT", "controlRow": 0, "targetRow": 1}
  ]
}

Rules:
1. Use mode=append for adding new behavior.
2. Use mode=replace only when user asks to rebuild/reset.
3. Return JSON only."""

        try:
            response = self.predict_with_retry(f"{prompt}\n\nUser Request: {user_request}{context_block}")
            spec = self.parse_json_safely(self.clean_json_response(response or ""))
            if not isinstance(spec, dict):
                raise ValueError("Invalid modification spec")

            operations = spec.get("operations") if isinstance(spec.get("operations"), list) else []
            typed_operations = [op for op in operations if isinstance(op, dict)]
            if not typed_operations:
                raise ValueError("No operations for modification")

            mode = str(spec.get("mode", "append")).strip().lower()
            append = mode != "replace"
            model = self._apply_operations(
                base_model,
                typed_operations,
                append=append,
                qubit_count_hint=spec.get("qubitCount"),
            )
            action_label = "updated" if append else "replaced"
            return {
                "action": "modify_model",
                "diagramType": self.get_diagram_type(),
                "model": model,
                "message": f"Quantum circuit {action_label} with {len(typed_operations)} operation(s).",
            }
        except Exception:
            return {
                "action": "modify_model",
                "diagramType": self.get_diagram_type(),
                "model": base_model,
                "message": "Could not parse requested quantum modification; kept the current circuit unchanged.",
            }

    def generate_fallback_element(self, request: str) -> Dict[str, Any]:
        model = self._apply_operations(
            _default_quantum_model(qubit_count=2),
            [{"column": 0, "row": 0, "gate": "H"}],
            append=False,
        )
        return {
            "action": "inject_element",
            "diagramType": self.get_diagram_type(),
            "model": model,
            "message": "Added a basic Hadamard gate (fallback).",
        }

    def generate_fallback_system(self) -> Dict[str, Any]:
        model = self._apply_operations(
            _default_quantum_model(qubit_count=2),
            [
                {"column": 0, "row": 0, "gate": "H"},
                {"column": 1, "gate": "CNOT", "controlRow": 0, "targetRow": 1},
                {"column": 2, "row": 0, "gate": "MEASURE"},
                {"column": 2, "row": 1, "gate": "MEASURE"},
            ],
            append=False,
        )
        return {
            "action": "inject_complete_system",
            "diagramType": self.get_diagram_type(),
            "model": model,
            "message": "Created a basic Bell-state circuit (fallback).",
        }
