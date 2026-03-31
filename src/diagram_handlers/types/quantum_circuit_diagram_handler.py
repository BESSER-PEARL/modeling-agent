"""
Quantum Circuit Diagram Handler
Handles generation of QuantumCircuitDiagram models.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..core.base_handler import BaseDiagramHandler, LLMPredictionError
from schemas import SingleQuantumGateSpec, SystemQuantumCircuitSpec, QuantumModificationSpec
from utilities.model_context import detailed_model_summary

logger = logging.getLogger(__name__)

DEFAULT_QUBITS = 5
MAX_QUBITS = 12

# ---------------------------------------------------------------------------
# Gate type → serialized col symbol mapping.
# These MUST match the frontend's mapGateToQuirkSymbol / typeMap exactly so
# that the circuit editor can deserialize whatever the handler produces.
# ---------------------------------------------------------------------------
GATE_SYMBOLS = {
    # ---- Probes ----
    "MEASURE": "Measure",
    "MEASURE_X": "Measure X",
    "MEASURE_Y": "Measure Y",
    # CONTROL → "•" and ANTI_CONTROL → "◦" handled in _operation_to_placements

    # ---- Half Turns ----
    "H": "H",
    "X": "X",
    "Y": "Y",
    "Z": "Z",
    "SWAP": "Swap",

    # ---- Quarter Turns ----
    "S": "S",                   # SGate type='Z^½', serialised via label fallback
    "S_DAG": "Z^-½",
    "V": "V",                   # VGate type='X^½', serialised via label fallback
    "V_DAG": "X^-½",
    "SQRT_Y": "Y^½",
    "SQRT_Y_DAG": "Y^-½",

    # ---- Eighth Turns ----
    "T": "Z^¼",
    "T_DAG": "Z^-¼",
    "SQRT_SQRT_X": "X^¼",
    "SQRT_SQRT_X_DAG": "X^-¼",
    "SQRT_SQRT_Y": "Y^¼",
    "SQRT_SQRT_Y_DAG": "Y^-¼",

    # ---- Spinning (time-dependent) ----
    "Z_POW_T": "Z^t",
    "Z_POW_NEG_T": "Z^-t",
    "Y_POW_T": "Y^t",
    "Y_POW_NEG_T": "Y^-t",
    "X_POW_T": "X^t",
    "X_POW_NEG_T": "X^-t",

    # ---- Parametrized Rotations ----
    "X_POW": "X^t",
    "Y_POW": "Y^t",
    "Z_POW": "Z^t",
    "EXP_X": "Exp(-iXt)",
    "EXP_Y": "Exp(-iYt)",
    "EXP_Z": "Exp(-iZt)",

    # ---- Frequency ----
    "QFT": "QFT",
    "QFT_DAG": "QFT†",
    "PHASE_GRADIENT": "Grad",
    "PHASE_GRADIENT_DAG": "Grad†",
    "PHASE_GRADIENT_INV": "Grad⁻¹",
    "PHASE_GRADIENT_INV_DAG": "Grad⁻¹†",

    # ---- Arithmetic ----
    "INC": "+=1",
    "DEC": "-=1",
    "ADD": "+=A",
    "SUB": "-=A",
    "MUL": "*=A",
    "ADD_AB": "+AB",
    "SUB_AB": "-AB",
    "MUL_INV": "×A⁻¹",

    # ---- Modular Arithmetic ----
    "MOD_ADD": "+A mod R",
    "MOD_SUB": "-A mod R",
    "MOD_MUL": "*A mod R",
    "MOD_INV_MUL": "/A mod R",
    "MOD_INC": "+1 mod R",
    "MOD_DEC": "-1 mod R",
    "MOD_MUL_B": "*B mod R",
    "MOD_MUL_B_INV": "*B A⁻¹ mod R",

    # ---- Compare / Logic ----
    "COMPARE": "A < B",
    "GREATER_THAN": "A > B",
    "LESS_EQUAL": "A ≤ B",
    "GREATER_EQUAL": "A ≥ B",
    "EQUAL": "A = B",
    "NOT_EQUAL": "A ≠ B",
    "COMPARE_A_LT": "Input < A",
    "COMPARE_A_GT": "Input > A",
    "COMPARE_A_EQ": "Input = A",
    "COUNT_1S": "Count 1s",
    "CYCLE_BITS": "Cycle",
    "XOR": "⊕",

    # ---- Order ----
    "REVERSE_BITS": "Reverse",
    "ROTATE_BITS_LEFT": "<<",
    "ROTATE_BITS_RIGHT": ">>",
    # INTERLEAVE / DEINTERLEAVE → "<<N" handled in _operation_to_placements

    # ---- Scalar ----
    "ONE": "1",              # string "1", distinct from integer 1 (empty)
    "MINUS_ONE": "-1",
    "PHASE_I": "i",
    "PHASE_MINUS_I": "-i",
    "PHASE_SQRT_I": "√i",
    "PHASE_SQRT_MINUS_I": "√-i",

    # ---- Displays ----
    "PROB": "Chance",
    "AMPLITUDE": "Amps",
    "BLOCH": "Bloch",
    "DENSITY": "Density",
}

# Default heights for multi-qubit gates (height > 1).
MULTI_QUBIT_HEIGHTS: Dict[str, int] = {
    "SWAP": 2,
    "QFT": 2, "QFT_DAG": 2,
    "PHASE_GRADIENT": 2, "PHASE_GRADIENT_DAG": 2,
    "PHASE_GRADIENT_INV": 2, "PHASE_GRADIENT_INV_DAG": 2,
    "ADD": 2, "SUB": 2, "MUL": 2, "MUL_INV": 2,
    "ADD_AB": 3, "SUB_AB": 3,
    "MOD_ADD": 2, "MOD_SUB": 2, "MOD_MUL": 2, "MOD_INV_MUL": 2,
    "MOD_INC": 2, "MOD_DEC": 2,
    "MOD_MUL_B": 3, "MOD_MUL_B_INV": 3,
    "COMPARE": 2, "GREATER_THAN": 2, "LESS_EQUAL": 2, "GREATER_EQUAL": 2,
    "EQUAL": 2, "NOT_EQUAL": 2,
    "COUNT_1S": 2, "CYCLE_BITS": 2, "XOR": 2,
    "REVERSE_BITS": 2, "ROTATE_BITS_LEFT": 2, "ROTATE_BITS_RIGHT": 2,
    "INTERLEAVE": 6, "DEINTERLEAVE": 6,
    "FUNCTION": 2, "ORACLE": 2, "UNITARY": 2,
}

# Gate types that use the __FUNC__ serialization prefix.
FUNCTION_GATE_TYPES = {"FUNCTION", "ORACLE", "UNITARY"}


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


_GATE_ALIASES: Dict[str, str] = {
    # Controlled gates
    "CX": "CNOT", "CNOT": "CNOT",
    "CZ": "CZ", "CONTROLLED_Z": "CZ",
    "CY": "CY", "CONTROLLED_Y": "CY",
    "SWAP_PAIR": "SWAP_PAIR", "SWAP2": "SWAP_PAIR",
    "TOFFOLI": "TOFFOLI", "CCX": "TOFFOLI", "CCNOT": "TOFFOLI",
    # Common readable aliases → canonical names
    "HADAMARD": "H",
    "NOT": "X", "PAULI_X": "X", "BIT_FLIP": "X",
    "PAULI_Y": "Y",
    "PAULI_Z": "Z", "PHASE_FLIP": "Z",
    "S_GATE": "S", "PHASE_90": "S", "SQRT_Z": "S",
    "S_DAGGER": "S_DAG", "SDG": "S_DAG",
    "SQRT_X": "V", "V_GATE": "V",
    "V_DAGGER": "V_DAG", "VDG": "V_DAG",
    "SQRT_Y_DAGGER": "SQRT_Y_DAG",
    "T_GATE": "T", "PHASE_45": "T",
    "T_DAGGER": "T_DAG", "TDG": "T_DAG",
    "RX": "X_POW", "RY": "Y_POW", "RZ": "Z_POW",
    "QUANTUM_FOURIER_TRANSFORM": "QFT",
    "QFT_DAGGER": "QFT_DAG", "INVERSE_QFT": "QFT_DAG", "IQFT": "QFT_DAG",
    "GRAD": "PHASE_GRADIENT",
    "INCREMENT": "INC", "DECREMENT": "DEC",
    "ADDITION": "ADD", "SUBTRACT": "SUB", "SUBTRACTION": "SUB",
    "MULTIPLY": "MUL", "MULTIPLICATION": "MUL",
    "LESS_THAN": "COMPARE",
    "REVERSE": "REVERSE_BITS",
    "ROL": "ROTATE_BITS_LEFT", "ROR": "ROTATE_BITS_RIGHT",
    "IDENTITY_SCALAR": "ONE",
    "CHANCE": "PROB", "AMPS": "AMPLITUDE",
    "MEAS": "MEASURE", "MEASUREMENT": "MEASURE",
    "CTRL": "CONTROL", "ANTI_CTRL": "ANTI_CONTROL",
    "FUNC": "FUNCTION",
}


def _normalize_gate_name(value: Any) -> str:
    if not isinstance(value, str):
        return "H"

    normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
    return _GATE_ALIASES.get(normalized, normalized)


def _ensure_column(cols: List[List[Any]], index: int, qubit_count: int) -> None:
    while len(cols) <= index:
        cols.append([1] * qubit_count)


def _place_symbol(col: List[Any], row: int, symbol: Any) -> None:
    if row < 0 or row >= len(col):
        return
    col[row] = symbol


def _operation_to_placements(operation: Dict[str, Any]) -> Tuple[Optional[int], List[Tuple[int, Any]], Optional[Dict[str, Any]]]:
    """Convert a high-level operation dict into concrete (row, symbol) placements.

    Returns ``(column_or_None, placements, gate_metadata_or_None)`` where
    *gate_metadata* is only set for function-style gates that need an entry in
    ``model["gateMetadata"]``.
    """
    gate_name = _normalize_gate_name(operation.get("gate"))
    explicit_column = operation.get("column")
    column = _to_int(explicit_column, -1) if explicit_column is not None else None

    # --- Controlled two-qubit gates (CNOT, CZ, CY) --------------------------
    if gate_name in {"CNOT", "CZ", "CY"}:
        control_row = _to_int(operation.get("controlRow"), 0)
        target_row = _to_int(operation.get("targetRow"), max(control_row + 1, 1))
        target_symbol = "X" if gate_name == "CNOT" else "Z" if gate_name == "CZ" else "Y"
        return column, [(control_row, "\u2022"), (target_row, target_symbol)], None

    # --- Toffoli (CCX): two controls + X target ------------------------------
    if gate_name == "TOFFOLI":
        ctrl1 = _to_int(operation.get("controlRow"), 0)
        ctrl2 = _to_int(operation.get("controlRow2"), max(ctrl1 + 1, 1))
        target_row = _to_int(operation.get("targetRow"), max(ctrl2 + 1, 2))
        return column, [(ctrl1, "\u2022"), (ctrl2, "\u2022"), (target_row, "X")], None

    # --- SWAP pair -----------------------------------------------------------
    if gate_name == "SWAP_PAIR":
        row = _to_int(operation.get("row"), 0)
        target_row = _to_int(operation.get("targetRow"), row + 1)
        return column, [(row, "Swap"), (target_row, "Swap")], None

    # --- Standalone CONTROL / ANTI_CONTROL -----------------------------------
    if gate_name == "CONTROL":
        row = _to_int(operation.get("row"), 0)
        return column, [(row, "\u2022")], None
    if gate_name == "ANTI_CONTROL":
        row = _to_int(operation.get("row"), 0)
        return column, [(row, "\u25E6")], None

    # --- Interleave / Deinterleave → "<<N" -----------------------------------
    if gate_name in {"INTERLEAVE", "DEINTERLEAVE"}:
        row = _to_int(operation.get("row"), 0)
        height = _to_int(operation.get("height"), MULTI_QUBIT_HEIGHTS.get(gate_name, 6))
        symbol = f"<<{height}"
        return column, [(row, symbol)], None

    # --- Function gates → "__FUNC__<label>" + gateMetadata -------------------
    if gate_name in FUNCTION_GATE_TYPES:
        row = _to_int(operation.get("row"), 0)
        label = str(operation.get("label", gate_name[0] if gate_name == "FUNCTION" else gate_name.capitalize()))
        height = _to_int(operation.get("height"), MULTI_QUBIT_HEIGHTS.get(gate_name, 2))
        symbol = f"__FUNC__{label}"
        metadata = {
            "label": label,
            "type": gate_name,
            "isFunctionGate": True,
            "height": height,
        }
        return column, [(row, symbol)], metadata

    # --- Default: look up from GATE_SYMBOLS ----------------------------------
    row = _to_int(operation.get("row"), 0)
    symbol = GATE_SYMBOLS.get(gate_name, gate_name if gate_name else "H")
    return column, [(row, symbol)], None


class QuantumCircuitDiagramHandler(BaseDiagramHandler):
    """Handler for Quantum circuit diagram generation."""

    def get_diagram_type(self) -> str:
        return "QuantumCircuitDiagram"

    def get_system_prompt(self) -> str:
        return """You are an expert quantum computing assistant that builds quantum circuits.

Return ONLY JSON with this shape:
{
  "operation": {
    "gate": "<GATE_NAME>",
    "row": 0,
    "column": 0,
    "controlRow": 0,
    "targetRow": 1,
    "label": "optional label for FUNCTION gates",
    "height": 2
  }
}

AVAILABLE GATES (use these exact names):
  Probes: MEASURE, MEASURE_X, MEASURE_Y, CONTROL, ANTI_CONTROL
  Half Turns: H, X, Y, Z, SWAP
  Quarter Turns: S, S_DAG, V, V_DAG, SQRT_Y, SQRT_Y_DAG
  Eighth Turns: T, T_DAG, SQRT_SQRT_X, SQRT_SQRT_X_DAG, SQRT_SQRT_Y, SQRT_SQRT_Y_DAG
  Parametrized: X_POW, Y_POW, Z_POW, EXP_X, EXP_Y, EXP_Z
  Spinning: Z_POW_T, Z_POW_NEG_T, Y_POW_T, Y_POW_NEG_T, X_POW_T, X_POW_NEG_T
  Frequency: QFT, QFT_DAG, PHASE_GRADIENT, PHASE_GRADIENT_DAG
  Arithmetic: INC, DEC, ADD, SUB, MUL, ADD_AB, SUB_AB, MUL_INV
  Modular: MOD_ADD, MOD_SUB, MOD_MUL, MOD_INV_MUL, MOD_INC, MOD_DEC
  Compare: COMPARE, GREATER_THAN, LESS_EQUAL, GREATER_EQUAL, EQUAL, NOT_EQUAL, XOR
  Order: INTERLEAVE, DEINTERLEAVE, REVERSE_BITS, ROTATE_BITS_LEFT, ROTATE_BITS_RIGHT
  Scalar: ONE, MINUS_ONE, PHASE_I, PHASE_MINUS_I, PHASE_SQRT_I, PHASE_SQRT_MINUS_I
  Function: FUNCTION (custom, set "label"), ORACLE, UNITARY
  Controlled: CNOT, CZ, CY (need controlRow + targetRow), TOFFOLI (needs controlRow, controlRow2, targetRow)

Rules:
1. Use controlRow/targetRow only for controlled gates (CNOT, CZ, CY, TOFFOLI) and SWAP.
2. For FUNCTION/ORACLE/UNITARY gates, set "label" and optionally "height".
3. For INTERLEAVE/DEINTERLEAVE, set "height" (default 6).
4. Keep row and column indexes zero-based.
5. Return JSON only."""

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
            _, placements, _ = _operation_to_placements(op)
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
        gate_metadata: Dict[str, Any] = normalized.get("gateMetadata", {}) if append else {}

        for op in operations:
            column, placements, metadata = _operation_to_placements(op)
            target_column = next_free_column if column is None or column < 0 else column
            _ensure_column(cols, target_column, qubit_count)
            column_values = cols[target_column]
            for row, symbol in placements:
                _place_symbol(column_values, row, symbol)
            # Store function-gate metadata so the frontend can restore nested-circuit info
            if metadata is not None and placements:
                first_row = placements[0][0]
                gate_metadata[f"{target_column}_{first_row}"] = metadata
            if column is None or column < 0:
                next_free_column += 1

        normalized["cols"] = cols
        normalized["qubitCount"] = qubit_count
        normalized["gateMetadata"] = gate_metadata
        return normalized

    def generate_single_element(self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        prompt = self.get_system_prompt()

        try:
            parsed = self.predict_structured(user_request, SingleQuantumGateSpec, system_prompt=prompt)
            spec = parsed.model_dump()

            operation = spec.get("operation")
            model = self._apply_operations(_default_quantum_model(), [operation], append=False)

            gate_name = _normalize_gate_name(operation.get("gate", "H"))
            row = operation.get("row", 0)
            gate_desc = gate_name
            if gate_name in {"CNOT", "CZ", "CY"}:
                ctrl = operation.get("controlRow", 0)
                tgt = operation.get("targetRow", 1)
                gate_desc = f"{gate_name} (control=q{ctrl}, target=q{tgt})"
            else:
                gate_desc = f"{gate_name} on q{row}"

            return {
                "action": "inject_element",
                "diagramType": self.get_diagram_type(),
                "model": model,
                "message": f"Added **{gate_desc}** to the circuit. Ask me to add more gates or build a complete algorithm!",
            }
        except LLMPredictionError:
            logger.error("[QuantumCircuit] generate_single_element LLM FAILED", exc_info=True)
            return self._error_response("I couldn't generate that quantum gate. Please try again or rephrase your request.")
        except Exception:
            logger.error("[QuantumCircuit] generate_single_element FAILED", exc_info=True)
            return self.generate_fallback_element(user_request)

    # Algorithm templates — only the one matching the user request is
    # injected into the prompt, saving ~800 tokens on average.
    _ALGORITHM_TEMPLATES = {
        "bell": (
            "**Bell State** (2 qubits):\n"
            "  Col 0: H on q0\n"
            "  Col 1: CNOT control=q0 target=q1\n"
            "  Col 2: MEASURE q0, q1"
        ),
        "ghz": (
            "**GHZ State** (3+ qubits):\n"
            "  Col 0: H on q0\n"
            "  Col 1..N-1: CNOT control=q0 target=q1, q2, ... qN-1 (one per column)\n"
            "  Last col: MEASURE all"
        ),
        "grover": (
            "**Grover's Search** — uses CONTROL + Z pattern (NOT CNOT).\n"
            "The oracle marks the target state with CZ; the diffusion operator is H-X-CZ-X-H.\n"
            "For N qubits the 'multi-controlled Z' is built by placing CONTROL on all qubits\n"
            "except the last target qubit, which gets Z.\n\n"
            "IMPORTANT: Use 'CONTROL' (row placement) + CZ or direct Z. Do NOT use CNOT for the oracle or diffusion.\n\n"
            "2-qubit Grover (searching |11>):\n"
            "  Col 0: H q0, H q1\n"
            "  Col 1: CZ control=q0 target=q1 (oracle)\n"
            "  Col 2: H q0, H q1 (diffusion start)\n"
            "  Col 3: X q0, X q1\n"
            "  Col 4: CZ control=q0 target=q1\n"
            "  Col 5: X q0, X q1\n"
            "  Col 6: H q0, H q1 (diffusion end)\n"
            "  Col 7: MEASURE q0, MEASURE q1\n\n"
            "3-qubit Grover (searching |111>):\n"
            "  Col 0: H q0, H q1, H q2\n"
            "  Col 1: CONTROL q0, CONTROL q1, Z q2 (oracle)\n"
            "  Col 2-6: H-X-CONTROL+Z-X-H diffusion\n"
            "  Col 7: MEASURE all\n"
            "  JSON for CONTROL+Z columns:\n"
            '    {"column": 1, "row": 0, "gate": "CONTROL"},\n'
            '    {"column": 1, "row": 1, "gate": "CONTROL"},\n'
            '    {"column": 1, "row": 2, "gate": "Z"}\n\n'
            "General pattern: N qubits (no ancilla). Oracle = CONTROL on q0..q(N-2) + Z on q(N-1).\n"
            "Diffusion = H all, X all, same CONTROL+Z pattern, X all, H all. qubitCount = N."
        ),
        "qft": (
            "**Quantum Fourier Transform (QFT)** (N qubits):\n"
            "  For each qubit k (0 to N-1): H on k, then controlled phase rotations from j > k\n"
            "  Then SWAP qubits to reverse order\n"
            "  3-qubit QFT: H q0, CZ q1->q0, CZ q2->q0, H q1, CZ q2->q1, H q2, SWAP q0<->q2"
        ),
        "deutsch": (
            "**Deutsch-Jozsa** (N+1 qubits, last is ancilla):\n"
            "  Col 0: X on ancilla\n"
            "  Col 1: H on ALL qubits\n"
            "  Oracle: CNOT from each input qubit to ancilla\n"
            "  Next: H on input qubits (not ancilla)\n"
            "  Last: MEASURE input qubits"
        ),
        "teleportation": (
            "**Quantum Teleportation** (3 qubits: q0=state, q1=Alice, q2=Bob):\n"
            "  Col 0: H q1, Col 1: CNOT q1->q2, Col 2: CNOT q0->q1,\n"
            "  Col 3: H q0, Col 4: MEASURE q0+q1, Col 5: CNOT q1->q2, Col 6: CZ q0->q2"
        ),
        "bernstein": (
            "**Bernstein-Vazirani** (N+1 qubits, last is ancilla):\n"
            "  X on ancilla, H on ALL, CNOT from secret-bit=1 qubits to ancilla,\n"
            "  H on input qubits, MEASURE input qubits"
        ),
        "superdense": (
            "**Superdense Coding** (2 qubits):\n"
            "  H q0, CNOT q0->q1, encode (X/Z on q0), CNOT q0->q1, H q0, MEASURE both"
        ),
        "phase": (
            "**Phase Estimation** (N+1 qubits):\n"
            "  H on counting qubits, controlled-U from counting to target,\n"
            "  Inverse QFT on counting qubits, MEASURE counting qubits"
        ),
    }

    _ALGORITHM_KEYWORDS = {
        "bell": ["bell state", "bell pair", "entangle two", "entangled pair"],
        "ghz": ["ghz", "greenberger"],
        "grover": ["grover", "search algorithm", "amplitude amplification"],
        "qft": ["qft", "fourier transform", "quantum fourier"],
        "deutsch": ["deutsch", "jozsa", "deutsch-jozsa"],
        "teleportation": ["teleport"],
        "bernstein": ["bernstein", "vazirani"],
        "superdense": ["superdense", "dense coding"],
        "phase": ["phase estimation"],
    }

    def _select_algorithm_templates(self, user_request: str) -> str:
        """Select only algorithm templates relevant to the user request.

        Returns the template text to inject into the prompt. For unknown
        requests, includes only Bell State and Grover's as compact examples.
        """
        request_lower = user_request.lower()
        matched = []
        for algo_key, keywords in self._ALGORITHM_KEYWORDS.items():
            if any(kw in request_lower for kw in keywords):
                matched.append(algo_key)

        if not matched:
            # Provide 2 compact examples for generic "create a quantum circuit" requests
            matched = ["bell", "grover"]

        templates = []
        for key in matched:
            if key in self._ALGORITHM_TEMPLATES:
                templates.append(self._ALGORITHM_TEMPLATES[key])

        header = "KNOWN QUANTUM ALGORITHMS — use these exact implementations when matching:\n\n"
        return header + "\n\n".join(templates)

    def generate_complete_system(self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        algorithm_section = self._select_algorithm_templates(user_request)

        prompt = f"""You are an expert quantum computing assistant that designs quantum circuits.

Before generating, think through:
- What algorithm does the user want?
- How many qubits are needed?
- What is the correct gate sequence for this algorithm?
- Are controlled gates (CNOT, CZ, TOFFOLI) placed correctly with control/target qubits?
- Does the circuit end with measurements where appropriate?

AVAILABLE GATES (by category):
  Probes: MEASURE, MEASURE_X, MEASURE_Y, CONTROL, ANTI_CONTROL
  Half Turns: H (Hadamard), X (NOT/bit-flip), Y, Z (phase-flip), SWAP
  Quarter Turns: S, S_DAG, V, V_DAG, SQRT_Y, SQRT_Y_DAG
  Eighth Turns: T, T_DAG, SQRT_SQRT_X, SQRT_SQRT_X_DAG, SQRT_SQRT_Y, SQRT_SQRT_Y_DAG
  Parametrized: X_POW, Y_POW, Z_POW, EXP_X, EXP_Y, EXP_Z
  Multi-qubit: QFT, QFT_DAG, PHASE_GRADIENT, ADD, SUB, MUL, INC, DEC, MOD_ADD, MOD_SUB, MOD_MUL
  Compare/Logic: COMPARE, GREATER_THAN, LESS_EQUAL, EQUAL, NOT_EQUAL, XOR
  Order: INTERLEAVE, DEINTERLEAVE, REVERSE_BITS, ROTATE_BITS_LEFT, ROTATE_BITS_RIGHT
  Function: FUNCTION (set "label" and "height"), ORACLE, UNITARY
  Controlled: CNOT (controlRow + targetRow), CZ, CY, TOFFOLI (controlRow, controlRow2, targetRow)

GATE USAGE RULES:
- Controlled gates (CNOT, CZ, CY): use "controlRow" and "targetRow" (no "row")
- TOFFOLI: use "controlRow", "controlRow2", and "targetRow"
- SWAP: use "row" and "targetRow"
- FUNCTION / ORACLE / UNITARY: set "label" and "height" (default 2)
- Multi-qubit gates auto-span rows based on default height
- Single-qubit gates: use "row"
- All indexes are zero-based

{algorithm_section}

Rules:
1. Choose the correct algorithm based on the user's request.
2. If the user asks for a specific algorithm, implement it faithfully.
3. If the user is vague, create a Bell state or Grover's search as a demo.
4. Keep qubit count minimal but sufficient."""

        system_prompt = prompt
        user_prompt = f"User Request: {user_request}"

        try:
            parsed = self.predict_structured(user_prompt, SystemQuantumCircuitSpec, system_prompt=system_prompt)
            spec = parsed.model_dump()

            operations = spec.get("operations") if isinstance(spec.get("operations"), list) else []
            typed_operations = [op for op in operations if isinstance(op, dict)]
            if not typed_operations:
                raise ValueError("No operations generated")

            algorithm_name = spec.get("algorithmName", "")

            model = self._apply_operations(
                _default_quantum_model(),
                typed_operations,
                append=False,
                qubit_count_hint=spec.get("qubitCount"),
            )

            step_count = len(model.get('cols', []))
            qubit_count = model.get('qubitCount', '?')
            algo_label = f" ({algorithm_name})" if algorithm_name else ""
            return {
                "action": "inject_complete_system",
                "diagramType": self.get_diagram_type(),
                "model": model,
                "message": (
                    f"Built a quantum circuit{algo_label} with "
                    f"**{step_count} column(s)** across **{qubit_count} qubit(s)**. "
                    f"You can ask me to modify it, add more gates, or explain how it works!"
                ),
            }
        except LLMPredictionError:
            logger.error("[QuantumCircuit] generate_complete_system LLM FAILED", exc_info=True)
            return self._error_response("I couldn't generate that quantum circuit. Please try again or rephrase your request.")
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

        prompt = """You are an expert quantum computing assistant that modifies quantum circuits.

Return ONLY JSON with this shape:
{
  "mode": "append|replace",
  "qubitCount": 3,
  "operations": [
    {"column": 0, "row": 0, "gate": "H"},
    {"gate": "CNOT", "controlRow": 0, "targetRow": 1}
  ]
}

AVAILABLE GATES (organised by category):
  Probes: MEASURE, MEASURE_X, MEASURE_Y, CONTROL, ANTI_CONTROL
  Half Turns: H, X, Y, Z, SWAP
  Quarter Turns: S, S_DAG, V, V_DAG, SQRT_Y, SQRT_Y_DAG
  Eighth Turns: T, T_DAG, SQRT_SQRT_X, SQRT_SQRT_X_DAG, SQRT_SQRT_Y, SQRT_SQRT_Y_DAG
  Parametrized: X_POW, Y_POW, Z_POW, EXP_X, EXP_Y, EXP_Z
  Spinning: Z_POW_T, Z_POW_NEG_T, Y_POW_T, Y_POW_NEG_T, X_POW_T, X_POW_NEG_T
  Frequency: QFT, QFT_DAG, PHASE_GRADIENT, PHASE_GRADIENT_DAG
  Arithmetic: INC, DEC, ADD, SUB, MUL, ADD_AB, SUB_AB, MUL_INV
  Modular: MOD_ADD, MOD_SUB, MOD_MUL, MOD_INV_MUL, MOD_INC, MOD_DEC
  Compare: COMPARE, GREATER_THAN, EQUAL, NOT_EQUAL, XOR, COUNT_1S
  Order: INTERLEAVE, DEINTERLEAVE, REVERSE_BITS, ROTATE_BITS_LEFT, ROTATE_BITS_RIGHT
  Scalar: ONE, MINUS_ONE, PHASE_I, PHASE_MINUS_I
  Function: FUNCTION (set "label" + "height"), ORACLE, UNITARY
  Controlled: CNOT, CZ, CY (controlRow + targetRow), TOFFOLI (controlRow, controlRow2, targetRow)

Modes:
- "append": Add new gates/columns to the END of the existing circuit.
  Use this for: "add a measurement", "extend with more gates", "add a CNOT".
- "replace": Rebuild the entire circuit from scratch.
  Use this for: "rebuild as Grover's", "replace with a Bell state",
  "change the whole circuit", "start over with", "redo as",
  "convert to", "implement Grover's algorithm".

When replacing, you can use any of the known quantum algorithms — build the full
circuit from scratch. Key algorithm patterns:
- **Grover (N qubits)**: H all → Oracle (CONTROL on q0..q(N-2) + Z on q(N-1)) →
  Diffusion (H all, X all, CONTROL+Z, X all, H all) → MEASURE all.
  Use CONTROL gate placements + Z target, NOT CNOT. qubitCount = N.
- **Bell (2 qubits)**: H q0 → CNOT(q0→q1) → MEASURE
- **QFT**: H + controlled rotations + SWAP to reverse order
- **Teleportation (3 qubits)**: Bell pair → CNOT+H → Measure → corrections

Rules:
1. Use mode=append to add new operations after the existing circuit.
2. Use mode=replace to rebuild the circuit entirely.
3. When appending, omit "column" to auto-place after existing gates.
4. When replacing, specify explicit column indexes starting from 0.
5. For controlled gates use controlRow + targetRow; for single gates use row.
6. For FUNCTION/ORACLE/UNITARY gates, set "label" and optionally "height".
7. Return JSON only."""

        try:
            user_prompt = f"User Request: {user_request}{context_block}"
            parsed = self.predict_structured(user_prompt, QuantumModificationSpec, system_prompt=prompt)
            spec = parsed.model_dump()

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
                "message": f"Quantum circuit **{action_label}** with {len(typed_operations)} operation(s). You can add more gates or change the circuit layout!",
            }
        except LLMPredictionError:
            logger.error("[QuantumCircuit] generate_modification LLM FAILED", exc_info=True)
            return self._error_response("I couldn't process that circuit modification. Please try again or rephrase your request.")
        except Exception:
            return {
                "action": "modify_model",
                "diagramType": self.get_diagram_type(),
                "model": base_model,
                "message": "I couldn't parse the requested modification, so the current circuit is unchanged. Try rephrasing, e.g. *'Add an X gate to qubit 1'*.",
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
            "message": "I created a basic Hadamard gate as a starting point. Describe your circuit (e.g. *'Create a Bell state circuit'* or *'Apply H to qubit 0 then CNOT on qubits 0,1'*) for a more complex result!",
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
            "message": "I created a basic Bell-state circuit as a starting point. Describe your quantum algorithm in more detail for a custom circuit!",
        }
