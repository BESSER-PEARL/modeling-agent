"""
Model Context Summaries
-----------------------
Functions that produce human-readable summaries of diagram models for LLM
prompts.  Both compact (one-line counts) and detailed (structural content)
variants live here so every handler and the workspace-context builder can
share the same logic.
"""

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Compact (one-line) model summary
# ---------------------------------------------------------------------------


def compact_model_summary(model_data: Any, diagram_type: str) -> str:
    """Return a one-line human-readable summary of a diagram model."""
    if not isinstance(model_data, dict):
        return f"{diagram_type}: no structured model available."

    if diagram_type in {"ClassDiagram", "ObjectDiagram", "StateMachineDiagram", "AgentDiagram"}:
        elements = model_data.get("elements")
        relationships = model_data.get("relationships")
        if isinstance(elements, dict) and isinstance(relationships, dict):
            if diagram_type == "ClassDiagram":
                # Count actual classes (not attributes/methods) for a clearer summary
                class_names = [
                    el.get("name") for el in elements.values()
                    if isinstance(el, dict) and el.get("type") == "Class"
                    and isinstance(el.get("name"), str) and el["name"].strip()
                ]
                class_count = len(class_names)
                if class_count > 0:
                    preview = ", ".join(class_names[:6])
                    extra = f" (+{class_count - 6} more)" if class_count > 6 else ""
                    return (
                        f"{diagram_type}: {class_count} class(es): "
                        f"{preview}{extra} and "
                        f"{len(relationships)} relationship(s)."
                    )
            return (
                f"{diagram_type}: {len(elements)} element(s), "
                f"{len(relationships)} relationship(s)."
            )

    if diagram_type == "GUINoCodeDiagram":
        pages = model_data.get("pages")
        if isinstance(pages, list):
            return f"{diagram_type}: {len(pages)} page(s)."

    if diagram_type == "QuantumCircuitDiagram":
        cols = model_data.get("cols")
        if isinstance(cols, list):
            return f"{diagram_type}: {len(cols)} circuit column(s)."

    return f"{diagram_type}: model metadata available."


# ---------------------------------------------------------------------------
# Private per-diagram-type summarisers
# ---------------------------------------------------------------------------


def _clean_attr_name(raw: str) -> str:
    """Strip visibility prefix (+/-/#/~) and type suffix from an attribute name."""
    name = raw.strip()
    if name and name[0] in "+-#~":
        name = name[1:].strip()
    if ":" in name:
        name = name.split(":", 1)[0].strip()
    return name


def _summarize_class_diagram(model: Dict[str, Any], *, max_classes: int = 20, max_attrs: int = 10) -> List[str]:
    """Summarize a ClassDiagram model: classes, attributes, methods, relationships."""
    elements = model.get("elements")
    relationships = model.get("relationships")
    if not isinstance(elements, dict):
        return []

    lines: List[str] = []

    # Collect classes
    class_data: Dict[str, Dict[str, Any]] = {}  # id -> {name, attrs, methods}
    for eid, el in elements.items():
        if not isinstance(el, dict) or el.get("type") != "Class":
            continue
        name = el.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        class_data[eid] = {"name": name.strip(), "attrs": [], "methods": []}

    # Attach attributes and methods
    for eid, el in elements.items():
        if not isinstance(el, dict):
            continue
        owner = el.get("owner")
        if not isinstance(owner, str) or owner not in class_data:
            continue
        el_type = el.get("type")
        raw_name = el.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue
        if el_type == "ClassAttribute":
            attr_type = el.get("attributeType", "")
            clean = _clean_attr_name(raw_name)
            if not attr_type and ":" in raw_name:
                attr_type = raw_name.rsplit(":", 1)[1].strip()
            type_str = f": {attr_type}" if attr_type else ""
            class_data[owner]["attrs"].append(f"{clean}{type_str}")
        elif el_type == "ClassMethod":
            clean = _clean_attr_name(raw_name)
            class_data[owner]["methods"].append(clean)

    # Format class lines
    class_items = list(class_data.items())
    for cid, cd in class_items[:max_classes]:
        parts = [f"Class {cd['name']}"]
        if cd["attrs"]:
            attrs_str = ", ".join(cd["attrs"][:max_attrs])
            if len(cd["attrs"]) > max_attrs:
                attrs_str += f" (+{len(cd['attrs']) - max_attrs} more)"
            parts.append(f"attributes: {attrs_str}")
        if cd["methods"]:
            methods_str = ", ".join(cd["methods"][:max_attrs])
            parts.append(f"methods: {methods_str}")
        lines.append(" | ".join(parts))
    if len(class_items) > max_classes:
        lines.append(f"  …and {len(class_items) - max_classes} more class(es)")

    # Format relationships
    if isinstance(relationships, dict):
        rel_items = list(relationships.values())
        for rel in rel_items[:15]:
            if not isinstance(rel, dict):
                continue
            source = rel.get("source")
            target = rel.get("target")
            if not isinstance(source, dict) or not isinstance(target, dict):
                continue
            src_id = source.get("element", "")
            tgt_id = target.get("element", "")
            src_name = class_data.get(src_id, {}).get("name", src_id)
            tgt_name = class_data.get(tgt_id, {}).get("name", tgt_id)
            rel_type = rel.get("type", "Association")
            rel_name = rel.get("name", "")
            src_mult = source.get("multiplicity", "")
            tgt_mult = target.get("multiplicity", "")
            mult_str = ""
            if src_mult or tgt_mult:
                mult_str = f" [{src_mult}..{tgt_mult}]"
            name_str = f' "{rel_name}"' if rel_name else ""
            lines.append(f"{rel_type}: {src_name} -> {tgt_name}{mult_str}{name_str}")
        if len(rel_items) > 15:
            lines.append(f"  …and {len(rel_items) - 15} more relationship(s)")

    return lines


def _summarize_state_machine(model: Dict[str, Any], *, max_items: int = 20) -> List[str]:
    """Summarize a StateMachineDiagram model: states with actions, transitions."""
    elements = model.get("elements")
    relationships = model.get("relationships")
    if not isinstance(elements, dict):
        return []

    lines: List[str] = []
    element_names: Dict[str, str] = {}  # id -> name

    # States
    for eid, el in elements.items():
        if not isinstance(el, dict):
            continue
        el_type = el.get("type")
        if el_type not in ("State", "StateInitialNode", "StateFinalNode"):
            continue
        name = el.get("name", "Unnamed")
        element_names[eid] = name
        parts = [f"State {name} ({el_type})"]
        entry = el.get("entryAction", "")
        exit_a = el.get("exitAction", "")
        do_act = el.get("doActivity", "")
        if entry:
            parts.append(f"entry={entry}")
        if exit_a:
            parts.append(f"exit={exit_a}")
        if do_act:
            parts.append(f"do={do_act}")
        lines.append(" | ".join(parts))

    # Transitions
    if isinstance(relationships, dict):
        for rel in relationships.values():
            if not isinstance(rel, dict):
                continue
            source = rel.get("source")
            target = rel.get("target")
            if not isinstance(source, dict) or not isinstance(target, dict):
                continue
            src_id = source.get("element", "")
            tgt_id = target.get("element", "")
            src_name = element_names.get(src_id, elements.get(src_id, {}).get("name", src_id))
            tgt_name = element_names.get(tgt_id, elements.get(tgt_id, {}).get("name", tgt_id))
            trigger = rel.get("name", "") or rel.get("trigger", "")
            guard = rel.get("guard", "")
            effect = rel.get("effect", "")
            parts = [f"Transition: {src_name} -> {tgt_name}"]
            detail_parts: List[str] = []
            if trigger:
                detail_parts.append(trigger)
            if guard:
                detail_parts.append(f"[{guard}]")
            if effect:
                detail_parts.append(f"/{effect}")
            if detail_parts:
                parts[0] += f" {' '.join(detail_parts)}"
            lines.append(parts[0])

    if len(lines) > max_items:
        overflow = len(lines) - max_items
        lines = lines[:max_items]
        lines.append(f"  …and {overflow} more state/transition item(s)")
    return lines


def _summarize_object_diagram(model: Dict[str, Any], *, max_objects: int = 15) -> List[str]:
    """Summarize an ObjectDiagram model: objects with attribute values."""
    elements = model.get("elements")
    if not isinstance(elements, dict):
        return []

    lines: List[str] = []
    for el in elements.values():
        if not isinstance(el, dict) or el.get("type") != "Object":
            continue
        name = el.get("name", "Unnamed")
        class_name = el.get("className", "")
        class_part = f": {class_name}" if class_name else ""
        attrs: List[str] = []
        for attr_id in el.get("attributes", []) or []:
            attr = elements.get(attr_id)
            if not isinstance(attr, dict):
                continue
            attr_name = attr.get("name", "")
            attr_value = attr.get("value", "")
            if attr_name:
                attrs.append(f"{attr_name}={attr_value}" if attr_value else attr_name)
        summary = f"Object {name}{class_part}"
        if attrs:
            attr_list = attrs[:8]
            if len(attrs) > 8:
                attr_list.append(f"…+{len(attrs) - 8} more")
            summary += f" | attributes: {', '.join(attr_list)}"
        lines.append(summary)

    if len(lines) > max_objects:
        overflow = len(lines) - max_objects
        lines = lines[:max_objects]
        lines.append(f"  …and {overflow} more object(s)")
    return lines


def _summarize_gui_model(model: Dict[str, Any]) -> List[str]:
    """Summarize a GUINoCodeDiagram model: pages and section types."""
    pages = model.get("pages")
    if not isinstance(pages, list):
        return []

    lines: List[str] = []
    for page in pages[:10]:
        if not isinstance(page, dict):
            continue
        page_name = page.get("name", "Unnamed")
        section_count = 0
        frames = page.get("frames")
        if isinstance(frames, list) and frames:
            comp = frames[0].get("component") if isinstance(frames[0], dict) else None
            if isinstance(comp, dict):
                components = comp.get("components")
                if isinstance(components, list):
                    section_count = len(components)
        lines.append(f"Page {page_name} ({section_count} section(s))")

    if len(pages) > 10:
        lines.append(f"  …and {len(pages) - 10} more page(s)")
    return lines


def _summarize_agent_diagram(model: Dict[str, Any]) -> List[str]:
    """Summarize an AgentDiagram model: states, intents, transitions."""
    elements = model.get("elements")
    relationships = model.get("relationships")
    if not isinstance(elements, dict):
        return []

    lines: List[str] = []

    states = [e.get("name") for e in elements.values()
              if isinstance(e, dict) and e.get("type") == "AgentState" and e.get("name")]
    if states:
        state_str = ', '.join(states[:10])
        if len(states) > 10:
            state_str += f" …+{len(states) - 10} more"
        lines.append(f"States: {state_str}")

    intents = [e.get("name") for e in elements.values()
               if isinstance(e, dict) and e.get("type") == "AgentIntent" and e.get("name")]
    if intents:
        intent_str = ', '.join(intents[:10])
        if len(intents) > 10:
            intent_str += f" …+{len(intents) - 10} more"
        lines.append(f"Intents: {intent_str}")

    if isinstance(relationships, dict):
        transitions: List[str] = []
        for rel in relationships.values():
            if not isinstance(rel, dict) or rel.get("type") != "AgentTransition":
                continue
            source = elements.get(rel.get("source", ""))
            target = elements.get(rel.get("target", ""))
            if isinstance(source, dict) and isinstance(target, dict):
                transitions.append(f"{source.get('name')} → {target.get('name')}")
        if transitions:
            trans_str = ', '.join(transitions[:5])
            if len(transitions) > 5:
                trans_str += f" …+{len(transitions) - 5} more"
            lines.append(f"Transitions: {trans_str}")

    return lines


# Mapping of Quirk-style gate symbols to human-readable names.
_QUIRK_SYMBOL_MAP: Dict[str, str] = {
    # Half Turns
    "H": "H (Hadamard)", "X": "X (Pauli-X/NOT)", "Y": "Y (Pauli-Y)", "Z": "Z (Pauli-Z)",
    "Swap": "SWAP",
    # Quarter Turns
    "S": "S (π/2 phase)", "Z^-½": "S† (−π/2 phase)",
    "V": "V (√X)", "X^-½": "V† (−√X)",
    "Y^½": "√Y", "Y^-½": "√Y†",
    # Eighth Turns
    "Z^¼": "T (π/4 phase)", "Z^-¼": "T† (−π/4 phase)",
    "X^¼": "√√X", "X^-¼": "√√X†",
    "Y^¼": "√√Y", "Y^-¼": "√√Y†",
    # Spinning (time-dependent)
    "Z^t": "Z^t (spinning)", "Z^-t": "Z^-t (spinning)",
    "Y^t": "Y^t (spinning)", "Y^-t": "Y^-t (spinning)",
    "X^t": "X^t (spinning)", "X^-t": "X^-t (spinning)",
    # Parametrized Rotations
    "Exp(-iXt)": "Exp(-iXt)", "Exp(-iYt)": "Exp(-iYt)", "Exp(-iZt)": "Exp(-iZt)",
    # Frequency
    "QFT": "QFT", "QFT†": "QFT†",
    "Grad": "Phase Gradient", "Grad†": "Phase Gradient†",
    "Grad⁻¹": "Phase Gradient⁻¹", "Grad⁻¹†": "Phase Gradient⁻¹†",
    # Arithmetic
    "+=1": "INC (+1)", "-=1": "DEC (−1)",
    "+=A": "ADD (+A)", "-=A": "SUB (−A)", "*=A": "MUL (×A)",
    "+AB": "ADD_AB", "-AB": "SUB_AB", "×A⁻¹": "MUL_INV",
    # Modular Arithmetic
    "+A mod R": "MOD_ADD", "-A mod R": "MOD_SUB",
    "*A mod R": "MOD_MUL", "/A mod R": "MOD_INV_MUL",
    "+1 mod R": "MOD_INC", "-1 mod R": "MOD_DEC",
    "*B mod R": "MOD_MUL_B", "*B A⁻¹ mod R": "MOD_MUL_B_INV",
    # Compare / Logic
    "A < B": "COMPARE (<)", "A > B": "GREATER_THAN",
    "A ≤ B": "LESS_EQUAL", "A ≥ B": "GREATER_EQUAL",
    "A = B": "EQUAL (=)", "A ≠ B": "NOT_EQUAL (≠)",
    "Input < A": "CMP_A_LT", "Input > A": "CMP_A_GT", "Input = A": "CMP_A_EQ",
    "Count 1s": "COUNT_1S", "Cycle": "CYCLE_BITS", "⊕": "XOR",
    # Order
    "Reverse": "REVERSE_BITS", "<<": "ROTATE_LEFT", ">>": "ROTATE_RIGHT",
    # Scalar
    "i": "PHASE_I", "-i": "PHASE_MINUS_I",
    "√i": "PHASE_√I", "√-i": "PHASE_√−I",
    # Probes / Displays
    "Measure": "MEASURE", "Measure X": "MEASURE_X", "Measure Y": "MEASURE_Y",
    "Chance": "PROB", "Amps": "AMP", "Bloch": "BLOCH", "Density": "DENSITY",
    # Control dots (Unicode bullet = frontend, asterisk = legacy handler)
    "\u2022": "● (control)", "\u25E6": "◦ (anti-control)", "*": "● (control)",
    # Legacy symbol compat (old handler used ASCII fractions)
    "Z^1/2": "S (π/2 phase)", "Z^-1/2": "S† (−π/2 phase)",
    "Z^1/4": "T (π/4 phase)", "Z^-1/4": "T† (−π/4 phase)",
    "QFT_dag": "QFT†",
}

# Short symbol map for compact gate counting
_QUIRK_SHORT_MAP: Dict[str, str] = {
    "H": "H", "X": "X", "Y": "Y", "Z": "Z", "Swap": "SWAP",
    "S": "S", "Z^-½": "S†", "V": "V", "X^-½": "V†",
    "Y^½": "√Y", "Y^-½": "√Y†",
    "Z^¼": "T", "Z^-¼": "T†",
    "X^¼": "√√X", "X^-¼": "√√X†", "Y^¼": "√√Y", "Y^-¼": "√√Y†",
    "Z^t": "Z^t", "Z^-t": "Z^-t", "Y^t": "Y^t", "Y^-t": "Y^-t",
    "X^t": "X^t", "X^-t": "X^-t",
    "Exp(-iXt)": "e^-iXt", "Exp(-iYt)": "e^-iYt", "Exp(-iZt)": "e^-iZt",
    "QFT": "QFT", "QFT†": "QFT†",
    "Grad": "Grad", "Grad†": "Grad†", "Grad⁻¹": "Grad⁻¹", "Grad⁻¹†": "Grad⁻¹†",
    "+=1": "INC", "-=1": "DEC", "+=A": "ADD", "-=A": "SUB", "*=A": "MUL",
    "+AB": "ADD_AB", "-AB": "SUB_AB", "×A⁻¹": "MUL_INV",
    "+A mod R": "MOD+", "-A mod R": "MOD-", "*A mod R": "MOD*", "/A mod R": "MOD/",
    "+1 mod R": "MOD_INC", "-1 mod R": "MOD_DEC",
    "*B mod R": "MOD*B", "*B A⁻¹ mod R": "MOD*B⁻¹",
    "A < B": "<", "A > B": ">", "A ≤ B": "≤", "A ≥ B": "≥",
    "A = B": "=", "A ≠ B": "≠",
    "Input < A": "CMP<A", "Input > A": "CMP>A", "Input = A": "CMP=A",
    "Count 1s": "#1s", "Cycle": "Cycle", "⊕": "XOR",
    "Reverse": "Rev", "<<": "ROL", ">>": "ROR",
    "i": "φ_i", "-i": "φ_-i", "√i": "φ_√i", "√-i": "φ_√-i",
    "Measure": "MEASURE", "Measure X": "MEAS_X", "Measure Y": "MEAS_Y",
    "Chance": "PROB", "Amps": "AMP", "Bloch": "BLOCH", "Density": "DENSITY",
    "\u2022": "●", "\u25E6": "◦", "*": "●",
    # Legacy compat
    "Z^1/2": "S", "Z^-1/2": "S†", "Z^1/4": "T", "Z^-1/4": "T†",
    "QFT_dag": "QFT†",
}


def _summarize_quantum_circuit(model: Dict[str, Any], *, max_cols: int = 30) -> List[str]:
    """Summarize a QuantumCircuitDiagram model with rich detail for LLM analysis.

    Includes qubit count, column-by-column gate listing, and a gate histogram
    so the LLM can identify which algorithm is implemented.
    """
    cols = model.get("cols")
    if not isinstance(cols, list):
        return []

    qubit_count = model.get("qubitCount", 0)
    if not isinstance(qubit_count, int) or qubit_count < 1:
        for col in cols:
            if isinstance(col, list):
                qubit_count = max(qubit_count, len(col))

    lines: List[str] = [f"Qubits: {qubit_count}, Columns (time steps): {len(cols)}"]

    # Gate histogram for high-level analysis
    gate_counts: Dict[str, int] = {}
    has_control = False
    has_measurement = False
    controlled_pairs: List[str] = []

    for col_idx, col in enumerate(cols):
        if not isinstance(col, list):
            continue
        # Detect control-target pairs in this column
        control_rows: List[int] = []
        target_rows: Dict[int, str] = {}
        for row_idx, cell in enumerate(col):
            if cell == 1 or cell is None:
                continue
            symbol = str(cell)
            # Handle special serialized symbols before map lookup
            if symbol.startswith("__FUNC__"):
                short = f"FUNC({symbol[8:]})"
            elif symbol.startswith("<<") and len(symbol) > 2:
                short = "INTERLEAVE"
            else:
                short = _QUIRK_SHORT_MAP.get(symbol, symbol)
            if short == "●":
                control_rows.append(row_idx)
                has_control = True
            elif short in {"MEASURE", "MEAS_X", "MEAS_Y"}:
                has_measurement = True
                gate_counts["MEASURE"] = gate_counts.get("MEASURE", 0) + 1
                target_rows[row_idx] = short
            else:
                gate_counts[short] = gate_counts.get(short, 0) + 1
                target_rows[row_idx] = short

        # Record controlled-gate relationships
        if control_rows and target_rows:
            for cr in control_rows:
                for tr, tg in target_rows.items():
                    if tg != "MEASURE":
                        controlled_pairs.append(f"q{cr}→q{tr} (C-{tg})")

    # Gate summary line
    if gate_counts:
        gate_parts = [f"{name}×{count}" for name, count in sorted(gate_counts.items())]
        lines.append(f"Gate counts: {', '.join(gate_parts)}")

    if controlled_pairs:
        lines.append(f"Controlled gates: {', '.join(controlled_pairs[:10])}")

    # Column-by-column detail
    for col_idx, col in enumerate(cols[:max_cols]):
        if not isinstance(col, list):
            continue
        gate_entries: List[str] = []
        for row_idx, cell in enumerate(col):
            if cell == 1 or cell is None:
                continue
            symbol = str(cell)
            if symbol.startswith("__FUNC__"):
                readable = f"FUNC({symbol[8:]})"
            elif symbol.startswith("<<") and len(symbol) > 2:
                readable = f"INTERLEAVE(h={symbol[2:]})"
            else:
                readable = _QUIRK_SYMBOL_MAP.get(symbol, symbol)
            gate_entries.append(f"q{row_idx}: {readable}")
        if gate_entries:
            lines.append(f"Col {col_idx}: {', '.join(gate_entries)}")

    if len(cols) > max_cols:
        lines.append(f"... and {len(cols) - max_cols} more column(s)")

    # High-level pattern hints to help the LLM identify algorithms
    hints: List[str] = []
    h_count = gate_counts.get("H", 0)
    if h_count > 0 and has_control and has_measurement:
        if h_count >= qubit_count and len(controlled_pairs) >= 1:
            hints.append("Pattern suggests: may involve superposition + entanglement + measurement")
    if h_count >= 2 * qubit_count:
        hints.append("Multiple H layers detected (common in Grover's diffusion or QFT)")
    if hints:
        lines.append("Analysis hints: " + "; ".join(hints))

    return lines


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def detailed_model_summary(model_data: Any, diagram_type: str) -> str:
    """Return a multi-line structural summary of a diagram model for LLM context.

    Unlike ``compact_model_summary`` (which returns just counts), this function
    includes class names, attributes, methods, relationships, state names,
    transitions, object attribute values, quantum gates, etc.  It is designed
    to be appended to LLM prompts so the model understands what already exists.

    All diagram types are handled by a single entry point — no need for
    per-handler summarisation code.
    """
    if not isinstance(model_data, dict):
        return f"{diagram_type}: no model data."

    if diagram_type == "ClassDiagram":
        lines = _summarize_class_diagram(model_data)
        if lines:
            return "Current class diagram:\n- " + "\n- ".join(lines)

    elif diagram_type == "StateMachineDiagram":
        lines = _summarize_state_machine(model_data)
        if lines:
            return "Current state machine:\n- " + "\n- ".join(lines)

    elif diagram_type == "ObjectDiagram":
        lines = _summarize_object_diagram(model_data)
        if lines:
            return "Current object diagram:\n- " + "\n- ".join(lines)

    elif diagram_type == "GUINoCodeDiagram":
        lines = _summarize_gui_model(model_data)
        if lines:
            return "Current GUI model:\n- " + "\n- ".join(lines)

    elif diagram_type == "QuantumCircuitDiagram":
        lines = _summarize_quantum_circuit(model_data)
        if lines:
            return "Current quantum circuit:\n- " + "\n- ".join(lines)

    elif diagram_type == "AgentDiagram":
        lines = _summarize_agent_diagram(model_data)
        if lines:
            return "Current agent diagram:\n- " + "\n- ".join(lines)

    # Fallback to compact
    return compact_model_summary(model_data, diagram_type)
