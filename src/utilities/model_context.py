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
            elif diagram_type == "StateMachineDiagram":
                # Count actual states only — exclude the StateInitialNode
                # pseudostate and per-state StateBody/StateFallbackBody/
                # StateCodeBlock sub-elements.
                state_names = [
                    el.get("name") for el in elements.values()
                    if isinstance(el, dict) and el.get("type") == "State"
                    and isinstance(el.get("name"), str) and el["name"].strip()
                ]
                state_count = len(state_names)
                if state_count > 0:
                    preview = ", ".join(state_names[:6])
                    extra = f" (+{state_count - 6} more)" if state_count > 6 else ""
                    return (
                        f"{diagram_type}: {state_count} state(s): "
                        f"{preview}{extra} and "
                        f"{len(relationships)} transition(s)."
                    )
            elif diagram_type == "AgentDiagram":
                # Count actual states/intents — exclude the StateInitialNode
                # pseudostate and per-element AgentStateBody/AgentIntentBody
                # sub-elements.
                state_count = sum(
                    1 for el in elements.values()
                    if isinstance(el, dict) and el.get("type") == "AgentState"
                    and isinstance(el.get("name"), str) and el["name"].strip()
                )
                intent_names = [
                    el.get("name") for el in elements.values()
                    if isinstance(el, dict) and el.get("type") == "AgentIntent"
                    and isinstance(el.get("name"), str) and el["name"].strip()
                ]
                if state_count > 0 or intent_names:
                    preview = ", ".join(intent_names[:6])
                    extra = f" (+{len(intent_names) - 6} more)" if len(intent_names) > 6 else ""
                    intent_part = f", intents: {preview}{extra}" if intent_names else ""
                    return (
                        f"{diagram_type}: {state_count} state(s), "
                        f"{len(intent_names)} intent(s){intent_part}."
                    )
            elif diagram_type == "ObjectDiagram":
                # Count actual objects only — exclude attribute sub-elements.
                object_names = [
                    el.get("name") for el in elements.values()
                    if isinstance(el, dict) and el.get("type") == "Object"
                    and isinstance(el.get("name"), str) and el["name"].strip()
                ]
                object_count = len(object_names)
                if object_count > 0:
                    preview = ", ".join(object_names[:6])
                    extra = f" (+{object_count - 6} more)" if object_count > 6 else ""
                    return f"{diagram_type}: {object_count} object(s): {preview}{extra}."
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

    # Explicit class COUNT header first — so factual queries ("how many
    # classes?") are answered from a stated number, and relationships are
    # never miscounted as classes.
    class_items = list(class_data.items())
    names_preview = ", ".join(cd["name"] for _, cd in class_items[:max_classes])
    if len(class_items) > max_classes:
        names_preview += f" (+{len(class_items) - max_classes} more)"
    lines.append(f"Classes ({len(class_items)}): {names_preview}")
    for cid, cd in class_items[:max_classes]:
        parts = [f"  - {cd['name']}"]
        if cd["attrs"]:
            attrs_str = ", ".join(cd["attrs"][:max_attrs])
            if len(cd["attrs"]) > max_attrs:
                attrs_str += f" (+{len(cd['attrs']) - max_attrs} more)"
            parts.append(f"attributes: {attrs_str}")
        if cd["methods"]:
            parts.append(f"methods: {', '.join(cd['methods'][:max_attrs])}")
        lines.append(" | ".join(parts))

    # Relationships — separate generalizations (inheritance) from associations
    # so "is X a subclass of Y?" is answerable and the two are never conflated.
    _GEN_TYPES = {"ClassInheritance", "ClassGeneralization", "ClassRealization"}
    _REL_LABEL = {
        "ClassComposition": "composition", "ClassAggregation": "aggregation",
        "ClassBidirectional": "association", "ClassUnidirectional": "directed association",
        "ClassDependency": "dependency", "ClassAssociation": "association",
    }
    gens: List[str] = []
    assocs: List[str] = []
    if isinstance(relationships, dict):
        for rel in relationships.values():
            if not isinstance(rel, dict):
                continue
            source = rel.get("source")
            target = rel.get("target")
            if not isinstance(source, dict) or not isinstance(target, dict):
                continue
            src_name = class_data.get(source.get("element", ""), {}).get("name", source.get("element", ""))
            tgt_name = class_data.get(target.get("element", ""), {}).get("name", target.get("element", ""))
            rtype = rel.get("type", "ClassBidirectional")
            if rtype in _GEN_TYPES:
                # Apollon inheritance arrow points child -> parent.
                gens.append(f"{src_name} extends {tgt_name}")
            else:
                label = _REL_LABEL.get(rtype, "association")
                src_mult = source.get("multiplicity", "")
                tgt_mult = target.get("multiplicity", "")
                mult = f" [{src_mult}..{tgt_mult}]" if (src_mult or tgt_mult) else ""
                rel_name = rel.get("name", "")
                name_str = f' "{rel_name}"' if rel_name else ""
                assocs.append(f"{src_name} -> {tgt_name} ({label}){mult}{name_str}")
    if gens:
        lines.append(f"Generalizations ({len(gens)}): " + "; ".join(gens[:15]))
    if assocs:
        more = f" (+{len(assocs) - 15} more)" if len(assocs) > 15 else ""
        lines.append(f"Relationships ({len(assocs)}): " + "; ".join(assocs[:15]) + more)

    return lines


def _summarize_state_machine(model: Dict[str, Any], *, max_items: int = 20) -> List[str]:
    """Summarize a StateMachineDiagram model: real states, bodies, transitions.

    Only genuine ``State`` elements are counted/listed as states — the editor
    also creates a ``StateInitialNode`` pseudostate plus per-state
    ``StateBody``/``StateFallbackBody``/``StateCodeBlock`` sub-elements, none
    of which are states themselves (mirrors how the class-diagram summary
    excludes attributes/methods from the class count).
    """
    elements = model.get("elements")
    relationships = model.get("relationships")
    if not isinstance(elements, dict):
        return []

    lines: List[str] = []
    element_names: Dict[str, str] = {}  # id -> name (any element, for transition lookups)
    state_data: Dict[str, Dict[str, Any]] = {}  # id -> {name, bodies, fallbacks, entry, exit, do}
    initial_count = 0
    final_count = 0

    for eid, el in elements.items():
        if not isinstance(el, dict):
            continue
        el_type = el.get("type")
        name = el.get("name") or ""
        if el_type == "State":
            if not name.strip():
                continue
            element_names[eid] = name.strip()
            state_data[eid] = {
                "name": name.strip(), "bodies": [], "fallbacks": [],
                "entry": el.get("entryAction", "") or "",
                "exit": el.get("exitAction", "") or "",
                "do": el.get("doActivity", "") or "",
            }
        elif el_type == "StateInitialNode":
            initial_count += 1
            element_names[eid] = name or "(initial)"
        elif el_type == "StateFinalNode":
            final_count += 1
            element_names[eid] = name or "(final)"

    # Attach per-state body/fallback function names (owner-based, mirrors
    # how the class-diagram summary attaches attributes/methods to classes).
    for el in elements.values():
        if not isinstance(el, dict):
            continue
        owner = el.get("owner")
        if not isinstance(owner, str) or owner not in state_data:
            continue
        el_type = el.get("type")
        body_name = el.get("name")
        if not isinstance(body_name, str) or not body_name.strip():
            continue
        if el_type == "StateBody":
            state_data[owner]["bodies"].append(body_name.strip())
        elif el_type == "StateFallbackBody":
            state_data[owner]["fallbacks"].append(body_name.strip())

    # Explicit state COUNT header first — so factual queries ("how many
    # states?") are answered from a stated number instead of the LLM
    # counting listed lines (which previously included the StateInitialNode
    # pseudostate as an extra "state").
    state_items = list(state_data.items())
    names_preview = ", ".join(sd["name"] for _, sd in state_items[:max_items])
    if len(state_items) > max_items:
        names_preview += f" (+{len(state_items) - max_items} more)"
    lines.append(f"States ({len(state_items)}): {names_preview}")
    for _, sd in state_items[:max_items]:
        parts = [f"  - {sd['name']}"]
        if sd["bodies"]:
            parts.append(f"body: {', '.join(sd['bodies'])}")
        if sd["fallbacks"]:
            parts.append(f"fallback: {', '.join(sd['fallbacks'])}")
        if sd["entry"]:
            parts.append(f"entry={sd['entry']}")
        if sd["exit"]:
            parts.append(f"exit={sd['exit']}")
        if sd["do"]:
            parts.append(f"do={sd['do']}")
        lines.append(" | ".join(parts))

    if initial_count or final_count:
        bits = []
        if initial_count:
            bits.append(f"{initial_count} initial pseudostate(s)")
        if final_count:
            bits.append(f"{final_count} final pseudostate(s)")
        lines.append("Pseudostates (not counted as states): " + ", ".join(bits))

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


def _summarize_agent_diagram(model: Dict[str, Any], *, max_items: int = 20) -> List[str]:
    """Summarize an AgentDiagram model: real states/intents (with their reply
    bodies / training phrases), and transitions.

    Only genuine ``AgentState``/``AgentIntent`` elements are counted/listed —
    the editor also creates a ``StateInitialNode`` pseudostate plus per-element
    ``AgentStateBody`` (bot reply text) / ``AgentIntentBody`` (training phrase)
    sub-elements, none of which are states or intents themselves (mirrors how
    the class-diagram summary excludes attributes/methods from the class
    count). Training phrases are surfaced so questions like "which intent
    handles the user saying hello" are answerable from the summary alone.
    """
    elements = model.get("elements")
    relationships = model.get("relationships")
    if not isinstance(elements, dict):
        return []

    lines: List[str] = []
    element_names: Dict[str, str] = {}  # id -> name (any element, for transition lookups)
    state_data: Dict[str, Dict[str, Any]] = {}   # id -> {name, replies}
    intent_data: Dict[str, Dict[str, Any]] = {}  # id -> {name, phrases}

    for eid, el in elements.items():
        if not isinstance(el, dict):
            continue
        el_type = el.get("type")
        name = el.get("name") or ""
        if el_type == "AgentState":
            if not name.strip():
                continue
            element_names[eid] = name.strip()
            state_data[eid] = {"name": name.strip(), "replies": []}
        elif el_type == "AgentIntent":
            if not name.strip():
                continue
            element_names[eid] = name.strip()
            intent_data[eid] = {"name": name.strip(), "phrases": []}
        elif el_type == "StateInitialNode":
            element_names[eid] = name or "(initial)"

    # Attach owned reply bodies / training phrases (owner-based, mirrors how
    # the class-diagram summary attaches attributes/methods to classes).
    for el in elements.values():
        if not isinstance(el, dict):
            continue
        owner = el.get("owner")
        el_type = el.get("type")
        text = el.get("name")
        if not isinstance(text, str) or not text.strip():
            continue
        if el_type == "AgentStateBody" and owner in state_data:
            state_data[owner]["replies"].append(text.strip())
        elif el_type == "AgentIntentBody" and owner in intent_data:
            intent_data[owner]["phrases"].append(text.strip())

    # Explicit COUNT headers first — mirrors the class-diagram summary so
    # factual queries ("how many intents?") are answered from a stated
    # number instead of the LLM guessing from a flat element dump.
    state_items = list(state_data.items())
    if state_items:
        names_preview = ", ".join(sd["name"] for _, sd in state_items[:max_items])
        if len(state_items) > max_items:
            names_preview += f" (+{len(state_items) - max_items} more)"
        lines.append(f"States ({len(state_items)}): {names_preview}")
        for _, sd in state_items[:max_items]:
            if sd["replies"]:
                lines.append(f"  - {sd['name']} | replies: {'; '.join(sd['replies'][:5])}")
            else:
                lines.append(f"  - {sd['name']}")

    intent_items = list(intent_data.items())
    if intent_items:
        names_preview = ", ".join(idata["name"] for _, idata in intent_items[:max_items])
        if len(intent_items) > max_items:
            names_preview += f" (+{len(intent_items) - max_items} more)"
        lines.append(f"Intents ({len(intent_items)}): {names_preview}")
        for _, idata in intent_items[:max_items]:
            if idata["phrases"]:
                lines.append(f"  - {idata['name']} | training phrases: {', '.join(idata['phrases'][:8])}")
            else:
                lines.append(f"  - {idata['name']}")

    # Transitions — real type is "AgentStateTransition" (plus
    # "AgentStateTransitionInit" wiring the initial pseudostate). Annotate
    # each with the intent that triggers it so "which intent leads to X" is
    # answerable directly from this line.
    if isinstance(relationships, dict):
        transitions: List[str] = []
        for rel in relationships.values():
            if not isinstance(rel, dict):
                continue
            if rel.get("type") not in ("AgentStateTransition", "AgentStateTransitionInit"):
                continue
            source = rel.get("source")
            target = rel.get("target")
            if not isinstance(source, dict) or not isinstance(target, dict):
                continue
            src_id = source.get("element", "")
            tgt_id = target.get("element", "")
            src_name = element_names.get(src_id, src_id)
            tgt_name = element_names.get(tgt_id, tgt_id)
            predefined = rel.get("predefined")
            predefined_type = predefined.get("predefinedType", "") if isinstance(predefined, dict) else ""
            intent_name = predefined.get("intentName", "") if isinstance(predefined, dict) else ""
            if intent_name:
                detail = f" (on intent: {intent_name})"
            elif predefined_type == "auto":
                detail = " (auto)"
            elif predefined_type:
                detail = f" ({predefined_type})"
            else:
                detail = ""
            transitions.append(f"{src_name} -> {tgt_name}{detail}")
        if transitions:
            trans_str = "; ".join(transitions[:15])
            more = f" (+{len(transitions) - 15} more)" if len(transitions) > 15 else ""
            lines.append(f"Transitions ({len(transitions)}): {trans_str}{more}")

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
# "Is this diagram non-trivial?" — used by describe-model to skip empty
# / seed-content diagrams so they don't drown out diagrams the user built.
# ---------------------------------------------------------------------------


def _quantum_circuit_is_nontrivial(model: Dict[str, Any]) -> bool:
    """Decide whether a quantum circuit was deliberately authored by the user.

    The editor often inserts ambient default content (e.g. a few qubits with a
    handful of single-qubit gates) when a tab is created.  We treat a circuit
    as trivial unless it has at least one of:

    - more than 3 gate operations total, OR
    - at least one entangling / multi-qubit gate (control dot + target, SWAP,
      multi-qubit functional block, etc.), OR
    - at least one measurement gate (suggests a real experiment), OR
    - more than 6 occupied columns (a wide circuit is unlikely to be seed).

    A circuit with 0 gates is always trivial.
    """
    cols = model.get("cols")
    if not isinstance(cols, list) or not cols:
        return False

    gate_total = 0
    occupied_cols = 0
    has_control = False
    has_swap = False
    has_measurement = False
    has_multiqubit_func = False

    for col in cols:
        if not isinstance(col, list):
            continue
        col_has_gate = False
        swaps_in_col = 0
        controls_in_col = 0
        targets_in_col = 0
        for cell in col:
            if cell == 1 or cell is None:
                continue
            symbol = str(cell)
            col_has_gate = True
            gate_total += 1
            if symbol == "•" or symbol == "*":
                has_control = True
                controls_in_col += 1
            elif symbol == "◦":
                has_control = True
                controls_in_col += 1
            elif symbol == "Swap":
                swaps_in_col += 1
                has_swap = True
            elif symbol in {"Measure", "Measure X", "Measure Y"}:
                has_measurement = True
            elif symbol.startswith("__FUNC__") or symbol.startswith("<<"):
                has_multiqubit_func = True
            else:
                targets_in_col += 1
        if col_has_gate:
            occupied_cols += 1
        # A column with both controls and targets is an actual entangling op.
        if controls_in_col >= 1 and targets_in_col >= 1:
            has_control = True
        if swaps_in_col >= 2:
            has_swap = True

    if gate_total == 0:
        return False

    if has_control or has_swap or has_measurement or has_multiqubit_func:
        return True

    if gate_total > 3:
        return True

    if occupied_cols > 6:
        return True

    return False


def is_diagram_nontrivial(model_data: Any, diagram_type: str) -> bool:
    """Return True if a diagram contains user-authored content worth describing.

    Used by the describe-model flow so empty / seed diagrams are skipped
    instead of being enumerated as "0 elements" noise.

    Rules (conservative — when in doubt, prefer True):

    - ClassDiagram: at least one user-named class.
    - StateMachineDiagram / AgentDiagram / ObjectDiagram: at least one
      element the user added (any element is enough).
    - GUINoCodeDiagram: at least one page.
    - QuantumCircuitDiagram: see :func:`_quantum_circuit_is_nontrivial` —
      filters out the 0-gate / sparse-default circuits the editor seeds.
    - Unknown types: True if the model dict is non-empty.
    """
    if not isinstance(model_data, dict) or not model_data:
        return False

    if diagram_type == "ClassDiagram":
        elements = model_data.get("elements")
        if not isinstance(elements, dict):
            return False
        for el in elements.values():
            if (
                isinstance(el, dict)
                and el.get("type") == "Class"
                and isinstance(el.get("name"), str)
                and el["name"].strip()
            ):
                return True
        return False

    if diagram_type == "ObjectDiagram":
        elements = model_data.get("elements")
        if not isinstance(elements, dict):
            return False
        for el in elements.values():
            if isinstance(el, dict) and el.get("type") == "Object":
                return True
        return False

    if diagram_type == "StateMachineDiagram":
        elements = model_data.get("elements")
        if not isinstance(elements, dict):
            return False
        # Any State element counts; pure initial/final markers without a
        # named state are seed content.
        for el in elements.values():
            if isinstance(el, dict) and el.get("type") == "State":
                return True
        return False

    if diagram_type == "AgentDiagram":
        elements = model_data.get("elements")
        if not isinstance(elements, dict):
            return False
        for el in elements.values():
            if isinstance(el, dict) and el.get("type") in {
                "AgentState", "AgentIntent", "AgentStateBody", "AgentIntentBody",
            }:
                return True
        return False

    if diagram_type == "GUINoCodeDiagram":
        pages = model_data.get("pages")
        return isinstance(pages, list) and len(pages) > 0

    if diagram_type == "QuantumCircuitDiagram":
        return _quantum_circuit_is_nontrivial(model_data)

    # Unknown diagram type — be permissive.
    return True


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
