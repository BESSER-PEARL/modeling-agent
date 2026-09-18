"""User-profile metamodel loader.

The BESSER editor's *User Profile* perspective (diagram type ``UserDiagram``)
draws its instance boxes from a fixed metamodel that the editor does **not**
transmit to the modeling agent.  We therefore bundle a copy of that metamodel
and load it here as the reference catalog for the user-profile handler —
analogous to how the object-diagram handler uses a reference class diagram.

Source of truth: this file is a verbatim copy of
``packages/editor/src/main/packages/user-modeling/usermetamodel_buml_short.json``
in the BESSER-Web-Modeling-Editor repository.  Keep them in sync — the
``classId`` / ``attributeId`` values the agent emits must match the ids the
editor uses, or the editor cannot link generated attribute rows back to the
schema.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_METAMODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "resources", "user_metamodel.json"
)

# Human-readable descriptions of the metamodel elements (the bundled metamodel
# itself carries no prose). Distilled from the User-Modeling-Language READMEs.
_SEMANTICS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "resources",
    "user_metamodel_semantics.json",
)

_ROOT_CLASS = "User"

# Cached after first successful load (the files never change at runtime).
_CACHED_METAMODEL: Optional[Dict[str, Any]] = None
_CACHED_SEMANTICS: Optional[Dict[str, Any]] = None
_CACHED_GUIDE: Optional[str] = None


def load_user_metamodel() -> Dict[str, Any]:
    """Return the bundled user-profile metamodel as a ClassDiagram-shaped dict.

    The returned object has the same ``{elements, relationships, ...}`` shape
    as any class diagram, so the object-diagram reference-catalog helpers work
    on it unchanged.  Returns an empty ``{"elements": {}}`` dict if the bundled
    resource is missing or unreadable (the handler degrades gracefully).
    """
    global _CACHED_METAMODEL
    if _CACHED_METAMODEL is not None:
        return _CACHED_METAMODEL

    try:
        with open(_METAMODEL_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or not isinstance(data.get("elements"), dict):
            logger.error(
                "[UserMetamodel] Bundled metamodel has unexpected shape; "
                "using empty catalog."
            )
            data = {"elements": {}, "relationships": {}}
    except (OSError, json.JSONDecodeError) as exc:
        logger.error(f"[UserMetamodel] Failed to load bundled metamodel: {exc}")
        data = {"elements": {}, "relationships": {}}

    _CACHED_METAMODEL = data
    return _CACHED_METAMODEL


def load_user_metamodel_semantics() -> Dict[str, Any]:
    """Return the curated element/attribute descriptions for the metamodel.

    Shape: ``{"overview": str, "classes": {name: desc}, "attributes":
    {"Class.attr": desc}}``.  Returns an empty dict if the resource is missing
    or unreadable (the guide then falls back to names/types only).
    """
    global _CACHED_SEMANTICS
    if _CACHED_SEMANTICS is not None:
        return _CACHED_SEMANTICS

    try:
        with open(_SEMANTICS_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            data = {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.error(f"[UserMetamodel] Failed to load metamodel semantics: {exc}")
        data = {}

    _CACHED_SEMANTICS = data
    return _CACHED_SEMANTICS


# ---------------------------------------------------------------------------
# Metamodel guide (grounding text for natural-language explanations)
# ---------------------------------------------------------------------------

def _build_catalog(
    metamodel: Dict[str, Any],
) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, List[str]], Dict[str, List[Tuple[str, bool]]], List[str]]:
    """Derive a compact catalog from the bundled metamodel.

    Returns ``(classes, enums, tree, ordered_names)`` where:
    - ``classes``   maps class name -> ``[(attr_name, attr_type), ...]``
    - ``enums``     maps enumeration name -> ``[literal, ...]``
    - ``tree``      maps parent class name -> ``[(child_name, is_many), ...]``
    - ``ordered_names`` is the class names in breadth-first order from ``User``.
    """
    elements = metamodel.get("elements", {})
    relationships = metamodel.get("relationships", {}) or {}

    id_to_name: Dict[str, str] = {}
    class_names: set = set()
    enum_names: set = set()
    for el in elements.values():
        if not isinstance(el, dict):
            continue
        name = el.get("name")
        if not isinstance(name, str):
            continue
        id_to_name[el.get("id")] = name
        if el.get("type") in ("Class", "AbstractClass"):
            class_names.add(name)
        elif el.get("type") == "Enumeration":
            enum_names.add(name)

    # Enumerations: literals are the child elements listed in `attributes`.
    enums: Dict[str, List[str]] = {}
    classes: Dict[str, List[Tuple[str, str]]] = {}
    for el in elements.values():
        if not isinstance(el, dict):
            continue
        name = el.get("name")
        kind = el.get("type")
        if kind == "Enumeration" and isinstance(name, str):
            literals: List[str] = []
            for cid in el.get("attributes", []) or []:
                child = elements.get(cid)
                if isinstance(child, dict) and isinstance(child.get("name"), str):
                    literals.append(child["name"].strip())
            enums[name] = literals
        elif kind in ("Class", "AbstractClass") and isinstance(name, str):
            attrs: List[Tuple[str, str]] = []
            for cid in el.get("attributes", []) or []:
                child = elements.get(cid)
                if isinstance(child, dict) and child.get("type") == "ClassAttribute":
                    attrs.append(
                        (str(child.get("name", "")).strip(), str(child.get("attributeType", "")).strip())
                    )
            classes[name] = attrs

    # Undirected association adjacency (endpoint multiplicity kept per side),
    # restricted to edges whose both ends are real classes and are not
    # inheritance edges. Then a BFS from User orients them parent -> child.
    adjacency: Dict[str, List[Tuple[str, str]]] = {c: [] for c in class_names}
    for rel in relationships.values():
        if not isinstance(rel, dict) or rel.get("type") == "ClassInheritance":
            continue
        src = rel.get("source", {}) or {}
        tgt = rel.get("target", {}) or {}
        sn = id_to_name.get(src.get("element"))
        tn = id_to_name.get(tgt.get("element"))
        if sn not in class_names or tn not in class_names:
            continue
        adjacency[sn].append((tn, str(tgt.get("multiplicity", ""))))
        adjacency[tn].append((sn, str(src.get("multiplicity", ""))))

    tree: Dict[str, List[Tuple[str, bool]]] = {}
    ordered_names: List[str] = []
    if _ROOT_CLASS in class_names:
        visited = {_ROOT_CLASS}
        queue: deque = deque([_ROOT_CLASS])
        while queue:
            parent = queue.popleft()
            ordered_names.append(parent)
            for child, child_mult in sorted(adjacency.get(parent, [])):
                if child in visited:
                    continue
                visited.add(child)
                tree.setdefault(parent, []).append((child, "*" in child_mult))
                queue.append(child)
        # Append any classes not reachable from User (defensive).
        for name in sorted(class_names):
            if name not in visited:
                ordered_names.append(name)
    else:
        ordered_names = sorted(class_names)

    return classes, enums, tree, ordered_names


def _format_attr_type(attr_type: str, enums: Dict[str, List[str]]) -> str:
    """Render an attribute's type, expanding enumerations to their values."""
    if attr_type in enums and enums[attr_type]:
        return f"{attr_type}: {' | '.join(enums[attr_type])}"
    return attr_type or "str"


def format_user_metamodel_guide() -> str:
    """Build a compact, human-readable reference for the user-profile metamodel.

    Combines the bundled metamodel structure (classes, attributes, types,
    enumeration values, and how elements connect) with the curated semantic
    descriptions.  Used to *ground* natural-language explanations of the
    modeling environment so the assistant only references elements that
    actually exist.  Pure text — no LLM, no network.
    """
    global _CACHED_GUIDE
    if _CACHED_GUIDE is not None:
        return _CACHED_GUIDE

    metamodel = load_user_metamodel()
    semantics = load_user_metamodel_semantics()
    class_desc = semantics.get("classes", {}) if isinstance(semantics, dict) else {}
    attr_desc = semantics.get("attributes", {}) if isinstance(semantics, dict) else {}

    classes, enums, tree, ordered_names = _build_catalog(metamodel)

    lines: List[str] = ["USER PROFILE METAMODEL — the elements a user profile is built from."]
    overview = semantics.get("overview") if isinstance(semantics, dict) else None
    if isinstance(overview, str) and overview.strip():
        lines.append("")
        lines.append(overview.strip())

    lines.append("")
    lines.append("ELEMENTS (each box in a user profile instantiates one of these classes):")
    used_enums: set = set()
    for name in ordered_names:
        attrs = classes.get(name, [])
        desc = class_desc.get(name, "")
        header = f"- {name}"
        if desc:
            header += f" — {desc}"
        lines.append(header)
        if not attrs:
            lines.append("    attributes: none (grouping/root element)")
            continue
        for attr_name, attr_type in attrs:
            if attr_type in enums:
                used_enums.add(attr_type)
            rendered = _format_attr_type(attr_type, enums)
            row = f"    - {attr_name} ({rendered})"
            adesc = attr_desc.get(f"{name}.{attr_name}")
            if adesc:
                row += f" — {adesc}"
            lines.append(row)

    # How elements connect.
    if tree:
        lines.append("")
        lines.append("HOW ELEMENTS CONNECT (every element attaches under the single root User):")
        for parent in ordered_names:
            children = tree.get(parent)
            if not children:
                continue
            parts = [f"{child} ({'0..*' if many else '0..1'})" for child, many in children]
            lines.append(f"- {parent} -> {', '.join(parts)}")

    lines.append("")
    lines.append(
        "Note: each attribute row is a matching CRITERION with a comparison operator "
        "(<, <=, ==, >=, >), e.g. \"age >= 18\" or \"level == B2\" — not a fixed value. "
        "A profile represents a GROUP of target users, not one individual."
    )

    _CACHED_GUIDE = "\n".join(lines)
    return _CACHED_GUIDE


# ---------------------------------------------------------------------------
# Help-question detection & prompt building
# ---------------------------------------------------------------------------

# Explicit phrases that mark a help question as being about the user-profile
# modeling environment even when the active diagram is not a UserDiagram.
# Deliberately specific (not bare class names like "Culture") to avoid
# misclassifying class-diagram help questions.
_USER_PROFILE_HELP_MARKERS = (
    "user profile",
    "user-profile",
    "user model",
    "user-model",
    "user metamodel",
    "user-metamodel",
    "user modeling",
    "user modelling",
    "target user",
    "persona",
)


def is_user_profile_help(message: Optional[str]) -> bool:
    """True if a help question is explicitly about user-profile modeling."""
    if not message:
        return False
    lowered = message.lower()
    return any(marker in lowered for marker in _USER_PROFILE_HELP_MARKERS)


def build_user_profile_help_prompt(message: str) -> str:
    """Build a grounded prompt for explaining the user-profile environment.

    Answers questions like "what is the Accessibility element?" or "which
    elements are necessary to model an old user with sight issues?" strictly
    against the bundled metamodel and its curated semantics.
    """
    guide = format_user_metamodel_guide()
    return (
        "You are an expert on the BESSER User Profile modeling language. A user "
        "profile describes a target group of users as boxes drawn from a fixed "
        "metamodel, and the user is asking about that modeling environment.\n\n"
        f'The user asked: "{message}"\n\n'
        "Use ONLY the metamodel below as your source of truth — its elements "
        "(classes), their attributes and value types, the enumeration values, "
        "and how the elements connect. Do not invent elements, attributes, or "
        "enumeration values that are not listed.\n\n"
        f"{guide}\n\n"
        "Answer the user's question based on this metamodel and the semantics of "
        "the element and attribute names:\n"
        "- If they ask what an element or attribute IS, explain what it "
        "represents, list its attributes and allowed values, and say how it "
        "connects to the rest (e.g. Accessibility groups the user's accessibility "
        "needs and sits between User and Disability).\n"
        "- If they ask WHICH elements are needed to model a described user, name "
        "the specific classes and the exact attributes / enumeration values to "
        "use, with an example criterion and operator (e.g. an old user → "
        "Personal_Information with age using '>' or '>='; sight issues → a "
        "Disability with affects == 'Sight'). Mention the grouping/root elements "
        "the boxes attach through (here, Disability attaches under Accessibility, "
        "which attaches under User).\n"
        "- Remember attribute rows are matching CRITERIA with a comparison "
        "operator, not fixed values.\n\n"
        "Keep the answer concise and well-formatted in Markdown. Where useful, "
        "tell them they can ask you to build it (e.g. \"create a profile for a "
        "user older than 65 with a sight disability\")."
    )
