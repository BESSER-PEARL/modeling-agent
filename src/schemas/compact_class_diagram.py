"""Compact structured-output schema for complete-system class generation.

WHY: complete-system generation cost ~56s, almost entirely completion tokens —
~4,300 tokens of verbose JSON (one object per attribute/method, long key names
repeated thousands of times). Measured on the live deployment (5 domains,
scratch_speed_experiment/): the SAME model emitting the SAME modeling content
through this compact schema needs ~1,600 tokens → ~2.4x faster, with
structured-output enforcement intact (0 malformed member strings in 5/5 runs).

HOW: 1-letter keys and string-encoded members ("price: float",
"decreasePrice(pct: float) -> None"). :func:`expand_compact_spec` converts the
compact form into the canonical :class:`SystemClassSpec` deterministically, so
everything downstream (guards, layout, frontend payload, generators) is
untouched. Every parse is tolerant — a malformed member degrades to a sane
default, never an exception.

The encoding covers everything SystemClassSpec can express EXCEPT method
implementation bodies (``implementationType``/``code``) — complete-system
generation never emits those (they are authored via modify flows).
"""

from __future__ import annotations

import re
from typing import List, Literal

from pydantic import BaseModel, Field

from schemas.class_diagram import (
    AttributeSpec,
    MethodParameterSpec,
    MethodSpec,
    OCLConstraintSpec,
    RelationshipSpec,
    SingleClassSpec,
    SystemClassSpec,
)


# ---------------------------------------------------------------------------
# The compact schema (what the LLM emits)
# ---------------------------------------------------------------------------

class CompactClassSpec(BaseModel):
    n: str = Field(min_length=1, max_length=30,
                   description="Class name in PascalCase, ONE word (e.g. Order)")
    a: List[str] = Field(description=(
        "Attributes, ONE string each: 'name: type'. Optional decorations: "
        "visibility prefix '+','-','#','~'; '/' prefix for derived; "
        "'= default' suffix for a default value; '?' suffix if optional. "
        "For an enumeration class the entries are BARE literal names "
        "(UPPER_CASE, no type)."))
    m: List[str] = Field(description=(
        "Methods, ONE string each: 'name(param: type, ...) -> returnType'. "
        "Omit '-> ...' for void; '()' when no parameters; optional "
        "visibility prefix; '{abstract}' suffix for abstract methods."))
    k: Literal["", "abstract", "enum"] = Field(
        description="Class kind: '' normal, 'abstract', or 'enum'.")


class CompactRelationshipSpec(BaseModel):
    f: str = Field(description="Source class name (for inheritance: the SUBCLASS)")
    t: str = Field(description="Target class name (for inheritance: the SUPERCLASS)")
    k: Literal["assoc", "comp", "aggr", "inher", "real", "dep"] = Field(
        description="Kind: assoc=Association, comp=Composition, "
                    "aggr=Aggregation, inher=Inheritance, real=Realization, "
                    "dep=Dependency")
    sm: str = Field(description="Source multiplicity (1, 0..1, 0..*, 1..*); '' for inheritance")
    tm: str = Field(description="Target multiplicity; '' for inheritance")
    l: str = Field(description="Relationship name; '' if none")


class CompactSystemClassSpec(BaseModel):
    """Compact complete class diagram — expanded via expand_compact_spec."""
    name: str = Field(description="Descriptive system name, PascalCase")
    classes: List[CompactClassSpec] = Field(min_length=1)
    rels: List[CompactRelationshipSpec]
    ocl: List[str] = Field(description=(
        "Full B-OCL invariants ('context X inv name: expr') ONLY for business "
        "rules the user EXPLICITLY stated; [] otherwise — never invent rules."))


# Appended to the system generation prompt when the compact schema is active.
# The COMPLETENESS emphasis is deliberate: the compact framing measurably
# nudged the model leaner (6.8 vs 10.4 classes) without it.
COMPACT_ENCODING_RULES = (
    "\n\nCOMPACT OUTPUT ENCODING: respond via the provided compact schema. "
    "Every modeling rule above still applies UNCHANGED — model the FULL "
    "domain with the same completeness as ever: all the classes the domain "
    "needs (typically 8-12 for a typical request), thorough attributes with "
    "types, meaningful methods, and every relationship with sensible "
    "multiplicities. ONLY the encoding is compact:\n"
    "- classes: {n: PascalCase name, a: attribute strings, m: method "
    "strings, k: ''|'abstract'|'enum'}\n"
    "- attribute string: 'name: type' (types: str, int, float, bool, "
    "datetime, date, time, or an enumeration name). Decorations when "
    "needed: visibility prefix '+'/'-'/'#'/'~', '/' prefix for derived, "
    "'= value' suffix for defaults, '?' suffix for optional. Enumeration "
    "classes list BARE literal names in a (UPPER_CASE, no type).\n"
    "- method string: 'name(param: type, ...) -> returnType' — omit "
    "'-> ...' when it returns nothing; '()' for no parameters. Parameter "
    "and return types are PLAIN type names — the '?' marker belongs to "
    "attribute entries only, never inside a method string.\n"
    "- rels: {f: source, t: target, k: assoc|comp|aggr|inher|real|dep, "
    "sm/tm: multiplicities ('' for inheritance), l: name or ''}. For "
    "inheritance f is the SUBCLASS and t the SUPERCLASS.\n"
    "- ocl: [] unless the user explicitly stated a business rule.\n"
    "Do not add prose or extra fields."
)


# ---------------------------------------------------------------------------
# Deterministic expansion (compact -> canonical SystemClassSpec)
# ---------------------------------------------------------------------------

_VISIBILITY = {"+": "public", "-": "private", "#": "protected", "~": "package"}

_REL_KIND = {
    "assoc": "Association",
    "comp": "Composition",
    "aggr": "Aggregation",
    "inher": "Inheritance",
    "real": "Realization",
    "dep": "Dependency",
}

_OCL_RE = re.compile(r"context\s+(\w+)\s+inv\s*(\w*)\s*:", re.IGNORECASE)


def _parse_attribute(raw: str, is_enum_class: bool) -> AttributeSpec:
    """'[vis][/]name: type [= default] [?]' → AttributeSpec. Never raises."""
    s = (raw or "").strip()
    visibility = "public"
    derived = False
    optional = False
    default = None

    if s[:1] in _VISIBILITY:
        visibility = _VISIBILITY[s[0]]
        s = s[1:].strip()
    if s.startswith("/"):
        derived = True
        s = s[1:].strip()
    if s.endswith("?"):
        optional = True
        s = s[:-1].strip()
    if "=" in s:
        s, default = s.split("=", 1)
        default = default.strip() or None
        s = s.strip()

    if ":" in s:
        name, type_ = s.split(":", 1)
        name = name.strip()
        type_ = type_.strip() or None
        # 'name: str?' — optional marker glued to the type instead of the
        # entry ('str?' is not a BUML type).
        if type_ and type_.endswith("?"):
            optional = True
            type_ = type_.rstrip("?").strip() or None
    else:
        # Bare name — an enum literal, or an untyped attribute.
        name, type_ = s, None

    if not name:
        name = "attribute"
    if is_enum_class:
        type_ = None  # enum literals carry no type by convention

    return AttributeSpec(
        name=name[:50], type=type_, visibility=visibility,
        isDerived=derived, defaultValue=default, isOptional=optional,
    )


def _parse_method(raw: str) -> MethodSpec:
    """'[vis]name(param: type, ...) [-> ret] [{abstract}]' → MethodSpec."""
    s = (raw or "").strip()
    visibility = "public"
    is_abstract = False
    return_type = "void"

    if s[:1] in _VISIBILITY:
        visibility = _VISIBILITY[s[0]]
        s = s[1:].strip()
    if "{abstract}" in s:
        is_abstract = True
        s = s.replace("{abstract}", "").strip()
    if "->" in s:
        s, return_type = s.rsplit("->", 1)
        return_type = return_type.strip() or "void"
        s = s.strip()

    def _clean_type(token: str, fallback: str) -> str:
        # 'str?' is not a BUML type — parameters/returns have no optionality
        # in the metamodel, so the stray marker is simply dropped.
        token = token.strip().rstrip("?").strip()
        return token or fallback

    params: List[MethodParameterSpec] = []
    if "(" in s:
        name, params_raw = s.split("(", 1)
        before, sep, trailer = params_raw.rpartition(")")
        if sep:
            params_raw = before
        else:
            trailer = ""  # unterminated '(' — treat the rest as params
        # 'name(...): ret' — colon-style return instead of '-> ret'.
        trailer = trailer.strip()
        if trailer.startswith(":") and return_type == "void":
            return_type = trailer[1:].strip() or "void"
        for chunk in params_raw.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" in chunk:
                p_name, p_type = chunk.split(":", 1)
                params.append(MethodParameterSpec(
                    name=(p_name.strip() or "param")[:50],
                    type=_clean_type(p_type, "String"),
                ))
            else:
                params.append(MethodParameterSpec(name=chunk[:50], type="String"))
    else:
        name = s

    name = name.strip() or "method"
    return_type = _clean_type(return_type, "void")
    return MethodSpec(
        name=name[:50], returnType=return_type, visibility=visibility,
        parameters=params, isAbstract=is_abstract,
    )


def expand_compact_spec(compact: CompactSystemClassSpec) -> SystemClassSpec:
    """Deterministically expand the compact form into the canonical spec.

    Constructing the canonical Pydantic models runs their validators (e.g.
    the multiplicity normalizer), so the expanded spec is exactly as safe as
    one the LLM had produced directly.
    """
    classes: List[SingleClassSpec] = []
    for c in compact.classes:
        is_enum = c.k == "enum"
        classes.append(SingleClassSpec(
            className=c.n[:30],
            attributes=[_parse_attribute(a, is_enum) for a in c.a],
            methods=[] if is_enum else [_parse_method(m) for m in c.m],
            isAbstract=c.k == "abstract",
            isEnumeration=is_enum,
        ))

    relationships: List[RelationshipSpec] = []
    for r in compact.rels:
        relationships.append(RelationshipSpec(
            type=_REL_KIND.get(r.k, "Association"),
            source=r.f,
            target=r.t,
            sourceMultiplicity=r.sm.strip() or "1",
            targetMultiplicity=r.tm.strip() or ("1" if r.k == "inher" else "*"),
            name=r.l.strip() or None,
        ))

    constraints: List[OCLConstraintSpec] = []
    for inv in compact.ocl:
        inv = (inv or "").strip()
        if not inv:
            continue
        match = _OCL_RE.search(inv)
        if not match:
            continue  # not a recognizable invariant — drop rather than crash
        constraints.append(OCLConstraintSpec(
            context=match.group(1),
            expression=inv,
            name=match.group(2) or None,
        ))

    return SystemClassSpec(
        systemName=compact.name,
        classes=classes,
        relationships=relationships,
        constraints=constraints,
    )
