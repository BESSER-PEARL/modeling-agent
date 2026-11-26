"""The requirements an application request states, and the class-diagram elements that realize them.

Before modeling a complete system, the class diagram handler extracts the
requirements of the user's request, lists the ones a class diagram can realize
in its structured pass and reports which elements realize each one. They travel
on the ``inject_complete_system`` reply (``requirements``, ``realizations``, and
the extraction's LLM in ``requirementsModel``); the editor records the
extraction as a ``RequirementExtraction`` that writes one ``RequirementArtifact``
per requirement, and each realization as a trace link from it
(``shared/traceability/traceRecorder.ts`` in the webapp).
"""

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Set

from common.agent_config import MAX_REQUIREMENT_CHARS, MAX_REQUIREMENTS
from common.errors import LLMPredictionError, non_recoverable_cause
from llm.model_config import MODEL_GENERATION_SMALL
from llm.provider import OutputTruncated
from prompts import requirements as requirement_prompts
from schemas import RequirementsExtractionSpec

from ..core.base_handler import BaseDiagramHandler

logger = logging.getLogger(__name__)

# The kinds a class diagram can realize; the structured pass lists only these. A
# screen, an external service, a quality or a technology is realized by the
# application, not by a class, so listing them costs tokens for no realization.
CLASS_REALIZABLE_KINDS = frozenset({
    "data", "uniqueness", "validation", "rule", "computed", "transition", "action",
})

# Completion tokens added to the structured pass per listed requirement, for its
# ``realizations`` entry (about 40-55 tokens: an id and every element modeled
# because of it, often 3-6 names). ``realizations`` is the reply's last field,
# so without this headroom a long list truncates the diagram. 128 requirements
# add 8192 to the 8192 of the complete-system schemas: the 16384 the GUI handler
# already asks of every provider.
REALIZATION_TOKENS_PER_REQUIREMENT = 64

_WHITESPACE_RE = re.compile(r"\s+")
_INVARIANT_REF_RE = re.compile(r"^inv\s+(\w+)$")
_INVARIANT_NAME_RE = re.compile(r"\binv\s+(\w+)\s*:")
_METHOD_REF_RE = re.compile(r"^(\w+)\.(\w+)\s*\(.*\)$")
_MEMBER_REF_RE = re.compile(r"^(\w+)\.(\w+)$")
_CLASS_REF_RE = re.compile(r"^(\w+)$")


def normalise_for_quote(text: str) -> str:
    """Text as a quote is compared with its request: lowercased, whitespace collapsed.

    The editor compares a requirement's quote with the prompt the same way before it
    records the prompt as the requirement's source.
    """
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def extract_requirements(handler: BaseDiagramHandler, request_text: str) -> List[Dict[str, str]]:
    """``[{"id": "R1", "text", "quote", "kind"}, ...]``: the requirements *request_text* states.

    One structured call on the small model tier, through *handler*'s
    ``predict_structured``. An item whose quote is not in the request was not
    stated there and is dropped, as is an empty or overlong one; the list is
    capped at ``MAX_REQUIREMENTS`` after that, so a full list still reads as
    full, and ids are given in order. A failed call, a reply cut off at the
    token cap included, returns ``[]`` with a warning: the diagram is still
    built, without requirements. A provider error no retry can fix (quota, a
    rejected key) is raised: the diagram's own call would only fail the same way.
    """
    try:
        parsed = handler.predict_structured(
            requirement_prompts.extraction_prompt(request_text),
            RequirementsExtractionSpec,
            system_prompt=requirement_prompts.EXTRACTION_SYSTEM_PROMPT,
            model=MODEL_GENERATION_SMALL,
        )
    except LLMPredictionError as exc:
        # A list too long for the token cap is not a provider fault: the diagram's
        # own call is a different reply and can still succeed.
        truncated = isinstance(exc, OutputTruncated) or isinstance(exc.__cause__, OutputTruncated)
        if not truncated and non_recoverable_cause(exc) is not None:
            raise
        logger.warning("[Requirements] extraction failed; building the diagram without requirements: %s", exc)
        return []

    source = normalise_for_quote(request_text)
    requirements: List[Dict[str, str]] = []
    for item in parsed.requirements:
        text, quote = item.text.strip(), item.quote.strip()
        if not text or len(text) > MAX_REQUIREMENT_CHARS:
            logger.warning("[Requirements] dropping a requirement of %d characters", len(text))
            continue
        if not quote or len(quote) > MAX_REQUIREMENT_CHARS:
            logger.warning("[Requirements] dropping %r: its quote has %d characters", text, len(quote))
            continue
        if normalise_for_quote(quote) not in source:
            logger.warning("[Requirements] dropping %r: its quote %r is not in the request", text, quote)
            continue
        if len(requirements) == MAX_REQUIREMENTS:
            logger.warning("[Requirements] more than %d requirements extracted; keeping the first %d",
                           MAX_REQUIREMENTS, MAX_REQUIREMENTS)
            break
        requirements.append({"id": f"R{len(requirements) + 1}", "text": text, "quote": quote, "kind": item.kind})
    return requirements


def extraction_model(handler: BaseDiagramHandler) -> Dict[str, str]:
    """``{"name", "provider"}`` of the LLM :func:`extract_requirements` runs on for
    *handler*'s request (a user's own key maps the tier to its model): the editor
    records the extraction as a call to it."""
    provider, name = handler.llm_provider.model_for(MODEL_GENERATION_SMALL)
    return {"name": name, "provider": provider}


def class_realizable(requirements: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """The *requirements* of a kind a class diagram can realize, in order."""
    return [r for r in requirements if r["kind"] in CLASS_REALIZABLE_KINDS]


class _SpecIndex:
    """The elements of a class-diagram system spec, by the names a realization uses."""

    def __init__(self, system_spec: Dict[str, Any]):
        self.classes = {c.get("className"): c for c in system_spec.get("classes", []) if c.get("className")}
        self.associations = [r for r in system_spec.get("relationships", [])
                             if r.get("type") not in ("Inheritance", "Realization", "Dependency")]
        # An invariant is named by its expression, as the editor names it; an
        # unnamed one cannot be referred to.
        self.invariants: Dict[str, str] = {}
        for constraint in system_spec.get("constraints", []):
            match = _INVARIANT_NAME_RE.search(constraint.get("expression") or "")
            if match:
                self.invariants[match.group(1)] = constraint.get("context")

    def resolve(self, reference: str) -> Optional[Dict[str, str]]:
        """The element *reference* names, as ``{"type", "name"[, "owner"]}``, or None."""
        reference = reference.strip()
        if match := _INVARIANT_REF_RE.match(reference):
            name = match.group(1)
            if name in self.invariants:
                return {"type": "Constraint", "name": name, "owner": self.invariants[name]}
            return None
        if match := _METHOD_REF_RE.match(reference):
            owner, name = match.groups()
            methods = self.classes.get(owner, {}).get("methods") or []
            if any(m.get("name") == name for m in methods):
                return {"type": "Method", "name": name, "owner": owner}
            return None
        if match := _MEMBER_REF_RE.match(reference):
            return self._member(*match.groups())
        if match := _CLASS_REF_RE.match(reference):
            cls = self.classes.get(match.group(1))
            if cls is None:
                return None
            return {"type": "Enumeration" if cls.get("isEnumeration") else "Class", "name": match.group(1)}
        return None

    def literals(self, enumeration: str) -> List[Dict[str, str]]:
        """The literals of *enumeration*, as :meth:`resolve` gives them."""
        attributes = self.classes.get(enumeration, {}).get("attributes") or []
        return [{"type": "EnumerationLiteral", "name": a["name"], "owner": enumeration}
                for a in attributes if a.get("name")]

    def _member(self, owner: str, name: str) -> Optional[Dict[str, str]]:
        """An attribute or enumeration literal of *owner*, else an association end *owner* navigates."""
        cls = self.classes.get(owner)
        if cls is None:
            return None
        if any(a.get("name") == name for a in cls.get("attributes") or []):
            return {"type": "EnumerationLiteral" if cls.get("isEnumeration") else "Property",
                    "name": name, "owner": owner}
        for rel in self.associations:
            ends = ((rel.get("source"), rel.get("name")), (rel.get("target"), rel.get("sourceRole")))
            if (owner, name) in ends and rel.get("name"):
                # The editor names an association by its target end (``rel.name``).
                return {"type": "BinaryAssociation", "name": rel["name"], "owner": rel.get("source")}
        return None


def resolve_realizations(
    realizations: Iterable[Dict[str, Any]], system_spec: Dict[str, Any], requirement_ids: Set[str],
) -> List[Dict[str, Any]]:
    """``[{"requirement": "R2", "elements": [{"type", "name"[, "owner"]}, ...]}]`` for the editor.

    Each reference is resolved against the FINAL *system_spec*, after every
    guard: a reference to an element a guard renamed or dropped, or to a
    requirement not in *requirement_ids* (the ones listed to the structured
    pass), is left out with a warning, never sent. ``type`` is the B-UML
    metaclass name the editor's trace records (``Class``, ``Enumeration``,
    ``Property``, ``EnumerationLiteral``, ``Method``, ``BinaryAssociation``,
    ``Constraint``); ``owner`` is the class holding the element (an invariant's
    context, an association's source). An enumeration named whole is realized
    with its values, so its literals realize the requirement too; when the
    realization names some of its literals, only those do (a rule about one
    status does not claim the others).
    """
    index = _SpecIndex(system_spec)
    elements_by_requirement: Dict[str, List[Dict[str, str]]] = {}
    for realization in realizations:
        requirement = (realization.get("requirement") or "").strip()
        if requirement not in requirement_ids:
            logger.warning("[Requirements] dropping a realization of unlisted requirement %r", requirement)
            continue
        resolved = []
        for reference in realization.get("elements") or []:
            element = index.resolve(reference)
            if element is None:
                logger.warning("[Requirements] %s: %r names no element of the diagram", requirement, reference)
            else:
                resolved.append(element)
        named_literals_of = {e["owner"] for e in resolved if e["type"] == "EnumerationLiteral"}
        elements = elements_by_requirement.setdefault(requirement, [])
        for element in resolved:
            whole_enumeration = element["type"] == "Enumeration" and element["name"] not in named_literals_of
            for realizing in [element, *(index.literals(element["name"]) if whole_enumeration else [])]:
                if realizing not in elements:
                    elements.append(realizing)
    return [{"requirement": rid, "elements": elements}
            for rid, elements in elements_by_requirement.items() if elements]
