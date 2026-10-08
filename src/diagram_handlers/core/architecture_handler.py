"""Shared flow for the architecture diagram handlers (Component, Deployment).

Both diagrams have the same shape — container elements, contained elements,
and dependency edges addressed by ``elementId`` / ``elementName`` — so the
complete-system, modification and fallback flow lives here once. Concrete
handlers supply the prompts, schemas, starter specs and reply messages, and
declare the base handler's reference-guardrail tables.
"""

import logging
from abc import abstractmethod
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel

from model_config import MODEL_GENERATION_LARGE, MODEL_REASONING
from utilities.model_context import detailed_model_summary

from .base_handler import BaseDiagramHandler, LLMPredictionError
from .prompt_fragments import EXACT_NAMES_RULE, MULTI_MOD_ARRAY_RULE, REMOVE_ELEMENT_RULE

logger = logging.getLogger(__name__)


# Rules every architecture modification prompt shares, composed once at
# import time so the system prompt stays byte-stable across calls.
ARCHITECTURE_MODIFY_SHARED_RULES = f"""SHARED RULES:
- modify_element: {EXACT_NAMES_RULE} For an UNNAMED element set target.elementId to the exact [id] from the context instead.
- remove_element: {REMOVE_ELEMENT_RULE} Use target.elementName, or target.elementId for unnamed elements. Connected dependencies are removed automatically.
- add_dependency / remove_dependency: set changes.source and changes.target to the endpoint names or exact [id]s.
- An element added earlier in the same response may be referenced by its name in later entries (for example add an element, then add_dependency to it).
- {MULTI_MOD_ARRAY_RULE}
- If the user asks for "a second", "another", or "one more" element, add exactly ONE unless they explicitly ask for more.

When the user asks to remove or modify an element, verify it exists in the current context listing first. If no entry matches (by name or id):
- Set elementFound: false
- Set modifications: [] (empty — do NOT substitute a different element)
- Set message to explain what was not found and list the current elements.
Partial matches are valid (e.g. "Planner" matching "Planner Agent"). Only set elementFound: false when there is genuinely no match.

If the user says 'undo', 'undo that', 'revert', or similar, do not emit any modifications. Reply with modifications: [], elementFound: false,
and set message to: 'To undo, use Ctrl+Z or the undo button in the editor toolbar.'"""


class ArchitectureDiagramHandler(BaseDiagramHandler):
    """Base for diagrams of elements + dependencies addressed by elementId/elementName."""

    _REF_TARGET_ACTIONS = frozenset({"modify_element", "remove_element"})
    _REF_ENDPOINT_ACTIONS = frozenset({"add_dependency", "remove_dependency"})
    _REF_RENAME_ACTIONS = frozenset({"modify_element"})
    _REF_REMOVE_ACTIONS = frozenset({"remove_element"})

    #: Human label used in replies and logs, e.g. "component diagram".
    _LABEL: str = ""
    _SYSTEM_SCHEMA: Type[BaseModel]
    _MODIFICATION_SCHEMA: Type[BaseModel]
    _MODIFY_SYSTEM_PROMPT: str = ""
    #: Two example edits shown when a modification cannot be applied.
    _MODIFY_EXAMPLES: str = ""

    @abstractmethod
    def _reasoning_prompt(self, request: str) -> str:
        """Free-text planning prompt for the two-pass reasoning pass."""

    @abstractmethod
    def _build_system_message(self, spec: Dict[str, Any]) -> str:
        """Reply text for a generated complete system."""

    @abstractmethod
    def generate_fallback_system(self) -> Dict[str, Any]:
        """Starter diagram returned when generation fails unexpectedly."""

    def generate_complete_system(
        self,
        user_request: str,
        existing_model: Dict[str, Any] = None,
        raw_request: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a complete diagram with two-pass structured output.

        ``raw_request`` is the user message before context enrichment; it
        drives the two-pass length check and keeps the reasoning prompt lean.
        """
        tag = self.get_diagram_type()
        logger.info(f"[{tag}] generate_complete_system called with: {user_request!r}")
        try:
            parsed = self.predict_two_pass_structured(
                user_request=user_request,
                system_prompt=self.get_system_prompt(),
                reasoning_prompt=self._reasoning_prompt(raw_request or user_request),
                response_schema=self._SYSTEM_SCHEMA,
                raw_request=raw_request,
                model=MODEL_GENERATION_LARGE,
                reasoning_model=MODEL_REASONING,
            )
            system_spec = parsed.model_dump()
            return {
                "action": "inject_complete_system",
                "systemSpec": system_spec,
                "diagramType": tag,
                "message": self._build_system_message(system_spec),
            }
        except LLMPredictionError as exc:
            logger.error(f"[{tag}] generate_complete_system LLM FAILED: {exc}")
            return self._error_response(
                f"I couldn't generate that {self._LABEL}. Please try again or rephrase your request.",
                code="llm_failure",
            )
        except Exception as exc:
            logger.error(f"[{tag}] generate_complete_system FAILED: {exc}", exc_info=True)
            return self.generate_fallback_system()

    def generate_modification(
        self, user_request: str, current_model: Dict[str, Any] = None, **kwargs,
    ) -> Dict[str, Any]:
        tag = self.get_diagram_type()
        # Passed explicitly: handlers are singletons shared by every session.
        elements: Dict[str, Any] = {}
        context_block = ""
        if isinstance(current_model, dict):
            raw = current_model.get("elements")
            if isinstance(raw, dict):
                elements = raw
            summary = detailed_model_summary(current_model, tag)
            if summary:
                context_block = f"\n\n{summary}"

        user_prompt = f"Modify the {self._LABEL}: {user_request}{context_block}"
        logger.info(f"[{tag}] generate_modification called with: {user_request!r}")
        try:
            result = self._execute_modification(
                user_prompt, self._MODIFY_SYSTEM_PROMPT, self._MODIFICATION_SCHEMA, elements=elements,
            )
            return self._validate_mod_refs(result, elements)
        except LLMPredictionError as exc:
            logger.error(f"[{tag}] generate_modification LLM FAILED: {exc}")
            return self._error_response(
                "I couldn't process that modification. Please try again or rephrase your request.",
            )
        except Exception as exc:
            logger.error(f"[{tag}] generate_modification FAILED: {exc}", exc_info=True)
            return {
                "action": "assistant_message",
                "message": (
                    "I couldn't apply that modification automatically. Could you rephrase it? "
                    f"For example: {self._MODIFY_EXAMPLES}."
                ),
            }

    def generate_fallback_element(self, request: str) -> Dict[str, Any]:
        return self.generate_single_element(request)
