"""Shared flow for the architecture diagram handlers (Component, Deployment).

Both diagrams have the same shape — container elements, contained elements,
and dependency edges addressed by ``elementId`` / ``elementName`` — so the
complete-system, modification and fallback flow lives here once. Concrete
handlers supply the prompts, schemas, starter specs and reply messages, and
declare the base handler's reference-guardrail tables.
"""

import logging
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple, Type

from pydantic import BaseModel

from llm.model_config import MODEL_GENERATION_LARGE, MODEL_REASONING
from utilities.model_context import detailed_model_summary

from .base_handler import BaseDiagramHandler
from protocol.actions import assistant_message, inject_complete_system

logger = logging.getLogger(__name__)


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
    _LLM_FAILURE_CODE = "llm_failure"

    @abstractmethod
    def _reasoning_prompt(self, request: str) -> str:
        """Free-text planning prompt for the two-pass reasoning pass."""

    @abstractmethod
    def _build_system_message(self, spec: Dict[str, Any]) -> str:
        """Reply text for a generated complete system."""

    @abstractmethod
    def generate_fallback_system(self) -> Dict[str, Any]:
        """Starter diagram returned when generation fails unexpectedly."""

    #: Name of the single-element starter when the request names none, e.g. "Component".
    _STARTER_DEFAULT_NAME: str = ""

    @abstractmethod
    def _starter_element(self, name: str) -> Dict[str, Any]:
        """Starter-diagram reply to a single-element request for ``name``."""

    def _build_system(
        self,
        user_request: str,
        existing_model: Dict[str, Any] = None,
        raw_request: Optional[str] = None,
        *,
        prepared=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a complete diagram with two-pass structured output.

        ``raw_request`` is the user message before context enrichment; it
        drives the two-pass length check and keeps the reasoning prompt lean.
        """
        tag = self.get_diagram_type()
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
        return inject_complete_system(system_spec, tag, self._build_system_message(system_spec))

    def _prepare_modification(
        self, user_request: str, current_model: Dict[str, Any] = None, **kwargs,
    ) -> Tuple[Dict[str, Any], str]:
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

        return elements, f"Modify the {self._LABEL}: {user_request}{context_block}"

    def _build_modification(
        self, user_request: str, current_model: Dict[str, Any] = None, *, prepared=None, **kwargs,
    ) -> Dict[str, Any]:
        elements, user_prompt = prepared
        result = self._execute_modification(
            user_prompt, self._MODIFY_SYSTEM_PROMPT, self._MODIFICATION_SCHEMA, elements=elements,
        )
        return self._validate_mod_refs(result, elements)

    def generate_fallback_modification(self, request: str) -> Dict[str, Any]:
        return assistant_message(
            "I couldn't apply that modification automatically. Could you rephrase it? "
            f"For example: {self._MODIFY_EXAMPLES}.",
        )

    # No LLM call: a single-element request gets a starter diagram.
    def _prepare_single_element(
        self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs,
    ) -> str:
        return self.extract_name_from_request(user_request, self._STARTER_DEFAULT_NAME)

    def _build_single_element(
        self, user_request: str, existing_model: Dict[str, Any] = None, *, prepared=None, **kwargs,
    ) -> Dict[str, Any]:
        return self._starter_element(prepared)

    def generate_fallback_element(self, request: str) -> Dict[str, Any]:
        return self._starter_element(self.extract_name_from_request(request, self._STARTER_DEFAULT_NAME))
