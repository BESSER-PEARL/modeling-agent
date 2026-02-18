"""
Base Diagram Handler
Provides common functionality for all diagram type handlers.

Positions are computed **after** the LLM returns semantic content by the
deterministic :pymod:`layout_engine` – the LLM is never asked to produce
pixel coordinates.
"""

import json
import logging
import os
import threading
import uuid
from typing import Dict, Any, List, Optional, Set
from abc import ABC, abstractmethod

from .layout_engine import apply_layout

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Concurrency guard – limits parallel LLM calls across all handlers.
# OpenAI rate-limits are per-org, so a global semaphore prevents bursts
# from concurrent WebSocket sessions exhausting the quota.
# ---------------------------------------------------------------------------
_LLM_CONCURRENCY_LIMIT = int(os.environ.get("LLM_CONCURRENCY_LIMIT", "4"))
_llm_semaphore = threading.Semaphore(_LLM_CONCURRENCY_LIMIT)


# ---------------------------------------------------------------------------
# Lightweight schema validation helpers
# ---------------------------------------------------------------------------

def _check_type(value: Any, expected: type, path: str) -> Optional[str]:
    """Return an error string if *value* is not an instance of *expected*."""
    if not isinstance(value, expected):
        return f"{path}: expected {expected.__name__}, got {type(value).__name__}"
    return None


def validate_spec(
    spec: Dict[str, Any],
    required_keys: Dict[str, type],
    optional_keys: Optional[Dict[str, type]] = None,
    label: str = "spec",
) -> List[str]:
    """Validate that *spec* contains *required_keys* with matching types.

    Returns a list of human-readable error strings (empty == valid).
    """
    errors: List[str] = []
    if not isinstance(spec, dict):
        return [f"{label}: expected a JSON object, got {type(spec).__name__}"]

    for key, expected_type in required_keys.items():
        if key not in spec:
            errors.append(f"{label}.{key}: missing required field")
        else:
            err = _check_type(spec[key], expected_type, f"{label}.{key}")
            if err:
                errors.append(err)

    if optional_keys:
        for key, expected_type in optional_keys.items():
            if key in spec:
                err = _check_type(spec[key], expected_type, f"{label}.{key}")
                if err:
                    errors.append(err)

    return errors


# Reusable required-key dicts for the most common specs -----------------

SINGLE_CLASS_REQUIRED = {"className": str}
SINGLE_CLASS_OPTIONAL = {"attributes": list, "methods": list}

SYSTEM_CLASS_REQUIRED = {"classes": list}
SYSTEM_CLASS_OPTIONAL = {"systemName": str, "relationships": list}

SINGLE_OBJECT_REQUIRED = {"objectName": str, "className": str}
SINGLE_OBJECT_OPTIONAL = {"classId": str, "attributes": list}

SYSTEM_OBJECT_REQUIRED = {"objects": list}
SYSTEM_OBJECT_OPTIONAL = {"systemName": str, "links": list}

SINGLE_STATE_REQUIRED = {"stateName": str}
SINGLE_STATE_OPTIONAL = {"stateType": str, "entryAction": str, "exitAction": str, "doActivity": str}

SYSTEM_STATE_REQUIRED = {"states": list}
SYSTEM_STATE_OPTIONAL = {"systemName": str, "transitions": list}

MODIFICATION_REQUIRED = {"modification": dict}
MODIFICATION_INNER_REQUIRED = {"action": str, "target": dict}


class BaseDiagramHandler(ABC):
    """Base class for all diagram type handlers"""

    def __init__(self, llm):
        """Initialize handler with LLM instance"""
        self.llm = llm

    @abstractmethod
    def get_diagram_type(self) -> str:
        """Return the diagram type this handler supports"""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this diagram type"""
        pass

    @abstractmethod
    def generate_single_element(self, user_request: str, existing_model: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Generate a single element for this diagram type."""
        pass

    @abstractmethod
    def generate_complete_system(self, user_request: str, existing_model: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a complete system/diagram with multiple elements."""
        pass

    @abstractmethod
    def generate_fallback_element(self, request: str) -> Dict[str, Any]:
        """Generate a fallback element when AI generation fails"""
        pass

    def generate_modification(self, user_request: str, current_model: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate modifications for existing diagram elements.
        Override this method in subclasses to provide diagram-specific modification logic.
        Default implementation returns a basic modification structure.
        """
        return {
            "action": "modify_model",
            "modification": {
                "action": "modify_element",
                "target": {"elementName": "unknown"},
                "changes": {"name": "modified"}
            },
            "diagramType": self.get_diagram_type(),
            "message": "Modification not implemented for this diagram type."
        }

    # ------------------------------------------------------------------
    # Layout helpers – deterministic positioning after LLM generation
    # ------------------------------------------------------------------

    def apply_single_layout(
        self, spec: Dict[str, Any], existing_model: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Apply the deterministic layout engine to a single-element spec."""
        return apply_layout(spec, self.get_diagram_type(), mode="single",
                            existing_model=existing_model)

    def apply_system_layout(
        self, system_spec: Dict[str, Any], existing_model: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Apply the deterministic layout engine to a complete-system spec."""
        return apply_layout(system_spec, self.get_diagram_type(), mode="system",
                            existing_model=existing_model)

    # ------------------------------------------------------------------
    # LLM call with retry
    # ------------------------------------------------------------------

    def predict_with_retry(self, prompt: str, max_retries: int = 1) -> str:
        """Call the LLM with automatic retry on transient failures.

        Acquires a module-level semaphore before calling the API so that
        concurrent WebSocket sessions don't exhaust OpenAI rate limits.

        Args:
            prompt: Full prompt to send.
            max_retries: Number of additional attempts after the first (default 1).

        Returns:
            Non-empty string response.

        Raises:
            ValueError: If all attempts return empty or fail.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1 + max_retries):
            try:
                _llm_semaphore.acquire()
                try:
                    response = self.llm.predict(prompt)
                finally:
                    _llm_semaphore.release()
                if response and response.strip():
                    return response
                last_error = ValueError("LLM returned empty response")
                logger.warning(
                    f"[{self.get_diagram_type()}] Empty LLM response "
                    f"(attempt {attempt + 1}/{1 + max_retries})"
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"[{self.get_diagram_type()}] LLM call failed "
                    f"(attempt {attempt + 1}/{1 + max_retries}): {exc}"
                )
        raise last_error or ValueError("LLM prediction failed after all retries")

    # ------------------------------------------------------------------
    # JSON / text utilities
    # ------------------------------------------------------------------

    def clean_json_response(self, response: str) -> str:
        """Clean JSON response from LLM — strip markdown fences, leading prose, etc."""
        text = response.strip()
        # Remove markdown code fences
        if text.startswith('```json'):
            text = text[7:]
        elif text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()
        # Skip any leading prose to find the JSON object/array
        for i, ch in enumerate(text):
            if ch in ('{', '['):
                text = text[i:]
                break
        return text.strip()

    def generate_uuid(self) -> str:
        """Generate a unique UUID"""
        return str(uuid.uuid4())

    def parse_json_safely(self, json_text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON with error handling"""
        try:
            result = json.loads(json_text)
            logger.debug(f"[BaseHandler] JSON parsed successfully, keys: {list(result.keys()) if isinstance(result, dict) else type(result).__name__}")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"[BaseHandler] JSON parse failed: {e}. Text (first 300 chars): {json_text[:300]!r}")
            return None

    def parse_and_validate(
        self,
        raw_response: str,
        required_keys: Dict[str, type],
        optional_keys: Optional[Dict[str, type]] = None,
        label: str = "LLM response",
    ) -> Dict[str, Any]:
        """Clean, parse, and validate an LLM response in one call.

        Returns the parsed dict on success.
        Raises ``ValueError`` with a descriptive message on failure.
        """
        json_text = self.clean_json_response(raw_response)
        spec = self.parse_json_safely(json_text)
        if spec is None:
            raise ValueError(f"Could not parse JSON from LLM response: {json_text[:200]}")

        errors = validate_spec(spec, required_keys, optional_keys, label=label)
        if errors:
            joined = "; ".join(errors)
            logger.warning(f"[{self.get_diagram_type()}] Schema validation failed: {joined}")
            raise ValueError(f"Schema validation failed: {joined}")

        return spec

    def validate_modification_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a modification response from the LLM.

        Supports both single ``modification`` (dict) and batch
        ``modifications`` (list of dicts).  Raises ``ValueError`` if the
        shape is invalid.
        """
        # Batch path: "modifications" is a list of inner objects
        if 'modifications' in spec and isinstance(spec['modifications'], list):
            for i, inner in enumerate(spec['modifications']):
                inner_errors = validate_spec(
                    inner, MODIFICATION_INNER_REQUIRED,
                    label=f"modifications[{i}]",
                )
                if inner_errors:
                    raise ValueError("; ".join(inner_errors))
            return spec

        # Single path: "modification" is a dict
        errors = validate_spec(spec, MODIFICATION_REQUIRED, label="modification")
        if errors:
            raise ValueError("; ".join(errors))

        inner = spec["modification"]
        inner_errors = validate_spec(inner, MODIFICATION_INNER_REQUIRED, label="modification.inner")
        if inner_errors:
            raise ValueError("; ".join(inner_errors))

        return spec

    def extract_name_from_request(self, request: str, default: str = "New") -> str:
        """Extract a name from user request"""
        words = request.split()
        for i, word in enumerate(words):
            if word.lower() in ['create', 'add', 'make', 'new', 'generate']:
                if i + 1 < len(words):
                    next_word = words[i + 1]
                    if next_word.lower() not in ['a', 'an', 'the', 'class', 'object', 'state', 'agent']:
                        if i + 2 < len(words):
                            return words[i + 2].capitalize()
                        return next_word.capitalize()
        return default
