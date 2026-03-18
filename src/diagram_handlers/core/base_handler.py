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
import random
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Type
from abc import ABC, abstractmethod

from pydantic import BaseModel

from .layout_engine import apply_layout

logger = logging.getLogger(__name__)

class LLMPredictionError(Exception):
    """Raised when the LLM fails to produce a usable response after retries."""
    pass


# ---------------------------------------------------------------------------
# Prompt-response cache stubs
# ---------------------------------------------------------------------------
# Prompt caching was removed because every prompt includes the current
# workspace context (model data) which changes on every interaction,
# yielding a ~0% cache hit rate.  These stubs remain so that callers
# (predict_with_retry, predict_structured) and tests don't need changes.

_PROMPT_CACHE_MAX_SIZE = 0
_PROMPT_CACHE_TTL = 3600
_prompt_cache: Dict[str, Any] = {}


def _normalize_cache_key(key: str) -> str:
    """Normalize whitespace (kept for backward compat / tests)."""
    return ' '.join(key.split())

def _cache_key(_prompt: str) -> str:
    """Return a constant — caching is disabled."""
    return ""

def _cache_get(_prompt: str) -> Optional[str]:
    return None

def _cache_put(_prompt: str, _response: str) -> None:
    pass


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

# ---------------------------------------------------------------------------
# Error taxonomy with user-facing recovery hints
# ---------------------------------------------------------------------------
_ERROR_RECOVERY: Dict[str, Dict[str, Any]] = {
    "llm_failure": {
        "message": "The AI service is temporarily unavailable.",
        "recovery": "try again",
        "retryable": True,
    },
    "parse_error": {
        "message": "I had trouble structuring that response.",
        "recovery": "try rephrasing your request more specifically",
        "retryable": True,
    },
    "validation_error": {
        "message": "The generated model had structural issues.",
        "recovery": "try simplifying your request",
        "retryable": True,
    },
    "timeout": {
        "message": "That request was too complex to process in time.",
        "recovery": "try breaking it into smaller steps",
        "retryable": True,
    },
    "schema_error": {
        "message": "The response format was unexpected.",
        "recovery": "try again with a simpler description",
        "retryable": True,
    },
    "context_error": {
        "message": "I couldn't find the diagram or element you're referring to.",
        "recovery": "make sure you have an active diagram open",
        "retryable": False,
    },
    "generation_error": {
        "message": "Something went wrong during generation.",
        "recovery": "try rephrasing your request",
        "retryable": True,
    },
    "unsupported": {
        "message": "This operation is not supported for this diagram type.",
        "recovery": "check the documentation for supported operations",
        "retryable": False,
    },
}


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

    def _error_response(
        self,
        error_code_or_message: str = "generation_error",
        details: str = "",
        *,
        message: Optional[str] = None,
        code: Optional[str] = None,
        retryable: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Return a structured error payload with recovery hints.

        Uses the module-level ``_ERROR_RECOVERY`` taxonomy to produce
        user-friendly messages and actionable recovery suggestions.

        **Backward compatible**: existing callers that pass a human-readable
        message as the first positional argument (with an optional ``code=``
        keyword) continue to work.  The method detects whether the first arg
        is a known error code or a legacy message string.

        Supported error codes (see ``_ERROR_RECOVERY``):
        - ``llm_failure``: LLM call failed after retries
        - ``parse_error``: JSON parsing failed
        - ``validation_error``: Schema validation failed
        - ``generation_error``: General generation failure (default)
        - ``timeout``: Request timed out
        - ``schema_error``: Unexpected response format
        - ``context_error``: Missing diagram/element context
        - ``unsupported``: Unsupported diagram type or operation

        Args:
            error_code_or_message: Either a key from ``_ERROR_RECOVERY`` **or**
                a legacy human-readable message string.  When a known error
                code is given the taxonomy message is used; otherwise the
                value is treated as the display message.
            details: Extra context appended to the taxonomy message.
            message: Explicit message override (takes priority over both the
                taxonomy message and the legacy first-arg message).
            code: Explicit error code override (takes priority over the
                first-arg code detection).
            retryable: Override for the retryable flag.
        """
        # Detect whether first arg is a known error code or a legacy message
        first_arg = error_code_or_message
        is_known_code = first_arg in _ERROR_RECOVERY

        if code is not None:
            # Explicit code= keyword — first arg is a legacy message
            effective_code = code
            legacy_message = first_arg if not is_known_code else None
        elif is_known_code:
            effective_code = first_arg
            legacy_message = None
        else:
            # First arg looks like a legacy message string (contains spaces, etc.)
            effective_code = "generation_error"
            legacy_message = first_arg

        recovery = _ERROR_RECOVERY.get(effective_code, {
            "message": "Something went wrong.",
            "recovery": "try rephrasing your request",
            "retryable": True,
        })
        effective_retryable = retryable if retryable is not None else recovery["retryable"]

        # Priority: explicit message= > legacy positional message > taxonomy + details
        if message is not None:
            effective_message = message
        elif legacy_message is not None:
            effective_message = legacy_message
        else:
            effective_message = f"{recovery['message']} {details}".strip()

        return {
            "action": "assistant_message",
            "error": True,
            "errorCode": effective_code,
            "message": effective_message,
            "suggestedRecovery": recovery["recovery"],
            "retryable": effective_retryable,
            # Keep legacy key for backward compat
            "error_code": effective_code,
            "diagramType": self.get_diagram_type(),
        }

    # ------------------------------------------------------------------
    # Friendly modification message helpers
    # ------------------------------------------------------------------

    _ACTION_LABELS = {
        'modify_class': 'Updated',
        'add_attribute': 'Added attribute to',
        'modify_attribute': 'Updated attribute in',
        'remove_attribute': 'Removed attribute from',
        'add_method': 'Added method to',
        'modify_method': 'Updated method in',
        'remove_method': 'Removed method from',
        'add_relationship': 'Added relationship to',
        'modify_relationship': 'Updated relationship in',
        'remove_relationship': 'Removed relationship from',
        'remove_element': 'Removed',
        'add_class': 'Added',
        'rename': 'Renamed',
        'modify_state': 'Updated',
        'add_state': 'Added',
        'remove_state': 'Removed',
        'add_transition': 'Added transition to',
        'modify_transition': 'Updated transition in',
        'remove_transition': 'Removed transition from',
        'modify_intent': 'Updated',
        'add_intent': 'Added',
        'modify_object': 'Updated',
        'add_object': 'Added',
        'add_link': 'Added link to',
    }

    @classmethod
    def _friendly_mod_message(cls, action: str, target_name: str) -> str:
        """Turn a raw action + target into a user-friendly message."""
        label = cls._ACTION_LABELS.get(action)
        if label:
            return f"{label} **{target_name}**."
        # Fallback: humanize the action string
        human = action.replace('_', ' ').capitalize()
        return f"{human} **{target_name}**."

    @classmethod
    def _friendly_batch_message(cls, mods: list) -> str:
        """Produce a friendly summary for a batch of modifications."""
        parts = []
        for m in mods:
            act = m.get('action', 'modification')
            t = m.get('target', {})
            name = (t.get('className') or t.get('stateName') or t.get('intentName')
                    or t.get('objectName') or t.get('attributeName') or t.get('methodName')
                    or 'element')
            parts.append(cls._friendly_mod_message(act, name))
        if len(parts) == 1:
            return parts[0]
        return f"Applied {len(parts)} changes:\n" + "\n".join(f"- {p}" for p in parts)

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
    # NOTE: This adds an extra LLM round-trip (2–4s latency).
    def predict_with_retry(self, prompt: str, max_retries: int = 1, *, use_cache: bool = True) -> str:
        """Call the LLM with automatic retry, cache check, and jittered exponential backoff.

        Rate-limit handling is delegated to OpenAI's API (429 responses)
        and the retry loop below — no local semaphore is used.

        Retry improvements:
        - Jitter added to backoff delays to prevent thundering-herd retries.
        - On parse-error retries, a simplified "JSON-only" prompt is tried as
          a self-healing mechanism.

        Args:
            prompt: Full prompt to send.
            max_retries: Number of additional attempts after the first (default 2).
            use_cache: Check/populate the prompt cache (default True).

        Returns:
            Non-empty string response.

        Raises:
            LLMPredictionError: If all attempts return empty or fail.
        """
        # --- Cache check ---
        if use_cache:
            cached = _cache_get(prompt)
            if cached:
                logger.info(f"[{self.get_diagram_type()}] Cache hit — returning cached response")
                return cached

        last_error: Optional[Exception] = None
        last_error_type: str = "unknown"
        total_attempts = 1 + max_retries
        for attempt in range(total_attempts):
            # Jittered exponential backoff before retries (0s, ~2-3s, ~4-5s, …)
            if attempt > 0:
                base_delay = 2 ** attempt
                jitter = random.uniform(0, 1)
                backoff = base_delay + jitter
                logger.info(
                    f"[{self.get_diagram_type()}] Retrying in {backoff:.1f}s "
                    f"(attempt {attempt + 1}/{total_attempts}, "
                    f"last_error_type={last_error_type})"
                )
                time.sleep(backoff)

            # On parse_error retries, try a simplified prompt that enforces JSON
            effective_prompt = prompt
            if attempt > 0 and last_error_type == "parse_error":
                effective_prompt = (
                    prompt + "\n\n"
                    "IMPORTANT: Return ONLY valid JSON, no markdown, no explanation."
                )
                logger.info(
                    f"[{self.get_diagram_type()}] Self-healing: using simplified "
                    f"JSON-only prompt on attempt {attempt + 1}"
                )

            try:
                logger.info(
                    f"[{self.get_diagram_type()}] LLM call started "
                    f"(attempt {attempt + 1}/{total_attempts}, "
                    f"prompt_len={len(effective_prompt)})"
                )
                response = self.llm.predict(effective_prompt)

                # Track tokens from the last API call if available
                try:
                    from tracking import get_tracker
                    client = getattr(self.llm, 'client', None)
                    if client is not None:
                        # OpenAI client stores last response in thread-local
                        # We estimate tokens from response length as a fallback
                        tracker = get_tracker()
                        est_prompt = len(effective_prompt) // 4
                        est_completion = len(response) // 4 if response else 0
                        tracker.record(
                            prompt_tokens=est_prompt,
                            completion_tokens=est_completion,
                            model=getattr(self.llm, 'name', 'gpt-4.1-mini'),
                        )
                except Exception:
                    pass  # tracking is best-effort

                if response and response.strip():
                    if use_cache:
                        _cache_put(prompt, response)
                    return response
                last_error = LLMPredictionError("LLM returned empty response")
                last_error_type = "llm_failure"
                logger.warning(
                    f"[{self.get_diagram_type()}] Empty LLM response "
                    f"(attempt {attempt + 1}/{total_attempts})"
                )
            except LLMPredictionError:
                raise
            except json.JSONDecodeError as exc:
                last_error = LLMPredictionError(f"JSON parse error: {exc}")
                last_error_type = "parse_error"
                logger.warning(
                    f"[{self.get_diagram_type()}] JSON parse error "
                    f"(attempt {attempt + 1}/{total_attempts}): {exc}"
                )
            except Exception as exc:
                exc_name = type(exc).__name__
                exc_str = str(exc).lower()
                # Rate-limit: don't retry, tell the user to wait
                if "RateLimitError" in exc_name or "429" in exc_str or "rate limit" in exc_str:
                    raise LLMPredictionError(
                        f"API rate limit reached. Please wait a moment and try again. ({exc})"
                    )
                last_error = LLMPredictionError(str(exc))
                last_error_type = "llm_failure"
                logger.warning(
                    f"[{self.get_diagram_type()}] LLM call failed "
                    f"(attempt {attempt + 1}/{total_attempts}): {exc}"
                )
        raise last_error or LLMPredictionError("LLM prediction failed after all retries")

    # ------------------------------------------------------------------
    # Structured output via OpenAI .parse() — eliminates JSON repair
    # ------------------------------------------------------------------

    # Schema names that produce small outputs (single element or modification).
    # These get a lower max_completion_tokens to speed up the API call.
    _SMALL_OUTPUT_SCHEMAS = {
        "SingleClassSpec", "SingleObjectSpec", "SingleStateSpec",
        "SingleGUIElementSpec", "SingleQuantumGateSpec", "AgentSingleElementSpec",
        "ClassModificationResponse", "ObjectModificationResponse",
        "StateMachineModificationResponse", "GUIModificationSpec",
        "QuantumModificationSpec", "AgentModificationResponse",
    }
    _SMALL_OUTPUT_MAX_TOKENS = 2048
    _LARGE_OUTPUT_MAX_TOKENS = 8192

    def predict_structured(
        self,
        prompt: str,
        response_schema: Type[BaseModel],
        *,
        max_retries: int = 1,
        use_cache: bool = True,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> BaseModel:
        """Call the LLM with OpenAI Structured Outputs, returning a validated Pydantic model.

        Uses ``client.beta.chat.completions.parse()`` which guarantees the
        response conforms to the Pydantic schema — no manual JSON cleaning,
        repair loops, or regex extraction needed.

        Falls back to ``predict_with_retry`` + manual parsing if the OpenAI
        client doesn't support ``.parse()`` (older SDK versions).

        Args:
            prompt: The user/request prompt.
            response_schema: Pydantic model class defining the expected output.
            max_retries: Number of retry attempts (default 2).
            use_cache: Check/populate prompt cache (default True).
            system_prompt: Optional system instruction prepended to messages.
            temperature: LLM temperature (default 0.2).

        Returns:
            Validated Pydantic model instance.

        Raises:
            LLMPredictionError: If all attempts fail.
        """
        # --- Cache check (keyed on prompt + schema name) ---
        cache_prompt = f"{prompt}||schema={response_schema.__name__}"
        if use_cache:
            cached = _cache_get(cache_prompt)
            if cached:
                try:
                    parsed = response_schema.model_validate_json(cached)
                    logger.info(
                        f"[{self.get_diagram_type()}] Structured cache hit "
                        f"(schema={response_schema.__name__})"
                    )
                    return parsed
                except Exception:
                    pass  # Cache entry invalid for this schema, re-generate

        # --- Check if client supports .parse() ---
        client = getattr(self.llm, 'client', None)
        has_parse = (
            client is not None
            and hasattr(client, 'beta')
            and hasattr(client.beta, 'chat')
        )

        if not has_parse:
            # Fallback: predict_with_retry + manual parse
            logger.info(
                f"[{self.get_diagram_type()}] Structured outputs unavailable, "
                "falling back to predict_with_retry"
            )
            return self._structured_fallback(
                prompt, response_schema, system_prompt, max_retries, use_cache,
            )

        # --- Structured parse with retry ---
        from tracking import get_tracker
        tracker = get_tracker()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error: Optional[Exception] = None
        total_attempts = 1 + max_retries

        for attempt in range(total_attempts):
            if attempt > 0:
                backoff = 2 ** attempt + random.uniform(0, 1)
                logger.info(
                    f"[{self.get_diagram_type()}] Structured retry in {backoff:.1f}s "
                    f"(attempt {attempt + 1}/{total_attempts})"
                )
                time.sleep(backoff)

            try:
                max_tokens = (
                    self._SMALL_OUTPUT_MAX_TOKENS
                    if response_schema.__name__ in self._SMALL_OUTPUT_SCHEMAS
                    else self._LARGE_OUTPUT_MAX_TOKENS
                )
                logger.info(
                    f"[{self.get_diagram_type()}] Structured LLM call started "
                    f"(attempt {attempt + 1}/{total_attempts}, "
                    f"schema={response_schema.__name__}, "
                    f"max_tokens={max_tokens})"
                )
                completion = client.beta.chat.completions.parse(
                    model=self.llm.name if hasattr(self.llm, 'name') else "gpt-4.1-mini",
                    messages=messages,
                    response_format=response_schema,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                )

                # Track tokens
                if hasattr(completion, 'usage') and completion.usage:
                    tracker.record_from_usage(
                        completion.usage,
                        model=self.llm.name if hasattr(self.llm, 'name') else "gpt-4.1-mini",
                    )

                parsed = completion.choices[0].message.parsed
                if parsed is None:
                    # Refusal or empty
                    refusal = getattr(completion.choices[0].message, 'refusal', None)
                    if refusal:
                        raise LLMPredictionError(f"LLM refused: {refusal}")
                    raise LLMPredictionError("LLM returned empty structured output")

                # Cache the JSON representation
                if use_cache:
                    _cache_put(cache_prompt, parsed.model_dump_json())

                logger.info(
                    f"[{self.get_diagram_type()}] Structured output success "
                    f"(schema={response_schema.__name__}, attempt={attempt + 1})"
                )
                return parsed

            except LLMPredictionError:
                raise
            except Exception as exc:
                # Non-retryable errors: bail immediately instead of wasting retries
                exc_name = type(exc).__name__
                if "BadRequestError" in exc_name or "AuthenticationError" in exc_name or "RateLimitError" in exc_name or "429" in str(exc):
                    raise LLMPredictionError(f"Structured parse failed (non-retryable): {exc}")
                last_error = LLMPredictionError(f"Structured parse failed: {exc}")
                logger.warning(
                    f"[{self.get_diagram_type()}] Structured parse attempt "
                    f"{attempt + 1}/{total_attempts} failed: {exc}"
                )

        raise last_error or LLMPredictionError("Structured prediction failed after all retries")

    def _structured_fallback(
        self,
        prompt: str,
        response_schema: Type[BaseModel],
        system_prompt: str,
        max_retries: int,
        use_cache: bool,
    ) -> BaseModel:
        """Fallback path when structured outputs API is unavailable.

        Uses predict_with_retry + JSON mode, then validates against the schema.
        Cache key is schema-qualified to match ``predict_structured``.
        """
        # Use the same schema-qualified cache key as predict_structured
        # so hits/puts are consistent across both code paths.
        cache_prompt = f"{prompt}||schema={response_schema.__name__}"
        if use_cache:
            cached = _cache_get(cache_prompt)
            if cached:
                try:
                    return response_schema.model_validate_json(cached)
                except Exception:
                    pass  # stale entry — regenerate

        # Build the schema description for the prompt so the LLM knows the shape
        schema_desc = ""
        try:
            schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
            schema_desc = f"\n\nExpected JSON schema:\n{schema_json}\n"
        except Exception:
            pass

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        full_prompt += schema_desc
        full_prompt += (
            "\n\nIMPORTANT: Return ONLY valid JSON matching this schema. "
            "No markdown, no explanation."
        )

        # predict_with_retry uses its own cache (sha256 of full_prompt) — disable
        # it to avoid double-caching with mismatched keys.
        response = self.predict_with_retry(full_prompt, max_retries=max_retries, use_cache=False)
        json_text = self.clean_json_response(response)

        try:
            parsed = response_schema.model_validate_json(json_text)
        except Exception as exc:
            # One more try: parse as dict and validate
            try:
                data = json.loads(json_text)
                parsed = response_schema.model_validate(data)
            except Exception:
                raise LLMPredictionError(
                    f"Failed to validate LLM response against "
                    f"{response_schema.__name__}: {exc}"
                )

        # Cache the validated result with the schema-qualified key
        if use_cache:
            _cache_put(cache_prompt, parsed.model_dump_json())

        return parsed

    # Threshold: requests shorter than this are "simple" and skip the
    # reasoning pass, saving one full LLM round-trip (~1-3 s).
    _TWO_PASS_MIN_LENGTH = 250

    def predict_two_pass_structured(
        self,
        user_request: str,
        system_prompt: str,
        reasoning_prompt: str,
        response_schema: Type[BaseModel],
        *,
        temperature: float = 0.2,
    ) -> BaseModel:
        """Two-pass generation with structured output for pass 2.

        Pass 1 (reasoning): Free-text design analysis via ``predict_with_retry``
        (handles retry and caching internally).
        Pass 2 (structured): Converts reasoning into a validated Pydantic model
        via ``predict_structured``, guaranteeing schema conformance.

        For short / simple requests (< _TWO_PASS_MIN_LENGTH chars) the
        reasoning pass is skipped and a single structured call is made
        directly — the LLM can handle simple systems without a chain-of-
        thought preamble.

        Falls back to single-pass structured if reasoning fails.
        """
        # Fast path: simple requests don't need a reasoning pass
        if len(user_request) < self._TWO_PASS_MIN_LENGTH:
            logger.info(
                f"[{self.get_diagram_type()}] Simple request ({len(user_request)} chars) "
                "— skipping reasoning pass, using single-pass structured"
            )
            return self.predict_structured(
                f"User Request: {user_request}",
                response_schema,
                system_prompt=system_prompt,
                temperature=temperature,
            )

        # Pass 1: Design reasoning (free-text)
        logger.info(f"[{self.get_diagram_type()}] Two-pass structured: reasoning pass")
        try:
            reasoning = self.predict_with_retry(reasoning_prompt, max_retries=1, use_cache=False)
        except Exception as exc:
            logger.warning(
                f"[{self.get_diagram_type()}] Reasoning pass failed ({exc}), "
                "falling back to single-pass structured"
            )
            return self.predict_structured(
                f"User Request: {user_request}",
                response_schema,
                system_prompt=system_prompt,
                temperature=temperature,
            )

        if not reasoning or not reasoning.strip():
            logger.warning(f"[{self.get_diagram_type()}] Reasoning pass empty, falling back")
            return self.predict_structured(
                f"User Request: {user_request}",
                response_schema,
                system_prompt=system_prompt,
                temperature=temperature,
            )

        logger.info(
            f"[{self.get_diagram_type()}] Two-pass structured: reasoning complete "
            f"({len(reasoning)} chars), starting structured pass"
        )

        # Pass 2: Structured output using the reasoning as context
        structured_prompt = (
            f"Design analysis (use this as your guide):\n{reasoning}\n\n"
            f"User Request: {user_request}\n\n"
            "Convert the design analysis above into the exact JSON format specified."
        )

        return self.predict_structured(
            structured_prompt,
            response_schema,
            system_prompt=system_prompt,
            temperature=temperature,
        )

    def _classify_error(self, exc: Exception) -> str:
        """Classify an exception into an error code from the recovery taxonomy."""
        exc_str = str(exc).lower()
        if "timeout" in exc_str or "timed out" in exc_str:
            return "timeout"
        if "json" in exc_str or "parse" in exc_str:
            return "parse_error"
        if "validation" in exc_str or "schema" in exc_str:
            return "validation_error"
        if isinstance(exc, LLMPredictionError):
            return "llm_failure"
        return "generation_error"

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
                return text[i:].strip()
        # No JSON opener found — return text as-is and let the caller handle it
        return text

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

    # ------------------------------------------------------------------
    # Two-pass generation (design reasoning → structured JSON)
    # ------------------------------------------------------------------

    def predict_two_pass(
        self,
        user_request: str,
        system_prompt: str,
        reasoning_prompt: str,
    ) -> str:
        """Two-pass LLM generation: first reason about the design, then generate JSON.

        Pass 1 (reasoning): The LLM thinks step-by-step about what the domain
        needs — which entities, attributes, and relationships make sense.
        This pass uses free-text output (no JSON mode).

        Pass 2 (structured): The LLM converts the reasoning into the target
        JSON schema.  The reasoning from pass 1 is included as context.

        Returns the raw JSON string from pass 2.
        """
        # Pass 1: Design reasoning (free-text) — delegates to predict_with_retry
        # which handles retry and backoff internally.
        logger.info(f"[{self.get_diagram_type()}] Two-pass: starting reasoning pass")
        try:
            reasoning = self.predict_with_retry(reasoning_prompt, max_retries=1, use_cache=False)
        except Exception as exc:
            logger.warning(
                f"[{self.get_diagram_type()}] Reasoning pass failed ({exc}), "
                "falling back to single-pass"
            )
            return self.predict_with_retry(f"{system_prompt}\n\nUser Request: {user_request}")

        if not reasoning or not reasoning.strip():
            logger.warning(f"[{self.get_diagram_type()}] Reasoning pass returned empty, falling back")
            return self.predict_with_retry(f"{system_prompt}\n\nUser Request: {user_request}")

        logger.info(
            f"[{self.get_diagram_type()}] Two-pass: reasoning complete "
            f"({len(reasoning)} chars), starting JSON pass"
        )

        # Pass 2: Structured JSON generation using the reasoning as context
        structured_prompt = (
            f"{system_prompt}\n\n"
            f"Design analysis (use this as your guide):\n{reasoning}\n\n"
            f"User Request: {user_request}\n\n"
            "Now convert the design analysis above into the exact JSON format specified. "
            "Return ONLY the JSON, no explanations."
        )

        return self.predict_with_retry(structured_prompt)

    # ------------------------------------------------------------------
    # Validation-feedback loop (critique → fix)
    # ------------------------------------------------------------------
    def validate_and_refine(
        self,
        spec: Dict[str, Any],
        user_request: str,
        diagram_type: str,
    ) -> Dict[str, Any]:
        """Validate a generated spec and refine it if quality issues are found.

        Sends the spec to the LLM for self-critique, asking it to check for:
        - Missing obvious attributes
        - Missing relationships between clearly related classes
        - Wrong multiplicities
        - Redundant or misplaced attributes
        - Naming convention issues

        Returns the refined spec (or the original if no issues found or
        refinement fails).
        """
        if diagram_type not in ("ClassDiagram",):
            # StateMachine validation is handled in StateMachineHandler._validate_and_refine_state_machine
            return spec

        classes = spec.get("classes", [])
        relationships = spec.get("relationships", [])

        # Skip validation for very small diagrams (1-3 classes) where the
        # extra LLM round-trip rarely finds issues.  4+ classes benefit from
        # relationship and attribute completeness checks.
        if len(classes) <= 20:
            return spec

        # Build a compact representation for the critique prompt
        class_summary = []
        for cls in classes:
            attrs = [a.get("name", "?") for a in cls.get("attributes", [])]
            class_summary.append(f"{cls.get('className', '?')}: {', '.join(attrs)}")
        rel_summary = []
        for rel in relationships:
            rel_summary.append(
                f"{rel.get('source', '?')} -> {rel.get('target', '?')} "
                f"({rel.get('type', '?')}, {rel.get('sourceMultiplicity', '?')}..{rel.get('targetMultiplicity', '?')})"
            )

        critique_prompt = f"""Review this UML class diagram for quality issues.

Classes:
{chr(10).join('- ' + s for s in class_summary)}

Relationships:
{chr(10).join('- ' + r for r in rel_summary) if rel_summary else '(none)'}

User's original request: "{user_request}"

Check for these issues:
1. MISSING RELATIONSHIPS: Are there classes that should clearly be connected but aren't? (e.g., Order and Customer without a relationship)
2. WRONG MULTIPLICITIES: Are any multiplicities incorrect? (e.g., Order-Product should typically be many-to-many via OrderItem, not one-to-one)
3. MISSING KEY ATTRIBUTES: Are any classes missing obviously essential attributes? (e.g., User without email, Order without date)
4. REDUNDANT ATTRIBUTES: Are any attributes duplicated across classes unnecessarily?

If you find issues, return a JSON object with fixes:
{{
  "has_issues": true,
  "add_relationships": [
    {{"type": "Association", "source": "Class1", "target": "Class2", "sourceMultiplicity": "1", "targetMultiplicity": "*", "name": "relName"}}
  ],
  "fix_multiplicities": [
    {{"source": "Class1", "target": "Class2", "sourceMultiplicity": "1", "targetMultiplicity": "*"}}
  ],
  "add_attributes": [
    {{"className": "ClassName", "attributes": [{{"name": "attr", "type": "String", "visibility": "public"}}]}}
  ]
}}

If the diagram looks good, return: {{"has_issues": false}}

Return ONLY the JSON, no explanations."""

        try:
            response = self.llm.predict(critique_prompt)

            if not response or not response.strip():
                return spec

            json_text = self.clean_json_response(response)
            critique = self.parse_json_safely(json_text)
            if not critique or not critique.get("has_issues"):
                logger.info(f"[{self.get_diagram_type()}] Validation: no issues found")
                return spec

            logger.info(f"[{self.get_diagram_type()}] Validation: applying refinements")

            # Apply relationship additions
            for new_rel in critique.get("add_relationships", []):
                if isinstance(new_rel, dict) and new_rel.get("source") and new_rel.get("target"):
                    # Check it doesn't already exist
                    exists = any(
                        r.get("source") == new_rel["source"] and r.get("target") == new_rel["target"]
                        for r in relationships
                    )
                    if not exists:
                        relationships.append(new_rel)
                        logger.info(
                            f"[{self.get_diagram_type()}] Added missing relationship: "
                            f"{new_rel['source']} -> {new_rel['target']}"
                        )

            # Apply multiplicity fixes
            for fix in critique.get("fix_multiplicities", []):
                if not isinstance(fix, dict):
                    continue
                for rel in relationships:
                    if rel.get("source") == fix.get("source") and rel.get("target") == fix.get("target"):
                        if "sourceMultiplicity" in fix:
                            rel["sourceMultiplicity"] = fix["sourceMultiplicity"]
                        if "targetMultiplicity" in fix:
                            rel["targetMultiplicity"] = fix["targetMultiplicity"]
                        logger.info(
                            f"[{self.get_diagram_type()}] Fixed multiplicity: "
                            f"{fix.get('source')} -> {fix.get('target')}"
                        )

            # Apply missing attributes
            for attr_fix in critique.get("add_attributes", []):
                if not isinstance(attr_fix, dict):
                    continue
                target_class = attr_fix.get("className")
                new_attrs = attr_fix.get("attributes", [])
                for cls in classes:
                    if cls.get("className") == target_class:
                        existing_names = {a.get("name") for a in cls.get("attributes", [])}
                        for attr in new_attrs:
                            if isinstance(attr, dict) and attr.get("name") not in existing_names:
                                cls.setdefault("attributes", []).append(attr)
                                logger.info(
                                    f"[{self.get_diagram_type()}] Added missing attribute: "
                                    f"{target_class}.{attr.get('name')}"
                                )

            spec["relationships"] = relationships
            return spec

        except Exception as exc:
            logger.warning(
                f"[{self.get_diagram_type()}] Validation-feedback failed ({exc}), "
                "returning original spec"
            )
            return spec

    # ------------------------------------------------------------------
    # Error recovery: JSON repair
    # ------------------------------------------------------------------

    def self_correct(
        self,
        original_response: str,
        validation_errors: List[str],
        system_prompt: str,
        user_request: str,
    ) -> Optional[str]:
        """Self-correcting error recovery: show the LLM its validation errors
        and ask it to fix them.

        This is called when a generated spec passes JSON parsing but fails
        schema or domain validation.  Instead of falling back to a static
        template, the LLM gets a second chance with explicit feedback about
        what went wrong.

        Returns the corrected JSON string, or ``None`` if correction fails.
        """
        error_list = "\n".join(f"- {e}" for e in validation_errors[:10])
        correction_prompt = (
            f"{system_prompt}\n\n"
            f"Your previous response had these validation errors:\n{error_list}\n\n"
            f"Original user request: {user_request}\n\n"
            f"Your previous (invalid) response:\n{original_response[:3000]}\n\n"
            "Fix ALL the validation errors listed above and return the corrected JSON. "
            "Return ONLY valid JSON, no explanations."
        )

        try:
            logger.info(
                f"[{self.get_diagram_type()}] Self-correcting: "
                f"{len(validation_errors)} error(s) to fix"
            )
            corrected = self.llm.predict(correction_prompt)

            if corrected and corrected.strip():
                cleaned = self.clean_json_response(corrected)
                # Verify it parses
                json.loads(cleaned)
                logger.info(f"[{self.get_diagram_type()}] Self-correction succeeded")
                return cleaned
        except Exception as exc:
            logger.warning(f"[{self.get_diagram_type()}] Self-correction failed: {exc}")
        return None

    def parse_validate_or_correct(
        self,
        raw_response: str,
        required_keys: Dict[str, type],
        optional_keys: Optional[Dict[str, type]] = None,
        label: str = "LLM response",
        system_prompt: str = "",
        user_request: str = "",
    ) -> Dict[str, Any]:
        """Parse, validate, and self-correct if validation fails.

        This is the most resilient parsing pipeline:
        1. Clean + parse JSON
        2. Validate against schema
        3. If validation fails, try self-correction (LLM fixes its own errors)
        4. If that fails, try JSON repair
        5. If everything fails, raise ValueError

        Returns the validated dict.
        """
        json_text = self.clean_json_response(raw_response)
        spec = self.parse_json_safely(json_text)

        if spec is None:
            # Try repair first
            schema_hint = f"Required keys: {list(required_keys.keys())}"
            repaired = self.repair_json_response(json_text, schema_hint)
            if repaired:
                spec = self.parse_json_safely(repaired)
            if spec is None:
                raise ValueError(f"Could not parse JSON: {json_text[:200]}")

        errors = validate_spec(spec, required_keys, optional_keys, label=label)
        if not errors:
            return spec

        # Try self-correction
        if system_prompt and user_request:
            corrected_json = self.self_correct(
                raw_response, errors, system_prompt, user_request,
            )
            if corrected_json:
                corrected_spec = self.parse_json_safely(corrected_json)
                if corrected_spec is not None:
                    retry_errors = validate_spec(
                        corrected_spec, required_keys, optional_keys, label=label,
                    )
                    if not retry_errors:
                        return corrected_spec
                    logger.warning(
                        f"[{self.get_diagram_type()}] Self-corrected response still has errors: "
                        f"{retry_errors}"
                    )

        joined = "; ".join(errors)
        logger.warning(f"[{self.get_diagram_type()}] Schema validation failed: {joined}")
        raise ValueError(f"Schema validation failed: {joined}")

    def repair_json_response(self, malformed_json: str, schema_hint: str) -> Optional[str]:
        """Attempt to repair malformed JSON by sending it back to the LLM.

        This is a last-resort recovery step when ``parse_json_safely`` fails.
        The LLM receives the broken JSON and the expected schema, and tries
        to fix syntax errors.

        Returns the repaired JSON string, or ``None`` if repair also fails.
        """
        repair_prompt = (
            "The following JSON is malformed and could not be parsed. "
            "Fix the syntax errors and return ONLY the corrected JSON, nothing else.\n\n"
            f"Expected schema: {schema_hint}\n\n"
            f"Malformed JSON:\n{malformed_json[:3000]}"
        )
        try:
            response = self.llm.predict(repair_prompt)
            if response and response.strip():
                cleaned = self.clean_json_response(response)
                # Verify it actually parses
                import json as _json
                _json.loads(cleaned)
                logger.info(f"[{self.get_diagram_type()}] JSON repair succeeded")
                return cleaned
        except Exception as exc:
            logger.warning(f"[{self.get_diagram_type()}] JSON repair failed: {exc}")
        return None

    def parse_and_validate_with_repair(
        self,
        raw_response: str,
        required_keys: Dict[str, type],
        optional_keys: Optional[Dict[str, type]] = None,
        label: str = "LLM response",
    ) -> Dict[str, Any]:
        """Like ``parse_and_validate`` but attempts JSON repair on parse failure.

        Falls back to ``repair_json_response`` when the initial parse fails,
        giving the LLM a second chance to produce valid JSON.
        """
        json_text = self.clean_json_response(raw_response)
        spec = self.parse_json_safely(json_text)

        if spec is None:
            # Attempt repair
            schema_hint = f"Required keys: {list(required_keys.keys())}"
            if optional_keys:
                schema_hint += f", Optional keys: {list(optional_keys.keys())}"
            repaired = self.repair_json_response(json_text, schema_hint)
            if repaired:
                spec = self.parse_json_safely(repaired)

        if spec is None:
            raise ValueError(f"Could not parse JSON from LLM response (even after repair): {json_text[:200]}")

        errors = validate_spec(spec, required_keys, optional_keys, label=label)
        if errors:
            joined = "; ".join(errors)
            logger.warning(f"[{self.get_diagram_type()}] Schema validation failed: {joined}")
            raise ValueError(f"Schema validation failed: {joined}")

        return spec

    # ------------------------------------------------------------------
    # Degraded generation fallback for complex failures
    # ------------------------------------------------------------------

    def _extract_element_names(self, user_request: str, element_label: str = "entity") -> List[str]:
        """Ask the LLM to extract element names from a user request.

        This is used by the degraded fallback path when full system generation
        fails.  The LLM is asked to return just the names (a simple task that
        is much less likely to fail than generating a full schema).

        Args:
            user_request: The original user request text.
            element_label: What to call the elements (e.g. "class", "state",
                "entity") in the extraction prompt.

        Returns:
            A list of name strings.  Empty list on failure.
        """
        extraction_prompt = (
            f"From this request, extract ONLY the {element_label} names the user wants. "
            "Return a JSON array of strings. Example: [\"User\", \"Product\", \"Order\"]\n\n"
            f"Request: {user_request}\n\n"
            "Return ONLY the JSON array, no explanations."
        )
        try:
            response = self.predict_with_retry(extraction_prompt, max_retries=1)
            cleaned = self.clean_json_response(response)
            names = json.loads(cleaned)
            if isinstance(names, list) and len(names) > 0:
                return [str(n) for n in names if isinstance(n, str) and n.strip()]
        except Exception as exc:
            logger.warning(
                f"[{self.get_diagram_type()}] Could not extract {element_label} names: {exc}"
            )
        return []

    def _generate_degraded_system(
        self,
        user_request: str,
        existing_model: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Attempt degraded generation: extract names first, then fill details.

        This two-step approach is a safety net when ``generate_complete_system``
        fails after all retries.  It is cheaper and more reliable because each
        LLM call is smaller.

        Subclasses that override ``generate_complete_system`` can call this
        from their exception handler as a last resort before falling back to
        ``generate_fallback_element``.

        Returns ``None`` if even the degraded approach fails, so the caller
        can fall through to a static fallback.
        """
        diagram_type = self.get_diagram_type()
        logger.info(f"[{diagram_type}] Attempting degraded generation fallback")

        # Step 1: Extract just the names (simple, high success rate)
        element_label = {
            "ClassDiagram": "class",
            "StateMachine": "state",
            "ObjectDiagram": "object",
            "AgentDiagram": "agent",
        }.get(diagram_type, "entity")

        names = self._extract_element_names(user_request, element_label)
        if not names:
            logger.warning(f"[{diagram_type}] Degraded fallback: no names extracted")
            return None

        # Step 2: Generate each element individually
        elements: List[Dict[str, Any]] = []
        for name in names[:10]:  # Cap to prevent excessive LLM calls
            try:
                single_prompt = (
                    f"{self.get_system_prompt()}\n\n"
                    f"User Request: Create a {name} {element_label} with appropriate "
                    f"attributes for a system about: {user_request}\n\n"
                    "Return ONLY valid JSON, no markdown, no explanation."
                )
                resp = self.predict_with_retry(single_prompt, max_retries=1)
                cleaned = self.clean_json_response(resp)
                spec = self.parse_json_safely(cleaned)
                if spec and isinstance(spec, dict):
                    spec.pop("position", None)
                    elements.append(spec)
                    logger.info(f"[{diagram_type}] Degraded fallback: generated {name}")
            except Exception as exc:
                logger.warning(
                    f"[{diagram_type}] Degraded fallback: failed to generate {name}: {exc}"
                )

        if not elements:
            return None

        logger.info(
            f"[{diagram_type}] Degraded fallback: produced "
            f"{len(elements)}/{len(names)} elements"
        )
        return {
            "elements": elements,
            "names": [n for n in names[:10] if n.strip()],
        }
