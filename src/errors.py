"""Unified error handling for the modeling agent.

Consolidates error classification, recovery hints, and response building
that was previously split between base_handler.py and execution.py.
"""
from enum import Enum
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    """Error taxonomy for the modeling agent."""
    GENERATION_ERROR = "generation_error"
    PARSE_ERROR = "parse_error"
    LLM_FAILURE = "llm_failure"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTH_ERROR = "auth_error"
    CONTEXT_ERROR = "context_error"
    SCHEMA_ERROR = "schema_error"
    UNSUPPORTED = "unsupported"
    PREREQUISITE_MISSING = "prerequisite_missing"
    HANDLER_MISSING = "handler_missing"
    GENERATION_HANDLER_ERROR = "generation_handler_error"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Unified recovery hints
# ---------------------------------------------------------------------------
# Merged from base_handler._ERROR_RECOVERY and execution._get_recovery_hint.
# Each entry carries a user-facing message, a recovery suggestion, and a
# retryable flag.

_RECOVERY_HINTS: Dict[str, Dict[str, Any]] = {
    # --- from base_handler._ERROR_RECOVERY ---
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
    "rate_limit": {
        "message": "The AI service is currently busy.",
        "recovery": "please wait a moment and try again",
        "retryable": True,
    },
    "auth_error": {
        "message": "The AI service is temporarily unavailable.",
        "recovery": "please try again later",
        "retryable": False,
    },
    # --- from execution._get_recovery_hint (additions not in base_handler) ---
    "prerequisite_missing": {
        "message": "A required diagram is missing.",
        "recovery": "create the required diagram first",
        "retryable": False,
    },
    "handler_missing": {
        "message": "This diagram type is not yet supported.",
        "recovery": "this diagram type is not yet supported",
        "retryable": False,
    },
    "generation_handler_error": {
        "message": "Code generation encountered an error.",
        "recovery": "try regenerating — if it persists, check the model for issues",
        "retryable": True,
    },
    "unknown": {
        "message": "Something went wrong.",
        "recovery": "try rephrasing your request",
        "retryable": True,
    },
}


def get_recovery_hint(error_code: ErrorCode) -> Dict[str, Any]:
    """Get the recovery hint for a given error code."""
    return _RECOVERY_HINTS.get(error_code.value, _RECOVERY_HINTS["unknown"])


def classify_error(error: Exception) -> ErrorCode:
    """Classify an exception into an ErrorCode.

    Consolidates _classify_error from both base_handler.py and execution.py.

    Classification priority:
    1. Exception class name matching (most specific)
    2. Error message string matching (fallback)
    """
    from diagram_handlers.core.base_handler import LLMPredictionError

    err_name = type(error).__name__.lower()
    err_msg = str(error).lower()

    # Timeout detection (both files check this)
    if "timeout" in err_name or "timeout" in err_msg or "timed out" in err_msg:
        return ErrorCode.TIMEOUT

    # Parse / JSON errors (both files check this)
    if "parse" in err_name or "json" in err_name or "parse" in err_msg or "decode" in err_msg:
        return ErrorCode.PARSE_ERROR

    # Validation / schema errors (base_handler also checks "schema")
    if "validation" in err_name or "validation" in err_msg or "invalid" in err_msg or "schema" in err_msg:
        return ErrorCode.VALIDATION_ERROR

    # LLM-specific exception class (base_handler checks isinstance)
    if isinstance(error, LLMPredictionError):
        return ErrorCode.LLM_FAILURE

    # LLM / API errors detected by message content (execution checks this)
    if any(kw in err_msg for kw in ("openai", "llm", "rate limit", "api")):
        return ErrorCode.LLM_FAILURE

    # Rate-limit specifically
    if "rate" in err_name and "limit" in err_name:
        return ErrorCode.RATE_LIMIT

    # Auth errors
    if "auth" in err_name or "unauthorized" in err_msg or "forbidden" in err_msg or "api_key" in err_msg or "invalid_api_key" in err_msg:
        return ErrorCode.AUTH_ERROR

    # Default: base_handler used generation_error, execution used unknown.
    # Use UNKNOWN for the unified version — callers can override if needed.
    return ErrorCode.UNKNOWN


def build_error_response(
    error_code: ErrorCode,
    message: str = "",
    *,
    diagram_type: str = "",
    details: str = "",
    retryable: Optional[bool] = None,
    operation: Optional[dict] = None,
) -> Dict[str, Any]:
    """Build a standardized error response payload.

    Replaces both:
    - base_handler._error_response()
    - execution._build_error_payload()

    The response uses ``"action": "assistant_message"`` to match what the
    frontend expects.
    """
    recovery = _RECOVERY_HINTS.get(error_code.value, _RECOVERY_HINTS["unknown"])

    if retryable is None:
        retryable = recovery.get("retryable", True)

    effective_message = message or recovery.get("message", "Something went wrong. Please try again.")
    if details:
        effective_message = f"{effective_message} {details}".strip()

    payload: Dict[str, Any] = {
        "action": "assistant_message",
        "error": True,
        "errorCode": error_code.value,
        "message": effective_message,
        "suggestedRecovery": recovery.get("recovery", "Try rephrasing your request."),
        "retryable": retryable,
        "diagramType": diagram_type,
    }

    if operation is not None:
        payload["operation"] = {
            "type": operation.get("type"),
            "diagramType": operation.get("diagramType"),
            "mode": operation.get("mode"),
        }

    return payload
