"""Unified LLM-based classifier for generation requests.

The modeling agent's intent routing (unified classifier, with BAF's
local Simple classifier as fallback) decides at the STATE level whether
a message is a generation request (``generation_intent``). Once routed
to ``generation_state``, this module makes ONE more LLM call that
decides the SUB-routing:

    route ∈ {"smart", "deterministic", "modeling", "other"}

and — in the same structured-output call — returns any fields the
caller needs so we never re-prompt the LLM later:

  * For ``route == "deterministic"``: which BESSER built-in generator
    (``generator_type``) so the caller dispatches to django / pydantic /
    sql / etc. without any further keyword matching.
  * For ``route == "smart"``: a polished ``refined_instructions`` prompt
    for the smart generator, plus a provider suggestion.

This replaces three legacy pieces: a keyword-based phrase list
(``_SMART_GEN_COMPLEX_PHRASES``, ``_COMPOUND_INTENT_RE``, an "unsupported
language" regex), a first smart-vs-deterministic LLM call, and a
separate ``refine_instructions`` LLM call. Net result: one LLM call
where we used to have three, no more keyword lists to maintain every
time a user asks for a new language, and intent recognition that scales
to any stack without code changes.

The user's Anthropic/OpenAI BYOK key is never used here — this call
runs on the modeling-agent operator's shared key (classifier tier, see
``model_config``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from protocol.types import AssistantRequest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------


# The names BESSER's deterministic generator registry understands. If
# the classifier picks ``deterministic`` with a type not in this list,
# the caller falls back to ``None`` and shows the user the generator
# menu. Keep in sync with ``generation_handler.GENERATOR_KEYWORDS``.
_DETERMINISTIC_GENERATOR_TYPES = Literal[
    "django",
    "backend",
    "web_app",
    "sql",
    "sqlalchemy",
    "python",
    "java",
    "pydantic",
    "jsonschema",
    "smartdata",
    "agent",
    "qiskit",
    "rest_api",
    "rdf",
    "export",
    "deploy",
]


class GenerationClassification(BaseModel):
    """Single source of truth for routing + sub-routing of a generation request."""

    route: Literal["smart", "deterministic", "modeling", "other"] = Field(
        ...,
        description=(
            "Pick exactly one:\n"
            "  'smart'         — user wants a custom codebase beyond any single "
            "BESSER built-in generator: ANY non-BESSER language or framework "
            "(rust, kotlin, swift, rails, flask, next.js, spring boot, go, "
            "elixir, php/laravel, c#, c++, angular, svelte, ios app, ...) OR "
            "a compound build ('full-stack fastapi + jwt + postgres', "
            "'dockerized next.js', 'react + node'). This path runs the "
            "LLM-augmented smart generator with the user's BYOK key.\n"
            "  'deterministic' — user wants ONE of BESSER's built-in generators "
            "EXACTLY: django, pydantic, python classes, java classes, sql, "
            "sqlalchemy, jsonschema, smartdata, agent, qiskit, web_app, "
            "backend, rest_api, rdf, export, deploy. If a non-BESSER language "
            "is named, it is 'smart', never 'deterministic'.\n"
            "  'modeling'      — user wants to CREATE or MODIFY a diagram "
            "(class diagram, state machine, GUI, agent diagram), not "
            "generate source code. 'generate a class diagram for X' is "
            "modeling, not deterministic.\n"
            "  'other'         — greeting, capability question, chat, "
            "anything that is not a code-generation or modeling request."
        ),
    )
    generator_type: Optional[_DETERMINISTIC_GENERATOR_TYPES] = Field(
        default=None,
        description=(
            "REQUIRED when route='deterministic'. One of BESSER's built-in "
            "generator names. LEAVE NULL for route != 'deterministic'."
        ),
    )
    refined_instructions: Optional[str] = Field(
        default=None,
        description=(
            "REQUIRED when route='smart'. A polished, implementation-focused "
            "prompt for the smart generator: 1-3 short paragraphs naming the "
            "stack (e.g. 'Ruby on Rails 7, PostgreSQL via Active Record, "
            "Devise auth'), any explicit non-functional requirements the "
            "user mentioned (JWT, Docker, migrations, tests), and NO "
            "invented requirements. Max 2000 chars. LEAVE NULL for "
            "route != 'smart'."
        ),
    )
    provider: Literal["anthropic", "openai"] = Field(
        default="anthropic",
        description=(
            "Suggested LLM provider for the smart generator when route='smart'. "
            "Default 'anthropic'. Ignored for other routes."
        ),
    )
    reason: str = Field(
        ...,
        description=(
            "One short sentence (max 160 chars) explaining the classification. "
            "Used for logs and surfaced to the user as a hint."
        ),
    )


_CLASSIFIER_SYSTEM_PROMPT = (
    "You are the sub-router for BESSER's generation flow. BAF has "
    "already classified the user's message as a generation-related "
    "request; your job is to decide which PATH inside the generation "
    "flow handles it, plus return any fields the dispatcher needs.\n\n"
    "CRITICAL BACKGROUND: BESSER has two kinds of generation paths.\n"
    "  * 'deterministic' = pure scaffolding from a template. No auth, "
    "    no JWT, no OAuth, no Docker, no custom middleware, no "
    "    migrations, no tests, no rate-limiting — JUST the baseline "
    "    code for that stack. If the user wants ANY feature beyond "
    "    the baseline, deterministic is the wrong path.\n"
    "  * 'smart' = scaffolding + custom features. Internally the smart "
    "    generator first runs a deterministic template, THEN uses an "
    "    LLM to add custom features on top. This is the right path "
    "    ANY TIME the user mentions extras the deterministic template "
    "    does not produce.\n\n"
    "Hard rules (in priority order):\n"
    "1. If the user names a language or framework OTHER than BESSER's "
    "   built-ins (django, pydantic, python classes, java classes, sql, "
    "   sqlalchemy, jsonschema, smartdata, agent, qiskit, web_app, "
    "   backend, rest_api, rdf) — route='smart'. This includes Ruby on "
    "   Rails, Rust, Kotlin, Swift, Go, Elixir, C++, C#, PHP, Laravel, "
    "   Flask, Express, Spring Boot, Next.js, Nest.js, Angular, Vue, "
    "   Svelte, iOS app, Android app, and anything else.\n"
    "2. If the user asks for a compound build ('backend + frontend', "
    "   'full-stack fastapi with jwt and postgres', 'dockerized X') — "
    "   route='smart'.\n"
    "3. IMPORTANT: If the user names a BESSER built-in AND asks for "
    "   extras the deterministic template does not include — "
    "   route='smart'. Extras include: authentication of any kind "
    "   (auth, authentication, login, signup, users, sign-in), JWT, "
    "   OAuth, Devise, session management, any permission/role system, "
    "   Docker, docker-compose, containerization, a specific database "
    "   other than the default (postgres, mysql, mongodb when the "
    "   default is sqlite), migrations / Alembic, unit or integration "
    "   tests, rate-limiting, custom middleware, CORS configuration, "
    "   deployment setup, CI/CD, or any other non-baseline feature. "
    "   Examples that MUST route='smart': 'web app with authentication', "
    "   'django with jwt', 'backend with docker', 'django with postgres "
    "   and tests', 'web_app with oauth'.\n"
    "4. If the user names EXACTLY a BESSER built-in with NO extras — "
    "   just the baseline scaffolding — route='deterministic' with "
    "   generator_type set. Examples: 'generate django', 'give me "
    "   pydantic classes', 'generate sql', 'python classes for this', "
    "   'run the web_app generator'.\n"
    "5. If the user wants a DIAGRAM (not source code) — route='modeling'.\n"
    "6. Greetings / questions / small-talk — route='other'.\n\n"
    "Pay attention to intent over literal words. 'generate me the class "
    "rust' means the user wants Rust code, not a UML class named Rust → "
    "route='smart'. 'generate a class diagram' means create a diagram → "
    "route='modeling'. 'give me pydantic classes' means the pydantic "
    "generator → route='deterministic', generator_type='pydantic'. "
    "'web app with authentication' means web_app scaffold + custom "
    "auth work the deterministic template can't do → route='smart', "
    "refined_instructions should name the stack (React + FastAPI) and "
    "the auth approach (e.g. JWT with passlib and python-jose).\n\n"
    "When route='smart', write refined_instructions that:\n"
    "  * names the stack explicitly (Rails, PostgreSQL, Devise auth, ...)\n"
    "  * includes any non-functional requirements the user mentioned\n"
    "  * stays under 2000 chars\n"
    "  * does NOT describe the class diagram in detail (the generator "
    "    already has the domain model)\n"
    "  * does NOT invent requirements the user didn't mention"
)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def classify_generation_request(
    request: AssistantRequest,
    llm_provider: Any,
) -> GenerationClassification:
    """Classify a generation request using one structured LLM call.

    Returns a safe fallback classification (``route='deterministic',
    generator_type=None, reason='LLM unavailable'``) if the LLM
    provider is missing or the call fails — the caller then shows the
    user the generator menu, which is the pre-LLM behaviour.

    Never raises — the router must never crash the agent.
    """
    if llm_provider is None:
        return _fallback_classification("LLM provider unavailable")

    try:
        messages = [
            {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_block(request),
            },
        ]
        result: GenerationClassification = llm_provider.parse(
            messages=messages,
            schema=GenerationClassification,
            temperature=0.0,
            max_tokens=800,
        )
        # Validate the LLM's output — if it returned ``deterministic``
        # with no generator_type, treat it as "deterministic but unknown"
        # so the caller falls back to the menu rather than attempting
        # to dispatch to None.
        if result.route == "smart" and not (result.refined_instructions or "").strip():
            # Smart without instructions = invalid. Fall back.
            logger.warning(
                "LLM classifier returned route='smart' with no refined_instructions; "
                "treating as deterministic-unknown"
            )
            return GenerationClassification(
                route="deterministic",
                generator_type=None,
                refined_instructions=None,
                reason="Smart path but no instructions returned",
            )
        return result
    except Exception:
        logger.exception("classify_generation_request failed; falling back")
        return _fallback_classification("LLM classifier failed")


def _fallback_classification(reason: str) -> GenerationClassification:
    """Safe default when the LLM is unavailable.

    We return 'deterministic' with no ``generator_type`` so the caller
    shows the user the built-in generator menu. This is the pre-LLM
    behaviour — the user picks a generator by name from a list.
    """
    return GenerationClassification(
        route="deterministic",
        generator_type=None,
        refined_instructions=None,
        reason=reason,
    )


def _build_user_block(request: AssistantRequest) -> str:
    """Compose the user message + workspace context for the classifier."""
    lines: List[str] = [
        "USER MESSAGE:",
        request.message or "",
    ]
    ctx = getattr(request, "context", None)
    if ctx is None:
        return "\n".join(lines)

    summary_lines: List[str] = []
    active_type = getattr(ctx, "active_diagram_type", None)
    if active_type:
        summary_lines.append(f"- active diagram: {active_type}")
    summaries = getattr(ctx, "diagram_summaries", None) or []
    if isinstance(summaries, list):
        for summary in summaries[:6]:
            if not isinstance(summary, dict):
                continue
            diagram_type = summary.get("type") or summary.get("diagramType")
            title = summary.get("title")
            element_count = summary.get("elementCount")
            bits = [b for b in (diagram_type, title) if b]
            if element_count is not None:
                bits.append(f"{element_count} elements")
            if bits:
                summary_lines.append("- " + " · ".join(str(b) for b in bits))
    if summary_lines:
        lines.append("")
        lines.append("WORKSPACE CONTEXT:")
        lines.extend(summary_lines)
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Payload assembly for the smart-gen path
# ---------------------------------------------------------------------


_DEFAULT_SMART_GEN_MODEL_BY_PROVIDER: Dict[str, str] = {
    # Match ``besser/generators/llm/llm_client.py::DEFAULT_MODELS`` and
    # the backend's config endpoint. If these diverge, a run with no
    # explicit ``llmModel`` override will fail with an "unknown model"
    # upstream error.
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
    "mistral": "mistral-large-latest",
}


def build_trigger_smart_generator_payload(
    classification: GenerationClassification,
    reason_prefix: str = "",
) -> Dict[str, Any]:
    """Assemble the WebSocket ``trigger_smart_generator`` action payload.

    Requires a classification whose ``route == 'smart'``.
    """
    if classification.route != "smart":
        raise ValueError(
            "build_trigger_smart_generator_payload called with non-smart classification"
        )
    instructions = (classification.refined_instructions or "").strip()
    if not instructions:
        raise ValueError("smart classification has no refined_instructions")

    provider = classification.provider or "anthropic"
    llm_model = _DEFAULT_SMART_GEN_MODEL_BY_PROVIDER.get(provider, "claude-sonnet-4-6")

    # Short, neutral run banner. The provider/free-tier choice and the BYOK
    # option were already conveyed by the confirmation copy shown before this
    # run (see ``_build_smart_gen_confirmation``), so this mid-run line stays
    # minimal instead of repeating the API-key explanation.
    return {
        "action": "trigger_smart_generator",
        "instructions": instructions,
        "provider": provider,
        "llmModel": llm_model,
        "message": "Generating your application from the spec…",
    }
