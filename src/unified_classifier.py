"""Unified message classifier — ONE LLM call replaces BAF's intent
classification + the smart-gen sub-router.

Before this module:
  * BAF's ``predict_intent`` fired on every message (LLM call #1),
    picking which *state* to transition to based on long ``description=``
    keyword blobs embedded in each intent declaration.
  * Inside ``generation_state``, ``classify_generation_request`` fired
    a second LLM call (#2) to pick smart vs deterministic and extract
    generator_type / refined_instructions.

This module collapses both into a single structured-output call that
returns EVERY field any downstream state body needs, cached per-message
so repeat transition conditions don't re-classify.

Architecture:

  1. ``classify_message(request, llm_provider)`` → ``UnifiedClassification``.
     One OpenAI call with a clean rule-based system prompt, forced
     structured output via Pydantic. Never raises — on any failure,
     returns a safe ``fallback_intent`` classification so the agent
     gracefully degrades to its own fallback body.

  2. ``get_or_classify(session, request, llm_provider)`` wraps that
     with a per-message cache. A BAF event's id is used as the cache
     key, so the first transition condition / state body to ask
     triggers the classification and everyone else reads the cached
     answer.

The schema is deliberately wide: it carries fields for generation
sub-routing AND diagram-creation targets in one object, so every state
body can read from the same source of truth.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from protocol.types import AssistantRequest
from session_keys import (
    UNIFIED_CLASSIFICATION,
    UNIFIED_CLASSIFICATION_EVENT_ID,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------


# Top-level intent — mirrors the ``new_intent`` declarations in
# ``modeling_agent.py``. If a new state is added there, mirror it here.
_INTENT_NAMES = Literal[
    "hello_intent",
    "create_complete_system_intent",
    "modify_model_intent",
    "modeling_help_intent",
    "describe_model_intent",
    "uml_spec_intent",
    "generation_intent",
    "workflow_intent",
    # Catch-all — BAF's own fallback state body runs when nothing
    # matches. Routed to whatever state fallback the current state
    # has (typically modeling_help_state).
    "fallback_intent",
]


# Deterministic generators BESSER has built-in. Must stay in sync with
# ``generation_handler.GENERATOR_KEYWORDS`` and the TypeScript frontend.
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


# Which diagram the user is talking about, for create_complete_system /
# modify_model / describe_model routes.
_TARGET_DIAGRAM_TYPES = Literal[
    "ClassDiagram",
    "ObjectDiagram",
    "StateMachineDiagram",
    "AgentDiagram",
    "GUINoCodeDiagram",
    "QuantumCircuitDiagram",
]


class UnifiedClassification(BaseModel):
    """Everything the state machine needs to route a message — in one shot."""

    intent: _INTENT_NAMES = Field(
        ...,
        description=(
            "State-level intent. Pick exactly one:\n"
            "  'hello_intent'                   — greeting / small-talk\n"
            "  'create_complete_system_intent'  — user wants a NEW diagram or "
            "complete system FROM SCRATCH (e.g. 'create a class diagram for a "
            "library', 'model a booking system', 'build a Grover algorithm "
            "circuit')\n"
            "  'modify_model_intent'            — user wants to ADD / REMOVE / "
            "CHANGE elements in an EXISTING diagram (e.g. 'add a class called "
            "User', 'remove the Book class', 'connect Author and Book')\n"
            "  'describe_model_intent'          — user is asking ABOUT their "
            "current diagram (e.g. 'what classes do I have?', 'list all "
            "states')\n"
            "  'modeling_help_intent'           — user asks for CONCEPTUAL "
            "help (e.g. 'how do I model inheritance?', 'explain UML "
            "composition')\n"
            "  'uml_spec_intent'                — user asks about the formal "
            "UML specification\n"
            "  'generation_intent'              — user wants SOURCE CODE in ANY "
            "language or stack, or to EXPORT / DEPLOY. Includes BESSER built-ins "
            "(django, pydantic, sql, ...) AND any other language (rails, rust, "
            "kotlin, next.js, go, ...).\n"
            "  'workflow_intent'                — user wants the FULL end-to-end "
            "pipeline in one flow (e.g. 'create a complete web app for X and "
            "generate all the code')\n"
            "  'fallback_intent'                — none of the above fit cleanly."
        ),
    )

    # --- Generation-only fields (populated when intent == 'generation_intent') ---

    generation_route: Optional[Literal["smart", "deterministic", "modeling", "other"]] = Field(
        default=None,
        description=(
            "REQUIRED when intent='generation_intent'. Sub-routing:\n"
            "  'deterministic' — user wants ONE BESSER built-in with NO extras "
            "(auth, JWT, Docker, migrations, …). Pure scaffolding.\n"
            "  'smart' — user wants a non-BESSER stack (rails, rust, kotlin, "
            "...) OR a BESSER built-in PLUS extras the template can't produce "
            "(auth, JWT, OAuth, Docker, custom DB, migrations, tests, rate-"
            "limiting, custom middleware).\n"
            "  'modeling' — this is actually a 'generate a diagram' request "
            "misrouted here (use this to redirect).\n"
            "  'other' — not a code-generation request at all."
        ),
    )
    generator_type: Optional[_DETERMINISTIC_GENERATOR_TYPES] = Field(
        default=None,
        description=(
            "REQUIRED when generation_route='deterministic'. Name of the "
            "BESSER built-in generator to run."
        ),
    )
    refined_instructions: Optional[str] = Field(
        default=None,
        description=(
            "REQUIRED when generation_route='smart'. A polished prompt for the "
            "smart generator naming the stack (e.g. 'Rails 7, PostgreSQL via "
            "Active Record, Devise auth') and any non-functional requirements "
            "the user mentioned. Max 2000 chars. Do NOT describe the class "
            "diagram in detail — the generator has the domain model. Do NOT "
            "invent requirements the user didn't mention."
        ),
    )
    provider: Literal["anthropic", "openai"] = Field(
        default="anthropic",
        description=(
            "Suggested LLM provider when generation_route='smart'. Ignored "
            "for other routes. The frontend's BYOK dropdown can override."
        ),
    )

    # --- Domain-mismatch fields (populated when generation_route='smart') ---
    # Used to refuse silent code-rewrites when the user's request describes
    # a different domain than their existing class diagram.

    domain_mismatch: Optional[bool] = Field(
        default=None,
        description=(
            "ONLY when generation_route='smart' AND a class diagram with at "
            "least one class is present in WORKSPACE CONTEXT. True if the "
            "user's request describes a domain that DOES NOT match the "
            "existing class diagram (e.g. classes are 'Team/Player' but the "
            "request is 'a shoe store'). False if the request fits the "
            "existing model OR the model is empty/absent. Be conservative: "
            "if unsure, return False. Leave NULL when route != 'smart' or "
            "when there's no existing class diagram to compare against."
        ),
    )
    suggested_new_domain: Optional[str] = Field(
        default=None,
        description=(
            "When domain_mismatch=True, a SHORT noun phrase naming the "
            "domain the user actually wants (e.g. 'a shoe store', 'a hotel "
            "booking system', 'a blog platform'). Used in the agent's "
            "follow-up question. Max 80 chars. Leave NULL otherwise."
        ),
    )

    # --- Modeling-side fields (create / modify / describe) ---

    target_diagram_type: Optional[_TARGET_DIAGRAM_TYPES] = Field(
        default=None,
        description=(
            "For create_complete_system_intent / modify_model_intent / "
            "describe_model_intent — which diagram the user is talking "
            "about. Leave NULL if the user didn't specify and the "
            "active diagram in the workspace context should be used."
        ),
    )

    reason: str = Field(
        ...,
        description=(
            "One short sentence (max 160 chars) explaining the classification. "
            "Used for logs and surfaced to users as a hint."
        ),
    )


_SYSTEM_PROMPT = (
    "You are an intent classifier. Classify the user's message into "
    "one of the listed intents and, when relevant, populate the "
    "sub-routing fields. Return the structured classification only — "
    "no prose, no questions, no follow-ups.\n\n"
    "=== TOP-LEVEL INTENT RULES (pick one) ===\n\n"
    "hello_intent: greetings, small-talk, capability questions "
    "('what can you do'), thanks, acknowledgements. A question about "
    "the user's OWN model or app is NEVER hello — in particular "
    "'where is the app?', 'how do I run / try / see / use it?', "
    "'can I try it?' are generation_intent, not hello.\n\n"
    "create_complete_system_intent: user wants a NEW diagram or "
    "complete system from scratch. Keywords that trigger this: "
    "'create a class diagram for', 'design a system', 'model a', "
    "'generate a class diagram', 'build a state machine for', "
    "'create the GUI for'. If they name a domain ('library', "
    "'e-commerce', 'hotel booking') and ask for a diagram or system, "
    "it's this. CRITICAL: 'generate a class diagram' is this intent, "
    "NOT generation_intent — they want a DIAGRAM, not source code.\n\n"
    "modify_model_intent: user wants to ADD / REMOVE / CHANGE "
    "elements in an existing diagram. 'add a class', 'remove the "
    "Book class', 'rename', 'delete', 'connect', 'add an attribute', "
    "'modify method', 'I also want to include', 'extend with', 'add "
    "a gate to the circuit'. Also single-element creation: 'create "
    "a class called User', 'make a state'.\n\n"
    "describe_model_intent: user asks QUESTIONS about their CURRENT "
    "diagram. 'how many classes', 'what attributes', 'list all', "
    "'tell me about my model', 'describe', 'summarize', 'what does "
    "this circuit do'. Always about what ALREADY EXISTS.\n\n"
    "modeling_help_intent: conceptual help, explanations, best "
    "practices. 'how do I', 'explain', 'what is', 'how does X work', "
    "'what are best practices for'. Conceptual, not about their "
    "specific model.\n\n"
    "uml_spec_intent: asks about the formal UML specification "
    "document. 'according to UML spec', 'what does UML standard say', "
    "'OMG specification'. Rare.\n\n"
    "generation_intent: user wants SOURCE CODE, EXPORT, or DEPLOY. "
    "Includes BESSER's built-in generators (django, pydantic, sql, "
    "sqlalchemy, python, java, web_app, backend, jsonschema, "
    "smartdata, agent, qiskit, rest_api, rdf) AND ANY OTHER language "
    "or framework (ruby on rails, rust, kotlin, swift, go, elixir, "
    "c#, c++, php, laravel, flask, express, next.js, spring boot, "
    "angular, vue, svelte, ios, android). Also: export to json/buml, "
    "deploy to render. ALSO includes asking to RUN, TRY, PREVIEW, "
    "LAUNCH, USE, or SEE the app, or 'where is the app?' — the user "
    "has a model and wants runnable output, which comes from "
    "generating code. NEVER use this when the user says 'generate a "
    "class diagram' — that's create_complete_system_intent.\n\n"
    "workflow_intent: user EXPLICITLY wants the FULL end-to-end flow "
    "in one go: 'create a complete web app for X and generate the "
    "code', 'build and deploy', 'end-to-end for a booking system'. "
    "Rare — only when they clearly want BOTH the model and the code.\n\n"
    "fallback_intent: none of the above fits cleanly.\n\n"
    "=== GENERATION SUB-ROUTING (populate when intent='generation_intent') ===\n\n"
    "CRITICAL BACKGROUND: BESSER has two generation paths.\n"
    "  * 'deterministic' = pure scaffolding from a template. No auth, "
    "no JWT, no OAuth, no Docker, no migrations, no tests — JUST the "
    "baseline. If the user wants ANY extra feature, deterministic is "
    "wrong.\n"
    "  * 'smart' = scaffolding + custom features. Internally runs a "
    "deterministic template first, then the LLM adds custom features "
    "on top.\n\n"
    "Sub-routing rules:\n"
    "1. Non-BESSER language/framework (rails, rust, kotlin, swift, "
    "go, elixir, php, laravel, flask, express, next.js, spring boot, "
    "angular, vue, svelte, ios app, android app, ...) → 'smart'.\n"
    "2. Compound build ('backend + frontend', 'full-stack fastapi "
    "with jwt + postgres', 'dockerized next.js') → 'smart'.\n"
    "3. BESSER built-in + EXTRAS (auth, JWT, OAuth, Docker, specific "
    "DB beyond default, migrations, tests, rate-limiting, middleware, "
    "CORS, CI/CD) → 'smart'. Examples: 'web app with authentication', "
    "'django with jwt', 'backend with docker'.\n"
    "4. BESSER built-in with NO extras → 'deterministic' with "
    "generator_type set. Examples: 'generate django', 'give me "
    "pydantic classes', 'generate sql'.\n"
    "4b. Vague 'how do I run / try / see / get the app' or 'where is "
    "the app' with NO stack named → 'deterministic' with "
    "generator_type=null (the agent then shows the generator menu so "
    "the user picks what to build).\n"
    "5. User actually wants a DIAGRAM (not source code) → 'modeling'.\n"
    "6. Greetings / small-talk leaking through → 'other'.\n\n"
    "=== domain_mismatch (populate when generation_route='smart') ===\n"
    "If WORKSPACE CONTEXT lists CLASS NAMES from an existing class "
    "diagram, judge whether those classes describe the SAME DOMAIN as "
    "the user's request:\n"
    "  * Classes 'Team', 'Player' + request 'build a shoe store webapp' "
    "→ domain_mismatch=True, suggested_new_domain='a shoe store'.\n"
    "  * Classes 'Book', 'Author' + request 'add JWT auth and Docker' "
    "→ domain_mismatch=False (request is about stack, not domain).\n"
    "  * No class names present (empty model or non-class diagram only) "
    "→ domain_mismatch=null.\n"
    "BE CONSERVATIVE: if the request could be applied on top of the "
    "existing classes, return False. Only flag True when the domain "
    "vocabulary clearly does not match.\n\n"
    "=== refined_instructions (populate when generation_route='smart') ===\n"
    "A polished, implementation-focused prompt for the smart generator:\n"
    "  * name the stack explicitly (Rails, PostgreSQL, Devise auth, ...)\n"
    "  * include non-functional requirements the user mentioned\n"
    "  * max 2000 chars\n"
    "  * do NOT describe the class diagram in detail (the generator "
    "    has the domain model)\n"
    "  * do NOT invent requirements the user didn't mention\n\n"
    "=== target_diagram_type (create / modify / describe) ===\n"
    "Which diagram the user is talking about. Leave NULL if they "
    "didn't specify — the state body will use the workspace's active "
    "diagram. Set it when they explicitly say 'the class diagram', "
    "'my state machine', 'the quantum circuit', etc.\n\n"
    "=== OUTPUT ===\n"
    "Return the structured classification. Always include 'reason' "
    "(≤160 chars) explaining your choice. Be decisive — do not "
    "second-guess; the state machine trusts your verdict."
)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def classify_message(
    request: AssistantRequest,
    llm_provider: Any,
) -> UnifiedClassification:
    """Classify a user message into a state-level intent + sub-routing fields.

    ONE classifier-tier structured-output call (see ``model_config``).
    Returns a safe
    ``fallback_intent`` classification if the provider is unavailable
    or the call fails — the caller should trust the returned object
    and dispatch based on ``intent``.

    Never raises — the classifier must never crash the agent.
    """
    if llm_provider is None:
        return _safe_fallback("LLM provider unavailable")
    message = (request.message or "").strip()
    if not message:
        return _safe_fallback("empty message")

    try:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_block(request)},
        ]
        result: UnifiedClassification = llm_provider.parse(
            messages=messages,
            schema=UnifiedClassification,
            temperature=0.0,
            max_tokens=800,
        )
        if result is None:
            return _safe_fallback("LLM returned no result")
        return _post_validate(result)
    except Exception:
        logger.exception("classify_message failed; falling back to fallback_intent")
        return _safe_fallback("LLM classifier failed")


def get_or_classify(
    session: Any,
    request: AssistantRequest,
    llm_provider: Any,
) -> UnifiedClassification:
    """Per-message cache wrapper around :func:`classify_message`.

    Uses the BAF event's id as the cache key. The first caller on a
    given message triggers the classification; subsequent callers on
    the SAME message read the cached result without an extra LLM call.

    Every transition condition and state body on a single incoming
    WebSocket message should go through this helper so the whole
    request consumes exactly ONE classification call.
    """
    event_id = _current_event_id(session)
    cached_event_id = session.get(UNIFIED_CLASSIFICATION_EVENT_ID)
    cached_classification = session.get(UNIFIED_CLASSIFICATION)
    if (
        event_id is not None
        and cached_event_id == event_id
        and cached_classification is not None
    ):
        return cached_classification

    # Frontend callbacks (``generator_result`` etc.) are protocol events,
    # not user prose — their routing is determined by the ``action``
    # field, so classifying their text is pure waste AND wrong: in
    # production a generation-completion echo was LLM-classified as
    # ``hello_intent`` and routed to greetings, so the generation
    # handler's frontend_event branch never ran.
    if getattr(request, "action", None) == "frontend_event":
        result = UnifiedClassification(
            intent="generation_intent",
            generation_route="other",
            reason="frontend_event callback — routed deterministically, no LLM call",
        )
    else:
        result = classify_message(request, llm_provider)
    if event_id is not None:
        session.set(UNIFIED_CLASSIFICATION, result)
        session.set(UNIFIED_CLASSIFICATION_EVENT_ID, event_id)
    return result


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _build_user_block(request: AssistantRequest) -> str:
    """Compose the user message + workspace context for the classifier."""
    lines = ["USER MESSAGE:", request.message or ""]
    ctx = getattr(request, "context", None)
    if ctx is None:
        return "\n".join(lines)

    summary_lines = []
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

    # Extract class names from the active class diagram so the classifier
    # can detect domain mismatches (e.g. user asks for a shoe store while
    # the diagram is Team/Player). We only pass class NAMES — attributes
    # and methods would bloat the prompt and aren't needed for the
    # mismatch judgement.
    class_names = _extract_class_names(ctx)
    if class_names:
        summary_lines.append(
            "- existing class names: " + ", ".join(class_names[:30])
        )

    if summary_lines:
        lines.append("")
        lines.append("WORKSPACE CONTEXT:")
        lines.extend(summary_lines)
    return "\n".join(lines)


def _extract_class_names(ctx: Any) -> list[str]:
    """Pull class names from the active ClassDiagram in the project snapshot.

    Returns an empty list when there is no ClassDiagram, the diagram is
    empty, or the snapshot shape is unexpected. Never raises.
    """
    try:
        diagram = ctx.get_diagram_from_snapshot("ClassDiagram")
    except Exception:
        return []
    if not isinstance(diagram, dict):
        return []
    model = diagram.get("model")
    if not isinstance(model, dict):
        return []
    elements = model.get("elements")
    if not isinstance(elements, dict):
        return []
    names: list[str] = []
    for elem in elements.values():
        if not isinstance(elem, dict):
            continue
        if elem.get("type") not in ("Class", "AbstractClass"):
            continue
        name = elem.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def _post_validate(result: UnifiedClassification) -> UnifiedClassification:
    """Defensive validation of LLM output.

    Catches the classic "LLM returned route='smart' but forgot to write
    instructions" and "intent='generation' but no generation_route"
    bugs. Demotes them to safer states rather than propagating broken
    classifications downstream.
    """
    if result.intent == "generation_intent":
        if result.generation_route is None:
            logger.warning(
                "LLM returned generation_intent with no generation_route; "
                "demoting to fallback_intent"
            )
            return UnifiedClassification(
                intent="fallback_intent",
                reason="classifier missed generation sub-routing",
            )
        if result.generation_route == "smart":
            if not (result.refined_instructions or "").strip():
                logger.warning(
                    "LLM returned smart route with no refined_instructions; "
                    "demoting to deterministic-unknown"
                )
                return UnifiedClassification(
                    intent="generation_intent",
                    generation_route="deterministic",
                    generator_type=None,
                    reason="classifier omitted smart instructions",
                )
        elif result.generation_route == "deterministic":
            # generator_type may be None — the caller will show the
            # generator menu. That's fine; no demotion needed.
            pass
    return result


def _safe_fallback(reason: str) -> UnifiedClassification:
    """Safe default when the LLM is unavailable.

    Returns ``fallback_intent`` so the agent runs its existing
    fallback state body. This is the pre-LLM-classifier behaviour —
    user gets a helpful message asking them to rephrase.
    """
    return UnifiedClassification(intent="fallback_intent", reason=reason)


def _current_event_id(session: Any) -> Optional[str]:
    """Best-effort event id used as the per-message cache key.

    BAF exposes the current event on ``session.event``; we try a few
    common attribute names and fall back to ``None`` (which disables
    caching — safe, just costs an extra LLM call if multiple callers
    on the same message ask independently).
    """
    event = getattr(session, "event", None)
    if event is None:
        return None
    for attr in ("id", "event_id", "uid"):
        value = getattr(event, attr, None)
        if value:
            return str(value)
    # Fall back to the raw event's id() — stable for the lifetime of
    # the event object, which is exactly one message dispatch.
    try:
        return f"obj:{id(event)}"
    except Exception:
        return None
