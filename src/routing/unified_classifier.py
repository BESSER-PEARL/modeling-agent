"""Unified message classifier — ONE LLM call replaces BAF's intent
classification + the smart-gen sub-router.

Before this module:
  * BAF's ``predict_intent`` fired on every message (LLM call #1),
    picking which *state* to transition to based on long ``description=``
    keyword blobs embedded in each intent declaration.
  * Inside ``generation_state``, a second sub-router prompt (since
    retired) fired another LLM call to pick smart vs deterministic and
    extract generator_type / refined_instructions.

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
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from llm.byok_config import get_current
from prompts.classifier import SYSTEM_PROMPT
from protocol.types import AssistantRequest
from routing.guards import AUTO_FIX_PREFIX, auto_fix_verdict, frontend_event_verdict  # noqa: F401  (AUTO_FIX_PREFIX re-exported)
from routing.generation_signals import (
    _GITHUB_CONTINUE_VERB_RE,
    _extract_github_reference,
    recent_smart_gen_for_project,
)
from utilities.original_request import remember_target, request_project_id, target_for_project
from common.session_keys import (
    LAST_SUGGESTED_PROMPTS,
    REPLY_LANGUAGE,
    UNIFIED_CLASSIFICATION,
    UNIFIED_CLASSIFICATION_EVENT_ID,
)
from common import pending_dialogs
from common.pending_dialogs import PENDING_GENERATOR_CONFIG, PLAN_GENERATION_CONFIRM_FLAG

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
    # User declines / opts out ("nothing", "no thanks", "never mind",
    # "I'm good") — acknowledge, do NOT build or modify anything.
    "decline_intent",
    # User asks for a non-software artifact (an actual image/picture, a poem,
    # a joke, general chit-chat) — redirect, do NOT model a diagram of it.
    "out_of_scope_intent",
    # User asks about the ASSISTANT itself — capabilities ("do you generate
    # websites?", "what can you do") or value ("why use you instead of
    # Claude/GPT?") — answer the pitch, do NOT build anything.
    "meta_question_intent",
    # Catch-all — BAF's own fallback state body runs when nothing
    # matches. Routed to whatever state fallback the current state
    # has (typically modeling_help_state).
    "fallback_intent",
]


# Deterministic generators BESSER has built-in. Must stay in sync with
# ``routing.generation_signals.GENERATOR_KEYWORDS`` and the TypeScript frontend.
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


# Languages / frameworks BESSER has NO deterministic generator for. Its built-in
# language generators are only Python and Java (plus SQL / Django / FastAPI /
# SQLAlchemy / Pydantic / …). An explicit request for anything else must go the
# smart (from-scratch, LLM-written) route — the LLM classifier is unreliable
# here, mapping "c classes" -> java and "c++ classes" -> python. Distinctive
# tokens are matched on a word boundary; ambiguous single words ("c", "go") only
# count when a language noun follows.
_UNSUPPORTED_STACK_RE = re.compile(
    r"\b(rust|kotlin|swift|scala|elixir|golang|ruby|php|dart|perl|haskell|zig|"
    r"nim|crystal|cpp|csharp|dotnet|fsharp|rails|nestjs|nextjs|express|"
    r"springboot|spring|laravel|symfony|angular|vue|svelte|nuxt|flutter|"
    r"objective-?c)\b",
    re.I,
)
_UNSUPPORTED_STACK_LITERALS = ("c++", "c#", ".net", "f#")
# Words that explicitly ask for screens. Shared with the planner's
# from-scratch web-app route (orchestrator/request_planner.py).
EXPLICIT_SCREEN_VOCAB_RE = re.compile(
    r"\b(gui|screens?|ui|user interface|mockups?|wireframes?|"
    r"pages?|frontend|front-end)\b",
    re.I,
)
_BARE_LANG_RE = re.compile(
    r"\b(?:c|go)\b[\s\-]{0,3}(?:classes|class|code|program|programs|language|"
    r"structs?|headers?|files?|app|application)\b",
    re.I,
)


def _names_unsupported_stack(message: str) -> bool:
    """True when the message explicitly names a language/stack BESSER can't
    generate deterministically (so it belongs on the smart/from-scratch route)."""
    low = (message or "").lower()
    if any(tok in low for tok in _UNSUPPORTED_STACK_LITERALS):
        return True
    if _UNSUPPORTED_STACK_RE.search(low):
        return True
    if _BARE_LANG_RE.search(low):
        return True
    return False


# Which diagram the user is talking about, for create_complete_system /
# modify_model / describe_model routes.
_TARGET_DIAGRAM_TYPES = Literal[
    "ClassDiagram",
    "ObjectDiagram",
    "StateMachineDiagram",
    "AgentDiagram",
    "GUINoCodeDiagram",
    "QuantumCircuitDiagram",
    "BPMN",
    # User Profile must be nameable here too: otherwise a request only reaches
    # its handler when KEYWORD_TARGETS catch "user profile" / "persona" /
    # "target user" at layer 2, and other phrasing falls through to the fallback.
    "UserDiagram",
    "ComponentDiagram",
    "DeploymentDiagram",
]


class UnifiedClassification(BaseModel):
    """Everything the state machine needs to route a message — in one shot."""

    intent: _INTENT_NAMES = Field(
        ...,
        description=(
            "State-level intent — pick exactly one. The SYSTEM PROMPT is the "
            "authoritative rulebook (single source of truth); one-line "
            "summaries: hello_intent = greeting/small-talk/thanks; "
            "create_complete_system_intent = new diagram or system from "
            "scratch; modify_model_intent = add/remove/change elements of an "
            "existing diagram; describe_model_intent = question or advice "
            "about the current diagram; modeling_help_intent = conceptual "
            "help, or how to run already-generated code; uml_spec_intent = "
            "formal UML specification; generation_intent = source code / "
            "export / deploy; decline_intent = user opts out ('nothing', 'no "
            "thanks'); out_of_scope_intent = wants a non-software artifact "
            "(picture/poem/joke); meta_question_intent = asks what the "
            "assistant can do or why to use it; fallback_intent = none fit."
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
            "Active Record') and any non-functional requirements the user "
            "mentioned. Invent nothing: authentication, login, sign-up, "
            "roles, JWT, responsive design, styling and navigation go in "
            "only if the user's own words asked for them, because the "
            "generator builds whatever this field names. Keep the user's "
            "concrete nouns (status values, named operations, stated rules). "
            "Max 2000 chars. Do NOT describe the class diagram in detail — "
            "the generator has the domain model."
        ),
    )
    provider: Literal["anthropic", "openai"] = Field(
        default="anthropic",
        description=(
            "Suggested LLM provider when generation_route='smart'. Ignored "
            "for other routes. The frontend's BYOK dropdown can override."
        ),
    )

    # --- Target fields (any intent; see TARGET in the system prompt) ---

    target_kind: Optional[Literal["web_app", "backend", "other"]] = Field(
        default=None,
        description=(
            "What KIND of software the user wants built, when the USER MESSAGE "
            "states it (any intent): 'web_app' = anything with a browser UI "
            "(incl. mobile-friendly/responsive web app, PWA, website, "
            "dashboard); 'backend' = API-only / headless service (incl. 'a REST "
            "API for my mobile app'); 'other' = a NAMED non-browser platform "
            "or form factor (native mobile, desktop/Windows/Mac, CLI, game, "
            "browser extension, embedded, library/SDK, bot). NULL unless the "
            "message names a platform or form factor: a bare 'app' / "
            "'application' plus a domain ('an app to manage a library') is NULL."
        ),
    )
    target_description: Optional[str] = Field(
        default=None,
        description=(
            "When target_kind is set: a SHORT English noun phrase naming what "
            "will be built, e.g. 'native Android/iOS app', 'Windows 11 desktop "
            "app', 'Rust CLI', 'Chrome extension', 'web app'. Max 60 chars. "
            "NULL otherwise."
        ),
    )
    architecture: Optional[str] = Field(
        default=None,
        description=(
            "A software architecture / pattern the USER MESSAGE explicitly "
            "asks for ('MVC', 'MVVM', 'hexagonal', 'clean architecture', "
            "'microservices'), as a short name. Any intent. NULL otherwise."
        ),
    )
    reply_language: Optional[str] = Field(
        default=None,
        description=(
            "ISO 639-1 code (e.g. 'en', 'es', 'fr', 'de', 'it') of the "
            "language the USER MESSAGE is written in. For a message too short "
            "or language-neutral to tell ('ok', '1', a class name) use the "
            "CONVERSATION LANGUAGE line when present, else NULL."
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

    model_disposition: Optional[Literal[
        "extend_existing",
        "replace_existing",
        "new_tab",
        "reuse_for_generation",
        "new_from_scratch",
    ]] = Field(
        default=None,
        description=(
            "How the request relates to the EXISTING workspace model — judge "
            "from WORKSPACE CONTEXT (does a usable model already exist?) plus "
            "the wording. Populate for create / modify / generation "
            "intents:\n"
            "  'extend_existing'      — add to / change the current model "
            "('add a Payment class', 'also include returns', 'expand this', "
            "'add auth to my model').\n"
            "  'replace_existing'     — user EXPLICITLY wants to discard the "
            "current model and start over ('scrap this and start fresh with', "
            "'replace it with a ...', 'redo it as').\n"
            "  'new_tab'              — build a SEPARATE new diagram alongside "
            "('in a new tab', 'a second diagram for', 'also model a ...').\n"
            "  'reuse_for_generation' — generate code / an app FROM the "
            "existing model WITHOUT changing it ('generate X from my model', "
            "'build an app from this', 'from my class diagram', 'use what I "
            "have', 'the diagram I've been working on').\n"
            "  'new_from_scratch'     — no usable model exists yet, OR the "
            "user clearly describes a brand-new system to build from nothing.\n"
            "Leave NULL ONLY when genuinely ambiguous — the agent will ask "
            "rather than guess destructively."
        ),
    )

    needs_clarification: bool = Field(
        default=False,
        description=(
            "Set True ONLY when acting would require a GUESS: no resolvable "
            "referent ('do the thing', 'make it better', 'fix it'), an "
            "unresolved pronoun with no antecedent ('rename it', 'connect "
            "them' with nothing prior to point at), two materially different "
            "readings, or a missing target the WORKSPACE CONTEXT cannot "
            "supply. Do NOT set it for requests you can reasonably act on, "
            "for plain confirmations ('yes'/'no'), or merely because a "
            "request is broad — bias toward acting when a sensible default "
            "exists. When True, also write clarifying_question."
        ),
    )
    clarifying_question: Optional[str] = Field(
        default=None,
        description=(
            "REQUIRED when needs_clarification=True. ONE short, specific "
            "question that, once answered, lets you act — reference the "
            "workspace where useful (e.g. 'Which class should I add the email "
            "attribute to — Customer or Order?'). Max 200 chars. LEAVE NULL "
            "otherwise."
        ),
    )

    # --- Pending-flow fields (populated when the user block carries a
    # PENDING QUESTION — i.e. the assistant just asked something) ---

    pending_flow_action: Optional[Literal["answer", "new_request"]] = Field(
        default=None,
        description=(
            "REQUIRED when the user block contains a PENDING QUESTION: "
            "'answer' if the USER MESSAGE replies to that question (however "
            "phrased); 'new_request' if it is an unrelated instruction or "
            "request. None when no question is pending."
        ),
    )
    pending_flow_answer: Optional[Literal[
        "replace", "keep", "new_tab", "auto", "llm", "confirm", "cancel",
        "other_answer",
    ]] = Field(
        default=None,
        description=(
            "When pending_flow_action='answer': the VALID ANSWER the message "
            "maps to (listed with the question). 'other_answer' for free-form "
            "values such as configuration input."
        ),
    )

    reason: str = Field(
        ...,
        description=(
            "One short sentence (max 160 chars) explaining the classification. "
            "Used for logs and surfaced to users as a hint."
        ),
    )


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------




def classify_message(
    request: AssistantRequest,
    llm_provider: Any,
    history: Optional[list] = None,
    recent_smart_gen: bool = False,
    pending_flow: Optional[dict] = None,
    remembered_target: Optional[dict] = None,
    conversation_language: Optional[str] = None,
) -> UnifiedClassification:
    """Classify a user message into a state-level intent + sub-routing fields.

    ONE classifier-tier structured-output call (see ``model_config``).
    Returns a safe
    ``fallback_intent`` classification if the provider is unavailable
    or the call fails — the caller should trust the returned object
    and dispatch based on ``intent``.

    ``history`` is an optional list of prior ``{"role", "content"}`` turns
    (most recent last) so the classifier can resolve referents like "the
    same", "continue", "it", or "do that for all" against what was just
    asked/done — this is what makes the agent feel like it remembers the
    session instead of treating every message cold.

    Never raises — the classifier must never crash the agent.
    """
    if llm_provider is None:
        return _safe_fallback("LLM provider unavailable")
    message = (request.message or "").strip()
    if not message:
        return _safe_fallback("empty message")

    auto_fix = auto_fix_verdict(message)
    if auto_fix is not None:
        return UnifiedClassification(**auto_fix)

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_block(
                request, history, recent_smart_gen, pending_flow,
                remembered_target, conversation_language,
            )},
        ]
        # Reasoning models (gpt-5* / o-series) burn hidden reasoning tokens
        # from the SAME completion budget. With only 800 tokens the visible
        # structured output is starved → parsed=None → _safe_fallback on
        # EVERY message. Give reasoning models headroom; keep the tight
        # budget for fast non-reasoning models like gpt-4o-mini.
        from llm.model_config import supports_custom_temperature

        from llm import byok
        classifier_model = getattr(llm_provider, "model_name", "") or ""
        cfg = get_current()
        if cfg is not None:
            # A user key sends this call to the user's model; size for that one.
            classifier_model = byok.resolve_model(cfg.provider, classifier_model, cfg.model)
        max_tokens = 800 if supports_custom_temperature(classifier_model) else 4000
        result: UnifiedClassification = llm_provider.parse(
            messages=messages,
            schema=UnifiedClassification,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        if result is None:
            return _safe_fallback("LLM returned no result")
        # The generated app keeps its target: a follow-up such as "add a
        # settings screen" edits that app's code, not the web GUI diagram.
        follow_up_kind = (remembered_target or {}).get("kind") if recent_smart_gen else None
        ctx = getattr(request, "context", None)
        return _post_validate(
            result, message, follow_up_target_kind=follow_up_kind,
            has_class_model=ctx is None or bool(_extract_class_names(ctx)),
        )
    except Exception:
        logger.exception("classify_message failed; falling back to fallback_intent")
        return _safe_fallback("LLM classifier failed")


def _pending_flow_context(session: Any) -> Optional[dict]:
    """Describe the question the assistant is currently awaiting an answer
    to: the first open dialog in ``pending_dialogs.PRECEDENCE``, the same
    order ``resume_pending`` hands it the message in. Returns
    ``{kind, question, valid_answers}`` or ``None``.
    """
    try:
        dialog = pending_dialogs.current(session)
        if dialog is None:
            return None
        if dialog.kind == pending_dialogs.GUI_CHOICE:
            return {
                "kind": "gui_choice",
                "question": (
                    "How should I create your screens — 1 basic CRUD pages "
                    "(one page per class), or 2 experimental AI design? "
                    "Answer 'auto' = option 1, basic CRUD pages; 'llm' = "
                    "option 2, experimental AI design (custom AI-designed "
                    "screens); 'cancel' = neither."
                ),
                "valid_answers": ["auto", "llm", "cancel"],
            }
        if dialog.kind == pending_dialogs.SYSTEM_CONFIRMATION:
            pending_cs = dialog.payload
            if isinstance(pending_cs, dict) and pending_cs.get("destructive_modify"):
                return {
                    "kind": "destructive_confirm",
                    "question": (
                        "This modification would delete most or all of the "
                        "existing model — do you want to proceed?"
                    ),
                    # 'confirm' maps to proceeding; 'replace' tolerated as a
                    # synonym by the handler.
                    "valid_answers": ["confirm", "cancel"],
                }
            _can_tab = bool(isinstance(pending_cs, dict) and pending_cs.get("can_add_tab"))
            return {
                "kind": "replace_confirm",
                "question": (
                    "A model already exists for this diagram — should I "
                    "replace it, keep it and add alongside"
                    + (", create it in a new diagram tab" if _can_tab else "")
                    + ", or cancel?"
                ),
                "valid_answers": (
                    ["replace", "keep", "new_tab", "cancel"] if _can_tab
                    else ["replace", "keep", "cancel"]
                ),
            }
        if dialog.kind == pending_dialogs.SMART_GEN:
            return {
                "kind": "smart_confirm",
                "question": (
                    "Do you want me to continue and run the Spec-Driven "
                    "generator on your model?"
                ),
                "valid_answers": ["confirm", "cancel"],
            }
        pending_gen = dialog.payload
        if pending_gen == "_awaiting_selection":
            return {
                "kind": "generator_menu",
                "question": (
                    "Which code generator should I run? (I listed the "
                    "available options.)"
                ),
                "valid_answers": ["other_answer", "cancel"],
            }
        _gen_cfg = pending_dialogs.get(session, PENDING_GENERATOR_CONFIG)
        if isinstance(_gen_cfg, dict) and _gen_cfg.get(PLAN_GENERATION_CONFIRM_FLAG):
            # Mixed-plan pause: the model was just built and the user was
            # asked whether to review/refine it or continue with the
            # planned code generation (see execution/planning.py).
            return {
                "kind": "generate_confirm",
                "question": (
                    f"The model is ready — I asked whether to continue "
                    f"with {pending_gen} code generation or keep refining "
                    "the model. A bare acknowledgment ('ok', 'sure', "
                    "'go ahead') means continue with generation."
                ),
                "valid_answers": ["confirm", "cancel"],
            }
        return {
            "kind": "generator_config",
            "question": (
                f"I asked for the {pending_gen} generator's remaining "
                "configuration (e.g. project name / dialect / options)."
            ),
            "valid_answers": ["other_answer", "cancel"],
        }
    except Exception:  # pragma: no cover — a session stub without .get
        return None


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

    frontend_event = frontend_event_verdict(request)
    if frontend_event is not None:
        result = UnifiedClassification(**frontend_event)
    else:
        _pending_flow = _pending_flow_context(session)
        # Project-scoped: the SMART-GEN FOLLOW-UP rule is about the app in the
        # project the user is looking at. LAST_SMART_GEN_AT alone survives a
        # project switch (the BAF session is keyed on the per-browser user_id),
        # so using it raw would tell the classifier "you just generated an app" in
        # a brand-new project.
        _recent_smart = recent_smart_gen_for_project(session, 15 * 60)
        _project_id = request_project_id(request)
        result = classify_message(
            request, llm_provider, _recent_history(session, request),
            recent_smart_gen=_recent_smart,
            pending_flow=_pending_flow,
            remembered_target=target_for_project(session, _project_id),
            conversation_language=session.get(REPLY_LANGUAGE),
        )
        result = _generate_button_guard(result, request, session)
        _log_classification(request, result)
        _remember_stated_target_and_language(session, request, result, _project_id)
    if event_id is not None:
        session.set(UNIFIED_CLASSIFICATION, result)
        session.set(UNIFIED_CLASSIFICATION_EVENT_ID, event_id)
    return result


def _remember_stated_target_and_language(
    session: Any, request: AssistantRequest, result: UnifiedClassification,
    project_id: Optional[str],
) -> None:
    """Keep the target the user stated for this project and the language they
    write in. Never raises."""
    try:
        remember_target(
            session, project_id, kind=result.target_kind,
            description=result.target_description, architecture=result.architecture,
        )
        # A quick-action click sends the button's English prompt: it says
        # nothing about the language the user writes in.
        clicked = (request.message or "").strip() in (session.get(LAST_SUGGESTED_PROMPTS) or ())
        if result.reply_language and not clicked:
            session.set(REPLY_LANGUAGE, result.reply_language)
    except Exception:  # pragma: no cover - best effort
        logger.debug("Remembering target/language failed", exc_info=True)


def _log_classification(
    request: AssistantRequest, result: UnifiedClassification
) -> None:
    """One concise INFO line per real classification — the agent's routing
    decision. Makes "why did it route there?" answerable from the logs. Never raises."""
    try:
        msg = (request.message or "").strip().replace("\n", " ")
        if len(msg) > 80:
            msg = msg[:77] + "..."
        bits = [f"intent={result.intent}"]
        if result.generation_route:
            bits.append(f"route={result.generation_route}")
        if result.generator_type:
            bits.append(f"gen={result.generator_type}")
        if result.target_diagram_type:
            bits.append(f"target={result.target_diagram_type}")
        if result.model_disposition:
            bits.append(f"disp={result.model_disposition}")
        if result.domain_mismatch:
            bits.append("domain_mismatch=True")
        if result.needs_clarification:
            bits.append("clarify=True")
        if result.target_kind:
            bits.append(f"app_target={result.target_kind}:{result.target_description!r}")
        if result.architecture:
            bits.append(f"arch={result.architecture!r}")
        if result.reply_language:
            bits.append(f"lang={result.reply_language}")
        logger.info("[classify] %s | %s", " ".join(bits), msg)
    except Exception:  # pragma: no cover - logging must never break routing
        pass


# The "Generate the code" suggestion offered after the agent builds a class
# diagram (suggestions.py). gpt-6-luna sent it to the generator menu
# (deterministic, generator_type=None) in 4 of 12 live runs; a rulebook rule
# scoped to "right after a build" also moved the hand-drawn and no-model
# readings of the same words, so the scope is checked here instead.
_GENERATE_BUTTON_PROMPT = "generate the application"


def _generate_button_guard(
    result: UnifiedClassification, request: AssistantRequest, session: Any
) -> UnifiedClassification:
    """Route the post-build button prompt to the Spec-Driven confirmation when
    the agent built this project's class diagram from the user's own request.
    Any other context keeps the classifier's verdict."""
    message = re.sub(r"[\s.!]+$", "", (request.message or "").strip().lower())
    if (
        message != _GENERATE_BUTTON_PROMPT
        or result.intent != "generation_intent"
        or result.generation_route == "smart"
    ):
        return result
    from utilities.original_request import original_request_for_project, request_project_id

    if not original_request_for_project(session, request_project_id(request)):
        return result
    logger.warning(
        "Generate-button guard: %s/%s -> smart (agent-built model in this project)",
        result.generation_route, result.generator_type,
    )
    return result.model_copy(update={
        "generation_route": "smart",
        "generator_type": None,
        "model_disposition": "reuse_for_generation",
        # The user's original request rides along separately
        # (generation_handler._original_for_smart_generation).
        "refined_instructions": (result.refined_instructions or "").strip()
        or "Build the application from the current class diagram.",
        "reason": "Generate-button guard: post-build 'generate the application' -> smart",
    })


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

# How many prior turns to feed the classifier, and how much of each.
_HISTORY_MAX_TURNS = 6
_HISTORY_MAX_CHARS = 240


def _recent_history(session: Any, request: AssistantRequest) -> Optional[list]:
    """Best-effort: the last few conversation turns for referent resolution.

    Read at classification time — which is BEFORE ``_common_preamble``
    records the *current* user message — so this returns the PRIOR turns
    only, exactly the context needed to resolve referents in the current
    message ("the same", "continue", "do it for all", ...).

    Lazy import keeps ``unified_classifier`` importable without the memory
    stack and sidesteps any import cycle. Never raises.
    """
    try:
        from memory import get_memory, memory_session_key

        mem = get_memory(memory_session_key(session, request))
        return mem.get_last_n(_HISTORY_MAX_TURNS)
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("recent_history unavailable (best-effort): %s", exc)
        return None


def _history_lines(history: Optional[list]) -> list:
    """Render prior turns as compact ``role: content`` lines (oldest first)."""
    if not history:
        return []
    out: list = []
    for turn in history[-_HISTORY_MAX_TURNS:]:
        if not isinstance(turn, dict):
            continue
        role = (turn.get("role") or "?").strip()
        content = (turn.get("content") or "").strip().replace("\n", " ")
        if not content:
            continue
        if len(content) > _HISTORY_MAX_CHARS:
            content = content[: _HISTORY_MAX_CHARS - 3].rstrip() + "..."
        out.append(f"{role}: {content}")
    return out


def _build_user_block(
    request: AssistantRequest, history: Optional[list] = None,
    recent_smart_gen: bool = False,
    pending_flow: Optional[dict] = None,
    remembered_target: Optional[dict] = None,
    conversation_language: Optional[str] = None,
) -> str:
    """Compose the user message + recent conversation + workspace context."""
    lines = ["USER MESSAGE:", request.message or ""]

    if remembered_target:
        # See REMEMBERED TARGET in the system prompt.
        parts = []
        if remembered_target.get("kind"):
            parts.append(
                f"target_kind={remembered_target['kind']}, "
                f"target_description={remembered_target.get('description') or ''!r}"
            )
        if remembered_target.get("architecture"):
            parts.append(f"architecture={remembered_target['architecture']!r}")
        lines.append("")
        lines.append(
            "REMEMBERED TARGET (stated earlier for this project): " + "; ".join(parts)
        )
    if conversation_language:
        lines.append("")
        lines.append(f"CONVERSATION LANGUAGE: {conversation_language}")

    if pending_flow:
        # The assistant is awaiting a reply — see PENDING QUESTION HANDLING
        # in the system prompt.
        lines.append("")
        lines.append(
            "PENDING QUESTION (the assistant just asked this and is awaiting "
            "a reply):"
        )
        lines.append(f"\u00ab{pending_flow.get('question', '')}\u00bb")
        lines.append(
            "VALID ANSWERS: "
            + ", ".join(pending_flow.get("valid_answers", []))
            + "  (set pending_flow_action='new_request' if the message is "
            "not a reply to this)"
        )

    if recent_smart_gen:
        # Structured recency signal (set when a smart generation completes) —
        # the primary cue for the SMART-GEN FOLLOW-UP rule, robust against
        # reply-copy rewording in the history snippets below.
        lines.append("")
        lines.append(
            "RECENT SMART GENERATION: a smart / Spec-Driven generation "
            "completed in this conversation within the last few minutes."
        )

    # Recent conversation turns so the classifier can resolve referents
    # ("the same", "continue", "do that for all", "it", "that one")
    # against what was just asked/done — the difference between an agent
    # that remembers the session and one that treats every message cold.
    hist_lines = _history_lines(history)
    if hist_lines:
        lines.append("")
        lines.append(
            "RECENT CONVERSATION (oldest first; resolve referents such as "
            '"the same", "continue", "it", "do that for all" against it, '
            "but classify the USER MESSAGE above — not these prior turns):"
        )
        lines.extend(hist_lines)

    ctx = getattr(request, "context", None)
    if ctx is None:
        return "\n".join(lines)

    summary_lines = []
    active_type = getattr(ctx, "active_diagram_type", None)
    if active_type:
        summary_lines.append(f"- active diagram: {active_type}")
    # FULL editor content: every relevant diagram type with its element
    # count + a few element names, and an explicit "not present" line for
    # missing ones — so the classifier can reason about what already exists
    # and about prerequisites (e.g. a webapp request when no GUI exists).
    summary_lines.extend(_workspace_summary_lines(ctx))
    # Full class-name list (up to 30) for the domain_mismatch judgement.
    class_names = _extract_class_names(ctx)
    if class_names:
        summary_lines.append(
            "- all class names: " + ", ".join(class_names[:30])
        )

    if summary_lines:
        lines.append("")
        lines.append("WORKSPACE CONTEXT (what is currently in the editor):")
        lines.extend(summary_lines)
    return "\n".join(lines)


# Sub-element types whose names are member rows, not top-level diagram
# elements — excluded from the per-diagram name preview.
_NAME_SKIP_TYPES = {
    "ClassAttribute", "ClassMethod", "ClassOCLConstraint",
    "ObjectAttribute", "ObjectMethod",
}

# The diagram types worth surfacing to the classifier, with a human unit.
_RELEVANT_DIAGRAM_TYPES = [
    ("ClassDiagram", "class(es)"),
    ("ObjectDiagram", "object(s)"),
    ("StateMachineDiagram", "state(s)"),
    ("AgentDiagram", "agent element(s)"),
    ("GUINoCodeDiagram", "GUI element(s)"),
    ("QuantumCircuitDiagram", "quantum element(s)"),
    ("BPMN", "process element(s)"),
    ("UserDiagram", "user profile element(s)"),
    ("ComponentDiagram", "component element(s)"),
    ("DeploymentDiagram", "deployment element(s)"),
]


def _diagram_element_names(model: Any, limit: int = 8) -> tuple[int, list[str]]:
    """(top-level element count, up to ``limit`` element names) for a model.

    Skips member rows (attributes/methods) and child elements so the names
    are the meaningful top-level entities. Never raises.
    """
    if not isinstance(model, dict):
        return 0, []
    elements = model.get("elements")
    if not isinstance(elements, dict):
        return 0, []
    count = 0
    names: list[str] = []
    for elem in elements.values():
        if not isinstance(elem, dict):
            continue
        if elem.get("type") in _NAME_SKIP_TYPES:
            continue
        if elem.get("owner"):
            continue
        count += 1
        name = elem.get("name")
        if isinstance(name, str) and name.strip() and len(names) < limit:
            names.append(name.strip())
    return count, names


def _workspace_summary_lines(ctx: Any) -> list[str]:
    """Per-diagram-type summary of everything currently in the editor.

    One line per relevant diagram type: present types show their element
    count + a few names (across tabs); missing types show "not present".
    Never raises.
    """
    out: list[str] = []
    for dtype, unit in _RELEVANT_DIAGRAM_TYPES:
        try:
            diagrams = ctx.get_all_diagrams_of_type(dtype)
        except Exception:
            diagrams = []
        total = 0
        names: list[str] = []
        for d in diagrams:
            if not isinstance(d, dict):
                continue
            c, n = _diagram_element_names(d.get("model"))
            total += c
            for name in n:
                if name not in names and len(names) < 8:
                    names.append(name)
        if total > 0:
            tabs = f" across {len(diagrams)} tabs" if len(diagrams) > 1 else ""
            shown = ", ".join(names)
            more = f", +{total - len(names)} more" if names and total > len(names) else ""
            detail = f" ({shown}{more})" if shown else ""
            out.append(f"- {dtype}: {total} {unit}{tabs}{detail}")
        else:
            out.append(f"- {dtype}: not present")
    return out


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


# Words that unambiguously signal a DEPLOY request in the safety nets
# below. Checked alongside "no export/save/download wording" so we never
# override a message that could genuinely mean export.
_DEPLOY_WORDS = ("deploy", "publish", "push", "ship", "go live", "launch")
_EXPORT_WORDS = ("export", "save", "download")


def _looks_like_unambiguous_deploy(lower_message: str) -> bool:
    """True when *lower_message* clearly means DEPLOY, not EXPORT.

    Used by :func:`_post_validate` to repair the classic mix-up where a
    deploy-shaped message ("deploy this model", "push this to prod") is
    mis-typed as ``generator_type='export'`` (structurally similar to
    "export the model") or dropped into the ``'other'`` (refusal) route.
    """
    return any(w in lower_message for w in _DEPLOY_WORDS) and not any(
        w in lower_message for w in _EXPORT_WORDS
    )


def _post_validate(
    result: UnifiedClassification, message: str = "", *,
    follow_up_target_kind: Optional[str] = None,
    has_class_model: bool = True,
) -> UnifiedClassification:
    """Defensive validation of LLM output.

    Catches the classic "LLM returned route='smart' but forgot to write
    instructions" and "intent='generation' but no generation_route"
    bugs. Repairs them in place rather than collapsing a valid intent to a
    worse one. Every rebuilt verdict keeps the target / language fields
    (see :func:`_rebuilt`).
    """
    result = _normalize_target_fields(result)
    # A "no" that names a non-web app to build instead ("no, una app nativa,
    # no una web app") is a correction, never a cancel. Live, the classifier
    # returned decline_intent for exactly that message while setting
    # target_kind, so its own field is the signal. Web and backend targets
    # have built-in generators and keep the classifier's verdict.
    if result.target_kind == "other" and (
        result.intent == "decline_intent" or result.pending_flow_answer == "cancel"
    ):
        logger.warning(
            "decline_intent that names a non-web target; treating it as a "
            "corrected %s request", "generation" if has_class_model else "create",
        )
        if not has_class_model:
            return _rebuilt(
                result,
                intent="create_complete_system_intent",
                target_diagram_type="ClassDiagram",
                model_disposition="new_from_scratch",
                pending_flow_action="new_request",
                reason="Correction with no model yet: build the model for the new target",
            )
        return _rebuilt(
            result,
            intent="generation_intent",
            generation_route="smart",
            model_disposition="reuse_for_generation",
            refined_instructions=(result.refined_instructions or message or "").strip(),
            pending_flow_action="new_request",
            reason="Correction: 'no' followed by a new target updates the request",
        )
    # Continue-from-GitHub guard (chat path for "continue from a repo"):
    # "continue from github.com/x/y" carries no modeling vocabulary, so the
    # LLM parks it in fallback ("none fit") or reads "continue" as a modify
    # of the current model — either way it never reaches the generation
    # handler, whose own deterministic guard emits the trigger_github_import
    # action. A message naming BOTH a GitHub repo reference AND a
    # continuation verb is unambiguous; reroute it deterministically. The
    # verb requirement keeps e.g. "create a diagram like github.com/x/y"
    # on its classified intent. pending_flow_action='new_request' frees the
    # message from any pending question's answer gate.
    if result.intent != "generation_intent":
        if _GITHUB_CONTINUE_VERB_RE.search(message or "") and (
            _extract_github_reference(message or "") is not None
        ):
            logger.warning(
                "Continue-from-GitHub message classified as %s; rerouting to "
                "generation_intent for the github-import guard", result.intent,
            )
            return _rebuilt(
                result,
                intent="generation_intent",
                generation_route="other",
                pending_flow_action="new_request",
                reason="GitHub-continuation guard: continue-from-repo routes to the generation handler",
            )
    # Data-model-first guard: a from-scratch build request can classify as
    # create_complete_system with target=GUINoCodeDiagram (the LLM inherits
    # the active tab / reads "app" as "screens"), which would jump straight to
    # AI-generating screens on an EMPTY data model — no class diagram, and
    # without asking the deterministic-vs-AI choice (that choice is gated on
    # an existing class diagram downstream).
    #
    # An APPLICATION request ("todo app", "webapp", "dashboard", "a system to
    # track X") is data-first: build the ClassDiagram (the specs) first; the user
    # reviews it, then generates the screens/code as an explicit next step (at
    # which point the GUI-mode choice fires because a class diagram now exists).
    # ONLY an explicit SCREEN/GUI request — naming screens, gui, ui, pages, a
    # mockup/wireframe, the frontend — goes straight to GUINoCodeDiagram.
    if (
        result.intent == "create_complete_system_intent"
        and result.target_diagram_type == "GUINoCodeDiagram"
        and result.model_disposition == "new_from_scratch"
        and not EXPLICIT_SCREEN_VOCAB_RE.search(message or "")
    ):
        logger.warning(
            "Data-model-first guard: from-scratch build request with no explicit "
            "screen/GUI vocabulary — retargeting create_complete_system "
            "GUINoCodeDiagram -> ClassDiagram (build the data model/specs first, "
            "not screens)"
        )
        return _rebuilt(
            result,
            intent="create_complete_system_intent",
            target_diagram_type="ClassDiagram",
            model_disposition=result.model_disposition,
            pending_flow_action=result.pending_flow_action,
            reason="Data-model-first: a from-scratch app/system request builds the ClassDiagram; screens come as an explicit next step",
        )
    if result.intent == "generation_intent":
        # GUI-diagram guard: "generate the GUI (model) / the
        # screens / the UI" means building the GUI DIAGRAM, but the LLM
        # sometimes reads it as web-app code generation — which then stashes
        # a smart-gen confirmation that re-creates itself on every retry
        # ("no, generate the gui model" loops the same prompt). Naming the
        # gui/screens/ui WITHOUT naming an app/code artifact is a modeling
        # request, deterministically.
        _low = (message or "").lower()
        # Both GUI-diagram guards below assume a web app: the GUI editor
        # designs web pages. A non-web app keeps its follow-ups in its code.
        _web_gui_guards = follow_up_target_kind != "other"
        if (
            _web_gui_guards
            and re.search(r"\b(gui|screens?|user interface|\bui\b)\b", _low)
            and re.search(r"\b(generate|create|build|make|design)\b", _low)
            and not re.search(
                r"\b(web ?app|application|app\b|code|backend|frontend|"
                r"django|sql|api)\b", _low)
        ):
            logger.warning(
                "Request asks for the GUI diagram, not code; rerouting "
                "generation_intent -> create_complete_system_intent (GUI)"
            )
            return _rebuilt(
                result,
                intent="create_complete_system_intent",
                target_diagram_type="GUINoCodeDiagram",
                model_disposition=result.model_disposition,
                pending_flow_action="new_request",
                reason="GUI-diagram guard: 'generate the gui/screens' is modeling, not code generation",
            )
        # Same guard, EDIT form ("add a Reports page" with an existing GUI
        # must not stash a spec-driven web-app confirmation). Adding/
        # removing/renaming a page or screen edits the GUI DIAGRAM — it is
        # never a code-generation request.
        if (
            _web_gui_guards
            and re.search(r"\b(pages?|screens?)\b", _low)
            and re.search(r"\b(add|insert|remove|delete|rename)\b", _low)
            and not re.search(
                r"\b(web ?app|application|code|backend|frontend|"
                r"django|sql|api)\b", _low)
        ):
            logger.warning(
                "Request edits GUI pages, not code; rerouting "
                "generation_intent -> modify_model_intent (GUI)"
            )
            return _rebuilt(
                result,
                intent="modify_model_intent",
                target_diagram_type="GUINoCodeDiagram",
                model_disposition="extend_existing",
                pending_flow_action="new_request",
                reason="GUI-diagram guard: adding/removing a page edits the GUI diagram, not code",
            )
        # Unsupported-language guard (see _names_unsupported_stack): BESSER's
        # deterministic generators only cover Python/Java (+ SQL/Django/...).
        # When the user explicitly asks for a language BESSER can't generate
        # (C, C++, Rust, Kotlin, Go, ...), the LLM maps it to the nearest
        # built-in (c->java, c++->python). Force the smart / from-scratch route
        # so the LLM writes the actual language instead.
        if (
            result.generation_route in ("deterministic", "smart")
            and result.generator_type not in ("export", "deploy")
            and _names_unsupported_stack(message or "")
        ):
            if result.generation_route != "smart":
                logger.warning(
                    "Request names a language/stack with no BESSER generator; "
                    "forcing smart/from-scratch (was route=%s type=%s)",
                    result.generation_route,
                    result.generator_type,
                )
            result.generation_route = "smart"
            result.generator_type = None
            if not (result.refined_instructions or "").strip():
                result.refined_instructions = (message or "").strip() or (
                    "Generate the requested code from the current model."
                )
        if result.generation_route is None:
            logger.warning(
                "LLM returned generation_intent with no generation_route; "
                "demoting to fallback_intent"
            )
            return _rebuilt(
                result,
                intent="fallback_intent",
                reason="classifier missed generation sub-routing",
            )
        if result.generation_route == "smart":
            lower_msg = (message or "").lower()
            # Safety net: "smartdata" is a deterministic BESSER built-in
            # NAME that happens to contain the substring "smart" — the
            # LLM sometimes lexically confuses that with the 'smart'
            # (spec-driven) route. Only correct it when the message names
            # smartdata AND doesn't also ask for something the template
            # can't do (those genuinely belong on the smart path).
            _smart_extras = (
                "auth", "jwt", "oauth", "docker", "container", "migrat",
                "rate limit", "middleware", "cors", "ci/cd", "role",
                "permission", "login", "signup", "sign-in",
            )
            if re.search(r"\bsmart\s*data\b", lower_msg) and not any(
                w in lower_msg for w in _smart_extras
            ):
                logger.warning(
                    "LLM routed 'smartdata' request to smart via substring "
                    "confusion; correcting to deterministic/smartdata"
                )
                result.generation_route = "deterministic"
                result.generator_type = "smartdata"
            elif not (result.refined_instructions or "").strip():
                # Keep the smart route — do NOT collapse to the
                # deterministic generator menu, which silently kills requests
                # like "dashboard pls" or "vibe-code me something cool from my
                # model". Fall back to the raw user message as the generator
                # instructions so the smart path still runs.
                logger.warning(
                    "LLM returned smart route with no refined_instructions; "
                    "synthesizing instructions from the user message"
                )
                result.refined_instructions = (
                    (message or "").strip()
                    or "Build a custom application from the current model."
                )
        elif result.generation_route == "deterministic":
            # Safety net: a DEPLOY-shaped message ("deploy this model",
            # "push this to prod") sometimes gets mis-typed as 'export'
            # (structurally similar to "export the model") or left with
            # generator_type=None (falls through to the generic menu).
            lower_msg = (message or "").lower()
            if result.generator_type in (None, "export") and _looks_like_unambiguous_deploy(
                lower_msg
            ):
                logger.warning(
                    "LLM routed a deploy-shaped message to generator_type=%s; "
                    "correcting to deploy",
                    result.generator_type,
                )
                result.generator_type = "deploy"
            # Otherwise generator_type may legitimately be None — the
            # caller will show the generator menu. That's fine; no
            # demotion needed.
        elif result.generation_route == "other":
            # Safety net: an unambiguous deploy request ("go ahead and
            # deploy it", "push this to prod") sometimes gets classified
            # as 'other' (no clear code-generation request) and the agent
            # refuses instead of deploying. Promote it the same way as
            # the 'deterministic' branch above.
            lower_msg = (message or "").lower()
            if _looks_like_unambiguous_deploy(lower_msg):
                logger.warning(
                    "LLM routed a deploy-shaped message to 'other'; "
                    "correcting to deterministic/deploy"
                )
                result.generation_route = "deterministic"
                result.generator_type = "deploy"
    return result


# ISO 639-1, optionally with a region ("es", "pt-BR").
_LANGUAGE_CODE_RE = re.compile(r"^([a-z]{2})(?:[-_][a-z]{2})?$")


def _normalize_target_fields(result: UnifiedClassification) -> UnifiedClassification:
    """Drop malformed target / language values (shape checks only)."""
    match = _LANGUAGE_CODE_RE.match((result.reply_language or "").strip().lower())
    result.reply_language = match.group(1) if match else None
    desc = (result.target_description or "").strip()
    result.target_description = desc[:120] if result.target_kind and desc else None
    arch = (result.architecture or "").strip()
    result.architecture = arch[:60] or None
    return result


def _rebuilt(result: UnifiedClassification, **fields: Any) -> UnifiedClassification:
    """A guard's replacement verdict, keeping what the user said about the
    target and the language they write in."""
    return UnifiedClassification(
        target_kind=result.target_kind,
        target_description=result.target_description,
        architecture=result.architecture,
        reply_language=result.reply_language,
        **fields,
    )


def _safe_fallback(reason: str) -> UnifiedClassification:
    """Safe default when the LLM is unavailable.

    Returns ``fallback_intent`` so the agent runs its existing
    fallback state body. This is the pre-LLM-classifier behaviour —
    user gets a helpful message asking them to rephrase.
    """
    # The "[classifier-error]" tag lets downstream adapters distinguish an
    # ERROR fallback (this function) from the LLM's deliberate
    # none-of-the-above verdict — see routing.guards.generation_route.
    return UnifiedClassification(
        intent="fallback_intent", reason=f"[classifier-error] {reason}")


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
