"""
State Bodies & Transition Wiring
---------------------------------
All BESSER agent state-body functions and the helper that wires
intent → state transitions.

Call :func:`register_all` from ``modeling_agent.py`` after states
and intents have been created.
"""

import functools
import json
import logging
import re
from typing import Any, Optional

from baf.core.session import Session
from baf.library.transition.events.base_events import ReceiveJSONEvent
from baf.nlp.rag.rag import RAGMessage

from common.agent_config import LLM_MAX_TOKENS_TEXT, LLM_TEXT_TEMPERATURE
from conversation import agent_context as ctx
from llm.model_config import MODEL_GENERATION_SMALL
from llm.provider import LLMProvider
from protocol.adapters import parse_assistant_request
from protocol.types import AssistantRequest
from protocol.message_limits import UserMessageTooLong
from protocol.v4_to_v3 import ModelConversionError
from memory import get_memory, memory_session_key
from routing.generation_signals import remember_repo_recap
from conversation.session_helpers import (
    get_user_message,
    reply_message,
    reply_payload,
    replay_last_reply,
    stream_llm_response,
    user_key_reply,
    json_intent_matches,
    json_no_intent_matched,
)
from app.confirmation import resume_pending
from execution import execute_planned_operations
from ingestion.file_handling import handle_file_attachments
from conversation.suggestions import get_suggested_actions
from diagram_handlers.registry.metadata import get_diagram_type_info
from codegen.generation_handler import (
    handle_generation_request,
    should_route_to_generation,
)
from orchestrator import determine_target_diagram_type
from prompts import describe_model as describe_prompts
from prompts import fallback as fallback_prompts
from prompts import modeling_help as help_prompts
from prompts import uml_rag as uml_prompts
from utilities.model_context import detailed_model_summary, is_diagram_nontrivial
from utilities.user_metamodel import format_user_metamodel_guide, is_user_profile_help
from routing import guards
from routing.intents import GENERATION_INTENT_NAME
from common.session_keys import (
    HAS_GREETED,
    LAST_EXECUTED_DIAGRAM_TYPE,
    LAST_MATCHED_INTENT,
    UNIFIED_CLASSIFICATION,
)
from common import pending_dialogs
from conversation.reply_copy import (
    DECLINE_ACK as _DECLINE_ACK,
    META_ANSWER as _META_ANSWER,
    OUT_OF_SCOPE_REDIRECT as _OUT_OF_SCOPE_REDIRECT,
    unsupported_editor_reply,
)
from routing.unified_classifier import get_or_classify
from protocol.actions import Action

try:
    from llm.provider import get_provider as _get_llm_provider
except ImportError:  # pragma: no cover — test env without BAF stack
    _get_llm_provider = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _text_reply(llm: Any, prompt: str) -> str:
    """One free-text call on the text LLM *llm*, on the user's key when one
    is active (the token cap and temperature mirror ``gpt_text``'s own)."""
    return LLMProvider(llm).complete(
        prompt, temperature=LLM_TEXT_TEMPERATURE, max_tokens=LLM_MAX_TOKENS_TEXT,
    )


def _conversation_memory(session: Session, request: AssistantRequest):
    """This conversation's memory (created with the summarizer), rebuilt from
    the client's ``recentHistory`` when the server no longer holds it.

    The client sends its latest turns on the first message after a
    (re)connect. They seed only an EMPTY memory (a restart, a conversation
    reaped while the tab was gone), before the new message is recorded and
    before the classifier reads the history; a memory that already holds the
    conversation keeps what the server recorded and ignores them.
    """
    key = memory_session_key(session, request)
    summarizer = getattr(ctx, 'gpt_text', None)
    summarize_fn = functools.partial(_text_reply, summarizer) if summarizer else None
    mem = get_memory(key, summarizer=summarize_fn)
    if request.recent_history:
        seeded = mem.seed(request.recent_history)
        if seeded:
            logger.info(
                "[Memory:%s] Rebuilt the conversation from the client's recent history (%d turn(s))",
                key, seeded,
            )
    return mem


# ------------------------------------------------------------------
# Common preamble helper
# ------------------------------------------------------------------

def _ensure_unified_classification(session: Session) -> bool:
    """Priority-0 transition hook: populates the per-message classifier cache.

    Normally returns False. Oversized messages return True to enter the
    current state's preamble, which reports the rejection without an LLM call.
    The normal purpose is the side effect: one LLM call
    per message that sets ``session[UNIFIED_CLASSIFICATION]``, so the
    subsequent ``json_intent_matches`` transitions read our classifier's
    verdict instead of BAF's description-based one.

    Runs for pending-flow replies TOO (ActiveFlow): the classifier sees
    the pending question in its context and judges answer-vs-new-request
    — that verdict is what lets ``json_intent_matches`` route a pivot
    normally instead of trapping it in the asking state. Skips only when
    the session already has a cached classification for this event id.
    """
    try:
        request = parse_assistant_request(session)
        if request.recent_history:
            # Before the classifier reads the conversation history.
            try:
                _conversation_memory(session, request)
            except Exception as exc:
                logger.warning("Rebuilding conversation memory from the client failed: %s", exc)
        if guards.skips_classification(request):
            return False
        provider = _get_llm_provider() if _get_llm_provider else None
        classification = get_or_classify(session, request, provider)
        logger.info(
            "unified classifier: intent=%s generation_route=%s generator_type=%s reason=%s",
            classification.intent,
            classification.generation_route,
            classification.generator_type,
            classification.reason,
        )
    except (UserMessageTooLong, ModelConversionError):
        return True
    except Exception:
        logger.exception("unified classifier hook failed; falling back to BAF")
    return False


def _common_preamble(session: Session) -> Optional[AssistantRequest]:
    """Run the shared preamble checks for every state body.

    Returns the parsed :class:`AssistantRequest` if the message should be
    handled normally, or ``None`` if a pending flow or file attachment
    already consumed it.
    """
    # Reconnect recovery: the frontend re-requests its last completed reply after
    # a mid-generation WebSocket reconnect dropped it. Re-send the buffered reply
    # and stop — never re-run generation or consume a pending flow.
    try:
        _replay_req = parse_assistant_request(session)
    except (UserMessageTooLong, ModelConversionError) as error:
        # reply_message parses the request again; use the wire envelope here
        # so rejection itself cannot recurse into the same parsing error.
        session.reply(json.dumps({"action": Action.ASSISTANT_MESSAGE, "message": str(error), "isError": True}))
        return None
    if getattr(_replay_req, "action", None) == Action.REPLAY_LAST_RESPONSE:
        replay_last_reply(session, _replay_req)
        return None
    remember_repo_recap(session)

    # The answer to an open question (replace/keep, GUI mode, a run
    # confirmation, the plan-pause go-ahead) is consumed here, whichever state
    # the classifier routed it to.
    if resume_pending(session):
        return None

    request = parse_assistant_request(session)

    if handle_file_attachments(session, request):
        return None

    # Record user message in conversation memory. Keyed on the stable
    # payload sessionId so memory survives WebSocket reconnects.
    if request.message:
        try:
            _conversation_memory(session, request).add_user(request.message)
        except Exception as exc:
            logger.debug(f"Recording user message in memory failed (best-effort): {exc}")

    # Ask-instead-of-guess: when the classifier judges the request genuinely
    # ambiguous (no resolvable referent, two readings, missing target the
    # workspace can't supply), stream its clarifying question and stop rather
    # than guessing a destructive mutation. Conservative by design — the
    # classifier is told to bias toward acting. Cached classification, so no
    # extra LLM call. Never fires on frontend_event callbacks; pending flows
    # were already handled above.
    if request.message and getattr(request, "action", None) != Action.FRONTEND_EVENT:
        try:
            _uc = get_or_classify(
                session, request,
                _get_llm_provider() if _get_llm_provider else None,
            )
            if getattr(_uc, "needs_clarification", False):
                _q = (getattr(_uc, "clarifying_question", None) or "").strip()
                if _q:
                    reply_message(session, _q)
                    return None
        except Exception as _clarify_err:
            logger.debug(f"clarify check skipped (best-effort): {_clarify_err}")

    return request


# ------------------------------------------------------------------
# "What's next" suggestions (powered by the suggestion engine)
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Quick info responses (no LLM call needed)
# ------------------------------------------------------------------

_QUICK_RESPONSES = {
    "what_is_besser": (
        "**BESSER** (Better Smart Software Engineering Research) is an open-source "
        "low-code platform for building software through model-driven engineering.\n\n"
        "It lets you:\n"
        "- Design domain models visually (class diagrams, state machines, GUIs, agents, quantum circuits, "
        "BPMN processes, component and deployment diagrams)\n"
        "- Generate production code automatically (Django, FastAPI, React, SQL, and more), "
        "or any other kind of app, such as a mobile or desktop app, written by AI from your model\n"
        "- Deploy full-stack web applications from your models\n\n"
        "Learn more at [besser.readthedocs.io](https://besser.readthedocs.io/) "
        "or try the online editor at [editor.besser-pearl.org](https://editor.besser-pearl.org/)."
    ),
    "what_can_you_do": (
        "Here's everything I can help you with:\n\n"
        "**Create diagrams:**\n"
        "- **Class Diagrams** — *\"Create an e-commerce system with customers, orders, and products\"*\n"
        "- **State Machines** — *\"Create an order processing workflow\"*\n"
        "- **Object Diagrams** — *\"Create instances of my classes\"*\n"
        "- **GUI / Web UI** — *\"Design a dashboard for my Product class\"*\n"
        "- **Agent Diagrams** — *\"Create a pizza-ordering chatbot agent\"*\n"
        "- **Quantum Circuits** — *\"Create Grover's search algorithm\"*\n"
        "- **BPMN Diagrams** — *\"Model an order fulfillment process\"*\n"
        "- **User Profiles** — *\"Create a target user profile for elderly users with sight issues\"*\n"
        "- **Component Diagrams** — *\"Create a component diagram for a RAG agent swarm\"*\n"
        "- **Deployment Diagrams** — *\"Create a deployment diagram for agent services\"*\n\n"
        "**Modify diagrams:**\n"
        "- *\"Add email attribute to User\"*, *\"Rename Order to Purchase\"*, *\"Add a transition from Idle to Active\"*\n\n"
        "**Generate code:**\n"
        "- *\"Generate Django\"*, *\"Generate React\"*, *\"Generate SQLAlchemy\"*, *\"Generate a web app\"*\n"
        "- Any other kind of app from your model: *\"Generate a mobile app\"*, "
        "*\"a Windows desktop app\"*, *\"a Python CLI\"*, a game…\n\n"
        "**Other:**\n"
        "- *\"Describe my diagram\"* — I'll analyze what you've built\n"
        "- *\"What is an association class?\"* — I can explain UML concepts\n"
        "- Attach a **PlantUML file** or **diagram image** and I'll convert it\n"
        "- Attach a **screenshot or mockup** of a page and I'll build it as a GUI\n"
        "- I reply in the language you write in\n\n"
        "What would you like to do?"
    ),
    "help": (
        "**Quick Start Guide:**\n\n"
        "1. **Describe your system** in plain language — I'll create the diagram\n"
        "   *Example: \"Create a library system with books, authors, and members\"*\n\n"
        "2. **Refine it** by asking for changes\n"
        "   *Example: \"Add a phone attribute to Member\"* or *\"Add inheritance between DigitalBook and Book\"*\n\n"
        "3. **Generate code** when you're ready\n"
        "   *Example: \"Generate a web app\"*, *\"Generate a mobile app\"* or *\"Generate a desktop app\"*\n\n"
        "**Tips:**\n"
        "- Be specific about what you want — more detail = better results\n"
        "- I support 10 diagram types: Class, State Machine, Object, GUI, Agent, "
        "Quantum Circuit, BPMN, User Profile, Component, and Deployment\n"
        "- You can switch between diagram types anytime\n"
        "- Ask *\"What can you do?\"* for a full list of capabilities"
    ),
    "how_to_run": (
        "**Running the app you generated:**\n\n"
        "1. **Unzip** the download.\n"
        "2. If it includes a **`docker-compose.yml`** (most web apps do):\n"
        "   ```\n"
        "   docker compose up --build\n"
        "   ```\n"
        "   then open the URL it prints — usually **http://localhost:8000** "
        "(backend) or **http://localhost:3000** (frontend).\n\n"
        "3. Otherwise, by stack:\n"
        "   - **FastAPI / Python backend** — `pip install -r requirements.txt`, "
        "then `uvicorn main:app --reload` (or `python main.py`).\n"
        "   - **Django** — `pip install -r requirements.txt`, "
        "`python manage.py migrate`, `python manage.py runserver`.\n"
        "   - **React / Next.js frontend** — `npm install`, then `npm run dev`.\n\n"
        "The generated project ships a **README** with the exact command — "
        "check it if you're unsure."
    ),
}

def _check_quick_response(message: str) -> Optional[str]:
    """Check if the message matches a quick-response pattern. Returns the response or None."""
    key = guards.quick_response_key(message)
    return _QUICK_RESPONSES[key] if key else None


# ------------------------------------------------------------------
# Global fallback
# ------------------------------------------------------------------

def _fallback_llm_reply(session: Session, user_message: str) -> None:
    """LLM answer for an unclassifiable message. Shared by the global fallback
    and by greetings_body when it receives another state's fallback traffic."""
    try:
        prompt = fallback_prompts.fallback_prompt(user_message)
        stream_llm_response(session, ctx.gpt_text, prompt)
    except Exception as e:
        logger.error(f"❌ Error in fallback LLM reply: {e}")
        reply_message(
            session,
            "I'm not sure how to help with that. Try asking me to create a class, "
            "design a system, build a quantum circuit, or modify your diagram.",
        )


def global_fallback_body(session: Session):
    """Handle unrecognized messages."""
    request = _common_preamble(session)
    if request is None:
        return

    user_message = request.message or "your message"

    # Check for quick info responses first (no LLM needed)
    quick = _check_quick_response(user_message)
    if quick:
        reply_message(session, quick)
        return

    _fallback_llm_reply(session, user_message)


# ------------------------------------------------------------------
# Greetings
# ------------------------------------------------------------------

GREETING_MESSAGE = (
    "Hi! I'm your BESSER modeling assistant. Describe your domain in plain "
    "words and I'll build the model with you, then refine it as you go.\n\n"
    "Try something like:\n"
    "- *\"a library with books, members and loans\"*\n"
    "- *\"add a due date to Loan\"*\n"
    "- *\"a state machine for an order's lifecycle\"*\n"
    "- *\"a pizza-ordering chatbot\"*\n\n"
    "I can also describe or review a model, design a quantum circuit, import a "
    "PlantUML / diagram image, or build a GUI from a screenshot or mockup. When "
    "you ask for it, I generate code too (SQL, Django, FastAPI, SQLAlchemy…) or "
    "a whole web, mobile or desktop app. I reply in the language you write in.\n\n"
    "What would you like to model today?"
)


def greetings_body(session: Session):
    """Send a greeting message when the user first connects or says hello."""

    if session.event is None:
        return

    request = _common_preamble(session)
    if request is None:
        return

    # "What can you do?"-style questions get the rich capability answer
    # (mirrors global_fallback_body's ordering).
    quick = _check_quick_response(request.message or "")
    if quick:
        reply_message(session, quick)
        session.set(HAS_GREETED, True)
        return

    # First contact: full greeting. Afterwards: ALWAYS reply — don't gate on
    # BAF's event.predicted_intent, which JSON events routed by the unified
    # classifier never carry, so the message would get no reply at all.
    if not session.get(HAS_GREETED):
        reply_message(session, GREETING_MESSAGE)
        session.set(HAS_GREETED, True)
        return

    _uc = session.get(UNIFIED_CLASSIFICATION)
    if getattr(_uc, "intent", None) == "fallback_intent":
        # This state is also the JSON-fallback DESTINATION for the uml_rag /
        # decline / out_of_scope / meta states: an unclassifiable message
        # lands here, and "Welcome back!" would be a non-sequitur — answer it.
        _fallback_llm_reply(session, request.message or "your message")
        return

    reply_message(session, "Welcome back! What would you like to work on?")


# ------------------------------------------------------------------
# Shared modeling-state body
# ------------------------------------------------------------------

_NON_MODELING_DECLINE = (
    "I'm a modeling assistant, so I can't take on other instructions or share how "
    "I work — but I'm glad to help you design a diagram. What would you like to "
    "model? For example: *Create a library management system*."
)


# Reply copy for the decline / out-of-scope / meta states lives in
# reply_copy.py (imported above); detection is the unified classifier's job.


_CONTRADICTORY_CREATE_CLARIFY = (
    "A diagram needs at least something to show, so I can't build an empty one. "
    "What would you like it to include? For example: *a Library with Book, "
    "Member and Loan classes*."
)

# guards.modeling_guard decision -> (log line, reply); None replies with the
# state's empty-message prompt.
_MODELING_GUARD_REPLIES = {
    "non_modeling": ("Non-modeling/injection input — declining and redirecting", _NON_MODELING_DECLINE),
    "outage_decline": ("Outage net: decline phrase — acknowledging", _DECLINE_ACK),
    "outage_artifact": ("Outage net: non-software artifact — redirecting", _OUT_OF_SCOPE_REDIRECT),
    "self_contradictory": ("Self-contradictory create — asking what to include", _CONTRADICTORY_CREATE_CLARIFY),
    "too_vague": ("Create request too vague to model — asking for a domain", None),
}


def _modeling_state_body(session: Session, intent_name: str, default_mode: str, empty_msg: str):
    """Unified handler for all modeling operations (system creation, modification)."""
    request = _common_preamble(session)
    if request is None:
        return

    # Honor the unified classifier's verdict over the state's default. Normally
    # they agree (routing is driven by the classifier), but a pending flow — e.g.
    # a GUI-choice prompt — suppresses intent routing (json_intent_matches), so a
    # follow-up modify can land in the CREATE state and get executed as a full
    # rebuild. Reconcile so "add Death to the PetStatus enum" modifies instead of
    # popping the replace/keep prompt.
    _uc = session.get(UNIFIED_CLASSIFICATION)
    _reconciled_intent, default_mode = guards.reconcile_intent(
        intent_name, default_mode, getattr(_uc, "intent", None),
    )
    if _reconciled_intent != intent_name:
        logger.info(
            "[Modeling] Reconciling state default '%s' -> unified intent '%s'",
            intent_name, _reconciled_intent,
        )
        intent_name = _reconciled_intent

    session.set(LAST_MATCHED_INTENT, intent_name)

    if not request.message:
        reply_message(session, empty_msg)
        return

    # Deterministic guards (routing/guards.py): injection input, the
    # classifier-outage net, and self-contradictory or contentless creates are
    # answered in prose instead of being modeled.
    _guard = guards.modeling_guard(request.message, intent_name, _uc)
    if _guard is not None:
        _log, _reply = _MODELING_GUARD_REPLIES[_guard]
        logger.info("[Modeling] %s", _log)
        reply_message(session, _reply or empty_msg)
        return

    try:
        execute_planned_operations(
            session=session,
            request=request,
            default_mode=default_mode,
            matched_intent=intent_name,
        )
        # "What's next?" suggestions are delivered as interactive QuickAction
        # buttons via suggestedActions in the result payload (execution.py).
        # No need for a separate text message.
        session.set(LAST_EXECUTED_DIAGRAM_TYPE, None)
    except Exception as e:
        logger.error(f"❌ Error in {intent_name}: {e}", exc_info=True)
        reply_message(session, "Something went wrong while processing your request. Could you try rephrasing it?")


def create_complete_system_body(session: Session):
    """Generate a complete system with multiple elements and relationships."""
    _modeling_state_body(
        session,
        intent_name='create_complete_system_intent',
        default_mode='complete_system',
        empty_msg="What system would you like me to design? For example: 'Create a library management system'",
    )


def modify_modeling_body(session: Session):
    """Apply modifications to an existing UML model."""
    _modeling_state_body(
        session,
        intent_name='modify_model_intent',
        default_mode='modify_model',
        empty_msg="What changes would you like me to make to the model?",
    )


# ------------------------------------------------------------------
# Modeling help
# ------------------------------------------------------------------

def modeling_help_body(session: Session):
    """Offer guidance or clarifying questions when the user needs modeling help."""
    request = _common_preamble(session)
    if request is None:
        return

    session.set(LAST_MATCHED_INTENT, 'modeling_help_intent')

    if not request.message:
        reply_message(
            session,
            "I can help you with UML modeling! Try asking me to create a class, "
            "design a system, or modify your diagram.",
        )
        return

    # Check for quick info responses first (no LLM needed)
    quick = _check_quick_response(request.message)
    if quick:
        reply_message(session, quick)
        return

    diagram_type = determine_target_diagram_type(request, last_intent='modeling_help_intent')
    diagram_info = get_diagram_type_info(diagram_type)

    # Build context-aware help prompt depending on the diagram type
    if diagram_type == "BPMN":
        help_prompt = help_prompts.bpmn_help_prompt(request.message)
    elif diagram_type == "ComponentDiagram":
        help_prompt = help_prompts.component_help_prompt(request.message)
    elif diagram_type == "DeploymentDiagram":
        help_prompt = help_prompts.deployment_help_prompt(request.message)
    elif diagram_type == "QuantumCircuitDiagram":
        help_prompt = help_prompts.quantum_help_prompt(request.message)
    elif diagram_type == "UserDiagram" or is_user_profile_help(request.message):
        # Explain the user-profile modeling environment (elements, attributes,
        # enum values, how they connect) grounded in the bundled metamodel.
        help_prompt = help_prompts.user_profile_help_prompt(request.message, format_user_metamodel_guide())
    else:
        help_prompt = help_prompts.generic_help_prompt(request.message, diagram_info)

    try:
        stream_llm_response(session, ctx.gpt_text, help_prompt)
    except Exception as e:
        logger.error(f"❌ Error in modeling_help_body: {e}", exc_info=True)
        reply_message(session, "I had trouble preparing guidance. Could you try rephrasing your question?")


# ------------------------------------------------------------------
# Code generation
# ------------------------------------------------------------------

def _build_full_project_summary(request: AssistantRequest) -> str:
    """Build a detailed summary of all *non-trivial* diagrams in the project.

    Empty diagrams (0 elements / 0 pages / 0 gates) and diagrams that only
    contain the editor's default seed content are skipped — listing them as
    "X is empty" drowns out the diagrams the user actually built.  See
    :func:`utilities.model_context.is_diagram_nontrivial` for the per-type
    rules.

    The active diagram is always included (even if empty) when the user is
    looking at it, so questions like "what's in my current diagram?" still
    get a meaningful answer.  A trailing note records any diagram types that
    were skipped, so the LLM can mention them in one short sentence.
    """
    sections: list[str] = []
    described_types: set[str] = set()
    skipped_types: set[str] = set()

    # Project metadata
    snapshot = request.context.project_snapshot
    if isinstance(snapshot, dict):
        name = snapshot.get("name")
        if isinstance(name, str) and name.strip():
            sections.append(f"**Project**: {name.strip()}")

    active_dt = request.context.active_diagram_type or request.diagram_type
    active_model = request.context.active_model or request.current_model

    # Track which diagram types we've already considered (avoid dupes)
    summarised: set[str] = set()

    # 1. Active diagram — always first.  Include it even if "trivial" so the
    #    user gets a reply when they're staring at an empty tab and ask
    #    "what's in this?".  But still record triviality for the closing note.
    if isinstance(active_model, dict) and active_dt:
        active_info = get_diagram_type_info(active_dt)
        if is_diagram_nontrivial(active_model, active_dt):
            sections.append(
                f"### Active diagram: {active_info['name']}\n"
                + detailed_model_summary(active_model, active_dt)
            )
            described_types.add(active_dt)
        else:
            # Empty active diagram — still mention it so the LLM doesn't
            # silently ignore the tab the user is looking at.
            sections.append(
                f"### Active diagram: {active_info['name']}\n"
                f"(This diagram is empty.)"
            )
        summarised.add(active_dt)

    # 2. All other diagrams from project snapshot — filter out trivial ones.
    if isinstance(snapshot, dict):
        diagrams = snapshot.get("diagrams")
        if isinstance(diagrams, dict):
            for dt, payload in diagrams.items():
                if dt in summarised:
                    continue
                dt_info = get_diagram_type_info(dt)
                if isinstance(payload, list):
                    # Multi-tab: summarise each tab that has *non-trivial* content
                    tabs_with_model = [
                        d for d in payload
                        if isinstance(d, dict) and isinstance(d.get("model"), dict)
                    ]
                    nontrivial_tabs = [
                        (i, tab) for i, tab in enumerate(tabs_with_model)
                        if is_diagram_nontrivial(tab["model"], dt)
                    ]
                    if not nontrivial_tabs:
                        if tabs_with_model:
                            skipped_types.add(dt_info["name"])
                        summarised.add(dt)
                        continue
                    for i, tab in nontrivial_tabs:
                        model = tab["model"]
                        tab_title = tab.get("title", "").strip()
                        summary = detailed_model_summary(model, dt)
                        if summary:
                            label = dt_info["name"]
                            if tab_title:
                                label = f"{dt_info['name']} — {tab_title}"
                            elif len(tabs_with_model) > 1:
                                label = f"{dt_info['name']} (tab {i})"
                            sections.append(f"### {label}\n{summary}")
                            described_types.add(dt)
                    # If some tabs were trivial and others weren't, don't
                    # bother flagging the trivial ones — too noisy.
                    summarised.add(dt)
                elif isinstance(payload, dict):
                    model = payload.get("model")
                    if not isinstance(model, dict):
                        summarised.add(dt)
                        continue
                    if not is_diagram_nontrivial(model, dt):
                        skipped_types.add(dt_info["name"])
                        summarised.add(dt)
                        continue
                    summary = detailed_model_summary(model, dt)
                    if summary:
                        sections.append(f"### {dt_info['name']}\n{summary}")
                        described_types.add(dt)
                    summarised.add(dt)

    if not sections:
        return ""

    # Closing note for the LLM about skipped/empty diagrams.  Only emit it
    # if at least one diagram WAS described — otherwise the regular "I don't
    # see any diagrams" path handles the empty-project case.
    if skipped_types and described_types:
        skipped_list = sorted(skipped_types)
        sections.append(
            "_Note: the following diagram types exist in the project but are "
            "empty or contain only default seed content, so they were not "
            "described above: "
            + ", ".join(skipped_list)
            + "._"
        )

    return "\n\n".join(sections)


def _recent_conversation_block(session: Session, request, max_turns: int = 6) -> str:
    """Recent prior turns as text, so a describe/evaluate answer can resolve
    referents ("it", "that", "the first one", "those") against what was just
    asked/answered. ``_common_preamble`` has already recorded the CURRENT user
    message, so drop that trailing entry — we want the PRIOR context. Never
    raises."""
    try:
        mem = get_memory(memory_session_key(session, request))
        turns = mem.get_last_n(max_turns + 1)
    except Exception:
        return ""
    cur = (request.message or "").strip()
    if turns and turns[-1].get("role") == "user" and (turns[-1].get("content") or "").strip() == cur:
        turns = turns[:-1]
    turns = turns[-max_turns:]
    lines = []
    for t in turns:
        role = (t.get("role") or "?").strip()
        content = (t.get("content") or "").strip().replace("\n", " ")
        if not content:
            continue
        if len(content) > 300:
            content = content[:297].rstrip() + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def describe_model_body(session: Session):
    """Answer user questions about the current diagram / project."""
    request = _common_preamble(session)
    if request is None:
        return

    session.set(LAST_MATCHED_INTENT, 'describe_model_intent')

    if not request.message:
        reply_message(
            session,
            "What would you like to know about your project? "
            "Try asking things like *\"how many classes do I have?\"*, "
            "*\"describe my diagram\"*, or *\"what diagrams are in my project?\"*.",
        )
        return

    # Build a comprehensive summary of the entire project
    full_summary = _build_full_project_summary(request)

    if not full_summary:
        reply_message(
            session,
            "I don\u2019t see any diagrams in your project yet. "
            "Create a diagram first, then ask me about it!",
        )
        return

    # Recent turns so follow-up questions with referents ("it", "the first
    # one", "those") resolve to the element the user actually means.
    history_block = _recent_conversation_block(session, request)
    history_section = (
        describe_prompts.history_section(history_block)
        if history_block
        else ""
    )
    # Whole, ahead of the history: its readers cut a turn to 300 characters.
    linked_repo = request.context.github_repo or {}
    if linked_repo.get("recap"):
        history_section = (
            describe_prompts.repo_recap_section(linked_repo["full_name"], linked_repo["recap"]) + history_section
        )

    # For User Profile projects, add the metamodel semantics so the assistant
    # can explain what the boxes on the canvas (Accessibility, Disability, …)
    # actually mean — the model itself carries only names and criteria.
    active_dt = request.context.active_diagram_type or request.diagram_type
    include_user_guide = active_dt == "UserDiagram" or is_user_profile_help(request.message)
    if not include_user_guide:
        snapshot = request.context.project_snapshot
        if isinstance(snapshot, dict):
            diagrams = snapshot.get("diagrams")
            if isinstance(diagrams, dict) and "UserDiagram" in diagrams:
                include_user_guide = True
    user_guide_block = (
        describe_prompts.user_guide_block(format_user_metamodel_guide())
        if include_user_guide
        else ""
    )

    qa_prompt = describe_prompts.qa_prompt(full_summary, user_guide_block, history_section, request.message)

    try:
        # Grounded analysis over real model data → SMALL generation tier
        # (help/fallback streaming stays on the classifier default).
        stream_llm_response(session, ctx.gpt_text, qa_prompt, model=MODEL_GENERATION_SMALL)
    except Exception as e:
        logger.error(f"❌ Error in describe_model_body: {e}", exc_info=True)
        reply_message(
            session,
            "I had trouble analysing your project. Could you try rephrasing your question?",
        )


# ------------------------------------------------------------------
# Code generation (continued)
# ------------------------------------------------------------------

def generation_body(session: Session):
    """Handle assistant-driven code generation orchestration."""
    request = _common_preamble(session)
    if request is None:
        return

    session.set(LAST_MATCHED_INTENT, GENERATION_INTENT_NAME)

    # If the request mixes modeling + generation ("create a class diagram and generate Django"),
    # route through the modeling pipeline first — the orchestrator builds the model and then
    # PAUSES for the user's go-ahead before generating (see execution/planning.py).
    if guards.is_mixed_modeling_and_generation(request.message or ""):
        logger.info("[GenerationBody] Mixed request detected — routing through modeling pipeline")
        reply_message(
            session,
            "I'll **create the diagram first** — once it's ready, you can review it "
            "or tell me to continue with **code generation**.",
        )
        try:
            execute_planned_operations(
                session=session,
                request=request,
                default_mode="complete_system",
                matched_intent=GENERATION_INTENT_NAME,
            )
        except Exception as error:
            logger.error(f"❌ Error in mixed request routing: {error}", exc_info=True)
            reply_message(session, "Something went wrong while processing your multi-step request.")
        return

    try:
        response_payload = handle_generation_request(session, request)
    except UserMessageTooLong as error:
        response_payload = {"action": Action.ASSISTANT_MESSAGE, "message": str(error), "isError": True}
    except Exception as error:
        logger.error(f"❌ Error in generation_body: {error}")
        response_payload = {
            "action": Action.AGENT_ERROR,
            "code": "generation_handler_error",
            "message": "Failed to process generation request.",
            "retryable": True,
        }

    if response_payload is None:
        # Already handled — e.g. a "generate" that was really a "create/design
        # a system" request got redirected and built the model inline, sending
        # its own reply. Nothing more to send.
        return
    if not isinstance(response_payload, dict):
        reply_message(session, "I could not process your generation request.")
        return

    # Attach contextual suggestions after code generation
    snapshot = request.context.project_snapshot
    avail_diagrams: list = []
    if isinstance(snapshot, dict):
        diagrams = snapshot.get("diagrams")
        if isinstance(diagrams, dict):
            for dtype, value in diagrams.items():
                if isinstance(value, list):
                    # Multi-tab: only include type when at least one tab has a model
                    if any(isinstance(d, dict) and d.get("model") for d in value):
                        avail_diagrams.append(dtype)
                elif isinstance(value, dict) and value.get("model"):
                    avail_diagrams.append(dtype)
    gen_suggestions = get_suggested_actions(
        diagram_type="",
        operation_mode="generation",
        available_diagrams=avail_diagrams,
    )
    if gen_suggestions:
        response_payload["suggestedActions"] = gen_suggestions

    reply_payload(session, response_payload)


# ------------------------------------------------------------------
# UML RAG
# ------------------------------------------------------------------

def uml_rag_body(session: Session):
    """Answer UML specification questions using RAG."""
    request = _common_preamble(session)
    if request is None:
        return

    session.set(LAST_MATCHED_INTENT, 'uml_spec_intent')

    user_message = request.message or get_user_message(session)

    if not user_message:
        reply_message(session, "Please ask a question about UML — for example *'What is an association class?'*.")
        return

    if ctx.uml_rag is None:
        logger.info("[UML_RAG] RAG unavailable — falling back to standard LLM")
        _uml_llm_reply(
            session,
            uml_prompts.fallback_prompt(user_message),
            note="*Note: UML knowledge base unavailable — answering from general knowledge.*\n\n",
        )
    else:
        try:
            rag_message: RAGMessage = session.run_rag(user_message)
            reply_message(session, rag_message.answer)
        except Exception as e:
            logger.error(f"❌ Error in uml_rag_body: {e}")
            _uml_llm_reply(session, uml_prompts.error_fallback_prompt(user_message))


def _uml_llm_reply(session: Session, prompt: str, *, note: str = "") -> None:
    """Answer from the LLM. BAF only logs what a state body raises, so a failing
    user key must reply here or the turn ends in silence."""
    try:
        answer = _text_reply(ctx.gpt_text, prompt)
    except Exception as e:
        key_reply = user_key_reply(e)
        if key_reply is None:
            raise
        logger.error(f"❌ UML answer failed on the user's key: {type(e).__name__}")
        reply_payload(session, key_reply)
        return
    reply_message(session, note + answer)


# ------------------------------------------------------------------
# Transition wiring
# ------------------------------------------------------------------

def decline_body(session: Session):
    """The user declined / opted out (decline_intent) — acknowledge and build
    nothing. Detection is the unified classifier's (it handles novel phrasings
    like "nah I'm good"); _modeling_state_body keeps only a classifier-OUTAGE
    net for the bare phrases."""
    request = _common_preamble(session)
    if request is None:
        return
    # If a config-collection flow is in progress (e.g. the user answered
    # "regular" to the JSON-schema mode prompt), the intent classifier may
    # misread the short answer as a decline. Route to the generation handler
    # so the pending flow can consume the answer instead of abandoning it.
    if pending_dialogs.is_open(session, pending_dialogs.GENERATOR):
        generation_body(session)
        return
    session.set(LAST_MATCHED_INTENT, 'decline_intent')
    reply_message(session, _DECLINE_ACK)


def out_of_scope_body(session: Session):
    """The user asked for a non-software artifact (out_of_scope_intent) — an
    actual image, creative writing, a joke. Redirect to modeling instead of
    building a diagram OF the request."""
    request = _common_preamble(session)
    if request is None:
        return
    session.set(LAST_MATCHED_INTENT, 'out_of_scope_intent')
    # In an editor the assistant can't model yet (e.g. the NN editor) the
    # request is in scope for BESSER, just not for the assistant: say so.
    _raw_context = (request.raw_payload or {}).get("context")
    _raw_active = _raw_context.get("activeDiagramType") if isinstance(_raw_context, dict) else None
    reply_message(session, unsupported_editor_reply(_raw_active) or _OUT_OF_SCOPE_REDIRECT)


def meta_question_body(session: Session):
    """The user asked about the assistant itself (meta_question_intent) —
    capabilities ("do you also generate websites?") or value proposition
    ("why use you instead of Claude/GPT?"). Answer the pitch; build nothing."""
    request = _common_preamble(session)
    if request is None:
        return
    session.set(LAST_MATCHED_INTENT, 'meta_question_intent')
    # A direct "do you …?" capability question deserves an explicit yes before
    # the pitch (presentation only — routing stays with the classifier).
    _prefix = "Yes. " if re.search(r"\bdo\s+you\b", request.message or "", re.IGNORECASE) else ""
    reply_message(session, _prefix + _META_ANSWER)


def route_to_generation(session: Session) -> bool:
    """Detect generation workflow requests or frontend callback events."""
    request = parse_assistant_request(session)
    return should_route_to_generation(session, request)


def add_unified_transitions(state, intents_map, fallback_state, generation_state):
    """Add both text and JSON event transitions for a state.

    Transition priority (first match wins):
    1. Intent-matched JSON transitions — the LLM-based intent classifier is
       the most accurate signal, so it gets first priority.
    2. Keyword-based generation route — catches generator keywords, frontend
       callback events, and pending-generator follow-ups that the intent
       classifier might not detect.
    3. Text-event intent transitions (backward compatibility).
    4. Fallback transitions.
    """
    # 0. Unified classifier hook — populates the per-message cache so
    #    subsequent ``json_intent_matches`` conditions read from our
    #    classifier's verdict instead of BAF's description-based one.
    #    Normally a pure side-effect condition. Oversized input self-transitions
    #    into the preamble's explicit rejection. One LLM call per valid message,
    #    regardless of how many transitions we have.
    state.when_event(ReceiveJSONEvent()) \
        .with_condition(_ensure_unified_classification) \
        .go_to(state)

    # 1. Intent-matched JSON transitions (highest priority for user messages)
    for intent, dest_state in intents_map.items():
        state.when_event(ReceiveJSONEvent()) \
            .with_condition(json_intent_matches, {'intent_name': intent.name}) \
            .go_to(dest_state)

    # 2. Keyword-based generation route (frontend events, pending config, etc.)
    state.when_event(ReceiveJSONEvent()) \
        .with_condition(route_to_generation) \
        .go_to(generation_state)

    # 3. Text event transitions (backward compatibility)
    for intent, dest_state in intents_map.items():
        state.when_intent_matched(intent).go_to(dest_state)

    # 4. Fallback transitions
    state.when_event(ReceiveJSONEvent()) \
        .with_condition(json_no_intent_matched) \
        .go_to(fallback_state)
    state.when_no_intent_matched().go_to(fallback_state)


def register_all(*, agent, states, intents):
    """Wire state bodies and transitions.

    Args:
        agent: The BESSER ``Agent`` instance.
        states: dict mapping state name → state object.
        intents: dict mapping intent name → intent object.
    """
    # -- Assign bodies --
    agent.set_global_fallback_body(global_fallback_body)
    states['greetings'].set_body(greetings_body)
    states['create_complete_system'].set_body(create_complete_system_body)
    states['modify_model'].set_body(modify_modeling_body)
    states['modeling_help'].set_body(modeling_help_body)
    states['describe_model'].set_body(describe_model_body)
    states['generation'].set_body(generation_body)
    states['uml_rag'].set_body(uml_rag_body)
    states['decline'].set_body(decline_body)
    states['out_of_scope'].set_body(out_of_scope_body)
    states['meta_question'].set_body(meta_question_body)

    # -- Wire transitions --
    intent_map = {
        intents['create_complete_system']: states['create_complete_system'],
        intents['modify_model']: states['modify_model'],
        intents['modeling_help']: states['modeling_help'],
        intents['describe_model']: states['describe_model'],
        intents['uml_spec']: states['uml_rag'],
        intents['generation']: states['generation'],
        intents['hello']: states['greetings'],
        intents['decline']: states['decline'],
        intents['out_of_scope']: states['out_of_scope'],
        intents['meta_question']: states['meta_question'],
    }

    generation_st = states['generation']

    for state_name, fallback_name in [
        ('greetings', 'modeling_help'),
        ('create_complete_system', 'create_complete_system'),
        ('modify_model', 'modify_model'),
        ('modeling_help', 'modeling_help'),
        ('describe_model', 'describe_model'),
        ('uml_rag', 'greetings'),
        ('generation', 'generation'),
        ('decline', 'greetings'),
        ('out_of_scope', 'greetings'),
        ('meta_question', 'greetings'),
    ]:
        add_unified_transitions(
            states[state_name], intent_map, states[fallback_name], generation_st,
        )
