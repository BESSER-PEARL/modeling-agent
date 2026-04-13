"""
State Bodies & Transition Wiring
---------------------------------
All BESSER agent state-body functions and the helper that wires
intent → state transitions.

Call :func:`register_all` from ``modeling_agent.py`` after states
and intents have been created.
"""

import logging
from typing import Any, Dict, Optional

from baf.core.session import Session
from baf.library.transition.events.base_events import ReceiveJSONEvent
from baf.nlp.rag.rag import RAGMessage

import agent_context as ctx
from protocol.adapters import parse_assistant_request
from protocol.types import AssistantRequest
from memory import get_memory
from session_helpers import (
    get_user_message,
    reply_message,
    reply_payload,
    stream_llm_response,
    json_intent_matches,
    json_no_intent_matched,
    route_to_generation,
)
from confirmation import handle_pending_system_confirmation, handle_pending_gui_choice
from execution import (
    execute_planned_operations,
    handle_file_attachments,
)
from suggestions import get_suggested_actions, format_suggestions_as_text
from diagram_handlers.registry.metadata import get_diagram_type_info
from handlers.generation_handler import (
    handle_generation_request,
    _looks_like_mixed_modeling_and_generation,
    detect_generator_type,
)
from handlers.validation_handler import validate_diagram
from orchestrator import determine_target_diagram_type
from utilities.model_context import detailed_model_summary
from routing.intents import GENERATION_INTENT_NAME
from session_keys import (
    HAS_GREETED,
    LAST_EXECUTED_DIAGRAM_TYPE,
    LAST_MATCHED_INTENT,
    PENDING_COMPLETE_SYSTEM,
    PENDING_GUI_CHOICE,
    WORKFLOW_PENDING_GENERATOR,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Common preamble helper
# ------------------------------------------------------------------

def _common_preamble(session: Session) -> Optional[AssistantRequest]:
    """Run the shared preamble checks for every state body.

    Returns the parsed :class:`AssistantRequest` if the message should be
    handled normally, or ``None`` if a pending flow or file attachment
    already consumed it.
    """
    if handle_pending_gui_choice(session):
        return None
    if handle_pending_system_confirmation(session):
        return None

    request = parse_assistant_request(session)

    if handle_file_attachments(session, request):
        return None

    # Record user message in conversation memory
    if request.message:
        try:
            session_id = getattr(session, 'id', None) or str(id(session))
            summarizer = getattr(ctx, 'gpt_text', None)
            summarize_fn = summarizer.predict if summarizer else None
            mem = get_memory(session_id, summarizer=summarize_fn)
            mem.add_user(request.message)
        except Exception as exc:
            logger.debug(f"Recording user message in memory failed (best-effort): {exc}")

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
        "- Design domain models visually (class diagrams, state machines, GUIs, agents, quantum circuits)\n"
        "- Generate production code automatically (Django, FastAPI, React, Flutter, SQL, and more)\n"
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
        "- **Quantum Circuits** — *\"Create Grover's search algorithm\"*\n\n"
        "**Modify diagrams:**\n"
        "- *\"Add email attribute to User\"*, *\"Rename Order to Purchase\"*, *\"Add a transition from Idle to Active\"*\n\n"
        "**Generate code:**\n"
        "- *\"Generate Django\"*, *\"Generate React\"*, *\"Generate SQLAlchemy\"*, *\"Generate a web app\"*\n\n"
        "**Other:**\n"
        "- *\"Describe my diagram\"* — I'll analyze what you've built\n"
        "- *\"What is an association class?\"* — I can explain UML concepts\n"
        "- Attach a **PlantUML file** or **diagram image** and I'll convert it\n\n"
        "What would you like to do?"
    ),
    "help": (
        "**Quick Start Guide:**\n\n"
        "1. **Describe your system** in plain language — I'll create the diagram\n"
        "   *Example: \"Create a library system with books, authors, and members\"*\n\n"
        "2. **Refine it** by asking for changes\n"
        "   *Example: \"Add a phone attribute to Member\"* or *\"Add inheritance between DigitalBook and Book\"*\n\n"
        "3. **Generate code** when you're ready\n"
        "   *Example: \"Generate Django\"* or *\"Generate a web app\"*\n\n"
        "**Tips:**\n"
        "- Be specific about what you want — more detail = better results\n"
        "- I support 6 diagram types: Class, State Machine, Object, GUI, Agent, and Quantum Circuit\n"
        "- You can switch between diagram types anytime\n"
        "- Ask *\"What can you do?\"* for a full list of capabilities"
    ),
}

# Patterns that trigger quick responses (checked in order)
_QUICK_PATTERNS = [
    # What is BESSER?
    (["what is besser", "what's besser", "tell me about besser", "explain besser", "about besser"],
     "what_is_besser"),
    # What can you do?
    (["what can you do", "what do you do", "your capabilities", "what are your features",
      "list your features", "show me what you can do", "what are you capable of",
      "how can you help", "what do you support", "what diagrams"],
     "what_can_you_do"),
    # Help
    (["help me", "i need help", "how does this work", "how do i use",
      "getting started", "quick start", "tutorial", "guide me"],
     "help"),
]


def _check_quick_response(message: str) -> Optional[str]:
    """Check if the message matches a quick-response pattern. Returns the response or None."""
    msg_lower = message.lower().strip()
    # Exact short matches
    if msg_lower in ("help", "?", "help!", "help?"):
        return _QUICK_RESPONSES["help"]
    for patterns, key in _QUICK_PATTERNS:
        if any(p in msg_lower for p in patterns):
            return _QUICK_RESPONSES[key]
    return None


# ------------------------------------------------------------------
# Global fallback
# ------------------------------------------------------------------

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

    try:
        prompt = (
            f"You are a modeling assistant that helps with UML diagrams, quantum circuits, "
            f"GUI design, agent diagrams, and code generation. The user said: '{user_message}'. "
            "If this is related to any kind of modeling (class diagrams, quantum circuits, "
            "state machines, GUI design, etc.), suggest how you can help them. "
            "Otherwise, politely explain your capabilities."
        )
        stream_llm_response(session, ctx.gpt_text, prompt)
    except Exception as e:
        logger.error(f"❌ Error in global_fallback_body: {e}")
        reply_message(
            session,
            "I'm not sure how to help with that. Try asking me to create a class, "
            "design a system, build a quantum circuit, or modify your diagram.",
        )


# ------------------------------------------------------------------
# Greetings
# ------------------------------------------------------------------

def greetings_body(session: Session):
    """Send a greeting message when the user first connects or says hello."""
    greeting_message = (
        "Hey there! I'm your modeling assistant.\n\n"
        "Here's what I can do:\n"
        "- **Create elements**: *\"Create a User class with name, email, and role\"*\n"
        "- **Build full systems**: *\"Design a library management system\"*\n"
        "- **Design chatbots**: *\"Create a pizza-ordering agent\"*\n"
        "- **Build UIs**: *\"Create a dashboard for my Product class\"*\n"
        "- **Quantum circuits**: *\"Create Grover's search algorithm\"* or *\"Build a Bell state circuit\"*\n"
        "- **Modify diagrams**: *\"Add a phone attribute to the Customer class\"*\n"
        "- **Describe models**: *\"What does my circuit do?\"* or *\"Describe my class diagram\"*\n"
        "- **Generate code**: *\"Generate SQLAlchemy\"* or *\"Generate Django\"*\n"
        "- **Model help**: *\"Explain Grover's algorithm\"* or *\"What is composition?\"*\n"
        "- **Import from files**: Attach a PlantUML, Knowledge Graph, or diagram image\n\n"
        "What would you like to create?"
    )

    if session.event is None:
        return

    request = _common_preamble(session)
    if request is None:
        return

    is_hello_intent = False
    if hasattr(session.event, 'predicted_intent') and session.event.predicted_intent:
        is_hello_intent = session.event.predicted_intent.intent.name == 'hello_intent'

    if is_hello_intent and not session.get(HAS_GREETED):
        reply_message(session, greeting_message)
        session.set(HAS_GREETED, True)
        return

    if is_hello_intent and session.get(HAS_GREETED):
        reply_message(session, "Welcome back! What would you like to work on?")
        return


# ------------------------------------------------------------------
# Shared modeling-state body
# ------------------------------------------------------------------

def _modeling_state_body(session: Session, intent_name: str, default_mode: str, empty_msg: str):
    """Unified handler for all modeling operations (system creation, modification)."""
    request = _common_preamble(session)
    if request is None:
        return

    session.set(LAST_MATCHED_INTENT, intent_name)

    if not request.message:
        reply_message(session, empty_msg)
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
    if diagram_type == "QuantumCircuitDiagram":
        help_prompt = (
            f'You are an expert quantum computing and quantum circuit modeling assistant. '
            f'The user asked: "{request.message}"\n\n'
            f'They are working with the Quantum Circuit Diagram editor.\n\n'
            "You have deep knowledge of:\n"
            "- Quantum gates (Hadamard, Pauli-X/Y/Z, CNOT, CZ, SWAP, S, T, QFT, etc.)\n"
            "- Quantum algorithms (Grover's search, Shor's factoring, QFT, Deutsch-Jozsa, "
            "Bernstein-Vazirani, quantum teleportation, superdense coding, phase estimation, VQE)\n"
            "- Quantum concepts (superposition, entanglement, interference, measurement, decoherence)\n"
            "- Circuit design principles (oracle construction, amplitude amplification, error correction)\n\n"
            "Provide clear, educational explanations. If they ask about a quantum algorithm, "
            "explain the key steps and intuition behind it. If they want to build something, "
            "tell them they can ask you to create it (e.g., 'Create a Grover\\'s search circuit').\n\n"
            "Keep your response conversational, encouraging, and technically accurate."
        )
    else:
        help_prompt = (
            f'You are an expert modeling assistant working with {diagram_info["name"]}. '
            f'The user asked: "{request.message}"\n\n'
            f'Current diagram type: {diagram_info["name"]} - {diagram_info["description"]}\n\n'
            "Provide helpful, practical advice about modeling for this diagram type. "
            "If they're asking about concepts, explain them clearly. "
            "If they want to create something, guide them on how to express their requirements.\n\n"
            "Keep your response conversational and encouraging. Suggest specific things they can ask you to create."
        )

    try:
        stream_llm_response(session, ctx.gpt_text, help_prompt)
    except Exception as e:
        logger.error(f"❌ Error in modeling_help_body: {e}", exc_info=True)
        reply_message(session, "I had trouble preparing guidance. Could you try rephrasing your question?")


# ------------------------------------------------------------------
# Code generation
# ------------------------------------------------------------------

def _build_full_project_summary(request: AssistantRequest) -> str:
    """Build a detailed summary of ALL diagrams in the project.

    Combines the active model (always included with full detail) with
    every other diagram found in the project snapshot so the LLM can
    answer cross-diagram questions.
    """
    sections: list[str] = []

    # Project metadata
    snapshot = request.context.project_snapshot
    if isinstance(snapshot, dict):
        name = snapshot.get("name")
        if isinstance(name, str) and name.strip():
            sections.append(f"**Project**: {name.strip()}")

    active_dt = request.context.active_diagram_type or request.diagram_type
    active_model = request.context.active_model or request.current_model

    # Track which diagram types we've already summarised (avoid dupes)
    summarised: set[str] = set()

    # 1. Active diagram — always first, always detailed
    if isinstance(active_model, dict):
        active_info = get_diagram_type_info(active_dt)
        sections.append(
            f"### Active diagram: {active_info['name']}\n"
            + detailed_model_summary(active_model, active_dt)
        )
        summarised.add(active_dt)

    # 2. All other diagrams from project snapshot
    if isinstance(snapshot, dict):
        diagrams = snapshot.get("diagrams")
        if isinstance(diagrams, dict):
            for dt, payload in diagrams.items():
                if dt in summarised:
                    continue
                dt_info = get_diagram_type_info(dt)
                if isinstance(payload, list):
                    # Multi-tab: summarise each tab that has a model
                    tabs_with_model = [
                        d for d in payload
                        if isinstance(d, dict) and isinstance(d.get("model"), dict)
                    ]
                    for i, tab in enumerate(tabs_with_model):
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
                    summarised.add(dt)
                elif isinstance(payload, dict):
                    model = payload.get("model")
                    if not isinstance(model, dict):
                        continue
                    summary = detailed_model_summary(model, dt)
                    if summary:
                        sections.append(f"### {dt_info['name']}\n{summary}")
                    summarised.add(dt)

    if not sections:
        return ""
    return "\n\n".join(sections)


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

    qa_prompt = (
        "You are an expert assistant for the BESSER Web Modeling Editor. "
        "The user has a project that may contain multiple diagrams "
        "(class, state machine, object, GUI, quantum circuit, agent).\n\n"
        f"Here is a detailed summary of their full project:\n\n"
        f"{full_summary}\n\n"
        f"The user asks: \"{request.message}\"\n\n"
        "Answer their question accurately based ONLY on the project data above. "
        "If they ask about a specific diagram type, focus on that one. "
        "If they ask a general question, consider all diagrams. "
        "Be specific \u2014 reference class names, attribute names, states, pages, "
        "gates, relationships, etc. by name.\n\n"
        "**For quantum circuits specifically**: when the user asks to 'describe' "
        "or 'explain' a quantum circuit, do more than just list the gates. "
        "Analyze the circuit and explain:\n"
        "- What algorithm or pattern it implements (Bell state, Grover's search, "
        "QFT, teleportation, entanglement, etc.)\n"
        "- The purpose of each stage (initialization, oracle, diffusion, measurement)\n"
        "- What the expected output/behavior would be\n"
        "- The role of key gates (e.g., 'H creates superposition', "
        "'CNOT entangles qubits')\n\n"
        "Keep the answer concise and well-formatted with Markdown."
    )

    try:
        stream_llm_response(session, ctx.gpt_text, qa_prompt)
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
    # route through the modeling pipeline first — it will handle both steps via the orchestrator.
    if _looks_like_mixed_modeling_and_generation(request.message or ""):
        logger.info("[GenerationBody] Mixed request detected — routing through modeling pipeline")
        reply_message(
            session,
            "I'll **create the diagram first**, then **generate the code**. Let me handle both steps.",
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
    except Exception as error:
        logger.error(f"❌ Error in generation_body: {error}")
        response_payload = {
            "action": "agent_error",
            "code": "generation_handler_error",
            "message": "Failed to process generation request.",
            "retryable": True,
        }

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
# End-to-end workflow: model -> validate -> generate
# ------------------------------------------------------------------

def workflow_body(session: Session):
    """End-to-end workflow: create model(s), validate, and generate code in one go."""
    request = _common_preamble(session)
    if request is None:
        return

    session.set(LAST_MATCHED_INTENT, 'workflow_intent')

    if not request.message:
        reply_message(
            session,
            "What would you like me to build end-to-end? For example: "
            "*\"Create a complete web app for a hotel booking system\"*",
        )
        return

    user_message = request.message

    # ── Step 0: Parse generator target from the user message ─────────
    target_generator = detect_generator_type(user_message)
    if not target_generator:
        # Default to web_app for generic "complete application" requests
        target_generator = "web_app"

    reply_message(
        session,
        f"Starting the **end-to-end workflow** for your request. "
        f"I will create the model(s), validate them, and generate **{target_generator}** code.\n\n"
        f"**Step 1/3** — Building your model...",
    )

    # ── Step 1: Create the model(s) via the existing planner ─────────
    try:
        execute_planned_operations(
            session=session,
            request=request,
            default_mode="complete_system",
            matched_intent="workflow_intent",
        )
    except Exception as e:
        logger.error(f"❌ [Workflow] Model creation failed: {e}", exc_info=True)
        reply_message(
            session,
            "Something went wrong while creating the model. "
            "Could you try rephrasing your request?",
        )
        return

    # If there's a pending confirmation (e.g. replace existing model),
    # we have to stop here — the user needs to respond first.
    if session.get(PENDING_COMPLETE_SYSTEM) or session.get(PENDING_GUI_CHOICE):
        logger.info("[Workflow] Paused — waiting for user confirmation before continuing")
        # Store workflow continuation state so we could resume later
        session.set(WORKFLOW_PENDING_GENERATOR, target_generator)
        return

    # ── Step 2: Validate the model ───────────────────────────────────
    reply_message(session, "**Step 2/3** — Running validation on your model...")

    # Collect the active model from the session context for validation
    active_model = request.context.active_model or request.current_model
    active_diagram_type = request.context.active_diagram_type or request.diagram_type

    # Also check the project snapshot for the model we just created
    snapshot = request.context.project_snapshot
    if not active_model and isinstance(snapshot, dict):
        # Prefer ClassDiagram as primary validation target
        for dt in ["ClassDiagram", active_diagram_type]:
            diagram = request.context.get_diagram_from_snapshot(dt)
            if isinstance(diagram, dict):
                candidate = diagram.get("model")
                if isinstance(candidate, dict):
                    active_model = candidate
                    active_diagram_type = dt
                    break

    validation_result = {"valid": True, "errors": [], "warnings": []}
    if isinstance(active_model, dict) and active_model:
        validation_result = validate_diagram(
            diagram_json=active_model,
            diagram_type=active_diagram_type,
        )

    # Report validation results
    if validation_result["errors"]:
        error_list = "\n".join(f"- {err}" for err in validation_result["errors"])
        warning_section = ""
        if validation_result["warnings"]:
            warning_list = "\n".join(f"- {w}" for w in validation_result["warnings"])
            warning_section = f"\n\n**Warnings:**\n{warning_list}"
        reply_message(
            session,
            f"Validation found **{len(validation_result['errors'])} error(s)**:\n"
            f"{error_list}{warning_section}\n\n"
            f"I recommend fixing these issues before generating code. "
            f"You can say *\"fix the validation errors\"* or modify the model manually.",
        )
        return

    # Validation passed
    warning_msg = ""
    if validation_result["warnings"]:
        warning_list = "\n".join(f"- {w}" for w in validation_result["warnings"])
        warning_msg = f"\n\n**Warnings** (non-blocking):\n{warning_list}"

    reply_message(
        session,
        f"Validation **passed** with 0 errors.{warning_msg}\n\n"
        f"**Step 3/3** — Generating **{target_generator}** code...",
    )

    # ── Step 3: Trigger code generation ──────────────────────────────
    from utilities.request_builders import build_generation_request

    generation_request = build_generation_request(
        request,
        generator_type=target_generator,
        config={},
        message_override=f"generate {target_generator}",
    )

    try:
        response_payload = handle_generation_request(session, generation_request)
    except Exception as error:
        logger.error(f"❌ [Workflow] Generation failed: {error}", exc_info=True)
        response_payload = {
            "action": "agent_error",
            "code": "generation_handler_error",
            "message": f"Failed to generate {target_generator} code.",
            "retryable": True,
        }

    if isinstance(response_payload, dict):
        # Add a completion summary to the payload message
        original_message = response_payload.get("message", "")
        response_payload["message"] = (
            f"{original_message}\n\n"
            f"**Workflow complete!** Your model was created, validated, and "
            f"**{target_generator}** code has been generated."
        )
        reply_payload(session, response_payload)
    else:
        reply_message(
            session,
            f"Code generation for **{target_generator}** did not return a valid result. "
            f"You can try again by saying *\"generate {target_generator}\"*.",
        )


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
        fallback_response = ctx.gpt_text.predict(
            f"You are a UML specification expert. Answer the following question about UML:\n\n"
            f"{user_message}\n\n"
            "Provide accurate information based on UML 2.x specifications. "
            "Be precise and reference specific UML concepts when applicable."
        )
        reply_message(
            session,
            "*Note: UML knowledge base unavailable — answering from general knowledge.*\n\n"
            + fallback_response,
        )
    else:
        try:
            rag_message: RAGMessage = session.run_rag(user_message)
            reply_message(session, rag_message.answer)
        except Exception as e:
            logger.error(f"❌ Error in uml_rag_body: {e}")
            fallback_response = ctx.gpt_text.predict(
                f"You are a UML specification expert. Answer the following question about UML:\n\n"
                f"{user_message}\n\n"
                "Provide accurate information based on UML 2.x specifications."
            )
            reply_message(session, fallback_response)


# ------------------------------------------------------------------
# Transition wiring
# ------------------------------------------------------------------

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
    states['workflow'].set_body(workflow_body)

    # -- Wire transitions --
    intent_map = {
        intents['create_complete_system']: states['create_complete_system'],
        intents['modify_model']: states['modify_model'],
        intents['modeling_help']: states['modeling_help'],
        intents['describe_model']: states['describe_model'],
        intents['uml_spec']: states['uml_rag'],
        intents['generation']: states['generation'],
        intents['hello']: states['greetings'],
        intents['workflow']: states['workflow'],
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
        ('workflow', 'workflow'),
    ]:
        add_unified_transitions(
            states[state_name], intent_map, states[fallback_name], generation_st,
        )
