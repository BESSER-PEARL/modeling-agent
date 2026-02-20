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

from besser.agent.core.session import Session
from besser.agent.library.transition.events.base_events import ReceiveJSONEvent
from besser.agent.nlp.rag.rag import RAGMessage

import agent_context as ctx
from protocol.adapters import parse_assistant_request
from protocol.types import AssistantRequest
from session_helpers import (
    get_user_message,
    reply_message,
    reply_payload,
    json_intent_matches,
    json_no_intent_matched,
    route_to_generation,
)
from confirmation import handle_pending_system_confirmation
from execution import (
    execute_planned_operations,
    handle_file_attachments,
)
from diagram_handlers.factory import get_diagram_type_info
from handlers.generation_handler import handle_generation_request
from orchestrator import determine_target_diagram_type
from routing.intents import GENERATION_INTENT_NAME

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Global fallback
# ------------------------------------------------------------------

def global_fallback_body(session: Session):
    """Handle unrecognized messages."""
    if handle_pending_system_confirmation(session):
        return

    request = parse_assistant_request(session)

    if handle_file_attachments(session, request):
        return

    user_message = request.message or "your message"
    try:
        answer = ctx.gpt_text.predict(
            f"You are a UML modeling assistant. The user said: '{user_message}'. "
            "If this is related to UML modeling, suggest how you can help them create models, classes, or diagrams. "
            "Otherwise, politely explain that you specialize in UML modeling assistance."
        )
        reply_message(session, answer)
    except Exception as e:
        logger.error(f"Error in global_fallback_body: {e}")
        reply_message(
            session,
            "I'm not sure how to help with that. Try asking me to create a class, "
            "design a system, or modify your diagram.",
        )


# ------------------------------------------------------------------
# Greetings
# ------------------------------------------------------------------

def greetings_body(session: Session):
    """Send a greeting message when the user first connects or says hello."""
    if handle_pending_system_confirmation(session):
        return

    greeting_message = (
        "Hey there! I'm your modeling assistant.\n\n"
        "Here's what I can do:\n"
        "- **Create elements**: *\"Create a User class with name, email, and role\"*\n"
        "- **Build full systems**: *\"Design a library management system\"*\n"
        "- **Design chatbots**: *\"Create a pizza-ordering agent\"*\n"
        "- **Build UIs**: *\"Create a dashboard for my Product class\"*\n"
        "- **Modify diagrams**: *\"Add a phone attribute to the Customer class\"*\n"
        "- **Generate code**: *\"Generate SQLAlchemy\"* or *\"Generate Django\"*\n"
        "- **UML help**: *\"What does UML say about composition?\"*\n"
        "- **Import from files**: Attach a PlantUML, Knowledge Graph, or diagram image\n\n"
        "What would you like to create?"
    )

    if session.event is None:
        return

    request = parse_assistant_request(session)
    if handle_file_attachments(session, request):
        return

    is_hello_intent = False
    if hasattr(session.event, 'predicted_intent') and session.event.predicted_intent:
        is_hello_intent = session.event.predicted_intent.intent.name == 'hello_intent'

    if is_hello_intent and not session.get('has_greeted'):
        reply_message(session, greeting_message)
        session.set('has_greeted', True)
        return

    if is_hello_intent and session.get('has_greeted'):
        reply_message(session, "Welcome back! What would you like to work on?")
        return


# ------------------------------------------------------------------
# Shared modeling-state body
# ------------------------------------------------------------------

def _modeling_state_body(session: Session, intent_name: str, default_mode: str, empty_msg: str):
    """Unified handler for all modeling operations (single element, system, modification)."""
    if handle_pending_system_confirmation(session):
        return

    session.set('last_matched_intent', intent_name)
    request = parse_assistant_request(session)

    if handle_file_attachments(session, request):
        return

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
    except Exception as e:
        logger.error(f"Error in {intent_name}: {e}", exc_info=True)
        reply_message(session, "Something went wrong while processing your request. Could you try rephrasing it?")


def create_single_element_body(session: Session):
    """Generate a single UML element based on the user's request."""
    _modeling_state_body(
        session,
        intent_name='create_single_element_intent',
        default_mode='single_element',
        empty_msg="What element would you like me to create? For example: 'Create a User class'",
    )


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
    if handle_pending_system_confirmation(session):
        return

    session.set('last_matched_intent', 'modeling_help_intent')
    request = parse_assistant_request(session)

    if handle_file_attachments(session, request):
        return

    if not request.message:
        reply_message(
            session,
            "I can help you with UML modeling! Try asking me to create a class, "
            "design a system, or modify your diagram.",
        )
        return

    diagram_type = determine_target_diagram_type(request, last_intent='modeling_help_intent')
    diagram_info = get_diagram_type_info(diagram_type)

    help_prompt = (
        f'You are a UML modeling expert assistant working with {diagram_info["name"]}. '
        f'The user asked: "{request.message}"\n\n'
        f'Current diagram type: {diagram_info["name"]} - {diagram_info["description"]}\n\n'
        "Provide helpful, practical advice about UML modeling for this diagram type. "
        "If they're asking about concepts, explain them clearly. "
        "If they want to create something, guide them on how to express their requirements.\n\n"
        "Keep your response conversational and encouraging. Suggest specific things they can ask you to create."
    )

    try:
        answer = ctx.gpt_text.predict(help_prompt)
        reply_message(session, answer)
    except Exception as e:
        logger.error(f"Error in modeling_help_body: {e}", exc_info=True)
        reply_message(session, "I had trouble preparing guidance. Could you try rephrasing your question?")


# ------------------------------------------------------------------
# Code generation
# ------------------------------------------------------------------

def generation_body(session: Session):
    """Handle assistant-driven code generation orchestration."""
    if handle_pending_system_confirmation(session):
        return

    session.set('last_matched_intent', GENERATION_INTENT_NAME)
    request = parse_assistant_request(session)

    if handle_file_attachments(session, request):
        return

    try:
        response_payload = handle_generation_request(session, request)
    except Exception as error:
        logger.error(f"Error in generation_body: {error}")
        response_payload = {
            "action": "agent_error",
            "code": "generation_handler_error",
            "message": "Failed to process generation request.",
            "retryable": True,
        }

    if not isinstance(response_payload, dict):
        reply_message(session, "I could not process your generation request.")
        return

    reply_payload(session, response_payload)


# ------------------------------------------------------------------
# UML RAG
# ------------------------------------------------------------------

def uml_rag_body(session: Session):
    """Answer UML specification questions using RAG."""
    if handle_pending_system_confirmation(session):
        return

    session.set('last_matched_intent', 'uml_spec_intent')
    request = parse_assistant_request(session)

    if handle_file_attachments(session, request):
        return

    user_message = request.message or get_user_message(session)

    if not user_message:
        reply_message(session, "Please ask a question about UML — for example *'What is an association class?'*.")
        return

    if ctx.uml_rag is None:
        fallback_response = ctx.gpt_text.predict(
            f"You are a UML specification expert. Answer the following question about UML:\n\n"
            f"{user_message}\n\n"
            "Provide accurate information based on UML 2.x specifications. "
            "Be precise and reference specific UML concepts when applicable."
        )
        reply_message(session, fallback_response)
    else:
        try:
            rag_message: RAGMessage = session.run_rag(user_message)
            reply_message(session, rag_message.answer)
        except Exception as e:
            logger.error(f"Error in uml_rag_body: {e}")
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
    """Add both text and JSON event transitions for a state."""
    # Direct generation route from JSON payload/callback events
    state.when_event(ReceiveJSONEvent()) \
        .with_condition(route_to_generation) \
        .go_to(generation_state)

    # Text event transitions (backward compatibility)
    for intent, dest_state in intents_map.items():
        state.when_intent_matched(intent).go_to(dest_state)

    # JSON event transitions (unified messages)
    for intent, dest_state in intents_map.items():
        state.when_event(ReceiveJSONEvent()) \
            .with_condition(json_intent_matches, {'intent_name': intent.name}) \
            .go_to(dest_state)

    # Fallback transitions
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
    states['create_single_element'].set_body(create_single_element_body)
    states['create_complete_system'].set_body(create_complete_system_body)
    states['modify_model'].set_body(modify_modeling_body)
    states['modeling_help'].set_body(modeling_help_body)
    states['generation'].set_body(generation_body)
    states['uml_rag'].set_body(uml_rag_body)

    # -- Wire transitions --
    intent_map = {
        intents['create_single_element']: states['create_single_element'],
        intents['create_complete_system']: states['create_complete_system'],
        intents['modify_model']: states['modify_model'],
        intents['modeling_help']: states['modeling_help'],
        intents['uml_spec']: states['uml_rag'],
        intents['generation']: states['generation'],
        intents['hello']: states['greetings'],
    }

    generation_st = states['generation']

    for state_name, fallback_name in [
        ('greetings', 'modeling_help'),
        ('create_single_element', 'create_single_element'),
        ('create_complete_system', 'create_complete_system'),
        ('modify_model', 'modify_model'),
        ('modeling_help', 'modeling_help'),
        ('uml_rag', 'greetings'),
        ('generation', 'generation'),
    ]:
        add_unified_transitions(
            states[state_name], intent_map, states[fallback_name], generation_st,
        )
