# Intelligent UML Modeling Assistant agent
# Supports: ClassDiagram, ObjectDiagram, StateMachineDiagram, AgentDiagram, GUINoCodeDiagram, QuantumCircuitDiagram

import logging
import json
import os
from typing import Dict, Any, List, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from chromadb.config import Settings as ChromaSettings
except Exception:
    ChromaSettings = None

from besser.agent import nlp
from besser.agent.core.agent import Agent
from besser.agent.core.session import Session
from besser.agent.library.transition.events.base_events import ReceiveJSONEvent
from besser.agent.exceptions.logger import logger
from besser.agent.nlp.intent_classifier.intent_classifier_configuration import LLMIntentClassifierConfiguration
from besser.agent.nlp.llm.llm_openai_api import LLMOpenAI
from besser.agent.nlp.rag.rag import RAGMessage, RAG

from diagram_handlers.factory import DiagramHandlerFactory, get_diagram_type_info
from protocol.adapters import parse_assistant_request
from protocol.types import AssistantRequest, WorkspaceContext
from routing.intents import GENERATION_INTENT_NAME
from handlers.generation_handler import (
    handle_generation_request,
    should_route_to_generation,
    detect_generator_type,
)
from orchestrator import (
    plan_assistant_operations,
    determine_target_diagram_type,
    resolve_diagram_id,
)
from utilities.model_helpers import (
    resolve_target_model,
    resolve_object_reference_diagram,
    count_reference_classes,
    build_workspace_context_block,
    build_request_for_target,
    build_generation_request,
    extract_class_metadata,
)


# Configure the logging module
logger.setLevel(logging.INFO)

# Create the agent
agent = Agent('uml_modeling_agent')

agent.load_properties('config.ini')
logger.info(f"Agent properties loaded from config.ini (name={agent.name})")

websocket_platform = agent.use_websocket_platform(use_ui=False)

# Disable Chroma telemetry to avoid runtime noise/errors with incompatible telemetry deps.
os.environ.setdefault('ANONYMIZED_TELEMETRY', 'False')
os.environ.setdefault('CHROMA_TELEMETRY_ENABLED', 'False')



# Maximum user message length (characters).  Messages beyond this are
# truncated to avoid blowing the LLM context window.  ~12 000 chars ≈
# ~3 000 tokens, leaving plenty of headroom inside the 1M context of
# gpt-4.1-mini while still fitting any realistic request.
MAX_USER_MESSAGE_CHARS = 12_000


def get_user_message(session: Session) -> str:
    """Extract normalized message using protocol adapters."""
    request = parse_assistant_request(session)
    message = request.message or ""
    if len(message) > MAX_USER_MESSAGE_CHARS:
        logger.warning(
            f"User message truncated from {len(message)} to {MAX_USER_MESSAGE_CHARS} chars"
        )
        message = message[:MAX_USER_MESSAGE_CHARS]
    return message


def get_diagram_type(session: Session, default: str = 'ClassDiagram') -> str:
    """Extract normalized diagram type using protocol adapters."""
    request = parse_assistant_request(session, default_diagram_type=default)
    return request.diagram_type or default


def get_current_model(session: Session) -> Optional[Dict[str, Any]]:
    """Extract normalized current model from protocol adapters."""
    request = parse_assistant_request(session)
    return request.current_model


# Intent matching condition functions for JSON events
def json_intent_matches(session: Session, params: Dict[str, Any]) -> bool:
    """Check if the predicted intent matches the target intent for JSON events."""
    target_intent_name = params.get('intent_name')
    if not target_intent_name:
        return False
    
    # The ReceiveJSONEvent should have predicted_intent after intent prediction
    if hasattr(session.event, 'predicted_intent') and session.event.predicted_intent:
        matched_intent = session.event.predicted_intent.intent
        return matched_intent.name == target_intent_name
    
    return False


def json_no_intent_matched(session: Session) -> bool:
    """Check if no specific intent was matched (fallback).
    
    Note: This function takes only session (no params) because it doesn't need any parameters.
    """
    if hasattr(session.event, 'predicted_intent') and session.event.predicted_intent:
        matched_intent = session.event.predicted_intent.intent
        return matched_intent.name == 'fallback_intent'
    return True


def _reply_message(session: Session, message: str):
    """Send assistant message, wrapped for v2 protocol clients."""
    request = parse_assistant_request(session)
    if request.is_v2:
        session.reply(json.dumps({
            "action": "assistant_message",
            "message": message
        }))
    else:
        session.reply(message)


def _reply_payload(session: Session, payload: Dict[str, Any]):
    """Send JSON payload response for both protocol versions."""
    logger.info(
        f"[Reply] Sending payload: action={payload.get('action')}, "
        f"diagramType={payload.get('diagramType')}, "
        f"message={str(payload.get('message', ''))[:100]!r}"
    )
    logger.debug(f"[Reply] Full payload keys: {list(payload.keys())}")
    session.reply(json.dumps(payload))


def route_to_generation(session: Session) -> bool:
    """Detect generation workflow requests or frontend callback events."""
    request = parse_assistant_request(session)
    return should_route_to_generation(session, request)


try:
    gpt = LLMOpenAI(
        agent=agent,
        name='gpt-4.1-mini',
        parameters={
            'temperature': 0.2,
            'max_completion_tokens': 8192,
            'response_format': {'type': 'json_object'},
        },
        num_previous_messages=4
    )

    # Keep a second LLM handle WITHOUT json_object enforcement for free-text
    # responses (help, greetings, RAG fallback) where JSON mode would break.
    gpt_text = LLMOpenAI(
        agent=agent,
        name='gpt-4.1-nano',
        parameters={
            'temperature': 0.4,
            'max_completion_tokens': 4096,
        },
        num_previous_messages=4
    )
    
    if gpt is None:
        raise Exception("LLM initialization returned None")
    
    logger.info("LLMs initialized: gpt-4.1-mini (json), gpt-4.1-nano (text), conversation memory=4")
    
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    logger.error("Check config.ini: nlp.openai.api_key must be a valid OpenAI API key.")
    logger.error("See https://platform.openai.com/api-keys")
    raise SystemExit(1)

# Create Vector Store for UML Specification RAG
try:
    chroma_kwargs = {
        'embedding_function': OpenAIEmbeddings(openai_api_key=agent.get_property(nlp.OPENAI_API_KEY)),
        'persist_directory': 'uml_vector_store',
    }
    if ChromaSettings is not None:
        chroma_kwargs['client_settings'] = ChromaSettings(anonymized_telemetry=False)

    vector_store: Chroma = Chroma(**chroma_kwargs)
    # Create text splitter (RAG creates a vector for each chunk)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    # Create the RAG for UML specification queries
    uml_rag = RAG(
        agent=agent,
        vector_store=vector_store,
        splitter=splitter,
        llm_name='gpt-4.1-mini',
        k=4,
        num_previous_messages=4
    )
    
    uml_rag.llm_prompt = """You are a UML (Unified Modeling Language) specification expert. Based on the context retrieved from the UML specification documents, answer the user's question about UML concepts, notation, semantics, or best practices.

If the context contains relevant information, use it to provide an accurate and detailed answer.
If you don't find the answer in the context, say that you don't have that specific information in the UML specification documents, but you can provide general guidance based on your knowledge.

Be precise and reference specific UML concepts when applicable. Use clear examples when helpful."""
    
    # Uncomment the following line to load UML specification PDFs into the vector store
    # uml_rag.load_pdfs('./uml_specs')
    
    logger.info("UML RAG initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize UML RAG: {e}. RAG features will be disabled.")
    uml_rag = None

# Initialize diagram handler factory
diagram_factory = DiagramHandlerFactory(gpt)
logger.info(f"Diagram handlers initialized: {', '.join(diagram_factory.get_supported_types())}")

ic_config = LLMIntentClassifierConfiguration(
    llm_name='gpt-4.1-mini',
    parameters={},
    use_intent_descriptions=True,
    use_training_sentences=False,
    use_entity_descriptions=True,
    use_entity_synonyms=False
)

agent.set_default_ic_config(ic_config)


def _resolve_class_diagram(request: AssistantRequest) -> Optional[Dict[str, Any]]:
    """Return the ClassDiagram model from the workspace context, if available."""
    snapshot = request.context.project_snapshot
    if isinstance(snapshot, dict):
        diagrams = snapshot.get("diagrams")
        if isinstance(diagrams, dict):
            cd = diagrams.get("ClassDiagram")
            if isinstance(cd, dict) and isinstance(cd.get("model"), dict):
                return cd["model"]
    if request.context.active_diagram_type == "ClassDiagram" and isinstance(request.current_model, dict):
        return request.current_model
    return None


def _execute_model_operation(
    session: Session,
    request: AssistantRequest,
    operation: Dict[str, Any],
    default_mode: str,
) -> Optional[str]:
    target_diagram_type = operation.get("diagramType")
    if not isinstance(target_diagram_type, str) or not target_diagram_type:
        target_diagram_type = determine_target_diagram_type(request, last_intent=session.get("last_matched_intent"))

    operation_mode = operation.get("mode")
    if not isinstance(operation_mode, str) or not operation_mode:
        operation_mode = default_mode

    operation_request = operation.get("request")
    if not isinstance(operation_request, str) or not operation_request.strip():
        operation_request = request.message
    operation_request = operation_request.strip()

    logger.info(
        f"[ModelOp] Executing: diagram={target_diagram_type}, mode={operation_mode}, "
        f"request={operation_request[:120]!r}"
    )

    # NOTE: We no longer send a standalone switch_diagram action here.
    # The result payload already carries `diagramType`, and the frontend's
    # handleInjection → ensureTargetDiagramReady will switch automatically.
    # Sending switch_diagram separately caused a race condition where the
    # frontend could process the switch and the injection asynchronously.

    # ── GUI Auto-Generate shortcut ──────────────────────────────────────
    # When the user asks for a complete GUI and there is an existing Class
    # Diagram with at least one class, delegate to the frontend's proven
    # autoGenerateGUIFromClassDiagram instead of LLM-generating components.
    # The frontend reads class metadata directly from the Apollon store and
    # builds navigation, tables, and method buttons via the GrapesJS API.
    if target_diagram_type == "GUINoCodeDiagram" and operation_mode in ("complete_system", None, ""):
        class_diagram_model = _resolve_class_diagram(request)
        if isinstance(class_diagram_model, dict):
            elements = class_diagram_model.get("elements")
            if isinstance(elements, dict) and len(elements) > 0:
                logger.info("[ModelOp] Routing GUI complete_system to frontend auto-generate")
                _reply_payload(session, {
                    "action": "auto_generate_gui",
                    "diagramType": "GUINoCodeDiagram",
                    "message": (
                        "I'll generate the GUI automatically from your Class Diagram. "
                        "Each class will get its own page with a data table and method buttons."
                    ),
                })
                return target_diagram_type

    handler = diagram_factory.get_handler(target_diagram_type)
    if not handler:
        logger.warning(f"[ModelOp] No handler for diagram type: {target_diagram_type}")
        _reply_message(
            session,
            f"{target_diagram_type} is not supported by the modeling handler yet.",
        )
        return None

    target_model = resolve_target_model(request, target_diagram_type)
    modeling_prompt = (
        f"{operation_request}\n\n"
        f"{build_workspace_context_block(request, target_diagram_type, target_model)}"
    )

    # ── Resolve class metadata for GUI diagram (charts/tables need it) ──
    gui_class_metadata = None
    if target_diagram_type == "GUINoCodeDiagram":
        class_diagram_model = _resolve_class_diagram(request)
        if isinstance(class_diagram_model, dict):
            gui_class_metadata = extract_class_metadata(class_diagram_model)
            if gui_class_metadata:
                logger.info(
                    f"[ModelOp] Resolved {len(gui_class_metadata)} class(es) for GUI chart binding"
                )

    logger.debug(f"[ModelOp] Modeling prompt ({len(modeling_prompt)} chars): {modeling_prompt[:300]!r}")
    logger.debug(f"[ModelOp] Target model present: {target_model is not None}, type: {type(target_model).__name__}")

    if operation_mode == "single_element":
        if target_diagram_type == "ObjectDiagram":
            reference_diagram = resolve_object_reference_diagram(request, target_model)
            reference_class_count = count_reference_classes(reference_diagram)
            if reference_class_count > 0:
                logger.info(
                    f"[ModelOp] ObjectDiagram reference resolved with {reference_class_count} class(es)."
                )
            else:
                logger.warning(
                    "[ModelOp] ObjectDiagram reference is missing or empty; output may drift."
                )
            result = handler.generate_single_element(
                modeling_prompt,
                reference_diagram=reference_diagram,
                existing_model=target_model,
            )
        else:
            result = handler.generate_single_element(
                modeling_prompt,
                existing_model=target_model,
                class_metadata=gui_class_metadata,
            )
    elif operation_mode == "modify_model":
        extra_kwargs = {"class_metadata": gui_class_metadata}
        if target_diagram_type == "ObjectDiagram":
            reference_diagram = resolve_object_reference_diagram(request, target_model)
            reference_class_count = count_reference_classes(reference_diagram)
            if reference_class_count > 0:
                logger.info(
                    f"[ModelOp] ObjectDiagram modify reference resolved with {reference_class_count} class(es)."
                )
            else:
                logger.warning(
                    "[ModelOp] ObjectDiagram modify reference is missing or empty; output may drift."
                )
            extra_kwargs["reference_diagram"] = reference_diagram
        result = handler.generate_modification(
            modeling_prompt,
            target_model,
            **extra_kwargs,
        )
    else:
        if target_diagram_type == "ObjectDiagram":
            reference_diagram = resolve_object_reference_diagram(request, target_model)
            reference_class_count = count_reference_classes(reference_diagram)
            if reference_class_count > 0:
                logger.info(
                    f"[ModelOp] ObjectDiagram reference resolved with {reference_class_count} class(es)."
                )
            else:
                logger.warning(
                    "[ModelOp] ObjectDiagram reference is missing or empty; output may drift."
                )
            result = handler.generate_complete_system(
                modeling_prompt,
                reference_diagram=reference_diagram,
                existing_model=target_model,
            )
        else:
            result = handler.generate_complete_system(
                modeling_prompt,
                existing_model=target_model,
                class_metadata=gui_class_metadata,
            )

    logger.info(
        f"[ModelOp] Handler result: action={result.get('action') if isinstance(result, dict) else 'N/A'}, "
        f"has_message={bool(result.get('message')) if isinstance(result, dict) else False}"
    )
    logger.debug(f"[ModelOp] Full result keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")

    if not isinstance(result, dict):
        _reply_message(session, f"I could not create a valid {target_diagram_type} response.")
        return None

    result["diagramType"] = target_diagram_type
    diagram_id = resolve_diagram_id(request, target_diagram_type)
    if isinstance(diagram_id, str):
        result["diagramId"] = diagram_id

    _reply_payload(session, result)
    return target_diagram_type


def _execute_planned_operations(
    session: Session,
    request: AssistantRequest,
    default_mode: str,
    matched_intent: Optional[str],
) -> None:
    operations = plan_assistant_operations(
        request=request,
        default_mode=default_mode,
        matched_intent=matched_intent,
        llm_predict=gpt.predict,
    )

    if not operations:
        _reply_message(session, "I couldn't determine an execution plan from your request.")
        return

    working_request = request

    for operation in operations:
        if not isinstance(operation, dict):
            continue

        operation_type = operation.get("type")
        if operation_type == "model":
            try:
                executed_target = _execute_model_operation(session, working_request, operation, default_mode=default_mode)
                if isinstance(executed_target, str) and executed_target:
                    working_request = build_request_for_target(working_request, executed_target)
            except Exception as error:
                logger.error(f"Error executing model operation {operation}: {error}")
                _reply_message(session, "I encountered an issue while applying a modeling step.")
            continue

        if operation_type == "generation":
            generator_type = operation.get("generatorType")
            if not isinstance(generator_type, str) or not generator_type:
                continue

            generation_message = operation.get("request") if isinstance(operation.get("request"), str) else None
            generation_request = build_generation_request(
                working_request,
                generator_type=generator_type,
                config=operation.get("config") if isinstance(operation.get("config"), dict) else {},
                message_override=generation_message,
            )
            try:
                response_payload = handle_generation_request(session, generation_request)
            except Exception as error:
                logger.error(f"Error executing generation operation {operation}: {error}")
                response_payload = {
                    "action": "agent_error",
                    "code": "generation_handler_error",
                    "message": f"Failed to process {generator_type} generation request.",
                    "retryable": True,
                }

            if isinstance(response_payload, dict):
                _reply_payload(session, response_payload)
            elif isinstance(response_payload, str):
                _reply_message(session, response_payload)

# STATES
greetings_state = agent.new_state('greetings_state', initial=True)
create_single_element_state = agent.new_state('create_single_element_state')
create_complete_system_state = agent.new_state('create_complete_system_state')
modify_model_state = agent.new_state('modify_model_state')
modeling_help_state = agent.new_state('modeling_help_state')
uml_rag_state = agent.new_state('uml_rag_state')
generation_state = agent.new_state('generation_state')

# INTENTS
hello_intent = agent.new_intent(
    name='hello_intent',
    description='The user greets you or wants to start a conversation'
)

create_single_element_intent = agent.new_intent(
    name='create_single_element_intent',
    description='The user wants to create exactly ONE single UML element. Examples: "create a class called User", "add a Person class", "make one state", "create an object instance". This is for creating ONE element only, NOT multiple elements or systems.'
)

create_complete_system_intent = agent.new_intent(
    name='create_complete_system_intent',
    description='The user wants to create a complete system, diagram, or multiple classes/elements. Keywords: "create a library system", "create a class diagram for", "design an e-commerce", "generate a banking application", "build a system", "create a diagram for", "model a", "create classes for". This is for creating MULTIPLE elements or a complete model.'
)

modify_model_intent = agent.new_intent(
    name='modify_model_intent',
    description='The user wants to modify, change, update, edit, add to, or connect elements in an EXISTING UML model. Keywords: "add relationship", "connect", "add inheritance", "add generalization", "relate", "modify class", "change attribute", "update method", "delete", "remove", "rename", "add association", "add composition", "add aggregation", "link classes"'
)

modeling_help_intent = agent.new_intent(
    name='modeling_help_intent',
    description='The user asks for help with UML modeling, design patterns, or modeling concepts'
)

uml_spec_intent = agent.new_intent(
    name='uml_spec_intent',
    description='The user asks theoretical questions about the official UML specification document, UML standards, or formal UML definitions. Keywords: "according to UML specification", "what does UML standard say", "UML 2.5 specification", "OMG specification", "formal UML definition". This is NOT for creating diagrams, only for asking about the UML specification document itself.'
)

generation_intent = agent.new_intent(
    name=GENERATION_INTENT_NAME,
    description='The user wants to generate code/artifacts (e.g., django, backend, web app, sql, jsonschema, qiskit, python, java, pydantic, agent).'
)

# STATE BODY DEFINITIONS


def global_fallback_body(session: Session):
    """Handle unrecognized messages."""
    user_message = get_user_message(session) or "your message"
    try:
        answer = gpt_text.predict(
            f"You are a UML modeling assistant. The user said: '{user_message}'. "
            "If this is related to UML modeling, suggest how you can help them create models, classes, or diagrams. "
            "Otherwise, politely explain that you specialize in UML modeling assistance."
        )
        _reply_message(session, answer)
    except Exception as e:
        logger.error(f"Error in global_fallback_body: {e}")
        _reply_message(session, "I'm not sure how to help with that. Try asking me to create a class, design a system, or modify your diagram.")

agent.set_global_fallback_body(global_fallback_body)

def greetings_body(session: Session):
    """Send a greeting message when the user first connects or says hello."""
    greeting_message = (
        "Hello! I'm your modeling assistant.\n\n"
        "I can help you:\n"
        "- Create classes: \"Create a User class\"\n"
        "- Build systems: \"Create a library management system\"\n"
        "- Create agent diagrams: \"Create an agent\"\n"
        "- Modify diagrams: \"Add transition from welcome to menu\"\n"
        "- UML specification: \"What does UML say about association classes?\"\n\n"
        "What would you like to create?"
    )

    # On initial state entry, session.event is None and connection isn't ready yet
    # Wait for the frontend's "hello" message to trigger the greeting
    if session.event is None:
        return
    
    # Check if this is a hello intent
    is_hello_intent = False
    if hasattr(session.event, 'predicted_intent') and session.event.predicted_intent:
        is_hello_intent = session.event.predicted_intent.intent.name == 'hello_intent'
    
    # If user said hello and we haven't greeted yet, send full greeting
    if is_hello_intent and not session.get('has_greeted'):
        _reply_message(session, greeting_message)
        session.set('has_greeted', True)
        return
    
    # If user said hello again after initial greeting, send short response
    if is_hello_intent and session.get('has_greeted'):
        _reply_message(session, "Hello again! How can I help you with UML modeling?")
        return


greetings_state.set_body(greetings_body)

def _modeling_state_body(session: Session, intent_name: str, default_mode: str, empty_msg: str):
    """Unified handler for all modeling operations (single element, system, modification)."""
    session.set('last_matched_intent', intent_name)
    request = parse_assistant_request(session)

    if not request.message:
        _reply_message(session, empty_msg)
        return

    try:
        _execute_planned_operations(
            session=session,
            request=request,
            default_mode=default_mode,
            matched_intent=intent_name,
        )
    except Exception as e:
        logger.error(f"Error in {intent_name}: {e}", exc_info=True)
        _reply_message(session, "Something went wrong while processing your request. Could you rephrase it?")


def create_single_element_body(session: Session):
    """Generate a single UML element based on the user's request."""
    _modeling_state_body(
        session,
        intent_name='create_single_element_intent',
        default_mode='single_element',
        empty_msg="What element would you like me to create? For example: 'Create a User class'",
    )

create_single_element_state.set_body(create_single_element_body)


def create_complete_system_body(session: Session):
    """Generate a complete system with multiple elements and relationships."""
    _modeling_state_body(
        session,
        intent_name='create_complete_system_intent',
        default_mode='complete_system',
        empty_msg="What system would you like me to design? For example: 'Create a library management system'",
    )

create_complete_system_state.set_body(create_complete_system_body)

def modify_modeling_body(session: Session):
    """Apply modifications to an existing UML model."""
    _modeling_state_body(
        session,
        intent_name='modify_model_intent',
        default_mode='modify_model',
        empty_msg="What changes would you like me to make to the model?",
    )

modify_model_state.set_body(modify_modeling_body)

def modeling_help_body(session: Session):
    """Offer guidance or clarifying questions when the user needs modeling help."""
    session.set('last_matched_intent', 'modeling_help_intent')
    request = parse_assistant_request(session)

    if not request.message:
        _reply_message(
            session,
            "I can help you with UML modeling! Try asking me to create a class, design a system, or modify your diagram.",
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
        answer = gpt_text.predict(help_prompt)
        _reply_message(session, answer)
    except Exception as e:
        logger.error(f"Error in modeling_help_body: {e}", exc_info=True)
        _reply_message(session, "I had trouble preparing guidance. Could you try again?")

modeling_help_state.set_body(modeling_help_body)


def generation_body(session: Session):
    """Handle assistant-driven code generation orchestration."""
    session.set('last_matched_intent', GENERATION_INTENT_NAME)
    request = parse_assistant_request(session)

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
        _reply_message(session, "I could not process your generation request.")
        return

    _reply_payload(session, response_payload)


generation_state.set_body(generation_body)


def add_unified_transitions(state, intents_map, fallback_state):
    """Add both text and JSON event transitions for a state.
    
    Args:
        state: The state to add transitions to
        intents_map: Dict mapping intent objects to destination states
        fallback_state: State to go to when no intent matches
    """
    # Direct generation route from JSON payload/callback events
    state.when_event(ReceiveJSONEvent())\
        .with_condition(route_to_generation)\
        .go_to(generation_state)

    # Text event transitions (backward compatibility)
    for intent, dest_state in intents_map.items():
        state.when_intent_matched(intent).go_to(dest_state)
    
    # JSON event transitions (unified messages)
    for intent, dest_state in intents_map.items():
        state.when_event(ReceiveJSONEvent())\
            .with_condition(json_intent_matches, {'intent_name': intent.name})\
            .go_to(dest_state)
    
    # Fallback transitions
    state.when_event(ReceiveJSONEvent())\
        .with_condition(json_no_intent_matched)\
        .go_to(fallback_state)
    state.when_no_intent_matched().go_to(fallback_state)


# Wire up identical intent → state routing for all modeling/rag/generation states.
# Each state can reach any other state; the fallback is itself (stay put).
_STANDARD_INTENT_MAP = {
    create_single_element_intent: create_single_element_state,
    create_complete_system_intent: create_complete_system_state,
    modify_model_intent: modify_model_state,
    modeling_help_intent: modeling_help_state,
    uml_spec_intent: uml_rag_state,
    generation_intent: generation_state,
    hello_intent: greetings_state,
}

for _state, _fallback in [
    (greetings_state, modeling_help_state),
    (create_single_element_state, create_single_element_state),
    (create_complete_system_state, create_complete_system_state),
    (modify_model_state, modify_model_state),
    (modeling_help_state, modeling_help_state),
    (uml_rag_state, greetings_state),
    (generation_state, generation_state),
]:
    add_unified_transitions(_state, _STANDARD_INTENT_MAP, _fallback)


# UML RAG STATE BODY

def uml_rag_body(session: Session):
    """Answer UML specification questions using RAG."""
    session.set('last_matched_intent', 'uml_spec_intent')
    user_message = get_user_message(session)
    
    if not user_message:
        _reply_message(session, "Please ask a question about UML specifications.")
        return
    
    if uml_rag is None:
        # Fallback if RAG is not initialized
        fallback_response = gpt_text.predict(
            f"""You are a UML specification expert. Answer the following question about UML:

{user_message}

Provide accurate information based on UML 2.x specifications. Be precise and reference specific UML concepts when applicable."""
        )
        _reply_message(session, fallback_response)
    else:
        try:
            rag_message: RAGMessage = session.run_rag(user_message)
            # Send only the answer text, not the full RAG JSON
            _reply_message(session, rag_message.answer)
        except Exception as e:
            logger.error(f"Error in uml_rag_body: {e}")
            # Fallback to LLM if RAG fails
            fallback_response = gpt_text.predict(
                f"""You are a UML specification expert. Answer the following question about UML:

{user_message}

Provide accurate information based on UML 2.x specifications."""
            )
            _reply_message(session, fallback_response)

uml_rag_state.set_body(uml_rag_body)


# RUN APPLICATION
if __name__ == '__main__':
    agent.run()
