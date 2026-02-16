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
    build_switch_diagram_action,
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



def get_user_message(session: Session) -> str:
    """Extract normalized message using protocol adapters."""
    request = parse_assistant_request(session)
    return request.message or ""


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
        name='gpt-4o-mini',
        parameters={
            'temperature': 0.3,
            'max_tokens': 4096
        },
        num_previous_messages=0
    )
    
    if gpt is None:
        raise Exception("LLM initialization returned None")
    
    logger.info("LLM initialized successfully")
    
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
        llm_name='gpt-4o-mini',
        k=4,
        num_previous_messages=0
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
    llm_name='gpt-4o-mini',
    parameters={},
    use_intent_descriptions=True,
    use_training_sentences=False,
    use_entity_descriptions=True,
    use_entity_synonyms=False
)

agent.set_default_ic_config(ic_config)


def _compact_model_summary(model_data: Any, diagram_type: str) -> str:
    if not isinstance(model_data, dict):
        return f"{diagram_type}: no structured model available."

    if diagram_type in {"ClassDiagram", "ObjectDiagram", "StateMachineDiagram", "AgentDiagram"}:
        elements = model_data.get("elements")
        relationships = model_data.get("relationships")
        if isinstance(elements, dict) and isinstance(relationships, dict):
            return (
                f"{diagram_type}: {len(elements)} element(s), "
                f"{len(relationships)} relationship(s)."
            )

    if diagram_type == "GUINoCodeDiagram":
        pages = model_data.get("pages")
        if isinstance(pages, list):
            return f"{diagram_type}: {len(pages)} page(s)."

    if diagram_type == "QuantumCircuitDiagram":
        cols = model_data.get("cols")
        if isinstance(cols, list):
            return f"{diagram_type}: {len(cols)} circuit column(s)."

    return f"{diagram_type}: model metadata available."


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _extract_element_position(element: Dict[str, Any]) -> Optional[Dict[str, Optional[int]]]:
    if not isinstance(element, dict):
        return None

    bounds = element.get("bounds")
    if isinstance(bounds, dict):
        x = _to_int(bounds.get("x"))
        y = _to_int(bounds.get("y"))
        if x is not None and y is not None:
            return {
                "x": x,
                "y": y,
                "width": _to_int(bounds.get("width")),
                "height": _to_int(bounds.get("height")),
            }

    position = element.get("position")
    if isinstance(position, dict):
        x = _to_int(position.get("x"))
        y = _to_int(position.get("y"))
        if x is not None and y is not None:
            return {"x": x, "y": y, "width": None, "height": None}

    return None


def _is_primary_layout_element(diagram_type: str, element: Dict[str, Any]) -> bool:
    element_type = element.get("type")
    owner = element.get("owner")
    owner_is_root = not isinstance(owner, str) or not owner

    diagram_primary_types = {
        "ClassDiagram": {"Class"},
        "ObjectDiagram": {"Object"},
        "StateMachineDiagram": {"State", "StateInitialNode", "StateFinalNode"},
        "AgentDiagram": {"AgentState", "AgentIntent", "StateInitialNode"},
    }

    primary_types = diagram_primary_types.get(diagram_type)
    if isinstance(element_type, str) and primary_types:
        return element_type in primary_types

    noisy_types = {
        "ClassAttribute",
        "ClassMethod",
        "AgentStateBody",
        "AgentStateFallbackBody",
        "AgentIntentBody",
    }
    if isinstance(element_type, str) and element_type in noisy_types:
        return False

    return owner_is_root


def _build_layout_anchor_lines(model_data: Any, diagram_type: str, limit: int = 18) -> List[str]:
    if not isinstance(model_data, dict):
        return []

    elements = model_data.get("elements")
    if not isinstance(elements, dict):
        return []

    anchors: List[tuple[int, int, str]] = []
    for element_id, element in elements.items():
        if not isinstance(element, dict):
            continue
        if not _is_primary_layout_element(diagram_type, element):
            continue

        position = _extract_element_position(element)
        if not position:
            continue

        x = position["x"]
        y = position["y"]
        if not isinstance(x, int) or not isinstance(y, int):
            continue

        width = position.get("width")
        height = position.get("height")
        size_part = (
            f", w={width}, h={height}"
            if isinstance(width, int) and isinstance(height, int)
            else ""
        )
        element_type = element.get("type") if isinstance(element.get("type"), str) else "Element"
        name = element.get("name") if isinstance(element.get("name"), str) and element.get("name") else element_id
        line = f"- {element_type} '{name}': x={x}, y={y}{size_part}"
        anchors.append((y, x, line))

    anchors.sort(key=lambda item: (item[0], item[1]))
    return [line for _, _, line in anchors[:limit]]


def _build_workspace_context_block(
    request: AssistantRequest,
    target_diagram_type: str,
    target_model: Optional[Dict[str, Any]] = None,
) -> str:
    lines: List[str] = []
    lines.append(f"Target diagram type: {target_diagram_type}")
    lines.append(f"Active diagram type: {request.context.active_diagram_type or request.diagram_type}")

    if request.context.active_diagram_id:
        lines.append(f"Active diagram id: {request.context.active_diagram_id}")

    active_model = request.context.active_model or request.current_model
    if active_model is not None:
        lines.append(_compact_model_summary(active_model, request.context.active_diagram_type or request.diagram_type))

    if target_model is None:
        target_model = _resolve_target_model(request, target_diagram_type)
    layout_anchors = _build_layout_anchor_lines(target_model, target_diagram_type)
    if layout_anchors:
        lines.append("Existing layout anchors (avoid overlap with these):")
        lines.extend(layout_anchors)

    snapshot = request.context.project_snapshot
    if isinstance(snapshot, dict):
        project_name = snapshot.get("name")
        project_description = snapshot.get("description")
        if isinstance(project_name, str) and project_name.strip():
            lines.append(f"Project name: {project_name.strip()}")
        if isinstance(project_description, str) and project_description.strip():
            lines.append(f"Project description: {project_description.strip()}")

        diagrams = snapshot.get("diagrams")
        if isinstance(diagrams, dict):
            diagram_lines: List[str] = []
            for diagram_type, payload in diagrams.items():
                if not isinstance(payload, dict):
                    continue
                title = payload.get("title")
                model = payload.get("model")
                title_part = f" ({title})" if isinstance(title, str) and title.strip() else ""
                diagram_lines.append(f"- {diagram_type}{title_part}: {_compact_model_summary(model, diagram_type)}")
            if diagram_lines:
                lines.append("Project diagrams overview:")
                lines.extend(diagram_lines[:10])

    summaries = request.context.diagram_summaries or []
    if summaries:
        compact_summaries: List[str] = []
        for item in summaries:
            if not isinstance(item, dict):
                continue
            diagram_type = item.get("diagramType")
            title = item.get("title")
            if isinstance(diagram_type, str):
                if isinstance(title, str) and title.strip():
                    compact_summaries.append(f"{diagram_type} ({title.strip()})")
                else:
                    compact_summaries.append(diagram_type)
        if compact_summaries:
            lines.append("Diagram summaries: " + ", ".join(compact_summaries[:10]))

    return "Workspace context:\n" + "\n".join(lines)


def _resolve_target_model(request: AssistantRequest, target_diagram_type: str) -> Optional[Dict[str, Any]]:
    if target_diagram_type == request.context.active_diagram_type and isinstance(request.current_model, dict):
        return request.current_model

    snapshot = request.context.project_snapshot
    if isinstance(snapshot, dict):
        diagrams = snapshot.get("diagrams")
        if isinstance(diagrams, dict):
            target = diagrams.get(target_diagram_type)
            if isinstance(target, dict) and isinstance(target.get("model"), dict):
                return target.get("model")

    if isinstance(request.current_model, dict):
        return request.current_model
    if isinstance(request.context.active_model, dict):
        return request.context.active_model
    return None


def _resolve_object_reference_diagram(
    request: AssistantRequest,
    target_model: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Resolve the best available ClassDiagram model for ObjectDiagram grounding.

    Priority:
    1) Object diagram's own `referenceDiagramData` (if already set)
    2) Active in-memory ClassDiagram model from current context
    3) Current model when it is a ClassDiagram
    4) Project snapshot ClassDiagram model
    """
    if isinstance(target_model, dict):
        reference_diagram = target_model.get("referenceDiagramData")
        if isinstance(reference_diagram, dict):
            return reference_diagram

    active_diagram_type = request.context.active_diagram_type or request.diagram_type
    active_model = request.context.active_model if isinstance(request.context.active_model, dict) else None
    if active_diagram_type == "ClassDiagram" and isinstance(active_model, dict):
        if isinstance(active_model.get("elements"), dict):
            return active_model

    if active_diagram_type == "ClassDiagram" and isinstance(request.current_model, dict):
        if isinstance(request.current_model.get("elements"), dict):
            return request.current_model

    if isinstance(request.context.project_snapshot, dict):
        diagrams = request.context.project_snapshot.get("diagrams")
        if isinstance(diagrams, dict):
            class_diagram = diagrams.get("ClassDiagram")
            if isinstance(class_diagram, dict) and isinstance(class_diagram.get("model"), dict):
                return class_diagram["model"]

    return None


def _count_reference_classes(reference_diagram: Optional[Dict[str, Any]]) -> int:
    if not isinstance(reference_diagram, dict):
        return 0
    elements = reference_diagram.get("elements")
    if not isinstance(elements, dict):
        return 0
    return sum(
        1
        for element in elements.values()
        if isinstance(element, dict) and element.get("type") == "Class"
    )


def _build_request_for_target(
    base_request: AssistantRequest,
    target_diagram_type: str,
    target_diagram_id: Optional[str] = None,
    target_model: Optional[Dict[str, Any]] = None,
) -> AssistantRequest:
    resolved_diagram_id = target_diagram_id or resolve_diagram_id(base_request, target_diagram_type)
    resolved_model = target_model if isinstance(target_model, dict) else _resolve_target_model(base_request, target_diagram_type)

    context = WorkspaceContext(
        active_diagram_type=target_diagram_type,
        active_diagram_id=resolved_diagram_id,
        active_model=resolved_model,
        project_snapshot=base_request.context.project_snapshot,
        diagram_summaries=base_request.context.diagram_summaries,
    )

    raw_payload = dict(base_request.raw_payload or {})
    raw_context = raw_payload.get("context")
    if not isinstance(raw_context, dict):
        raw_context = {}
    raw_context.update(
        {
            "activeDiagramType": target_diagram_type,
            "activeDiagramId": resolved_diagram_id,
            "activeModel": resolved_model,
            "projectSnapshot": base_request.context.project_snapshot,
            "diagramSummaries": base_request.context.diagram_summaries,
        }
    )
    raw_payload["context"] = raw_context
    raw_payload["diagramType"] = target_diagram_type

    return AssistantRequest(
        action=base_request.action,
        protocol_version=base_request.protocol_version,
        client_mode=base_request.client_mode,
        session_id=base_request.session_id,
        message=base_request.message,
        diagram_type=target_diagram_type,
        diagram_id=resolved_diagram_id,
        current_model=resolved_model,
        context=context,
        raw_payload=raw_payload,
    )


def _build_generation_request(
    base_request: AssistantRequest,
    generator_type: str,
    config: Optional[Dict[str, Any]] = None,
    message_override: Optional[str] = None,
) -> AssistantRequest:
    config = config or {}
    override_message = message_override.strip() if isinstance(message_override, str) else ""
    if override_message:
        message = override_message if detect_generator_type(override_message) else f"generate {generator_type} {override_message}"
    else:
        inline_config: List[str] = []
        for key, value in config.items():
            if value is None:
                continue
            inline_config.append(f"{key}={value}")
        inline = " ".join(inline_config).strip()
        message = f"generate {generator_type}" + (f" {inline}" if inline else "")

    active_model = base_request.context.active_model if isinstance(base_request.context.active_model, dict) else base_request.current_model

    raw_payload = {
        "action": "user_message",
        "protocolVersion": "2.0",
        "clientMode": base_request.client_mode,
        "sessionId": base_request.session_id,
        "message": message,
        "context": {
            "activeDiagramType": base_request.context.active_diagram_type,
            "activeDiagramId": base_request.context.active_diagram_id,
            "activeModel": active_model,
            "projectSnapshot": base_request.context.project_snapshot,
            "diagramSummaries": base_request.context.diagram_summaries,
        },
    }

    return AssistantRequest(
        action="user_message",
        protocol_version="2.0",
        client_mode=base_request.client_mode,
        session_id=base_request.session_id,
        message=message,
        diagram_type=base_request.context.active_diagram_type or base_request.diagram_type,
        diagram_id=base_request.context.active_diagram_id or base_request.diagram_id,
        current_model=active_model,
        context=WorkspaceContext(
            active_diagram_type=base_request.context.active_diagram_type or base_request.diagram_type,
            active_diagram_id=base_request.context.active_diagram_id or base_request.diagram_id,
            active_model=active_model,
            project_snapshot=base_request.context.project_snapshot,
            diagram_summaries=base_request.context.diagram_summaries,
        ),
        raw_payload=raw_payload,
    )


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

    if request.context.active_diagram_type and request.context.active_diagram_type != target_diagram_type:
        switch_action = build_switch_diagram_action(
            target_diagram_type,
            reason=f"Your request includes {target_diagram_type}.",
        )
        _reply_payload(session, switch_action)

    handler = diagram_factory.get_handler(target_diagram_type)
    if not handler:
        logger.warning(f"[ModelOp] No handler for diagram type: {target_diagram_type}")
        _reply_message(
            session,
            f"{target_diagram_type} is not supported by the modeling handler yet.",
        )
        return None

    target_model = _resolve_target_model(request, target_diagram_type)
    modeling_prompt = (
        f"{operation_request}\n\n"
        f"{_build_workspace_context_block(request, target_diagram_type, target_model)}"
    )

    logger.debug(f"[ModelOp] Modeling prompt ({len(modeling_prompt)} chars): {modeling_prompt[:300]!r}")
    logger.debug(f"[ModelOp] Target model present: {target_model is not None}, type: {type(target_model).__name__}")

    if operation_mode == "single_element":
        if target_diagram_type == "ObjectDiagram":
            reference_diagram = _resolve_object_reference_diagram(request, target_model)
            reference_class_count = _count_reference_classes(reference_diagram)
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
            )
    elif operation_mode == "modify_model":
        result = handler.generate_modification(modeling_prompt, target_model)
    else:
        if target_diagram_type == "ObjectDiagram":
            reference_diagram = _resolve_object_reference_diagram(request, target_model)
            reference_class_count = _count_reference_classes(reference_diagram)
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
                    working_request = _build_request_for_target(working_request, executed_target)
            except Exception as error:
                logger.error(f"Error executing model operation {operation}: {error}")
                _reply_message(session, "I encountered an issue while applying a modeling step.")
            continue

        if operation_type == "generation":
            generator_type = operation.get("generatorType")
            if not isinstance(generator_type, str) or not generator_type:
                continue

            generation_message = operation.get("request") if isinstance(operation.get("request"), str) else None
            generation_request = _build_generation_request(
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
        answer = gpt.predict(
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
        answer = gpt.predict(help_prompt)
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
        fallback_response = gpt.predict(
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
            fallback_response = gpt.predict(
                f"""You are a UML specification expert. Answer the following question about UML:

{user_message}

Provide accurate information based on UML 2.x specifications."""
            )
            _reply_message(session, fallback_response)

uml_rag_state.set_body(uml_rag_body)


# RUN APPLICATION
if __name__ == '__main__':
    agent.run()
