"""
Request Builders
----------------
Functions that create or re-target ``AssistantRequest`` objects for
cross-diagram orchestration and code-generation workflows.
"""

from typing import Any, Dict, List, Optional

from protocol.types import AssistantRequest, WorkspaceContext
from handlers.generation_handler import detect_generator_type
from orchestrator import resolve_diagram_id
from .model_resolution import resolve_target_model


def build_request_for_target(
    base_request: AssistantRequest,
    target_diagram_type: str,
    target_diagram_id: Optional[str] = None,
    target_model: Optional[Dict[str, Any]] = None,
) -> AssistantRequest:
    """Build a new AssistantRequest re-targeted to *target_diagram_type*."""
    resolved_diagram_id = target_diagram_id or resolve_diagram_id(base_request, target_diagram_type)
    resolved_model = target_model if isinstance(target_model, dict) else resolve_target_model(base_request, target_diagram_type)

    context = WorkspaceContext(
        active_diagram_type=target_diagram_type,
        active_diagram_id=resolved_diagram_id,
        active_model=resolved_model,
        project_snapshot=base_request.context.project_snapshot,
        diagram_summaries=base_request.context.diagram_summaries,
        current_diagram_indices=base_request.context.current_diagram_indices,
    )

    raw_payload = dict(base_request.raw_payload or {})
    raw_context = raw_payload.get("context")
    if not isinstance(raw_context, dict):
        raw_context = {}
    raw_context.update(
        {
            "activeDiagramType": target_diagram_type,
            "activeDiagramId": resolved_diagram_id,
            "projectSnapshot": base_request.context.project_snapshot,
            "diagramSummaries": base_request.context.diagram_summaries,
        }
    )
    # Remove legacy activeModel if it was present in the original payload.
    raw_context.pop("activeModel", None)
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


def build_generation_request(
    base_request: AssistantRequest,
    generator_type: str,
    config: Optional[Dict[str, Any]] = None,
    message_override: Optional[str] = None,
) -> AssistantRequest:
    """Build a synthetic AssistantRequest suitable for generation handlers."""
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

    # Derive the model from current_model (already resolved from snapshot by the adapter).
    resolved_model = base_request.current_model if isinstance(base_request.current_model, dict) else None

    raw_payload = {
        "action": "user_message",
        "protocolVersion": "2.0",
        "clientMode": base_request.client_mode,
        "sessionId": base_request.session_id,
        "message": message,
        "context": {
            "activeDiagramType": base_request.context.active_diagram_type,
            "activeDiagramId": base_request.context.active_diagram_id,
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
        current_model=resolved_model,
        context=WorkspaceContext(
            active_diagram_type=base_request.context.active_diagram_type or base_request.diagram_type,
            active_diagram_id=base_request.context.active_diagram_id or base_request.diagram_id,
            active_model=resolved_model,
            project_snapshot=base_request.context.project_snapshot,
            diagram_summaries=base_request.context.diagram_summaries,
            current_diagram_indices=base_request.context.current_diagram_indices,
        ),
        raw_payload=raw_payload,
    )
