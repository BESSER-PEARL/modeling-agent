"""File-attachment processing and conversion.

Handles uploaded files (images, PDFs, etc.) by converting them to diagram specs
via the file conversion handler, then either injecting directly or prompting
the user if the target diagram already has content.
"""

import logging

from baf.core.session import Session

import agent_context as ctx
from protocol.types import AssistantRequest
from session_helpers import reply_payload
from model_utils import model_has_elements
from handlers.file_conversion_handler import convert_file_to_diagram_spec

from .model_operations import _build_existing_model_confirmation

logger = logging.getLogger(__name__)


def handle_file_attachments(session: Session, request: AssistantRequest) -> bool:
    """Process file attachments if present.  Returns True if attachments were handled."""
    if not request.has_attachments:
        return False

    from utilities.model_resolution import resolve_target_model
    from utilities.model_context import compact_model_summary

    openai_key = ctx.openai_api_key

    for attachment in request.attachments:
        logger.info(
            f"[FileConversion] Processing attachment: {attachment.filename} "
            f"({attachment.mime_type}, {len(attachment.content_b64)} b64 chars)"
        )
        result = convert_file_to_diagram_spec(
            file_content_b64=attachment.content_b64,
            filename=attachment.filename,
            llm_predict=ctx.gpt_predict_json,
            openai_api_key=openai_key,
        )

        # Errors are sent directly
        if result.get("action") != "inject_complete_system":
            reply_payload(session, result)
            continue

        # Check if the target diagram already has elements
        target_diagram_type = result.get("diagramType", "ClassDiagram")
        existing_model = resolve_target_model(request, target_diagram_type)

        if model_has_elements(existing_model):
            summary = compact_model_summary(existing_model, target_diagram_type)

            _build_existing_model_confirmation(
                session=session,
                request=request,
                target_diagram_type=target_diagram_type,
                existing_summary=summary,
                pending_data={
                    'message': attachment.filename,
                    'diagram_type': target_diagram_type,
                    'precomputed_payload': result,
                },
                source_description=f"I extracted a {target_diagram_type} from '{attachment.filename}'",
            )
            break
        else:
            # No existing model — inject directly
            reply_payload(session, result)

    return True
