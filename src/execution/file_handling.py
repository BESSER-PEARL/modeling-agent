"""File-attachment processing and conversion.

Handles uploaded files (images, PDFs, etc.) by converting them to diagram specs
via the file conversion handler, then either injecting directly or prompting
the user if the target diagram already has content. Screenshots and mockups of
a user interface become a GUI through the normal GUI creation path (#643).
"""

import logging
from typing import Optional

from baf.core.session import Session

import agent_context as ctx
from byok import user_openai_key
from model_config import MODEL_GENERATION_SMALL
from protocol.types import AssistantRequest, FileAttachment
from session_helpers import reply_payload
from model_utils import model_has_elements
from unified_classifier import get_or_classify
from utilities.image_inputs import data_url
from handlers.file_conversion_handler import (
    GUI_MOCKUP_ACTION,
    convert_file_to_diagram_spec,
    detect_file_type,
    image_mime_type,
)

from .model_operations import _build_existing_model_confirmation, execute_model_operation

logger = logging.getLogger(__name__)

_GUI = "GUINoCodeDiagram"
_MOCKUP_REQUEST = "Build the GUI shown in the attached mockup."


def _requested_diagram_type(session: Session, request: AssistantRequest) -> Optional[str]:
    """The diagram type the user's typed message asks for, if any.

    Reads the classifier verdict for this message (cached per event, so no
    extra call); an attachment sent without text has none.
    """
    if not (request.message or "").strip():
        return None
    try:
        from llm.provider import get_provider
        return get_or_classify(session, request, get_provider()).target_diagram_type
    except Exception:
        logger.warning("[FileConversion] No classifier verdict for the message", exc_info=True)
        return None


def _image_data_url(attachment: FileAttachment) -> str:
    mime = attachment.mime_type
    if not mime.startswith("image/"):
        mime = image_mime_type(attachment.filename)
    return data_url(mime, attachment.content_b64)


def handle_file_attachments(session: Session, request: AssistantRequest) -> bool:
    """Process file attachments if present.  Returns True if attachments were handled."""
    if not request.has_attachments:
        return False

    from utilities.model_resolution import resolve_target_model
    from utilities.model_context import compact_model_summary

    # The user's own OpenAI key pays for the vision path when it can; the
    # text path is BYOK-routed inside gpt_predict_json.
    openai_key = user_openai_key() or ctx.openai_api_key

    # Cap attachment count + per-file size BEFORE decoding — each attachment
    # triggers a sequential vision/LLM call, so an unbounded batch blocks the
    # single-threaded agent for minutes and runs up cost.
    MAX_ATTACHMENTS = 5
    MAX_FILE_B64_CHARS = 14_000_000  # ~10 MB once decoded
    attachments = list(request.attachments)
    if len(attachments) > MAX_ATTACHMENTS:
        logger.warning(
            "[FileConversion] %d attachments — processing first %d only",
            len(attachments), MAX_ATTACHMENTS,
        )
        reply_payload(session, {
            "action": "assistant_message",
            "message": (
                f"You attached {len(attachments)} files — I'll process the first "
                f"{MAX_ATTACHMENTS} to stay responsive. Send the rest in a follow-up."
            ),
        })
        attachments = attachments[:MAX_ATTACHMENTS]

    requested_type = _requested_diagram_type(session, request)
    mockups = []  # UI screenshots / mockups, built together as one GUI
    awaiting_answer = False

    for attachment in attachments:
        if len(attachment.content_b64 or "") > MAX_FILE_B64_CHARS:
            logger.warning(
                "[FileConversion] Skipping oversized attachment %s (%d b64 chars)",
                attachment.filename, len(attachment.content_b64 or ""),
            )
            reply_payload(session, {
                "action": "assistant_message",
                "message": f"Skipped **{attachment.filename}** — it's too large to process.",
            })
            continue
        logger.info(
            f"[FileConversion] Processing attachment: {attachment.filename} "
            f"({attachment.mime_type}, {len(attachment.content_b64)} b64 chars)"
        )
        is_image = detect_file_type(attachment.filename) == "image"
        if is_image and requested_type == _GUI:
            mockups.append(_image_data_url(attachment))
            continue
        # Text-path conversions produce a full diagram model → SMALL
        # generation tier instead of the classifier default (model_config).
        _predict_json = ctx.gpt_predict_json
        result = convert_file_to_diagram_spec(
            file_content_b64=attachment.content_b64,
            filename=attachment.filename,
            llm_predict=lambda prompt: _predict_json(prompt, model=MODEL_GENERATION_SMALL),
            openai_api_key=openai_key,
            target_diagram_type=requested_type,
        )
        if is_image and result.get("action") == GUI_MOCKUP_ACTION:
            mockups.append(_image_data_url(attachment))
            continue

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
            awaiting_answer = True
            break
        else:
            # No existing model — inject directly
            reply_payload(session, result)

    if mockups:
        if awaiting_answer:
            # One pending question at a time: the GUI would ask its own.
            reply_payload(session, {
                "action": "assistant_message",
                "message": (
                    "Once you've answered, send the screenshot again and I'll "
                    "build the GUI from it."
                ),
            })
        else:
            # The normal GUI creation path: same replace/keep question, class
            # diagram binding and progress as a typed request.
            execute_model_operation(
                session=session,
                request=request,
                operation={
                    "diagramType": _GUI,
                    "mode": "complete_system",
                    "request": (request.message or "").strip() or _MOCKUP_REQUEST,
                    "images": mockups,
                },
                default_mode="complete_system",
            )

    return True
