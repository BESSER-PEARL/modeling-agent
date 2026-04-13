"""Progress reporting for multi-step operation plans."""

import logging

from baf.core.session import Session

from session_helpers import reply_progress

logger = logging.getLogger(__name__)


def _report_progress(session: Session, current_idx: int, total: int, operation: dict):
    """Send a lightweight progress indicator for multi-step plans.

    Only emits a message when there are 2+ steps — single-step plans
    don't need a "Step 1/1" indicator.
    """
    if total <= 1:
        return

    op_type = operation.get("type", "unknown")
    diagram_type = operation.get("diagramType", "")

    progress_msg = f"Step {current_idx + 1}/{total}: "
    if op_type == "model":
        progress_msg += f"Creating {diagram_type}\u2026" if diagram_type else "Processing model\u2026"
    elif op_type == "generation":
        gen_type = operation.get("generatorType", "code")
        progress_msg += f"Generating {gen_type}\u2026"
    else:
        progress_msg += "Processing\u2026"

    reply_progress(session, progress_msg, step=current_idx + 1, total=total)
