"""
Model Resolution Utilities
---------------------------
**Backward-compatibility shim.**

All functions that used to live in this monolithic module have been split
into focused sub-modules inside ``utilities/``:

- ``model_context``      — compact & detailed model summaries
- ``layout_helpers``     — position extraction, layout anchors
- ``model_resolution``   — target-model & reference-diagram resolution
- ``class_metadata``     — ClassDiagram metadata extraction for GUI charts
- ``workspace_context``  — workspace context block builder
- ``request_builders``   — AssistantRequest factory helpers

This file re-exports every public name so that existing ``from
utilities.model_helpers import X`` statements keep working without any
changes.
"""

from __future__ import annotations

from typing import Any

# Re-export: model context (summaries)
from .model_context import (                        # noqa: F401
    compact_model_summary,
    detailed_model_summary,
)

# Re-export: layout helpers
from .layout_helpers import (                       # noqa: F401
    to_int,
    extract_element_position,
    is_primary_layout_element,
    build_layout_anchor_lines,
)

# Re-export: model resolution
from .model_resolution import (                     # noqa: F401
    resolve_target_model,
    resolve_object_reference_diagram,
    count_reference_classes,
)

# Re-export: class metadata
from .class_metadata import (                       # noqa: F401
    extract_class_metadata,
    format_class_metadata_for_prompt,
)

# Re-export: workspace context
from .workspace_context import (                    # noqa: F401
    build_workspace_context_block,
)


def build_request_for_target(*args: Any, **kwargs: Any):  # noqa: D401
    """Lazy proxy for ``request_builders.build_request_for_target``."""
    from .request_builders import build_request_for_target as _build_request_for_target

    return _build_request_for_target(*args, **kwargs)


def build_generation_request(*args: Any, **kwargs: Any):  # noqa: D401
    """Lazy proxy for ``request_builders.build_generation_request``."""
    from .request_builders import build_generation_request as _build_generation_request

    return _build_generation_request(*args, **kwargs)
