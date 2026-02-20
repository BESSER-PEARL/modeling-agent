"""
Utilities Package
Common helper functions extracted from the main agent module.

Sub-modules:
  model_context      — compact & detailed model summaries
  layout_helpers     — position extraction, layout anchors
  model_resolution   — target-model & reference-diagram resolution
  class_metadata     — ClassDiagram metadata for GUI charts
  workspace_context  — workspace context block builder
  request_builders   — AssistantRequest factory helpers

For backward compatibility ``model_helpers`` re-exports all public names.
"""

# All public names are re-exported through the backward-compat shim so that
# ``from utilities.model_helpers import X`` still works.  The package-level
# ``__init__`` also exports them for ``from utilities import X``.

from .model_context import (
    compact_model_summary,
    detailed_model_summary,
)
from .layout_helpers import (
    to_int,
    extract_element_position,
    is_primary_layout_element,
    build_layout_anchor_lines,
)
from .model_resolution import (
    resolve_target_model,
    resolve_object_reference_diagram,
    count_reference_classes,
)
from .class_metadata import (
    extract_class_metadata,
    format_class_metadata_for_prompt,
)
from .workspace_context import (
    build_workspace_context_block,
)
from .request_builders import (
    build_request_for_target,
    build_generation_request,
)

__all__ = [
    'compact_model_summary',
    'detailed_model_summary',
    'to_int',
    'extract_element_position',
    'is_primary_layout_element',
    'build_layout_anchor_lines',
    'resolve_target_model',
    'resolve_object_reference_diagram',
    'count_reference_classes',
    'extract_class_metadata',
    'format_class_metadata_for_prompt',
    'build_workspace_context_block',
    'build_request_for_target',
    'build_generation_request',
]
