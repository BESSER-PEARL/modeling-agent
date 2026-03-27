"""Execution engine for the modeling agent.

This package handles model operations, file processing, and multi-operation planning.

- :mod:`.model_operations` -- single model-operation execution and helpers
- :mod:`.file_handling` -- file-attachment processing and conversion
- :mod:`.planning` -- orchestrator-driven multi-step dispatch, parallel execution
- :mod:`.progress` -- progress reporting for multi-step plans
"""

from .model_operations import execute_model_operation
from .file_handling import handle_file_attachments
from .planning import execute_planned_operations

__all__ = [
    "execute_model_operation",
    "handle_file_attachments",
    "execute_planned_operations",
]
