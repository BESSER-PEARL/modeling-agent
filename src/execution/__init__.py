"""Execution engine for the modeling agent.

This package handles model operations and multi-operation planning.

- :mod:`.model_operations` -- single model-operation execution and helpers
- :mod:`.planning` -- orchestrator-driven multi-step dispatch, parallel execution
- :mod:`.progress` -- progress reporting for multi-step plans
"""

from .model_operations import execute_model_operation
from .planning import execute_planned_operations

__all__ = [
    "execute_model_operation",
    "execute_planned_operations",
]
