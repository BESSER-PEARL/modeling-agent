"""Model utility functions shared between execution and confirmation modules.

Extracted to break the circular import dependency between execution.py and confirmation.py.
"""
from typing import Any, Dict, Optional


def model_has_elements(model: Optional[Dict[str, Any]]) -> bool:
    """Return True when *model* contains at least one user-visible element."""
    if not isinstance(model, dict):
        return False
    elements = model.get('elements')
    return isinstance(elements, dict) and len(elements) > 0
