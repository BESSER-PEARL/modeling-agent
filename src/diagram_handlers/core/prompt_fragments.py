"""Shared prompt fragments used across all diagram handlers.

Centralizes repeated instructions to avoid duplication and ensure consistency.
"""

POSITION_DISCLAIMER = (
    'Do NOT include any "position" field — positioning is handled automatically.'
)

REMOVE_ELEMENT_RULE = (
    "For remove_element, only specify the target — no \"changes\" needed."
)
