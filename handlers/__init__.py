"""Specialized assistant handlers."""

from .generation_handler import (
    detect_generator_type,
    should_route_to_generation,
    handle_generation_request,
)

__all__ = [
    "detect_generator_type",
    "should_route_to_generation",
    "handle_generation_request",
]
