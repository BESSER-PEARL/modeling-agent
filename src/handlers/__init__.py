"""Specialized assistant handlers."""

from .generation_handler import (
    detect_generator_type,
    should_route_to_generation,
    handle_generation_request,
)
from .smart_generation_handler import (
    build_trigger_smart_generator_payload,
    GenerationClassification,
)

__all__ = [
    "detect_generator_type",
    "should_route_to_generation",
    "handle_generation_request",
    "GenerationClassification",
    "build_trigger_smart_generator_payload",
]
