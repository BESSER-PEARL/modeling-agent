"""Over-eagerness guard: a create request with no domain is too vague to model.

Catches pure-filler creates ("create", "make an app") so the agent asks what to
build instead of hallucinating a default model — while any real domain noun
lets the request proceed unchanged.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from state_bodies import _create_request_is_too_vague as _vague  # noqa: E402


def test_pure_filler_creates_are_caught():
    for msg in ["create", "app", "make an app", "build a system", "generate",
                "new project", "design something", "can you make me an app"]:
        assert _vague(msg), f"expected too-vague: {msg!r}"


def test_requests_with_a_domain_proceed():
    for msg in ["create a hospital system",
                "create a library",
                "create a task management web application",
                "create a class diagram for a library",
                "design a banking system with accounts and transactions",
                "build an e-commerce store with products and orders"]:
        assert not _vague(msg), f"expected NOT vague: {msg!r}"
