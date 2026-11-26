"""Over-eagerness guard: a create request with no domain is too vague to model.

Catches pure-filler creates ("create", "make an app") so the agent asks what to
build instead of hallucinating a default model — while any real domain noun
lets the request proceed unchanged.
"""

from routing.guards import is_too_vague_create


def test_pure_filler_creates_are_caught():
    for msg in ["create", "app", "make an app", "build a system", "generate",
                "new project", "design something", "can you make me an app"]:
        assert is_too_vague_create(msg), f"expected too-vague: {msg!r}"


def test_requests_with_a_domain_proceed():
    for msg in ["create a hospital system",
                "create a library",
                "create a task management web application",
                "create a class diagram for a library",
                "design a banking system with accounts and transactions",
                "build an e-commerce store with products and orders"]:
        assert not is_too_vague_create(msg), f"expected NOT vague: {msg!r}"


def test_a_short_or_non_latin_domain_word_is_content():
    """The guard answered "too vague" before the classifier for real requests
    whose domain word is short or not written in Latin letters."""
    for msg in ["Make an HR app", "Create an AI app", "Make a to-do app", "Create a 3D app",
                "Φτιάξε μια εφαρμογή για ξενοδοχείο", "做一个图书馆应用"]:
        assert not is_too_vague_create(msg), f"expected NOT vague: {msg!r}"
