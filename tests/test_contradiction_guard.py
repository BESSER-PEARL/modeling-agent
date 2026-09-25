"""Self-contradiction guard: a CREATE request that negates the very content it
asks to create ("a class diagram with no classes", "an empty diagram", "don't
model anything") can't yield a real model — the agent should clarify instead of
building a token 1-class model. HIGH-PRECISION: real, positive requests that
merely contain a negation ("a shop with no online payments") must NOT be caught.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from state_bodies import _create_is_self_contradictory as _contra  # noqa: E402


def test_degenerate_creates_are_caught():
    for msg in [
        "create a class diagram with absolutely no classes",
        "create a class diagram with no classes",
        "make a diagram without any classes",
        "create an object diagram with zero elements",
        "build a diagram with no entities",
        "create a class diagram without any elements",
        "design a class diagram but don't model anything",
        "create a diagram but do not model anything at all",
        "create an empty class diagram",
        "make an empty diagram",
        "create an empty state machine diagram",
        "model nothing",
    ]:
        assert _contra(msg), f"expected self-contradictory: {msg!r}"


def test_real_positive_requests_are_not_caught():
    for msg in [
        "create a class diagram for a shop with no online payments",
        "create a shop with no user accounts, just products",
        "create a library with books and members",
        "create a class diagram for a hotel with no smoking rooms",
        "model a system about school with only 3 classes",
        "create a todo app, nothing fancy",
        "create a simple blog, add nothing fancy to it",
        "design an e-commerce store with products, orders and payments",
        "create a class diagram with a User and an Order class",
        "build a diagram with no more than five classes",
    ]:
        assert not _contra(msg), f"expected NOT self-contradictory: {msg!r}"
