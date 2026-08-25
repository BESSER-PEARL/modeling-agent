"""Decline / no-op guard: a bare "nothing" / "no" / "never mind" is the user
opting out — it must NOT be routed into a create (which would pop the
replace/keep-existing prompt on the user's model). A message that merely
contains such a word alongside a real request is not a decline.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from state_bodies import _request_is_decline as _decline  # noqa: E402


def test_bare_declines_are_caught():
    for msg in ["nothing", "Nothing.", "no", "nope", "no thanks!", "never mind",
                "nvm", "I'm done", "that's all", "stop", "cancel",
                "nothing else", "not now", "maybe later"]:
        assert _decline(msg), f"expected decline: {msg!r}"


def test_real_requests_are_not_declines():
    for msg in ["nothing fancy, just a todo app",
                "i want a library system",
                "create a library app",
                "no user accounts, just products",
                "add nothing to the diagram — remove the Loan class"]:
        assert not _decline(msg), f"expected NOT a decline: {msg!r}"
