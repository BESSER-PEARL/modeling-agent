"""_reconcile_intent: the unified classifier is the authoritative create-vs-modify
router. A pending flow (e.g. a GUI-choice prompt) can suppress intent routing so a
follow-up modify lands in the CREATE state; the reconciliation honors the
classifier so "add Death to the PetStatus enum" modifies instead of rebuilding.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from state_bodies import _reconcile_intent as _rec  # noqa: E402


def test_modify_stuck_in_create_is_reconciled():
    # The reported bug: classifier says modify, but the message landed in create.
    assert _rec("create_complete_system_intent", "complete_system",
                "modify_model_intent") == ("modify_model_intent", "modify_model")


def test_create_stuck_in_modify_is_reconciled():
    assert _rec("modify_model_intent", "modify_model",
                "create_complete_system_intent") == (
                    "create_complete_system_intent", "complete_system")


def test_agreement_is_left_untouched():
    assert _rec("create_complete_system_intent", "complete_system",
                "create_complete_system_intent") == (
                    "create_complete_system_intent", "complete_system")
    assert _rec("modify_model_intent", "modify_model",
                "modify_model_intent") == ("modify_model_intent", "modify_model")


def test_no_or_other_classification_keeps_state_default():
    for uc in (None, "generation_intent", "fallback_intent", "hello_intent"):
        assert _rec("create_complete_system_intent", "complete_system", uc) == (
            "create_complete_system_intent", "complete_system")
        assert _rec("modify_model_intent", "modify_model", uc) == (
            "modify_model_intent", "modify_model")
