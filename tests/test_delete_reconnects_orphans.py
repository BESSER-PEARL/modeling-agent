"""Deleting a class must not silently orphan the ones it connected.

Live case 2026-09-16. A library model had Book -> BookCopy -> Loan. The user
said "remove book copy"; the agent removed BookCopy and its two relationships
and reported a clean success. The result was structurally valid but
meaningless: Loan still linked to Member and Librarian, but to nothing
borrowable. The user's verdict was "it didn't fix the model, it's a little bit
stupid" -- and they were right.

The frontend fix stopped the modifier DESTROYING neighbours. This covers the
other half: the agent must reconnect what the deletion strands, or at minimum
name it, instead of reporting success.
"""
from src.diagram_handlers.core.prompt_fragments import DELETE_CLASS_CASCADE_RULE
from src.diagram_handlers.types.class_diagram_handler import _CLASS_KEY_RULES_BLOCK


def test_rule_still_requires_removing_connected_relationships():
    """The original cascade guidance must survive."""
    rule = DELETE_CLASS_CASCADE_RULE
    assert "remove_element for EVERY relationship connected to it" in rule


def test_rule_forbids_deleting_neighbouring_classes():
    lowered = DELETE_CLASS_CASCADE_RULE.lower()
    assert "only the class the user named" in lowered
    assert "never remove a neighbouring class" in lowered


def test_rule_requires_reconnecting_orphans():
    rule = DELETE_CLASS_CASCADE_RULE
    assert "add_relationship" in rule
    lowered = rule.lower()
    assert "orphans" in lowered or "orphan" in lowered
    # The worked example is the case that actually happened.
    assert "BookCopy" in rule and "Loan" in rule and "Book" in rule


def test_rule_requires_naming_stranded_classes_when_unclear():
    lowered = DELETE_CLASS_CASCADE_RULE.lower()
    assert "stranded" in lowered
    assert "clean success" in lowered


def test_rule_reaches_the_class_diagram_prompt():
    """A rule that never lands in the prompt changes nothing."""
    prompt = _CLASS_KEY_RULES_BLOCK
    assert "add_relationship" in prompt
    assert "never remove a neighbouring class" in prompt.lower()
    assert "stranded" in prompt.lower()
