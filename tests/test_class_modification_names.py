"""Tests for the class-diagram modification placeholder-leak fix.

Covers the hackathon bug where the LLM hallucinates a placeholder string into
``target.className`` for an ``add_class`` operation (live evidence:
``RolePermissionAssociationClassNamePlaceholderHere``), and the related
"ChatbotHandlerClassNamePlaceholder" case.

Asserts:
- A placeholder-laden ``target.className`` is cleaned (nulled).
- ``add_class`` resolves the real name from ``changes.className``.
- Self-consistency: a real name in only one of target/changes is preserved.
- No surviving name contains "placeholder".
- Placeholder attribute names are dropped.
- An association-class request yields a clean linking-class name and a clear
  success message.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schemas.class_diagram import (
    ClassModification,
    ClassModificationChanges,
    ClassModificationResponse,
    _clean_name,
    _is_placeholder,
)
from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _no_placeholder_anywhere(obj) -> bool:
    """Recursively assert no string value contains 'placeholder'."""
    if isinstance(obj, str):
        return "placeholder" not in obj.lower()
    if isinstance(obj, dict):
        return all(_no_placeholder_anywhere(v) for v in obj.values())
    if isinstance(obj, list):
        return all(_no_placeholder_anywhere(v) for v in obj)
    return True


@pytest.fixture
def handler():
    return ClassDiagramHandler(llm=None)


# ---------------------------------------------------------------------------
# _is_placeholder / _clean_name primitives
# ---------------------------------------------------------------------------

class TestPlaceholderDetection:
    @pytest.mark.parametrize("value", [
        "RolePermissionAssociationClassNamePlaceholderHere",
        "ChatbotHandlerClassNamePlaceholder",
        "PlaceholderHere",
        "ClassNameHere",
        "NameHere",
        "YourClassName",
        "NewClassName",
        "<ClassName>",
    ])
    def test_detects_placeholder(self, value):
        assert _is_placeholder(value) is True

    @pytest.mark.parametrize("value", [
        "Todo", "User", "Order", "Item", "Box", "Permission",
    ])
    def test_does_not_flag_real_class_names(self, value):
        assert _is_placeholder(value) is False

    @pytest.mark.parametrize("value", [
        "RolePermission", "User", "Order", "Payment", "Enrollment",
    ])
    def test_clean_name_keeps_real_names(self, value):
        assert _clean_name(value) == value

    @pytest.mark.parametrize("value", [None, ""])
    def test_clean_name_empty_stays_falsy(self, value):
        assert not _clean_name(value)

    def test_clean_name_nulls_placeholder(self):
        assert _clean_name("RolePermissionAssociationClassNamePlaceholderHere") is None
        assert _clean_name("ChatbotHandlerClassNamePlaceholder") is None

    def test_clean_name_strips_json_artifacts_then_keeps_real(self):
        assert _clean_name("User},") == "User"


# ---------------------------------------------------------------------------
# Schema-level add_class resolution
# ---------------------------------------------------------------------------

class TestAddClassNameResolution:
    def test_live_bug_payload_is_cleaned(self):
        """The exact live payload: junk in target, real name in changes."""
        m = ClassModification(
            action="add_class",
            target={"className": "RolePermissionAssociationClassNamePlaceholderHere"},
            changes={
                "className": "RolePermission",
                "attributes": [
                    {"name": "roleId", "type": "String"},
                    {"name": "permissionId", "type": "String"},
                ],
            },
        )
        assert m.target.className is None
        assert m.changes.className == "RolePermission"
        assert _no_placeholder_anywhere(m.model_dump())

    def test_name_only_in_target_is_promoted(self):
        """Self-consistency: real name only in target → resolved into changes."""
        m = ClassModification(
            action="add_class",
            target={"className": "Invoice"},
            changes={"attributes": [{"name": "total", "type": "float"}]},
        )
        assert m.changes.className == "Invoice"
        assert m.target.className is None

    def test_name_only_in_changes_is_kept(self):
        m = ClassModification(
            action="add_class",
            target={},
            changes={"className": "Customer"},
        )
        assert m.changes.className == "Customer"
        assert m.target.className is None

    def test_chatbot_handler_placeholder_everywhere_is_nulled(self):
        """Long placeholder (>30 chars) in BOTH fields must not raise and must null."""
        m = ClassModification(
            action="add_class",
            target={"className": "ChatbotHandlerClassNamePlaceholder"},
            changes={"className": "ChatbotHandlerClassNamePlaceholder"},
        )
        assert m.changes.className is None
        assert m.target.className is None
        assert _no_placeholder_anywhere(m.model_dump())

    def test_placeholder_attribute_is_dropped(self):
        m = ClassModification(
            action="add_class",
            target={},
            changes={
                "className": "Foo",
                "attributes": [
                    {"name": "PlaceHolderName"},
                    {"name": "realAttr"},
                ],
            },
        )
        names = [a.name for a in m.changes.attributes]
        assert names == ["realAttr"]
        assert _no_placeholder_anywhere(m.model_dump())

    def test_non_add_class_keeps_target_class(self):
        """add_attribute and friends must keep target.className intact."""
        m = ClassModification(
            action="add_attribute",
            target={"className": "User", "attributeName": "email"},
            changes={"type": "String"},
        )
        assert m.target.className == "User"

    def test_changes_className_placeholder_nulled_before_length_check(self):
        """A placeholder >30 chars in changes.className must not raise."""
        c = ClassModificationChanges(className="RolePermissionAssociationClassNamePlaceholderHere")
        assert c.className is None

    def test_full_response_no_placeholder_survives(self):
        resp = ClassModificationResponse(modifications=[
            {
                "action": "add_class",
                "target": {"className": "SomethingClassNamePlaceholderHere"},
                "changes": {"className": "RolePermission"},
            },
            {
                "action": "add_relationship",
                "target": {"sourceClass": "Role", "targetClass": "RolePermission"},
                "changes": {"relationshipType": "Association"},
            },
        ])
        assert _no_placeholder_anywhere(resp.model_dump())
        assert resp.modifications[0].changes.className == "RolePermission"
        assert resp.modifications[0].target.className is None


# ---------------------------------------------------------------------------
# Handler-level success message
# ---------------------------------------------------------------------------

class TestModificationMessage:
    def test_association_class_message_names_endpoints(self, handler):
        spec = {
            "action": "modify_model",
            "modifications": [
                {"action": "add_class", "target": {"className": None},
                 "changes": {"className": "RolePermission",
                             "attributes": [{"name": "roleId", "type": "String"},
                                            {"name": "permissionId", "type": "String"}]}},
                {"action": "add_relationship",
                 "target": {"sourceClass": "Role", "targetClass": "RolePermission"},
                 "changes": {"relationshipType": "Association"}},
                {"action": "add_relationship",
                 "target": {"sourceClass": "RolePermission", "targetClass": "Permission"},
                 "changes": {"relationshipType": "Association"}},
            ],
            "message": "Applied 3 changes:\n- Added **element**.",
        }
        handler._apply_clean_modification_message(
            spec, "create an association class between Role and Permission")
        msg = spec["message"]
        assert "RolePermission" in msg
        assert "Role" in msg and "Permission" in msg
        assert "linking class" in msg.lower()
        assert "placeholder" not in msg.lower()

    def test_plain_add_class_message_uses_real_name(self, handler):
        spec = {
            "action": "modify_model",
            "modification": {"action": "add_class", "target": {"className": None},
                             "changes": {"className": "Invoice",
                                         "attributes": [{"name": "total", "type": "float"}]}},
            "message": "Added **element**.",
        }
        handler._apply_clean_modification_message(spec, "add an Invoice class")
        assert "Invoice" in spec["message"]
        assert "placeholder" not in spec["message"].lower()

    def test_message_never_contains_placeholder_even_if_one_slips_through(self, handler):
        """Defensive: a placeholder name in changes must not reach the message."""
        spec = {
            "action": "modify_model",
            "modification": {"action": "add_class", "target": {"className": None},
                             "changes": {"className": "FooClassNamePlaceholder"}},
            "message": "Added **element**.",
        }
        handler._apply_clean_modification_message(spec, "add a Foo class")
        # The placeholder name is rejected → message left as-is (no junk injected).
        assert "placeholder" not in spec["message"].lower()

    def test_non_add_class_message_untouched(self, handler):
        spec = {
            "action": "modify_model",
            "modification": {"action": "add_attribute",
                             "target": {"className": "User", "attributeName": "email"},
                             "changes": {"type": "String"}},
            "message": "Added attribute to **User**.",
        }
        handler._apply_clean_modification_message(spec, "add email to User")
        assert spec["message"] == "Added attribute to **User**."


# ---------------------------------------------------------------------------
# Association-class request end-to-end (schema → linking class name)
# ---------------------------------------------------------------------------

class TestAssociationClassRequest:
    def test_linking_class_gets_clean_name(self):
        """Schema resolves a clean linking-class name from a junk-target payload."""
        resp = ClassModificationResponse(modifications=[
            {"action": "add_class",
             "target": {"className": "RolePermissionAssociationClassNamePlaceholderHere"},
             "changes": {"className": "RolePermission",
                         "attributes": [{"name": "roleId", "type": "String"},
                                        {"name": "permissionId", "type": "String"}]}},
            {"action": "add_relationship",
             "target": {"sourceClass": "Role", "targetClass": "RolePermission"},
             "changes": {"relationshipType": "Association"}},
            {"action": "add_relationship",
             "target": {"sourceClass": "RolePermission", "targetClass": "Permission"},
             "changes": {"relationshipType": "Association"}},
        ])
        add_cls = resp.modifications[0]
        assert add_cls.changes.className == "RolePermission"
        assert not _is_placeholder(add_cls.changes.className)
        assert _no_placeholder_anywhere(resp.model_dump())
