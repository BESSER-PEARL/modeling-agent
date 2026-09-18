"""Tests for the enum-relationship guard (#33).

A class diagram must never carry a relationship (Association, Composition,
Aggregation, …) whose source or target is an Enumeration. Enums are used only
as attribute *types* (e.g. Task.status : TaskStatus), never as a relationship
endpoint.

These tests exercise the deterministic post-process guards directly (no LLM):

- ``_rewrite_enum_relationships`` for the complete-system generation path
  (operates on a SystemClassSpec-shaped dict).
- ``_rewrite_enum_relationship_mods`` for the modification path (operates on a
  ClassModificationResponse-shaped dict, considering enums in the current
  model and enums added in the same batch).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler


@pytest.fixture
def handler():
    return ClassDiagramHandler(llm=None)


# ---------------------------------------------------------------------------
# Complete-system path: _rewrite_enum_relationships
# ---------------------------------------------------------------------------

class TestRewriteEnumRelationshipsSystem:
    def _spec(self):
        return {
            "systemName": "Tasks",
            "classes": [
                {
                    "className": "Task",
                    "attributes": [{"name": "title", "type": "String"}],
                    "methods": [],
                    "isEnumeration": False,
                },
                {
                    "className": "TaskStatus",
                    "attributes": [
                        {"name": "OPEN"}, {"name": "DONE"},
                    ],
                    "methods": [],
                    "isEnumeration": True,
                },
            ],
            "relationships": [
                {
                    "type": "Association",
                    "source": "Task",
                    "target": "TaskStatus",
                    "sourceMultiplicity": "1",
                    "targetMultiplicity": "1",
                },
            ],
        }

    def test_relationship_to_enum_is_removed(self, handler):
        spec = self._spec()
        handler._rewrite_enum_relationships(spec)
        # The relationship to the enum must be gone.
        assert spec["relationships"] == []

    def test_relationship_to_enum_becomes_typed_attribute(self, handler):
        spec = self._spec()
        handler._rewrite_enum_relationships(spec)
        task = next(c for c in spec["classes"] if c["className"] == "Task")
        enum_attrs = [a for a in task["attributes"] if a.get("type") == "TaskStatus"]
        assert len(enum_attrs) == 1
        assert enum_attrs[0]["name"] == "taskStatus"  # camelCase of enum name

    def test_enum_as_source_also_rewritten(self, handler):
        spec = self._spec()
        # Flip direction: enum is the source.
        spec["relationships"][0]["source"] = "TaskStatus"
        spec["relationships"][0]["target"] = "Task"
        handler._rewrite_enum_relationships(spec)
        assert spec["relationships"] == []
        task = next(c for c in spec["classes"] if c["className"] == "Task")
        assert any(a.get("type") == "TaskStatus" for a in task["attributes"])

    def test_non_enum_relationships_are_preserved(self, handler):
        spec = self._spec()
        spec["classes"].append({
            "className": "Project",
            "attributes": [],
            "methods": [],
            "isEnumeration": False,
        })
        spec["relationships"].append({
            "type": "Composition",
            "source": "Project",
            "target": "Task",
            "sourceMultiplicity": "1",
            "targetMultiplicity": "*",
        })
        handler._rewrite_enum_relationships(spec)
        # The Project->Task relationship survives; only the enum one is dropped.
        assert len(spec["relationships"]) == 1
        assert spec["relationships"][0]["source"] == "Project"
        assert spec["relationships"][0]["target"] == "Task"

    def test_no_duplicate_attribute_when_already_typed(self, handler):
        spec = self._spec()
        task = next(c for c in spec["classes"] if c["className"] == "Task")
        # Task already has a status attribute typed as the enum.
        task["attributes"].append({"name": "status", "type": "TaskStatus"})
        handler._rewrite_enum_relationships(spec)
        enum_attrs = [a for a in task["attributes"] if a.get("type") == "TaskStatus"]
        # No second enum-typed attribute is added.
        assert len(enum_attrs) == 1
        assert enum_attrs[0]["name"] == "status"

    def test_enum_to_enum_relationship_is_dropped(self, handler):
        spec = self._spec()
        spec["classes"].append({
            "className": "Priority",
            "attributes": [{"name": "LOW"}, {"name": "HIGH"}],
            "methods": [],
            "isEnumeration": True,
        })
        spec["relationships"] = [{
            "type": "Association",
            "source": "TaskStatus",
            "target": "Priority",
        }]
        handler._rewrite_enum_relationships(spec)
        # No real class to attach to → relationship simply dropped, no attrs added.
        assert spec["relationships"] == []

    def test_no_enums_is_noop(self, handler):
        spec = {
            "systemName": "Plain",
            "classes": [
                {"className": "A", "attributes": [], "methods": [], "isEnumeration": False},
                {"className": "B", "attributes": [], "methods": [], "isEnumeration": False},
            ],
            "relationships": [{"type": "Association", "source": "A", "target": "B"}],
        }
        handler._rewrite_enum_relationships(spec)
        assert len(spec["relationships"]) == 1


# ---------------------------------------------------------------------------
# Modification path: _rewrite_enum_relationship_mods
# ---------------------------------------------------------------------------

class TestRewriteEnumRelationshipMods:
    def _model_with_enum(self):
        """Current model that already contains a TaskStatus enumeration."""
        return {
            "elements": {
                "task-1": {"id": "task-1", "name": "Task", "type": "Class"},
                "status-1": {"id": "status-1", "name": "TaskStatus", "type": "Enumeration"},
            },
            "relationships": {},
        }

    def test_add_relationship_to_existing_enum_becomes_add_attribute(self, handler):
        spec = {
            "modifications": [
                {
                    "action": "add_relationship",
                    "target": {"sourceClass": "Task", "targetClass": "TaskStatus"},
                    "changes": {"relationshipType": "Association"},
                },
            ],
        }
        handler._rewrite_enum_relationship_mods(spec, self._model_with_enum())
        mods = spec["modifications"]
        assert len(mods) == 1
        assert mods[0]["action"] == "add_attribute"
        assert mods[0]["target"]["className"] == "Task"
        assert mods[0]["changes"]["type"] == "TaskStatus"
        assert mods[0]["changes"]["name"] == "taskStatus"

    def test_enum_added_in_same_batch_is_detected(self, handler):
        """An enum created earlier in the SAME batch is recognised, so a later
        add_relationship to it is rewritten."""
        spec = {
            "modifications": [
                {
                    "action": "add_class",
                    "target": {},
                    "changes": {
                        "className": "Priority",
                        "isEnumeration": True,
                        "attributes": [{"name": "LOW"}, {"name": "HIGH"}],
                    },
                },
                {
                    "action": "add_relationship",
                    "target": {"sourceClass": "Task", "targetClass": "Priority"},
                    "changes": {"relationshipType": "Association"},
                },
            ],
        }
        # current_model has Task (so the attribute can attach to it)
        model = {
            "elements": {"task-1": {"id": "task-1", "name": "Task", "type": "Class"}},
            "relationships": {},
        }
        handler._rewrite_enum_relationship_mods(spec, model)
        mods = spec["modifications"]
        actions = [m["action"] for m in mods]
        # add_class stays, add_relationship becomes add_attribute.
        assert actions == ["add_class", "add_attribute"]
        assert mods[1]["target"]["className"] == "Task"
        assert mods[1]["changes"]["type"] == "Priority"

    def test_non_enum_relationship_unchanged(self, handler):
        spec = {
            "modifications": [
                {
                    "action": "add_relationship",
                    "target": {"sourceClass": "Task", "targetClass": "Project"},
                    "changes": {"relationshipType": "Association"},
                },
            ],
        }
        handler._rewrite_enum_relationship_mods(spec, self._model_with_enum())
        assert spec["modifications"][0]["action"] == "add_relationship"

    def test_single_modification_form_rewritten(self, handler):
        spec = {
            "modification": {
                "action": "add_relationship",
                "target": {"sourceClass": "Task", "targetClass": "TaskStatus"},
                "changes": {"relationshipType": "Association"},
            },
        }
        handler._rewrite_enum_relationship_mods(spec, self._model_with_enum())
        # Single form is preserved (still one mod) but rewritten to add_attribute.
        assert "modification" in spec
        assert spec["modification"]["action"] == "add_attribute"
        assert spec["modification"]["changes"]["type"] == "TaskStatus"

    def test_no_enums_is_noop(self, handler):
        spec = {
            "modifications": [
                {
                    "action": "add_relationship",
                    "target": {"sourceClass": "Task", "targetClass": "Project"},
                    "changes": {"relationshipType": "Association"},
                },
            ],
        }
        model = {
            "elements": {"task-1": {"id": "task-1", "name": "Task", "type": "Class"}},
            "relationships": {},
        }
        handler._rewrite_enum_relationship_mods(spec, model)
        assert spec["modifications"][0]["action"] == "add_relationship"
