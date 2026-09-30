"""Tests for the relationship-name dedup guard.

A generated class diagram must never carry two associations with the same name:
they are ambiguous on the canvas and collide when the BUML domain model is built
(association names must be unique). ``_dedupe_relationship_names`` renames each
duplicate deterministically — preferring a ``sourceTarget`` camelCase name,
falling back to a numeric suffix — while leaving unnamed associations alone.

Exercises the deterministic post-process guard directly (no LLM), mirroring
test_enum_relationship_guard.py.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler


@pytest.fixture
def handler():
    return ClassDiagramHandler(llm=None)


def _names(spec):
    return [r.get("name") for r in spec["relationships"]]


class TestDedupeRelationshipNames:
    def test_duplicate_names_are_made_unique(self, handler):
        spec = {
            "relationships": [
                {"name": "task", "source": "User", "target": "Task", "type": "Association"},
                {"name": "task", "source": "Project", "target": "Task", "type": "Association"},
                {"name": "task", "source": "Comment", "target": "Task", "type": "Association"},
            ]
        }
        handler._dedupe_relationship_names(spec)
        names = _names(spec)
        # First occurrence keeps the original name; the rest get unique names.
        assert names[0] == "task"
        assert len(names) == len({n.lower() for n in names})

    def test_prefers_source_target_derived_name(self, handler):
        spec = {
            "relationships": [
                {"name": "task", "source": "User", "target": "Task", "type": "Association"},
                {"name": "task", "source": "Project", "target": "Task", "type": "Association"},
            ]
        }
        handler._dedupe_relationship_names(spec)
        assert _names(spec) == ["task", "projectTask"]

    def test_numeric_fallback_when_no_endpoints(self, handler):
        spec = {
            "relationships": [
                {"name": "link", "type": "Association"},
                {"name": "link", "type": "Association"},
            ]
        }
        handler._dedupe_relationship_names(spec)
        assert _names(spec) == ["link", "link2"]

    def test_unnamed_relationships_left_alone(self, handler):
        spec = {
            "relationships": [
                {"name": None, "source": "User", "target": "Task", "type": "Association"},
                {"name": "", "source": "Project", "target": "Task", "type": "Association"},
                {"source": "Comment", "target": "Task", "type": "Association"},
            ]
        }
        handler._dedupe_relationship_names(spec)
        assert _names(spec) == [None, "", None]

    def test_case_insensitive_collision(self, handler):
        spec = {
            "relationships": [
                {"name": "Owner", "source": "Project", "target": "User", "type": "Association"},
                {"name": "owner", "source": "Task", "target": "User", "type": "Association"},
            ]
        }
        handler._dedupe_relationship_names(spec)
        names = _names(spec)
        assert names[0] == "Owner"
        assert names[1] != "owner"  # the case-only duplicate was renamed
        assert len({n.lower() for n in names}) == 2

    def test_distinct_names_unchanged(self, handler):
        spec = {
            "relationships": [
                {"name": "owns", "source": "User", "target": "Project", "type": "Association"},
                {"name": "assignedTo", "source": "Task", "target": "User", "type": "Association"},
            ]
        }
        handler._dedupe_relationship_names(spec)
        assert _names(spec) == ["owns", "assignedTo"]

    def test_no_relationships_is_safe(self, handler):
        spec = {"classes": []}
        handler._dedupe_relationship_names(spec)  # must not raise
        assert "relationships" not in spec or spec.get("relationships") is None
