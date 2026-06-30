"""Tests for the User Profile (UserDiagram) handler, schemas, and routing."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _mods(result):
    """Normalize a modify_model result to a flat list of modifications."""
    if "modifications" in result:
        return result["modifications"]
    return [result["modification"]]


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _load_orchestrator():
    """Load workspace_orchestrator directly (avoids baf import in __init__)."""
    import importlib.util
    import os

    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    spec = importlib.util.spec_from_file_location(
        "orchestrator.workspace_orchestrator",
        os.path.join(src, "orchestrator", "workspace_orchestrator.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("orchestrator.workspace_orchestrator", mod)
    spec.loader.exec_module(mod)
    return mod


def test_user_profile_routing_explicit():
    mod = _load_orchestrator()
    from protocol.types import AssistantRequest, WorkspaceContext

    assert mod.determine_target_diagram_type(
        AssistantRequest(
            message="create a user profile for a teenage Spanish speaker",
            context=WorkspaceContext(),
        )
    ) == "UserDiagram"


def test_user_profile_routing_persona():
    mod = _load_orchestrator()
    from protocol.types import AssistantRequest, WorkspaceContext

    assert mod.determine_target_diagram_type(
        AssistantRequest(
            message="build a persona for an elderly low-vision user",
            context=WorkspaceContext(),
        )
    ) == "UserDiagram"


def test_user_diagram_supported_and_registered():
    from protocol.types import SUPPORTED_DIAGRAM_TYPES
    from diagram_handlers.registry.factory import DiagramHandlerFactory

    assert "UserDiagram" in SUPPORTED_DIAGRAM_TYPES

    class _LLM:
        name = "gpt-4.1-mini"
        client = None

    factory = DiagramHandlerFactory(_LLM())
    handler = factory.get_handler("UserDiagram")
    assert handler is not None
    assert handler.get_diagram_type() == "UserDiagram"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

def test_user_profile_attribute_operator_default():
    from schemas import UserProfileAttributeSpec

    a = UserProfileAttributeSpec(name="age", value="18")
    assert a.operator == "=="


def test_user_profile_system_spec_requires_profiles():
    from schemas import SystemUserProfileSpec
    import pytest

    with pytest.raises(Exception):
        SystemUserProfileSpec(profiles=[])  # min_length=1


# ---------------------------------------------------------------------------
# Metamodel loader
# ---------------------------------------------------------------------------

def test_metamodel_loads_expected_classes():
    from utilities.user_metamodel import load_user_metamodel

    m = load_user_metamodel()
    elements = m.get("elements", {})
    class_names = {
        el.get("name")
        for el in elements.values()
        if isinstance(el, dict) and el.get("type") == "Class"
    }
    assert {"User", "Personal_Information", "Language"} <= class_names


# ---------------------------------------------------------------------------
# Handler generation
# ---------------------------------------------------------------------------

def test_complete_system_resolves_ids_operators_and_drops_unknowns():
    from schemas import SystemUserProfileSpec
    from diagram_handlers.types.user_profile_handler import UserProfileDiagramHandler

    handler = UserProfileDiagramHandler(llm=None)

    def fake_predict_structured(prompt, schema, system_prompt="", **kw):
        assert schema is SystemUserProfileSpec
        return SystemUserProfileSpec(
            systemName="Teen Spanish speaker",
            profiles=[
                {"className": "Personal_Information", "profileName": "pi1",
                 "attributes": [{"name": "age", "operator": ">=", "value": "13"}]},
                {"className": "Language", "profileName": "lang1",
                 "attributes": [
                     {"name": "iso693_3", "operator": "==", "value": "Spanish"},
                     {"name": "level", "operator": "==", "value": "B2"},
                     {"name": "not_a_real_attr", "operator": "==", "value": "x"},
                 ]},
                {"className": "NotARealClass", "profileName": "bad1", "attributes": []},
            ],
            links=[{"source": "pi1", "target": "lang1", "relationshipType": "speaks"}],
        )

    handler.predict_structured = fake_predict_structured
    result = handler.generate_complete_system("a teenager who speaks Spanish at B2")

    assert result["action"] == "inject_complete_system"
    assert result["diagramType"] == "UserDiagram"

    profiles = result["systemSpec"]["profiles"]
    by_class = {p["className"] for p in profiles}
    # Unknown class dropped; User root + Competence intermediate auto-inserted.
    assert "NotARealClass" not in by_class
    assert {"User", "Personal_Information", "Competence", "Language"} <= by_class
    # Every box has a resolved metamodel classId.
    assert all(p["classId"] for p in profiles)
    # Unknown attribute dropped; known attrs carry attributeId + type + operator.
    lang = next(p for p in profiles if p["className"] == "Language")
    assert [a["name"] for a in lang["attributes"]] == ["iso693_3", "level"]
    assert all(a["attributeId"] for a in lang["attributes"])
    assert next(a for a in lang["attributes"] if a["name"] == "level")["type"] == "CEFR"
    pi = next(p for p in profiles if p["className"] == "Personal_Information")
    assert pi["attributes"][0]["operator"] == ">="
    # Layout assigned positions.
    assert all("position" in p for p in profiles)

    # Structure is a tree rooted at the single User box, wired via the
    # metamodel associations: User->Personal_Information, User->Competence,
    # Competence->Language.
    name_to_class = {p["profileName"]: p["className"] for p in profiles}
    user_names = [p["profileName"] for p in profiles if p["className"] == "User"]
    assert len(user_names) == 1
    edges = {(name_to_class[l["source"]], name_to_class[l["target"]])
             for l in result["systemSpec"]["links"]}
    assert ("User", "Personal_Information") in edges
    assert ("User", "Competence") in edges
    assert ("Competence", "Language") in edges


def test_complete_system_is_rooted_at_user_for_single_class():
    """Even a one-attribute request yields User + the owning box + a link."""
    from schemas import SystemUserProfileSpec
    from diagram_handlers.types.user_profile_handler import UserProfileDiagramHandler

    handler = UserProfileDiagramHandler(llm=None)
    handler.predict_structured = lambda prompt, schema, system_prompt="", **kw: SystemUserProfileSpec(
        systemName="Aged 90",
        profiles=[{"className": "Personal_Information",
                   "attributes": [{"name": "age", "operator": "==", "value": "90"}]}],
        links=[],
    )
    result = handler.generate_complete_system("a user aged 90")
    profiles = result["systemSpec"]["profiles"]
    classes = sorted(p["className"] for p in profiles)
    assert classes == ["Personal_Information", "User"]
    links = result["systemSpec"]["links"]
    assert len(links) == 1
    name_to_class = {p["profileName"]: p["className"] for p in profiles}
    src, tgt = links[0]["source"], links[0]["target"]
    assert (name_to_class[src], name_to_class[tgt]) == ("User", "Personal_Information")


def test_box_includes_all_metamodel_attributes():
    """A class box must list every metamodel attribute (none are optional),
    filling supplied criteria and leaving the rest blank."""
    from schemas import SystemUserProfileSpec
    from diagram_handlers.types.user_profile_handler import UserProfileDiagramHandler

    handler = UserProfileDiagramHandler(llm=None)
    handler.predict_structured = lambda prompt, schema, system_prompt="", **kw: SystemUserProfileSpec(
        systemName="Dyslexic user",
        profiles=[{"className": "Disability",
                   "attributes": [{"name": "name", "operator": "==", "value": "Dyslexia"}]}],
        links=[],
    )
    result = handler.generate_complete_system("a dyslexic user")
    disability = next(p for p in result["systemSpec"]["profiles"] if p["className"] == "Disability")
    names = {a["name"] for a in disability["attributes"]}
    # Disability defines name, description, affects — all three must be present.
    assert names == {"name", "description", "affects"}
    assert all(a["attributeId"] for a in disability["attributes"])
    by_name = {a["name"]: a for a in disability["attributes"]}
    assert by_name["name"]["value"] == "Dyslexia"
    assert by_name["description"]["value"] == ""  # unspecified -> blank row
    assert by_name["affects"]["type"] == "AspectsEnum"


def test_modification_add_object_emits_full_attribute_set(monkeypatch):
    from diagram_handlers.types.user_profile_handler import UserProfileDiagramHandler
    from schemas import (
        UserProfileModificationResponse, UserProfileModification,
        UserProfileModificationTarget, UserProfileModificationChanges,
    )

    response = UserProfileModificationResponse(
        modifications=[
            UserProfileModification(
                action="add_object",
                target=UserProfileModificationTarget(profileName="disability1"),
                changes=UserProfileModificationChanges(
                    className="Disability",
                    attributes=[{"name": "name", "operator": "==", "value": "Dyslexia"}],
                ),
            )
        ]
    )
    monkeypatch.setattr(UserProfileDiagramHandler, "predict_structured",
                        lambda self, *a, **k: response)
    result = UserProfileDiagramHandler(None).generate_modification("add a dyslexia disability")
    mods = _mods(result)
    disability = next(m for m in mods
                      if m["action"] == "add_object" and m["changes"]["className"] == "Disability")
    names = {a["name"] for a in disability["changes"]["attributes"]}
    assert names == {"name", "description", "affects"}


def test_generate_modification_add_object_resolves_catalog(monkeypatch):
    from diagram_handlers.types.user_profile_handler import UserProfileDiagramHandler
    from schemas import (
        UserProfileModificationResponse,
        UserProfileModification,
        UserProfileModificationTarget,
        UserProfileModificationChanges,
    )

    response = UserProfileModificationResponse(
        modifications=[
            UserProfileModification(
                action="add_object",
                target=UserProfileModificationTarget(profileName="language2"),
                changes=UserProfileModificationChanges(
                    className="Language",
                    attributes=[{"name": "iso693_3", "operator": "==", "value": "French"},
                                {"name": "level", "operator": ">=", "value": "C1"}],
                ),
            )
        ]
    )

    def fake_predict(self, user_prompt, schema_cls, **kwargs):
        return response

    monkeypatch.setattr(UserProfileDiagramHandler, "predict_structured", fake_predict)
    handler = UserProfileDiagramHandler(None)
    result = handler.generate_modification("add a French C1 language box", current_model=None)

    assert result["action"] == "modify_model"
    mods = _mods(result)
    lang = next(m for m in mods
                if m["action"] == "add_object" and m["changes"]["className"] == "Language")
    changes = lang["changes"]
    # classId + per-attribute attributeId resolved from the bundled metamodel.
    assert changes["classId"]
    assert all(a.get("attributeId") for a in changes["attributes"])
    assert {a["name"]: a["operator"] for a in changes["attributes"]}["level"] == ">="


def _add_disability_response():
    from schemas import (
        UserProfileModificationResponse, UserProfileModification,
        UserProfileModificationTarget, UserProfileModificationChanges,
    )
    return UserProfileModificationResponse(modifications=[
        UserProfileModification(
            action="add_object",
            target=UserProfileModificationTarget(profileName="disability1"),
            changes=UserProfileModificationChanges(
                className="Disability",
                attributes=[{"name": "name", "operator": "==", "value": "Paraplegia"}],
            ),
        )
    ])


def test_modification_creates_missing_ancestor_and_links(monkeypatch):
    """Adding a Disability to a User-only model also creates Accessibility and
    wires User->Accessibility->Disability."""
    from diagram_handlers.types.user_profile_handler import UserProfileDiagramHandler

    monkeypatch.setattr(UserProfileDiagramHandler, "predict_structured",
                        lambda self, *a, **k: _add_disability_response())

    model = {"elements": {"u": {"type": "UserModelName", "name": "user_1",
                                "className": "User", "attributes": []}},
             "relationships": {}}
    result = UserProfileDiagramHandler(None).generate_modification(
        "add the disability paraplegia", current_model=model)

    mods = _mods(result)
    added_classes = {m["changes"]["className"] for m in mods if m["action"] == "add_object"}
    assert added_classes == {"Accessibility", "Disability"}
    link_pairs = {(m["target"]["sourceProfile"], m["target"]["targetProfile"])
                  for m in mods if m["action"] == "add_link"}
    assert ("User", "Accessibility") in link_pairs
    assert ("Accessibility", "disability1") in link_pairs


def test_modification_reuses_existing_ancestor(monkeypatch):
    """When Accessibility already exists and is linked to User, adding a
    Disability only adds the box + a single link to the existing Accessibility."""
    from diagram_handlers.types.user_profile_handler import UserProfileDiagramHandler

    monkeypatch.setattr(UserProfileDiagramHandler, "predict_structured",
                        lambda self, *a, **k: _add_disability_response())

    model = {
        "elements": {
            "u": {"type": "UserModelName", "name": "user_1", "className": "User", "attributes": []},
            "a": {"type": "UserModelName", "name": "accessibility", "className": "Accessibility", "attributes": []},
        },
        "relationships": {"r1": {"type": "ObjectLink",
                                 "source": {"element": "u"}, "target": {"element": "a"}}},
    }
    result = UserProfileDiagramHandler(None).generate_modification(
        "add the disability paraplegia", current_model=model)

    mods = _mods(result)
    added_classes = [m["changes"]["className"] for m in mods if m["action"] == "add_object"]
    # Accessibility is NOT re-created.
    assert added_classes == ["Disability"]
    link_pairs = {(m["target"]["sourceProfile"], m["target"]["targetProfile"])
                  for m in mods if m["action"] == "add_link"}
    # Only the new Disability link; no duplicate User->Accessibility link.
    assert link_pairs == {("Accessibility", "disability1")}


def test_fallback_system_envelope():
    from diagram_handlers.types.user_profile_handler import UserProfileDiagramHandler

    r = UserProfileDiagramHandler(None).generate_fallback_system()
    assert r["action"] == "inject_complete_system"
    assert r["diagramType"] == "UserDiagram"
    assert "profiles" in r["systemSpec"]


# ---------------------------------------------------------------------------
# Summaries & suggestions
# ---------------------------------------------------------------------------

def test_detailed_summary_lists_boxes_and_criteria():
    from utilities.model_context import detailed_model_summary

    model = {
        "elements": {
            "u1": {"type": "UserModelName", "name": "user_1", "className": "User",
                   "attributes": []},
            "pi": {"type": "UserModelName", "name": "personal_information1",
                   "className": "Personal_Information", "attributes": ["a1"]},
            "a1": {"type": "UserModelAttribute", "name": "age < 18", "owner": "pi"},
        },
        "relationships": {},
    }
    summary = detailed_model_summary(model, "UserDiagram")
    assert "Personal_Information" in summary
    assert "age < 18" in summary


def test_user_profile_suggestions_have_prompts():
    from suggestions import get_suggested_actions

    actions = get_suggested_actions("UserDiagram", "complete_system", [])
    assert actions
    for action in actions:
        assert action.get("prompt"), f"Chip '{action.get('label')}' has empty prompt"
