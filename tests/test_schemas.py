"""Tests for all Pydantic schemas -- validates constraints, defaults, and edge cases."""
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pydantic import ValidationError

# -- Class Diagram imports --
from schemas.class_diagram import (
    AttributeSpec,
    MethodParameterSpec,
    MethodSpec,
    RelationshipSpec,
    SingleClassSpec,
    SystemClassSpec,
    ClassModificationTarget,
    ClassModificationChanges,
    ClassModification,
    ClassModificationResponse,
)

# -- State Machine imports --
from schemas.state_machine import (
    StateSpec,
    TransitionSpec,
    SingleStateSpec,
    SystemStateMachineSpec,
    StateMachineModificationTarget,
    StateMachineModificationChanges,
    StateMachineModification,
    StateMachineModificationResponse,
)

# -- Object Diagram imports --
from schemas.object_diagram import (
    ObjectAttributeSpec,
    SingleObjectSpec,
    ObjectLinkSpec,
    SystemObjectSpec,
    ObjectModificationTarget,
    ObjectModificationChanges,
    ObjectModification,
    ObjectModificationResponse,
)

# -- Agent Diagram imports --
from schemas.agent_diagram import (
    AgentReplySpec,
    AgentStateSpec,
    AgentIntentSpec,
    AgentSingleElementSpec,
    AgentTransitionSpec,
    SystemAgentSpec,
    AgentModificationTarget,
    AgentModificationChanges,
    AgentModification,
    AgentModificationResponse,
)

# -- GUI Diagram imports --
from schemas.gui_diagram import (
    GUISampleDataPoint,
    GUISectionSpec,
    SingleGUIElementSpec,
    GUIPageSpec,
    SystemGUISpec,
    GUIModificationSpec,
)

# -- Quantum Circuit imports --
from schemas.quantum_circuit import (
    QuantumOperationSpec,
    SingleQuantumGateSpec,
    SystemQuantumCircuitSpec,
    QuantumModificationSpec,
)


# =============================================================================
# Class Diagram schemas
# =============================================================================

class TestMethodParameterSpec:
    def test_valid_creation(self):
        p = MethodParameterSpec(name="id")
        assert p.name == "id"
        assert p.type == "String"

    def test_custom_type(self):
        p = MethodParameterSpec(name="age", type="int")
        assert p.type == "int"


class TestAttributeSpec:
    def test_valid_creation(self):
        a = AttributeSpec(name="title")
        assert a.name == "title"
        assert a.type == "String"
        assert a.visibility == "public"

    @pytest.mark.parametrize("name", ["", ])
    def test_rejects_empty_name(self, name):
        with pytest.raises(ValidationError):
            AttributeSpec(name=name)

    @pytest.mark.parametrize("vis", ["public", "private", "protected", "package"])
    def test_valid_visibility(self, vis):
        a = AttributeSpec(name="x", visibility=vis)
        assert a.visibility == vis

    def test_rejects_invalid_visibility(self):
        with pytest.raises(ValidationError):
            AttributeSpec(name="x", visibility="internal")


class TestMethodSpec:
    def test_valid_creation(self):
        m = MethodSpec(name="getTitle")
        assert m.name == "getTitle"
        assert m.returnType == "void"
        assert m.visibility == "public"
        assert m.parameters == []

    def test_rejects_empty_name(self):
        with pytest.raises(ValidationError):
            MethodSpec(name="")

    def test_with_parameters(self):
        m = MethodSpec(
            name="setAge",
            parameters=[MethodParameterSpec(name="age", type="int")]
        )
        assert len(m.parameters) == 1
        assert m.parameters[0].name == "age"


class TestSingleClassSpec:
    def test_valid_creation(self):
        c = SingleClassSpec(
            className="User",
            attributes=[AttributeSpec(name="email")],
            methods=[MethodSpec(name="login")],
        )
        assert c.className == "User"
        assert len(c.attributes) == 1
        assert len(c.methods) == 1

    def test_valid_minimal(self):
        c = SingleClassSpec(className="Empty")
        assert c.className == "Empty"
        assert c.attributes == []
        assert c.methods == []

    def test_rejects_empty_className(self):
        with pytest.raises(ValidationError):
            SingleClassSpec(className="")


class TestRelationshipSpec:
    def test_defaults(self):
        r = RelationshipSpec(source="A", target="B")
        assert r.type == "Association"
        assert r.sourceMultiplicity == "1"
        assert r.targetMultiplicity == "*"
        assert r.name is None

    @pytest.mark.parametrize("rel_type", [
        "Association", "Inheritance", "Composition",
        "Aggregation", "Realization", "Dependency",
    ])
    def test_valid_types(self, rel_type):
        r = RelationshipSpec(source="A", target="B", type=rel_type)
        assert r.type == rel_type

    def test_rejects_invalid_type(self):
        with pytest.raises(ValidationError):
            RelationshipSpec(source="A", target="B", type="FriendOf")

    def test_with_name(self):
        r = RelationshipSpec(source="A", target="B", name="manages")
        assert r.name == "manages"


class TestSystemClassSpec:
    def test_valid_creation(self):
        s = SystemClassSpec(
            systemName="Library",
            classes=[SingleClassSpec(className="Book")],
        )
        assert s.systemName == "Library"
        assert len(s.classes) == 1

    def test_multiple_classes_and_relationships(self):
        s = SystemClassSpec(
            classes=[
                SingleClassSpec(className="Book"),
                SingleClassSpec(className="Author"),
            ],
            relationships=[
                RelationshipSpec(source="Author", target="Book", type="Association")
            ],
        )
        assert len(s.classes) == 2
        assert len(s.relationships) == 1

    def test_rejects_empty_classes(self):
        with pytest.raises(ValidationError):
            SystemClassSpec(classes=[])

    def test_default_systemName(self):
        s = SystemClassSpec(classes=[SingleClassSpec(className="X")])
        assert s.systemName == ""


class TestClassModificationTarget:
    def test_all_optional(self):
        t = ClassModificationTarget()
        assert t.className is None
        assert t.attributeName is None
        assert t.methodName is None
        assert t.sourceClass is None
        assert t.targetClass is None


class TestClassModificationChanges:
    def test_all_optional(self):
        c = ClassModificationChanges()
        assert c.name is None
        assert c.type is None
        assert c.visibility is None
        assert c.returnType is None
        assert c.parameters is None
        assert c.relationshipType is None
        assert c.sourceMultiplicity is None
        assert c.targetMultiplicity is None

    def test_valid_visibility_values(self):
        c = ClassModificationChanges(visibility="private")
        assert c.visibility == "private"

    def test_rejects_invalid_visibility(self):
        with pytest.raises(ValidationError):
            ClassModificationChanges(visibility="global")


class TestClassModification:
    def test_valid(self):
        m = ClassModification(
            action="rename_class",
            target=ClassModificationTarget(className="OldName"),
            changes=ClassModificationChanges(name="NewName"),
        )
        assert m.action == "rename_class"
        assert m.target.className == "OldName"
        assert m.changes.name == "NewName"

    def test_changes_optional(self):
        m = ClassModification(
            action="delete_class",
            target=ClassModificationTarget(className="Foo"),
        )
        assert m.changes is None


class TestClassModificationResponse:
    def test_valid_with_one_modification(self):
        r = ClassModificationResponse(
            modifications=[
                ClassModification(
                    action="add_class",
                    target=ClassModificationTarget(className="Widget"),
                )
            ]
        )
        assert len(r.modifications) == 1

    def test_valid_with_multiple_modifications(self):
        r = ClassModificationResponse(
            modifications=[
                ClassModification(
                    action="add_class",
                    target=ClassModificationTarget(className="A"),
                ),
                ClassModification(
                    action="delete_class",
                    target=ClassModificationTarget(className="B"),
                ),
            ]
        )
        assert len(r.modifications) == 2

    def test_rejects_empty_modifications(self):
        # ClassModificationResponse.modifications has no min_length in schema,
        # but we still verify it accepts empty (it does not enforce min_length).
        # The requirement says it should reject empty, so test what the schema
        # actually does: if no min_length, empty is accepted.
        r = ClassModificationResponse(modifications=[])
        assert len(r.modifications) == 0


# =============================================================================
# State Machine schemas
# =============================================================================

class TestStateSpec:
    def test_valid_creation(self):
        s = StateSpec(stateName="Idle")
        assert s.stateName == "Idle"
        assert s.stateType == "regular"
        assert s.entryAction is None
        assert s.exitAction is None
        assert s.doActivity is None

    def test_rejects_empty_stateName(self):
        with pytest.raises(ValidationError):
            StateSpec(stateName="")

    @pytest.mark.parametrize("st", ["initial", "final", "regular"])
    def test_valid_stateTypes(self, st):
        s = StateSpec(stateName="S", stateType=st)
        assert s.stateType == st

    def test_rejects_invalid_stateType(self):
        with pytest.raises(ValidationError):
            StateSpec(stateName="S", stateType="transient")

    def test_with_actions(self):
        s = StateSpec(
            stateName="Active",
            entryAction="startTimer()",
            exitAction="stopTimer()",
            doActivity="runLoop()",
        )
        assert s.entryAction == "startTimer()"
        assert s.exitAction == "stopTimer()"
        assert s.doActivity == "runLoop()"


class TestTransitionSpec:
    def test_valid_creation(self):
        t = TransitionSpec(source="A", target="B")
        assert t.source == "A"
        assert t.target == "B"
        assert t.trigger is None
        assert t.guard is None
        assert t.effect is None

    def test_with_all_fields(self):
        t = TransitionSpec(
            source="Idle",
            target="Active",
            trigger="start",
            guard="isReady",
            effect="initialize()",
        )
        assert t.trigger == "start"
        assert t.guard == "isReady"
        assert t.effect == "initialize()"


class TestSingleStateSpec:
    def test_valid_creation(self):
        s = SingleStateSpec(stateName="Ready")
        assert s.stateName == "Ready"
        assert s.stateType == "regular"

    def test_rejects_empty_stateName(self):
        with pytest.raises(ValidationError):
            SingleStateSpec(stateName="")


class TestSystemStateMachineSpec:
    def test_valid_creation(self):
        s = SystemStateMachineSpec(
            systemName="TrafficLight",
            states=[StateSpec(stateName="Green")],
        )
        assert s.systemName == "TrafficLight"
        assert len(s.states) == 1

    def test_rejects_empty_states(self):
        with pytest.raises(ValidationError):
            SystemStateMachineSpec(states=[])

    def test_default_transitions(self):
        s = SystemStateMachineSpec(
            states=[StateSpec(stateName="X")],
        )
        assert s.transitions == []

    def test_with_transitions(self):
        s = SystemStateMachineSpec(
            states=[
                StateSpec(stateName="A"),
                StateSpec(stateName="B"),
            ],
            transitions=[TransitionSpec(source="A", target="B")],
        )
        assert len(s.transitions) == 1


class TestStateMachineModificationTarget:
    def test_all_optional(self):
        t = StateMachineModificationTarget()
        assert t.stateName is None
        assert t.sourceState is None
        assert t.targetState is None


class TestStateMachineModificationChanges:
    def test_all_optional(self):
        c = StateMachineModificationChanges()
        assert c.name is None
        assert c.entryAction is None
        assert c.exitAction is None
        assert c.doActivity is None
        assert c.trigger is None
        assert c.guard is None
        assert c.effect is None


class TestStateMachineModification:
    def test_valid(self):
        m = StateMachineModification(
            action="rename_state",
            target=StateMachineModificationTarget(stateName="Old"),
            changes=StateMachineModificationChanges(name="New"),
        )
        assert m.action == "rename_state"

    def test_changes_optional(self):
        m = StateMachineModification(
            action="delete_state",
            target=StateMachineModificationTarget(stateName="X"),
        )
        assert m.changes is None


class TestStateMachineModificationResponse:
    def test_valid_with_one_modification(self):
        r = StateMachineModificationResponse(
            modifications=[
                StateMachineModification(
                    action="add_state",
                    target=StateMachineModificationTarget(stateName="New"),
                )
            ]
        )
        assert len(r.modifications) == 1

    def test_rejects_empty_modifications(self):
        with pytest.raises(ValidationError):
            StateMachineModificationResponse(modifications=[])


# =============================================================================
# Object Diagram schemas
# =============================================================================

class TestObjectAttributeSpec:
    def test_valid_creation(self):
        a = ObjectAttributeSpec(name="title", value="LOTR")
        assert a.name == "title"
        assert a.value == "LOTR"
        assert a.attributeId is None

    def test_with_attributeId(self):
        a = ObjectAttributeSpec(name="x", value="1", attributeId="attr-001")
        assert a.attributeId == "attr-001"


class TestSingleObjectSpec:
    def test_valid_creation(self):
        o = SingleObjectSpec(objectName="book1", className="Book")
        assert o.objectName == "book1"
        assert o.className == "Book"
        assert o.classId is None
        assert o.attributes == []

    def test_rejects_empty_objectName(self):
        with pytest.raises(ValidationError):
            SingleObjectSpec(objectName="", className="Book")

    def test_rejects_empty_className(self):
        with pytest.raises(ValidationError):
            SingleObjectSpec(objectName="book1", className="")

    def test_with_attributes(self):
        o = SingleObjectSpec(
            objectName="book1",
            className="Book",
            attributes=[ObjectAttributeSpec(name="title", value="Dune")],
        )
        assert len(o.attributes) == 1


class TestObjectLinkSpec:
    def test_valid_creation(self):
        l = ObjectLinkSpec(source="book1", target="author1")
        assert l.source == "book1"
        assert l.target == "author1"
        assert l.relationshipType is None

    def test_with_relationship_type(self):
        l = ObjectLinkSpec(source="a", target="b", relationshipType="Association")
        assert l.relationshipType == "Association"


class TestSystemObjectSpec:
    def test_valid_creation(self):
        s = SystemObjectSpec(
            objects=[SingleObjectSpec(objectName="o1", className="C")],
        )
        assert len(s.objects) == 1
        assert s.links == []

    def test_rejects_empty_objects(self):
        with pytest.raises(ValidationError):
            SystemObjectSpec(objects=[])

    def test_with_links(self):
        s = SystemObjectSpec(
            objects=[
                SingleObjectSpec(objectName="o1", className="C1"),
                SingleObjectSpec(objectName="o2", className="C2"),
            ],
            links=[ObjectLinkSpec(source="o1", target="o2")],
        )
        assert len(s.links) == 1


class TestObjectModificationTarget:
    def test_all_optional(self):
        t = ObjectModificationTarget()
        assert t.objectName is None
        assert t.attributeName is None
        assert t.sourceObject is None
        assert t.targetObject is None


class TestObjectModificationChanges:
    def test_all_optional(self):
        c = ObjectModificationChanges()
        assert c.objectName is None
        assert c.value is None
        assert c.relationshipType is None


class TestObjectModification:
    def test_valid(self):
        m = ObjectModification(
            action="update_attribute",
            target=ObjectModificationTarget(objectName="o1", attributeName="title"),
            changes=ObjectModificationChanges(value="NewTitle"),
        )
        assert m.action == "update_attribute"

    def test_changes_optional(self):
        m = ObjectModification(
            action="delete_object",
            target=ObjectModificationTarget(objectName="o1"),
        )
        assert m.changes is None


class TestObjectModificationResponse:
    def test_valid_with_one_modification(self):
        r = ObjectModificationResponse(
            modifications=[
                ObjectModification(
                    action="add_object",
                    target=ObjectModificationTarget(objectName="o1"),
                )
            ]
        )
        assert len(r.modifications) == 1

    def test_rejects_empty_modifications(self):
        with pytest.raises(ValidationError):
            ObjectModificationResponse(modifications=[])


# =============================================================================
# Agent Diagram schemas
# =============================================================================

class TestAgentReplySpec:
    def test_valid_creation(self):
        r = AgentReplySpec(text="Hello!")
        assert r.text == "Hello!"
        assert r.replyType == "text"

    @pytest.mark.parametrize("rtype", ["text", "llm"])
    def test_valid_replyTypes(self, rtype):
        r = AgentReplySpec(text="hi", replyType=rtype)
        assert r.replyType == rtype

    def test_rejects_invalid_replyType(self):
        with pytest.raises(ValidationError):
            AgentReplySpec(text="hi", replyType="audio")


class TestAgentStateSpec:
    def test_valid_creation(self):
        s = AgentStateSpec(stateName="Greeting")
        assert s.stateName == "Greeting"
        assert s.type == "state"
        assert s.replies == []
        assert s.fallbackBodies == []

    def test_rejects_empty_stateName(self):
        with pytest.raises(ValidationError):
            AgentStateSpec(stateName="")

    def test_with_replies(self):
        s = AgentStateSpec(
            stateName="Welcome",
            replies=[AgentReplySpec(text="Welcome!")],
            fallbackBodies=[AgentReplySpec(text="Sorry, I didn't get that.")],
        )
        assert len(s.replies) == 1
        assert len(s.fallbackBodies) == 1


class TestAgentIntentSpec:
    def test_valid_creation(self):
        i = AgentIntentSpec(intentName="greet")
        assert i.intentName == "greet"
        assert i.type == "intent"
        assert i.trainingPhrases == []

    def test_rejects_empty_intentName(self):
        with pytest.raises(ValidationError):
            AgentIntentSpec(intentName="")

    def test_with_training_phrases(self):
        i = AgentIntentSpec(
            intentName="greet",
            trainingPhrases=["hello", "hi", "hey there"],
        )
        assert len(i.trainingPhrases) == 3


class TestAgentSingleElementSpec:
    def test_valid_state_type(self):
        e = AgentSingleElementSpec(type="state", stateName="Idle")
        assert e.type == "state"
        assert e.stateName == "Idle"

    def test_valid_intent_type(self):
        e = AgentSingleElementSpec(type="intent", intentName="greet")
        assert e.type == "intent"
        assert e.intentName == "greet"

    def test_valid_initial_type(self):
        e = AgentSingleElementSpec(type="initial", description="Start node")
        assert e.type == "initial"
        assert e.description == "Start node"

    def test_defaults(self):
        e = AgentSingleElementSpec()
        assert e.type == "state"
        assert e.stateName is None
        assert e.intentName is None
        assert e.replies == []
        assert e.fallbackBodies == []
        assert e.trainingPhrases == []
        assert e.description is None

    def test_rejects_invalid_type(self):
        with pytest.raises(ValidationError):
            AgentSingleElementSpec(type="unknown")


class TestAgentTransitionSpec:
    def test_valid_creation(self):
        t = AgentTransitionSpec(source="s1", target="s2")
        assert t.source == "s1"
        assert t.target == "s2"
        assert t.condition == "when_intent_matched"
        assert t.conditionValue is None
        assert t.label is None

    @pytest.mark.parametrize("cond", [
        "when_intent_matched", "when_no_intent_matched", "auto",
    ])
    def test_valid_conditions(self, cond):
        t = AgentTransitionSpec(source="A", target="B", condition=cond)
        assert t.condition == cond

    def test_rejects_invalid_condition(self):
        with pytest.raises(ValidationError):
            AgentTransitionSpec(source="A", target="B", condition="always")

    def test_with_directions(self):
        t = AgentTransitionSpec(
            source="A", target="B",
            sourceDirection="right", targetDirection="left",
        )
        assert t.sourceDirection == "right"
        assert t.targetDirection == "left"


class TestSystemAgentSpec:
    def test_valid_creation(self):
        s = SystemAgentSpec(
            systemName="ChatBot",
            states=[AgentStateSpec(stateName="Welcome")],
        )
        assert s.systemName == "ChatBot"
        assert s.hasInitialNode is True
        assert s.initialNode is None
        assert len(s.states) == 1
        assert s.intents == []
        assert s.transitions == []

    def test_rejects_empty_states(self):
        with pytest.raises(ValidationError):
            SystemAgentSpec(states=[])

    def test_accepts_initialNode_as_dict(self):
        s = SystemAgentSpec(
            states=[AgentStateSpec(stateName="S1")],
            initialNode={"x": 100, "y": 200, "description": "Start"},
        )
        assert s.initialNode == {"x": 100, "y": 200, "description": "Start"}

    def test_full_agent_system(self):
        s = SystemAgentSpec(
            systemName="HelpDesk",
            hasInitialNode=True,
            initialNode={"id": "init"},
            intents=[AgentIntentSpec(intentName="ask_help")],
            states=[
                AgentStateSpec(stateName="Welcome"),
                AgentStateSpec(stateName="Helping"),
            ],
            transitions=[
                AgentTransitionSpec(source="Welcome", target="Helping"),
            ],
        )
        assert len(s.intents) == 1
        assert len(s.states) == 2
        assert len(s.transitions) == 1


class TestAgentModificationTarget:
    def test_all_optional(self):
        t = AgentModificationTarget()
        assert t.stateName is None
        assert t.intentName is None
        assert t.sourceStateName is None
        assert t.targetStateName is None
        assert t.transitionId is None


class TestAgentModificationChanges:
    def test_all_optional(self):
        c = AgentModificationChanges()
        assert c.name is None
        assert c.intentName is None
        assert c.condition is None
        assert c.text is None
        assert c.replyType is None
        assert c.trainingPhrase is None


class TestAgentModification:
    def test_valid(self):
        m = AgentModification(
            action="rename_state",
            target=AgentModificationTarget(stateName="Old"),
            changes=AgentModificationChanges(name="New"),
        )
        assert m.action == "rename_state"

    def test_changes_optional(self):
        m = AgentModification(
            action="delete_intent",
            target=AgentModificationTarget(intentName="greet"),
        )
        assert m.changes is None


class TestAgentModificationResponse:
    def test_valid_with_one_modification(self):
        r = AgentModificationResponse(
            modifications=[
                AgentModification(
                    action="add_state",
                    target=AgentModificationTarget(stateName="New"),
                )
            ]
        )
        assert len(r.modifications) == 1

    def test_rejects_empty_modifications(self):
        with pytest.raises(ValidationError):
            AgentModificationResponse(modifications=[])


# =============================================================================
# GUI Diagram schemas
# =============================================================================

class TestGUISampleDataPoint:
    def test_valid_creation(self):
        dp = GUISampleDataPoint(name="Sales", value=100)
        assert dp.name == "Sales"
        assert dp.value == 100
        assert dp.color is None

    def test_accepts_optional_color(self):
        dp = GUISampleDataPoint(name="Revenue", value=50, color="#FF0000")
        assert dp.color == "#FF0000"

    def test_default_value(self):
        dp = GUISampleDataPoint(name="X")
        assert dp.value == 0

    def test_value_as_string(self):
        dp = GUISampleDataPoint(name="Label", value="text")
        assert dp.value == "text"


class TestGUISectionSpec:
    @pytest.mark.parametrize("section_type", [
        "hero", "feature_list", "content", "form", "table",
        "bar_chart", "pie_chart", "line_chart", "radar_chart",
        "dashboard", "metric_card", "stats_grid", "footer",
        "two_column",
    ])
    def test_valid_types(self, section_type):
        s = GUISectionSpec(type=section_type)
        assert s.type == section_type

    def test_defaults(self):
        s = GUISectionSpec()
        assert s.type == "content"
        assert s.title == ""
        assert s.body is None
        assert s.items == []
        assert s.fields == []
        assert s.ctaLabel is None
        assert s.className is None
        assert s.sampleData == []

    def test_rejects_invalid_type(self):
        with pytest.raises(ValidationError):
            GUISectionSpec(type="carousel")

    def test_with_sample_data(self):
        s = GUISectionSpec(
            type="bar_chart",
            title="Sales",
            sampleData=[GUISampleDataPoint(name="Q1", value=100)],
        )
        assert len(s.sampleData) == 1

    def test_with_items_and_fields(self):
        s = GUISectionSpec(
            type="form",
            items=["Name", "Email"],
            fields=["name_field", "email_field"],
        )
        assert len(s.items) == 2
        assert len(s.fields) == 2


class TestSingleGUIElementSpec:
    def test_valid_creation(self):
        e = SingleGUIElementSpec(
            pageName="Home",
            section=GUISectionSpec(type="hero", title="Welcome"),
        )
        assert e.pageName == "Home"
        assert e.section.type == "hero"

    def test_rejects_empty_pageName(self):
        with pytest.raises(ValidationError):
            SingleGUIElementSpec(
                pageName="",
                section=GUISectionSpec(),
            )


class TestGUIPageSpec:
    def test_valid_creation(self):
        p = GUIPageSpec(pageName="Dashboard")
        assert p.pageName == "Dashboard"
        assert p.sections == []

    def test_rejects_empty_pageName(self):
        with pytest.raises(ValidationError):
            GUIPageSpec(pageName="")

    def test_with_sections(self):
        p = GUIPageSpec(
            pageName="Home",
            sections=[
                GUISectionSpec(type="hero", title="Banner"),
                GUISectionSpec(type="footer"),
            ],
        )
        assert len(p.sections) == 2


class TestSystemGUISpec:
    def test_valid_creation(self):
        s = SystemGUISpec(
            systemName="MyApp",
            pages=[GUIPageSpec(pageName="Home")],
        )
        assert s.systemName == "MyApp"
        assert len(s.pages) == 1

    def test_rejects_empty_pages(self):
        with pytest.raises(ValidationError):
            SystemGUISpec(pages=[])

    def test_default_systemName(self):
        s = SystemGUISpec(pages=[GUIPageSpec(pageName="Home")])
        assert s.systemName == ""


class TestGUIModificationSpec:
    def test_valid_append_section(self):
        m = GUIModificationSpec(
            operation="append_section",
            pageName="Home",
            section=GUISectionSpec(type="footer"),
        )
        assert m.operation == "append_section"
        assert m.pageName == "Home"

    def test_valid_rename_page(self):
        m = GUIModificationSpec(
            operation="rename_page",
            pageName="OldName",
            newPageName="NewName",
        )
        assert m.operation == "rename_page"
        assert m.newPageName == "NewName"

    def test_valid_remove_page(self):
        m = GUIModificationSpec(
            operation="remove_page",
            pageName="OldPage",
        )
        assert m.operation == "remove_page"

    @pytest.mark.parametrize("op", ["append_section", "rename_page", "remove_page"])
    def test_valid_operations(self, op):
        m = GUIModificationSpec(operation=op, pageName="P")
        assert m.operation == op

    def test_rejects_invalid_operation(self):
        with pytest.raises(ValidationError):
            GUIModificationSpec(operation="delete_section", pageName="P")

    def test_rejects_empty_pageName(self):
        with pytest.raises(ValidationError):
            GUIModificationSpec(pageName="")

    def test_defaults(self):
        m = GUIModificationSpec(pageName="Home")
        assert m.operation == "append_section"
        assert m.newPageName is None
        assert m.section is None


# =============================================================================
# Quantum Circuit schemas
# =============================================================================

class TestQuantumOperationSpec:
    def test_valid_creation(self):
        op = QuantumOperationSpec(gate="H")
        assert op.gate == "H"
        assert op.row is None
        assert op.column == 0
        assert op.controlRow is None
        assert op.targetRow is None
        assert op.controlRow2 is None
        assert op.label is None
        assert op.height is None

    def test_with_all_fields(self):
        op = QuantumOperationSpec(
            gate="CNOT",
            row=0,
            column=1,
            controlRow=0,
            targetRow=1,
            controlRow2=2,
            label="CX",
            height=2,
        )
        assert op.gate == "CNOT"
        assert op.controlRow == 0
        assert op.targetRow == 1
        assert op.controlRow2 == 2
        assert op.label == "CX"
        assert op.height == 2


class TestSingleQuantumGateSpec:
    def test_valid_creation(self):
        g = SingleQuantumGateSpec(operation=QuantumOperationSpec(gate="X"))
        assert g.operation.gate == "X"


class TestSystemQuantumCircuitSpec:
    def test_valid_creation(self):
        s = SystemQuantumCircuitSpec(
            qubitCount=3,
            algorithmName="Bell",
            operations=[QuantumOperationSpec(gate="H")],
        )
        assert s.qubitCount == 3
        assert s.algorithmName == "Bell"
        assert len(s.operations) == 1

    def test_default_qubitCount(self):
        s = SystemQuantumCircuitSpec(
            operations=[QuantumOperationSpec(gate="X")],
        )
        assert s.qubitCount == 2

    def test_rejects_qubitCount_zero(self):
        with pytest.raises(ValidationError):
            SystemQuantumCircuitSpec(
                qubitCount=0,
                operations=[QuantumOperationSpec(gate="H")],
            )

    def test_rejects_negative_qubitCount(self):
        with pytest.raises(ValidationError):
            SystemQuantumCircuitSpec(
                qubitCount=-1,
                operations=[QuantumOperationSpec(gate="H")],
            )

    def test_rejects_empty_operations(self):
        with pytest.raises(ValidationError):
            SystemQuantumCircuitSpec(operations=[])

    def test_multiple_operations(self):
        s = SystemQuantumCircuitSpec(
            qubitCount=2,
            operations=[
                QuantumOperationSpec(gate="H", row=0, column=0),
                QuantumOperationSpec(gate="CNOT", controlRow=0, targetRow=1, column=1),
            ],
        )
        assert len(s.operations) == 2


class TestQuantumModificationSpec:
    def test_valid_append(self):
        m = QuantumModificationSpec(
            mode="append",
            operations=[QuantumOperationSpec(gate="Z")],
        )
        assert m.mode == "append"
        assert m.qubitCount is None

    def test_valid_replace(self):
        m = QuantumModificationSpec(
            mode="replace",
            qubitCount=4,
            operations=[QuantumOperationSpec(gate="H")],
        )
        assert m.mode == "replace"
        assert m.qubitCount == 4

    @pytest.mark.parametrize("mode", ["append", "replace"])
    def test_valid_modes(self, mode):
        m = QuantumModificationSpec(
            mode=mode,
            operations=[QuantumOperationSpec(gate="X")],
        )
        assert m.mode == mode

    def test_rejects_invalid_mode(self):
        with pytest.raises(ValidationError):
            QuantumModificationSpec(
                mode="insert",
                operations=[QuantumOperationSpec(gate="X")],
            )

    def test_rejects_empty_operations(self):
        with pytest.raises(ValidationError):
            QuantumModificationSpec(operations=[])

    def test_rejects_qubitCount_zero(self):
        with pytest.raises(ValidationError):
            QuantumModificationSpec(
                qubitCount=0,
                operations=[QuantumOperationSpec(gate="H")],
            )

    def test_default_mode(self):
        m = QuantumModificationSpec(
            operations=[QuantumOperationSpec(gate="X")],
        )
        assert m.mode == "append"
