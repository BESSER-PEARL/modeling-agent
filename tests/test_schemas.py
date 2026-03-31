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
    StateCodeBlockSpec,
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
    AgentInitialNodeSpec,
    AgentRagSpec,
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
            action="modify_class",
            target=ClassModificationTarget(className="OldName"),
            changes=ClassModificationChanges(name="NewName"),
        )
        assert m.action == "modify_class"
        assert m.target.className == "OldName"
        assert m.changes.name == "NewName"

    def test_changes_optional(self):
        m = ClassModification(
            action="remove_element",
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
                    action="remove_element",
                    target=ClassModificationTarget(className="B"),
                ),
            ]
        )
        assert len(r.modifications) == 2

    def test_rejects_empty_modifications(self):
        with pytest.raises(ValidationError):
            ClassModificationResponse(modifications=[])


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
            action="modify_state",
            target=StateMachineModificationTarget(stateName="Old"),
            changes=StateMachineModificationChanges(name="New"),
        )
        assert m.action == "modify_state"

    def test_changes_optional(self):
        m = StateMachineModification(
            action="remove_element",
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
            action="modify_attribute_value",
            target=ObjectModificationTarget(objectName="o1", attributeName="title"),
            changes=ObjectModificationChanges(value="NewTitle"),
        )
        assert m.action == "modify_attribute_value"

    def test_changes_optional(self):
        m = ObjectModification(
            action="remove_element",
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
        assert isinstance(s.initialNode, AgentInitialNodeSpec)
        assert s.initialNode.description == "Start"

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
            action="modify_state",
            target=AgentModificationTarget(stateName="Old"),
            changes=AgentModificationChanges(name="New"),
        )
        assert m.action == "modify_state"

    def test_changes_optional(self):
        m = AgentModification(
            action="remove_element",
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


# =============================================================================
# Class Diagram — new field tests
# =============================================================================


class TestAttributeSpecNewFields:
    """Tests for isDerived, defaultValue, isOptional on AttributeSpec."""

    def test_defaults_for_new_fields(self):
        a = AttributeSpec(name="title")
        assert a.isDerived is False
        assert a.defaultValue is None
        assert a.isOptional is False

    def test_isDerived_true(self):
        a = AttributeSpec(name="fullName", isDerived=True)
        assert a.isDerived is True

    def test_defaultValue_string(self):
        a = AttributeSpec(name="status", defaultValue="active")
        assert a.defaultValue == "active"

    def test_isOptional_true(self):
        a = AttributeSpec(name="nickname", isOptional=True)
        assert a.isOptional is True

    def test_all_new_fields_together(self):
        a = AttributeSpec(
            name="score",
            type="float",
            isDerived=True,
            defaultValue="0.0",
            isOptional=True,
        )
        assert a.isDerived is True
        assert a.defaultValue == "0.0"
        assert a.isOptional is True


class TestMethodSpecNewFields:
    """Tests for isAbstract, implementationType, code on MethodSpec."""

    def test_defaults_for_new_fields(self):
        m = MethodSpec(name="doWork")
        assert m.isAbstract is False
        assert m.implementationType == "none"
        assert m.code is None

    def test_isAbstract_true(self):
        m = MethodSpec(name="execute", isAbstract=True)
        assert m.isAbstract is True

    def test_implementationType_code_with_body(self):
        m = MethodSpec(
            name="calc",
            implementationType="code",
            code="def calc(self): return 1",
        )
        assert m.implementationType == "code"
        assert m.code == "def calc(self): return 1"

    def test_implementationType_bal(self):
        m = MethodSpec(name="process", implementationType="bal")
        assert m.implementationType == "bal"
        assert m.code is None

    @pytest.mark.parametrize("impl", ["none", "code", "bal", "state_machine", "quantum_circuit"])
    def test_valid_implementationTypes(self, impl):
        m = MethodSpec(name="m", implementationType=impl)
        assert m.implementationType == impl

    def test_rejects_invalid_implementationType(self):
        with pytest.raises(ValidationError):
            MethodSpec(name="m", implementationType="javascript")


class TestSingleClassSpecNewFields:
    """Tests for isAbstract and isEnumeration on SingleClassSpec."""

    def test_defaults_for_new_fields(self):
        c = SingleClassSpec(className="Foo")
        assert c.isAbstract is False
        assert c.isEnumeration is False

    def test_isAbstract_true(self):
        c = SingleClassSpec(className="Shape", isAbstract=True)
        assert c.isAbstract is True

    def test_isEnumeration_true(self):
        c = SingleClassSpec(className="Color", isEnumeration=True)
        assert c.isEnumeration is True

    def test_abstract_class_with_methods(self):
        c = SingleClassSpec(
            className="Vehicle",
            isAbstract=True,
            methods=[MethodSpec(name="move", isAbstract=True)],
        )
        assert c.isAbstract is True
        assert c.methods[0].isAbstract is True

    def test_enumeration_with_attributes_as_values(self):
        c = SingleClassSpec(
            className="Priority",
            isEnumeration=True,
            attributes=[
                AttributeSpec(name="LOW"),
                AttributeSpec(name="MEDIUM"),
                AttributeSpec(name="HIGH"),
            ],
        )
        assert c.isEnumeration is True
        assert len(c.attributes) == 3


class TestClassModificationChangesNewFields:
    """Tests for isDerived, defaultValue, isOptional, isAbstract, isEnumeration, implementationType, code."""

    def test_new_fields_default_none(self):
        c = ClassModificationChanges()
        assert c.isDerived is None
        assert c.defaultValue is None
        assert c.isOptional is None
        assert c.isAbstract is None
        assert c.isEnumeration is None
        assert c.implementationType is None
        assert c.code is None

    def test_isDerived_set(self):
        c = ClassModificationChanges(isDerived=True)
        assert c.isDerived is True

    def test_defaultValue_set(self):
        c = ClassModificationChanges(defaultValue="pending")
        assert c.defaultValue == "pending"

    def test_isOptional_set(self):
        c = ClassModificationChanges(isOptional=True)
        assert c.isOptional is True

    def test_isAbstract_set(self):
        c = ClassModificationChanges(isAbstract=True)
        assert c.isAbstract is True

    def test_isEnumeration_set(self):
        c = ClassModificationChanges(isEnumeration=True)
        assert c.isEnumeration is True

    def test_implementationType_and_code(self):
        c = ClassModificationChanges(
            implementationType="code",
            code="def run(self): pass",
        )
        assert c.implementationType == "code"
        assert c.code == "def run(self): pass"

    def test_all_new_fields_together(self):
        c = ClassModificationChanges(
            isDerived=True,
            defaultValue="0",
            isOptional=False,
            isAbstract=True,
            isEnumeration=False,
            implementationType="bal",
            code=None,
        )
        assert c.isDerived is True
        assert c.defaultValue == "0"
        assert c.isOptional is False
        assert c.isAbstract is True
        assert c.isEnumeration is False
        assert c.implementationType == "bal"
        assert c.code is None


# =============================================================================
# State Machine — new field tests
# =============================================================================


class TestStateMachineModificationChangesNewFields:
    """Tests for stateType, code, language on StateMachineModificationChanges."""

    def test_new_fields_default_none(self):
        c = StateMachineModificationChanges()
        assert c.stateType is None
        assert c.code is None
        assert c.language is None

    def test_stateType_set(self):
        c = StateMachineModificationChanges(stateType="initial")
        assert c.stateType == "initial"

    def test_code_and_language(self):
        c = StateMachineModificationChanges(
            code="print('hello')",
            language="python",
        )
        assert c.code == "print('hello')"
        assert c.language == "python"


class TestStateMachineModificationAddState:
    """Tests for StateMachineModification with action='add_state'."""

    def test_add_state_with_changes(self):
        m = StateMachineModification(
            action="add_state",
            target=StateMachineModificationTarget(stateName="Processing"),
            changes=StateMachineModificationChanges(
                stateType="regular",
                entryAction="log('entered')",
            ),
        )
        assert m.action == "add_state"
        assert m.target.stateName == "Processing"
        assert m.changes.stateType == "regular"
        assert m.changes.entryAction == "log('entered')"


class TestStateMachineModificationAddCodeBlock:
    """Tests for StateMachineModification with action='add_code_block'."""

    def test_add_code_block(self):
        m = StateMachineModification(
            action="add_code_block",
            target=StateMachineModificationTarget(stateName="Handler"),
            changes=StateMachineModificationChanges(
                name="process_data",
                code="def process_data(ctx):\n    return ctx['data']",
                language="python",
            ),
        )
        assert m.action == "add_code_block"
        assert m.changes.name == "process_data"
        assert m.changes.code.startswith("def process_data")
        assert m.changes.language == "python"


class TestStateCodeBlockSpec:
    """Tests for StateCodeBlockSpec schema."""

    def test_valid_creation(self):
        cb = StateCodeBlockSpec(name="setup", code="x = 1")
        assert cb.name == "setup"
        assert cb.code == "x = 1"
        assert cb.language == "python"

    def test_custom_language(self):
        cb = StateCodeBlockSpec(name="init", code="val x = 1", language="kotlin")
        assert cb.language == "kotlin"

    def test_multiline_code(self):
        code = "def run():\n    print('hello')\n    return True"
        cb = StateCodeBlockSpec(name="runner", code=code)
        assert cb.code == code


class TestSystemStateMachineSpecCodeBlocks:
    """Tests for codeBlocks field on SystemStateMachineSpec."""

    def test_default_empty_codeBlocks(self):
        s = SystemStateMachineSpec(
            states=[StateSpec(stateName="Idle")],
        )
        assert s.codeBlocks == []

    def test_with_codeBlocks(self):
        s = SystemStateMachineSpec(
            states=[StateSpec(stateName="Active")],
            codeBlocks=[
                StateCodeBlockSpec(name="setup", code="x = 0"),
                StateCodeBlockSpec(name="teardown", code="cleanup()"),
            ],
        )
        assert len(s.codeBlocks) == 2
        assert s.codeBlocks[0].name == "setup"
        assert s.codeBlocks[1].name == "teardown"


# =============================================================================
# Object Diagram — new field tests
# =============================================================================


class TestObjectModificationAddObject:
    """Tests for ObjectModification with action='add_object'."""

    def test_add_object(self):
        m = ObjectModification(
            action="add_object",
            target=ObjectModificationTarget(objectName="book1"),
            changes=ObjectModificationChanges(
                className="Book",
                attributes=[
                    ObjectAttributeSpec(name="title", value="Dune"),
                    ObjectAttributeSpec(name="year", value="1965"),
                ],
            ),
        )
        assert m.action == "add_object"
        assert m.changes.className == "Book"
        assert len(m.changes.attributes) == 2
        assert m.changes.attributes[0].name == "title"
        assert m.changes.attributes[0].value == "Dune"


class TestObjectModificationChangesNewFields:
    """Tests for className and attributes on ObjectModificationChanges."""

    def test_className_default_none(self):
        c = ObjectModificationChanges()
        assert c.className is None
        assert c.attributes is None

    def test_className_set(self):
        c = ObjectModificationChanges(className="User")
        assert c.className == "User"

    def test_attributes_list(self):
        c = ObjectModificationChanges(
            attributes=[
                ObjectAttributeSpec(name="name", value="Alice"),
            ],
        )
        assert len(c.attributes) == 1
        assert c.attributes[0].value == "Alice"


# =============================================================================
# Agent Diagram — new field tests
# =============================================================================


class TestAgentReplySpecNewFields:
    """Tests for new replyType values (rag, db_reply, code) and ragDatabaseName."""

    def test_replyType_rag(self):
        r = AgentReplySpec(text="Searching knowledge base...", replyType="rag", ragDatabaseName="MyKB")
        assert r.replyType == "rag"
        assert r.ragDatabaseName == "MyKB"

    def test_replyType_db_reply(self):
        r = AgentReplySpec(text="SELECT * FROM users", replyType="db_reply")
        assert r.replyType == "db_reply"
        assert r.ragDatabaseName is None

    def test_replyType_code(self):
        r = AgentReplySpec(text="print('hello')", replyType="code")
        assert r.replyType == "code"

    @pytest.mark.parametrize("rtype", ["text", "llm", "rag", "db_reply", "code"])
    def test_all_valid_replyTypes(self, rtype):
        r = AgentReplySpec(text="t", replyType=rtype)
        assert r.replyType == rtype

    def test_ragDatabaseName_default_none(self):
        r = AgentReplySpec(text="hi")
        assert r.ragDatabaseName is None


class TestAgentRagSpec:
    """Tests for AgentRagSpec schema."""

    def test_valid_creation(self):
        rag = AgentRagSpec(name="CustomerKB")
        assert rag.name == "CustomerKB"

    def test_another_name(self):
        rag = AgentRagSpec(name="ProductDocs")
        assert rag.name == "ProductDocs"


class TestAgentModificationAddState:
    """Tests for AgentModification with action='add_state'."""

    def test_add_state_with_replies(self):
        m = AgentModification(
            action="add_state",
            target=AgentModificationTarget(stateName="greeting"),
            changes=AgentModificationChanges(
                replies=[
                    AgentReplySpec(text="Hello!", replyType="text"),
                    AgentReplySpec(text="How can I help?", replyType="llm"),
                ],
            ),
        )
        assert m.action == "add_state"
        assert len(m.changes.replies) == 2
        assert m.changes.replies[0].text == "Hello!"
        assert m.changes.replies[1].replyType == "llm"


class TestAgentModificationAddIntent:
    """Tests for AgentModification with action='add_intent'."""

    def test_add_intent_with_training_phrases(self):
        m = AgentModification(
            action="add_intent",
            target=AgentModificationTarget(intentName="OrderFood"),
            changes=AgentModificationChanges(
                trainingPhrases=["I want to order", "get me food", "order pizza"],
            ),
        )
        assert m.action == "add_intent"
        assert len(m.changes.trainingPhrases) == 3

    def test_add_intent_minimal(self):
        m = AgentModification(
            action="add_intent",
            target=AgentModificationTarget(intentName="Greet"),
        )
        assert m.action == "add_intent"
        assert m.changes is None


class TestAgentModificationAddRagElement:
    """Tests for AgentModification with action='add_rag_element'."""

    def test_add_rag_element(self):
        m = AgentModification(
            action="add_rag_element",
            target=AgentModificationTarget(stateName="queryState"),
            changes=AgentModificationChanges(name="CustomerKB"),
        )
        assert m.action == "add_rag_element"
        assert m.changes.name == "CustomerKB"


class TestAgentModificationChangesNewFields:
    """Tests for replies and trainingPhrases on AgentModificationChanges."""

    def test_replies_default_none(self):
        c = AgentModificationChanges()
        assert c.replies is None

    def test_trainingPhrases_default_none(self):
        c = AgentModificationChanges()
        assert c.trainingPhrases is None

    def test_replies_set(self):
        c = AgentModificationChanges(
            replies=[AgentReplySpec(text="Welcome!", replyType="text")],
        )
        assert len(c.replies) == 1
        assert c.replies[0].text == "Welcome!"

    def test_trainingPhrases_set(self):
        c = AgentModificationChanges(
            trainingPhrases=["hi", "hello", "hey"],
        )
        assert len(c.trainingPhrases) == 3

    def test_replies_and_trainingPhrases_together(self):
        c = AgentModificationChanges(
            replies=[AgentReplySpec(text="Sure!", replyType="text")],
            trainingPhrases=["help me", "I need help"],
        )
        assert len(c.replies) == 1
        assert len(c.trainingPhrases) == 2


class TestSystemAgentSpecRagElements:
    """Tests for ragElements field on SystemAgentSpec."""

    def test_default_empty_ragElements(self):
        s = SystemAgentSpec(
            states=[AgentStateSpec(stateName="welcome")],
        )
        assert s.ragElements == []

    def test_with_ragElements(self):
        s = SystemAgentSpec(
            states=[AgentStateSpec(stateName="queryState")],
            ragElements=[
                AgentRagSpec(name="CustomerKB"),
                AgentRagSpec(name="ProductDocs"),
            ],
        )
        assert len(s.ragElements) == 2
        assert s.ragElements[0].name == "CustomerKB"
        assert s.ragElements[1].name == "ProductDocs"

    def test_full_agent_with_rag(self):
        s = SystemAgentSpec(
            systemName="SupportBot",
            states=[
                AgentStateSpec(
                    stateName="ragQuery",
                    replies=[AgentReplySpec(text="Looking it up...", replyType="rag", ragDatabaseName="KB")],
                ),
            ],
            ragElements=[AgentRagSpec(name="KB")],
        )
        assert s.systemName == "SupportBot"
        assert s.ragElements[0].name == "KB"
        assert s.states[0].replies[0].replyType == "rag"
        assert s.states[0].replies[0].ragDatabaseName == "KB"
