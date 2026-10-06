"""Pins for the React Flow v4 acceptance findings (agent behaviour wave, 2026-10).

Each test reproduces one live finding from the browser sweep
(``wave/agent-behaviour/results.jsonl``) and failed against the pre-fix code.
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

import confirmation
from protocol.types import AssistantRequest, WorkspaceContext
from session_keys import (
    MISMATCH_REGEN_PENDING,
    PENDING_COMPLETE_SYSTEM,
    PENDING_GUI_CHOICE,
    PENDING_SMART_GEN_INSTRUCTIONS,
    PENDING_SMART_GEN_PROVIDER,
    PENDING_SMART_GEN_TIMESTAMP,
    PENDING_WEBAPP_GENERATE,
    UNIFIED_CLASSIFICATION,
)
from unified_classifier import UnifiedClassification
from utilities.model_context import (
    compact_model_summary,
    detailed_model_summary,
    is_diagram_nontrivial,
)

from tests.conftest import FakeSession, MINIMAL_CLASS_MODEL


def _req(message, diagram_type="ClassDiagram", raw_payload=None):
    return AssistantRequest(
        message=message,
        context=WorkspaceContext(active_diagram_type=diagram_type),
        raw_payload=raw_payload or {},
    )


def _arm_stale_smart_gen(session):
    """The state c14 left behind: an injected "run Django, push, deploy"
    request stashed a Spec-Driven run behind a one-click Continue."""
    session.set(PENDING_SMART_GEN_INSTRUCTIONS, "Generate a Django app, push to acme/hotel-app, deploy")
    session.set(PENDING_SMART_GEN_PROVIDER, "openai")
    session.set(PENDING_SMART_GEN_TIMESTAMP, time.time())


def _payloads(mock):
    return [c.args[1] for c in mock.call_args_list if len(c.args) > 1 and isinstance(c.args[1], dict)]


def _is_smart_gen_prompt(payload):
    return any(
        isinstance(a, dict) and a.get("prompt") == "generate anyway with my current model"
        for a in payload.get("suggestedActions") or []
    )


# ---------------------------------------------------------------------------
# 2. A stale Spec-Driven stash must not resurface or linger
# ---------------------------------------------------------------------------

class TestStaleSmartGenStash:
    def _answer_confirmation(self, session, uc, message):
        session.set(UNIFIED_CLASSIFICATION, uc)
        with patch.object(confirmation, "parse_assistant_request", return_value=_req(message)), \
             patch.object(confirmation, "reply_message"), \
             patch.object(confirmation, "reply_payload") as payload, \
             patch.object(confirmation, "execute_model_operation", return_value="ClassDiagram"):
            assert confirmation.handle_pending_system_confirmation(session) is True
        return payload

    @pytest.mark.parametrize("answer", ["new_tab", "replace"])
    def test_answering_replace_or_new_tab_does_not_offer_the_old_run(self, answer):
        """c16: clicking "Create in a new diagram tab" for a hospital model
        replied "Model rebuilt and ready ... Do you want to continue?" with a
        Continue button for the run injected 10 turns earlier."""
        session = FakeSession()
        _arm_stale_smart_gen(session)
        session.set(PENDING_COMPLETE_SYSTEM, {
            "message": "create a hospital management system", "diagram_type": "ClassDiagram",
            "operation": {}, "default_mode": "complete_system", "can_add_tab": True,
        })
        uc = UnifiedClassification(
            intent="create_complete_system_intent", reason="t",
            pending_flow_action="answer", pending_flow_answer=answer,
        )
        payload = self._answer_confirmation(session, uc, answer.replace("_", " "))
        assert not any(_is_smart_gen_prompt(p) for p in _payloads(payload))

    def _preamble(self, session, message, uc):
        session.set(UNIFIED_CLASSIFICATION, uc)
        from handlers import generation_handler as gh
        with patch("protocol.adapters.parse_assistant_request", return_value=_req(message)), \
             patch("session_helpers.reply_message"), patch("session_helpers.reply_payload"):
            return gh.handle_pending_smart_gen_confirmation(session)

    def test_a_new_create_request_abandons_the_stash(self):
        """o01: the next modeling request routed as new_request, but the stash
        survived (and kept the classifier on gen=django for several turns)."""
        session = FakeSession()
        _arm_stale_smart_gen(session)
        handled = self._preamble(
            session, "Create an object diagram: a Hotel 'Grand Plaza' with two Rooms",
            UnifiedClassification(intent="create_complete_system_intent", reason="t",
                                  pending_flow_action="new_request"),
        )
        assert handled is False
        assert not session.get(PENDING_SMART_GEN_INSTRUCTIONS)

    def test_a_modify_request_abandons_the_stash(self):
        session = FakeSession()
        _arm_stale_smart_gen(session)
        self._preamble(
            session, "Rename Shipped to InTransit",
            UnifiedClassification(intent="modify_model_intent", reason="t",
                                  pending_flow_action="new_request"),
        )
        assert not session.get(PENDING_SMART_GEN_INSTRUCTIONS)

    def test_the_mismatch_rebuild_prompt_keeps_its_stash(self):
        session = FakeSession()
        _arm_stale_smart_gen(session)
        session.set(MISMATCH_REGEN_PENDING, "create a class diagram for a shoe store")
        self._preamble(
            session, "create a class diagram for a shoe store",
            UnifiedClassification(intent="create_complete_system_intent", reason="t",
                                  pending_flow_action="new_request"),
        )
        assert session.get(PENDING_SMART_GEN_INSTRUCTIONS)
        assert session.get(MISMATCH_REGEN_PENDING)

    def test_an_off_topic_interjection_keeps_the_stash(self):
        session = FakeSession()
        _arm_stale_smart_gen(session)
        self._preamble(
            session, "tell me a joke",
            UnifiedClassification(intent="out_of_scope_intent", reason="t",
                                  pending_flow_action="new_request"),
        )
        assert session.get(PENDING_SMART_GEN_INSTRUCTIONS)


# ---------------------------------------------------------------------------
# 3 + 6. Model summaries see v3 abstract classes, enums, OCL and ObjectName
# ---------------------------------------------------------------------------

def _library_v3():
    """The class diagram after c02/c03, as v4_to_v3 delivers it."""
    els = {
        "book": {"id": "book", "name": "Book", "type": "Class", "owner": None},
        "b-pages": {"id": "b-pages", "name": "pages", "type": "ClassAttribute",
                    "owner": "book", "attributeType": "int"},
        "author": {"id": "author", "name": "Author", "type": "AbstractClass", "owner": None},
        "a-name": {"id": "a-name", "name": "name", "type": "ClassAttribute",
                   "owner": "author", "attributeType": "str"},
        "genre": {"id": "genre", "name": "Genre", "type": "Enumeration", "owner": None},
        "g1": {"id": "g1", "name": "FICTION", "type": "ClassAttribute", "owner": "genre"},
        "g2": {"id": "g2", "name": "SCIENCE", "type": "ClassAttribute", "owner": "genre"},
        "ocl": {"id": "ocl", "name": "", "type": "ClassOCLConstraint", "owner": None,
                "constraint": "context Book inv positive_pages: self.pages > 0"},
    }
    rels = {
        "r1": {"id": "r1", "type": "ClassBidirectional", "name": "writes",
               "source": {"element": "author", "multiplicity": "1..*"},
               "target": {"element": "book", "multiplicity": "0..*"}},
        "l1": {"id": "l1", "type": "ClassOCLLink",
               "source": {"element": "ocl"}, "target": {"element": "book"}},
    }
    return {"version": "3.0.0", "type": "ClassDiagram", "elements": els, "relationships": rels}


class TestClassSummaryV3Types:
    def test_detailed_summary_keeps_abstract_enum_and_ocl(self):
        summary = detailed_model_summary(_library_v3(), "ClassDiagram")
        assert "Author (abstract)" in summary
        assert "Genre {FICTION, SCIENCE}" in summary
        assert "self.pages > 0" in summary
        # The association resolves to names, the OCL link is not an association.
        assert "Author -> Book (association)" in summary
        assert "ocl -> Book" not in summary
        assert "Relationships (1)" in summary

    def test_compact_summary_counts_the_abstract_class(self):
        assert "Author" in compact_model_summary(_library_v3(), "ClassDiagram")

    def test_a_diagram_of_only_abstract_classes_is_nontrivial(self):
        model = {"elements": {"a": {"name": "Shape", "type": "AbstractClass"}}, "relationships": {}}
        assert is_diagram_nontrivial(model, "ClassDiagram") is True


def _hotel_objects_v3():
    return {
        "type": "ObjectDiagram",
        "elements": {
            "o1": {"id": "o1", "name": "guest1", "type": "ObjectName", "owner": None,
                   "className": "Guest", "attributes": ["o1a"]},
            "o1a": {"id": "o1a", "name": "lastName = Smith", "type": "ObjectAttribute", "owner": "o1"},
        },
        "relationships": {},
    }


class TestObjectSummaryV3Types:
    def test_detailed_summary_lists_objectname_boxes(self):
        summary = detailed_model_summary(_hotel_objects_v3(), "ObjectDiagram")
        assert "Object guest1: Guest" in summary
        assert "lastName = Smith" in summary

    def test_compact_and_nontrivial_see_objects(self):
        assert "1 object(s): guest1" in compact_model_summary(_hotel_objects_v3(), "ObjectDiagram")
        assert is_diagram_nontrivial(_hotel_objects_v3(), "ObjectDiagram") is True


# ---------------------------------------------------------------------------
# 4. A relationship change next to a class rename is kept
# ---------------------------------------------------------------------------

def _class_mod_spec(handler, mods, current_model):
    from schemas import ClassModificationResponse

    parsed = ClassModificationResponse.model_validate({"modifications": mods})
    with patch.object(handler, "predict_structured", return_value=parsed):
        return handler.generate_modification("rename and change", current_model=current_model)


def _ops(spec):
    if isinstance(spec.get("modifications"), list):
        return spec["modifications"]
    return [spec["modification"]] if isinstance(spec.get("modification"), dict) else []


class TestRenameKeepsRealRelationshipChanges:
    _MODEL = {
        "elements": {
            "a": {"id": "a", "name": "Author", "type": "AbstractClass", "owner": None},
            "b": {"id": "b", "name": "Book", "type": "Class", "owner": None},
        },
        "relationships": {
            "r": {"id": "r", "type": "ClassBidirectional",
                  "source": {"element": "a", "multiplicity": "0..*"},
                  "target": {"element": "b", "multiplicity": "0..*"}},
        },
    }

    def test_multiplicity_change_survives_the_rename(self):
        """c04: "Rename Author to Writer ... every Book has at least one writer
        (1..*)" applied only the rename; the multiplicity change vanished."""
        from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler

        spec = _class_mod_spec(ClassDiagramHandler(llm=None), [
            {"action": "modify_class", "target": {"className": "Author"},
             "changes": {"name": "Writer", "isAbstract": False}},
            {"action": "modify_relationship",
             "target": {"sourceClass": "Writer", "targetClass": "Book"},
             "changes": {"sourceMultiplicity": "1..*"}},
        ], self._MODEL)
        actions = [m["action"] for m in _ops(spec)]
        assert "modify_relationship" in actions
        assert "Applied 2 changes" in spec["message"]

    def test_a_pure_rename_echo_is_still_stripped(self):
        from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler

        spec = _class_mod_spec(ClassDiagramHandler(llm=None), [
            {"action": "modify_class", "target": {"className": "Author"},
             "changes": {"name": "Writer"}},
            {"action": "modify_relationship",
             "target": {"sourceClass": "Author", "targetClass": "Book"},
             "changes": {"name": "Writer"}},
        ], self._MODEL)
        assert [m["action"] for m in _ops(spec)] == ["modify_class"]
        assert "Applied" not in spec["message"]

    def test_message_does_not_count_ops_dropped_as_phantoms(self):
        """The reply was built before phantom-target ops were dropped."""
        from diagram_handlers.types.class_diagram_handler import ClassDiagramHandler

        spec = _class_mod_spec(ClassDiagramHandler(llm=None), [
            {"action": "add_attribute", "target": {"className": "Book"},
             "changes": {"name": "title", "type": "str"}},
            {"action": "remove_element", "target": {"className": "Book", "attributeName": "isbn"}},
            {"action": "modify_class", "target": {"className": "Book"},
             "changes": {"isAbstract": True}},
        ], self._MODEL)
        assert "Applied 2 changes" in spec["message"]
        assert "Applied 3 changes" not in spec["message"]


# ---------------------------------------------------------------------------
# 5. BPMN flows to nodes added in the same request are kept
# ---------------------------------------------------------------------------

class TestBpmnSameRequestNodes:
    _MODEL = {
        "elements": {
            "t1": {"id": "t1", "name": "Issue Refund", "type": "BPMNTask"},
            "e1": {"id": "e1", "name": "Refund Issued", "type": "BPMNEndEvent"},
        },
        "relationships": {},
    }

    def _run(self, mods):
        from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
        from schemas import BPMNModificationResponse

        handler = BPMNDiagramHandler(None)
        parsed = BPMNModificationResponse.model_validate({"modifications": mods, "message": "x"})
        with patch.object(handler, "predict_structured", return_value=parsed):
            return handler.generate_modification("insert a task", current_model=self._MODEL)

    def test_flows_to_a_node_added_in_the_same_request_survive(self):
        """b02: "insert a new task 'Notify Customer' between 'Issue Refund' and
        'Refund Issued'" - both new flows were dropped (only existing nodes
        were consulted) while the reply still claimed 5 changes."""
        spec = self._run([
            {"action": "remove_flow", "target": {},
             "changes": {"source": "Issue Refund", "target": "Refund Issued"}},
            {"action": "add_task", "target": {"nodeName": "Notify Customer"}},
            {"action": "add_flow", "target": {},
             "changes": {"source": "Issue Refund", "target": "Notify Customer"}},
            {"action": "add_flow", "target": {},
             "changes": {"source": "Notify Customer", "target": "Refund Issued"}},
        ])
        assert len(_ops(spec)) == 4

    def test_reply_is_rebuilt_after_a_drop(self):
        spec = self._run([
            {"action": "add_task", "target": {"nodeName": "Notify Customer"}},
            {"action": "add_flow", "target": {},
             "changes": {"source": "Notify Customer", "target": "Ghost Node"}},
        ])
        assert [m["action"] for m in _ops(spec)] == ["add_task"]
        assert "Applied 2 changes" not in spec["message"]
        assert "skipped 1" in spec["message"]


# ---------------------------------------------------------------------------
# 7. "Your screens are ready" is not announced while asking how to build them
# ---------------------------------------------------------------------------

class TestWebappPromptWaitsForTheScreens:
    def test_no_screens_ready_while_the_gui_choice_is_open(self):
        """c12: one reply said "How would you like me to create your screens?"
        and "Your screens are ready" together."""
        session = FakeSession()
        session.set(PENDING_WEBAPP_GENERATE, True)

        def _gui_op_asks(**kwargs):
            session.set(PENDING_GUI_CHOICE, {"operation_request": "screens"})
            return None

        with patch.object(confirmation, "execute_model_operation", side_effect=_gui_op_asks), \
             patch.object(confirmation, "emit_webapp_generate_prompt") as emit, \
             patch.object(confirmation, "reply_message"), \
             patch("utilities.request_builders.build_request_for_target", side_effect=lambda r, t: r):
            confirmation._resume_remaining_ops(
                session, [{"type": "model", "diagramType": "GUINoCodeDiagram"}],
                _req("create a web app"), "ClassDiagram", "complete_system",
                "create a web app", {},
            )
        emit.assert_not_called()
        assert session.get(PENDING_WEBAPP_GENERATE) is True

    def test_auto_answer_consumes_the_flag(self):
        session = FakeSession()
        session.set(PENDING_WEBAPP_GENERATE, True)
        session.set(PENDING_GUI_CHOICE, {"operation_request": "create a web app"})
        session.set(UNIFIED_CLASSIFICATION, UnifiedClassification(
            intent="modify_model_intent", reason="t",
            pending_flow_action="answer", pending_flow_answer="auto"))
        with patch.object(confirmation, "parse_assistant_request", return_value=_req("1")), \
             patch.object(confirmation, "reply_payload"), \
             patch.object(confirmation, "_build_auto_gui_message", return_value="built"):
            assert confirmation.handle_pending_gui_choice(session) is True
        assert not session.get(PENDING_WEBAPP_GENERATE)


# ---------------------------------------------------------------------------
# 8. "In a new diagram tab, ..." is honoured without re-asking
# ---------------------------------------------------------------------------

class TestNewTabDisposition:
    def _execute(self, monkeypatch, uc):
        import execution.model_operations as modeling

        handler = MagicMock()
        handler.generate_complete_system.return_value = {
            "action": "inject_complete_system", "message": "System created",
            "systemSpec": {"classes": []},
        }
        factory = MagicMock()
        factory.get_handler.return_value = handler
        sent = []
        monkeypatch.setattr(modeling.ctx, "diagram_factory", factory)
        monkeypatch.setattr(modeling, "reply_progress", lambda *a: None)
        monkeypatch.setattr(modeling, "reply_message", lambda *a: None)
        monkeypatch.setattr(modeling, "reply_payload", lambda s, p: sent.append(p))
        monkeypatch.setattr(modeling, "resolve_target_model", lambda *a: MINIMAL_CLASS_MODEL)
        monkeypatch.setattr(modeling, "build_workspace_context_block", lambda *a: "")
        session = FakeSession()
        session.set(UNIFIED_CLASSIFICATION, uc)
        request = AssistantRequest(
            message="In a new diagram tab, create a hospital management system",
            context=WorkspaceContext(active_diagram_type="ClassDiagram",
                                     project_snapshot={"id": "p", "diagrams": {}}),
        )
        modeling.execute_model_operation(
            session, request, {"diagramType": "ClassDiagram", "mode": "complete_system"},
            "complete_system",
        )
        return session, sent

    def test_new_tab_disposition_skips_the_question(self, monkeypatch):
        """c15: the classifier set disp=new_tab, the agent still asked
        replace / keep / new tab."""
        session, sent = self._execute(monkeypatch, UnifiedClassification(
            intent="create_complete_system_intent", reason="t",
            target_diagram_type="ClassDiagram", model_disposition="new_tab"))
        assert not session.get(PENDING_COMPLETE_SYSTEM)
        created = [p for p in sent if p.get("action") == "inject_complete_system"]
        assert created and created[0].get("createNewTab") is True

    def test_other_dispositions_still_ask(self, monkeypatch):
        session, _ = self._execute(monkeypatch, UnifiedClassification(
            intent="create_complete_system_intent", reason="t",
            target_diagram_type="ClassDiagram", model_disposition="new_from_scratch"))
        assert session.get(PENDING_COMPLETE_SYSTEM)


# ---------------------------------------------------------------------------
# 9. State machine transition schema stays stable and documented
# ---------------------------------------------------------------------------

class TestStateMachineTransitionSchema:
    def test_transition_actions_share_source_target_state_shape(self):
        from schemas.state_machine import StateMachineModificationResponse

        parsed = StateMachineModificationResponse.model_validate({"modifications": [
            {"action": "add_transition",
             "target": {"sourceState": "Delivered", "targetState": "Returned"},
             "changes": {"trigger": "return"}},
            {"action": "modify_transition",
             "target": {"sourceState": "Created", "targetState": "Paid"},
             "changes": {"trigger": "payOrder"}},
        ]})
        dumped = parsed.model_dump()["modifications"]
        assert dumped[0]["target"]["sourceState"] == "Delivered"
        assert dumped[1]["changes"]["trigger"] == "payOrder"

    def test_modify_prompt_explains_modify_transition_and_transition_removal(self):
        from diagram_handlers.types.state_machine_handler import MODIFY_SYSTEM_PROMPT_STATE_MACHINE as p

        assert "modify_transition: identify the existing transition by target.sourceState" in p
        assert "to remove a transition set target.sourceState and target.targetState" in p


# ---------------------------------------------------------------------------
# 10. Replies say what was applied, in words
# ---------------------------------------------------------------------------

class TestReplyCopy:
    def test_state_machine_batch_names_states_not_element(self):
        """s02 reply: "Updated Shipped. Added Returned. Added transition to
        element. Removed element." """
        from diagram_handlers.types.state_machine_handler import StateMachineHandler

        msg = StateMachineHandler(None)._friendly_batch_message([
            {"action": "modify_state", "target": {"stateName": "Shipped"}, "changes": {"name": "InTransit"}},
            {"action": "add_state", "target": {"stateName": "Returned"}, "changes": {}},
            {"action": "add_transition", "target": {"sourceState": "Delivered", "targetState": "Returned"},
             "changes": {"trigger": "return"}},
            {"action": "remove_element", "target": {"sourceState": "Paid", "targetState": "Created"}},
        ])
        assert "element" not in msg
        assert "Renamed **Shipped** to **InTransit**" in msg
        assert "Added transition **Delivered → Returned**" in msg
        assert "Removed **transition Paid → Created**" in msg

    def test_object_attribute_value_change_is_worded(self):
        """o02 reply: "Modify attribute value guest1." """
        from diagram_handlers.types.object_diagram_handler import ObjectDiagramHandler

        msg = ObjectDiagramHandler(None)._friendly_batch_message([
            {"action": "modify_attribute_value",
             "target": {"objectName": "guest1", "attributeName": "lastName"},
             "changes": {"value": "Johnson"}},
        ])
        assert msg == "Updated **lastName of guest1**."

    def test_nn_editor_gets_an_unsupported_editor_reply(self):
        """n02: in the Neural Network editor the assistant answered "That's a
        bit outside what I do"."""
        import state_bodies

        raw = {"context": {"activeDiagramType": "NNDiagram"}}
        session = FakeSession()
        with patch.object(state_bodies, "_common_preamble", return_value=_req("Create a CNN", raw_payload=raw)), \
             patch.object(state_bodies, "reply_message") as reply:
            state_bodies.out_of_scope_body(session)
        text = reply.call_args.args[1]
        assert "doesn't support the **Neural Network** editor yet" in text
        assert "outside what I do" not in text

    def test_out_of_scope_elsewhere_keeps_the_redirect(self):
        import state_bodies

        raw = {"context": {"activeDiagramType": "ClassDiagram"}}
        with patch.object(state_bodies, "_common_preamble", return_value=_req("write a poem", raw_payload=raw)), \
             patch.object(state_bodies, "reply_message") as reply:
            state_bodies.out_of_scope_body(FakeSession())
        assert "outside what I do" in reply.call_args.args[1]


class TestRemovalGuardChildScope:
    """Removing transitions or links names their endpoints; that is not a top-level removal."""

    def test_transition_and_link_endpoints_are_child_scope(self):
        from execution.model_operations import _is_top_level_removal

        assert not _is_top_level_removal({"sourceState": "Red", "targetState": "Green"})
        assert not _is_top_level_removal({"sourceObject": "lib1", "targetObject": "book1"})
        assert _is_top_level_removal({"stateName": "Red"})
