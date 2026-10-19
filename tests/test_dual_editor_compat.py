"""Outbound specs must work in both the production (v3) and the React Flow (v4) editor.

The agent ships before the v4 editor, so every spec it emits is applied by the
v3 modifiers (``features/assistant/services/modifiers`` on WME ``main``) first.
"""

from unittest.mock import patch

import pytest


def _sm_model():
    """The traffic light the v3 repro used: an unnamed initial node points at Red."""
    def state(sid, name):
        return {"id": sid, "type": "State", "name": name, "owner": None}

    def transition(rid, src, dst, name=""):
        return {"id": rid, "type": "StateTransition", "name": name,
                "source": {"element": src}, "target": {"element": dst}}

    return {
        "type": "StateMachineDiagram",
        "elements": {
            "init": {"id": "init", "type": "StateInitialNode", "name": "", "owner": None},
            "red": state("red", "Red"),
            "green": state("green", "Green"),
            "amber": state("amber", "Amber"),
        },
        "relationships": {
            "t-init": transition("t-init", "init", "red"),
            "t-rg": transition("t-rg", "red", "green", "chronometer_finished"),
            "t-ga": transition("t-ga", "green", "amber", "chronometer_finished"),
            "t-ar": transition("t-ar", "amber", "red", "chronometer_finished"),
        },
    }


def _sm_modify(mods, model=None):
    from diagram_handlers.types.state_machine_handler import StateMachineHandler
    from schemas.state_machine import StateMachineModificationResponse

    parsed = StateMachineModificationResponse.model_validate({"modifications": mods})
    handler = StateMachineHandler(None)
    with patch.object(StateMachineHandler, "predict_structured", return_value=parsed):
        spec = handler.generate_modification("remove it", current_model=model or _sm_model())
    return spec, spec.get("modifications") or [spec.get("modification")]


class TestStateMachineTransitionRemoval:
    """v3 ``remove_element`` reads only ``stateName``: an endpoint-only target
    matched the unnamed initial pseudostate and deleted it (r9/prod repro)."""

    def test_endpoint_removal_becomes_remove_transition_by_id(self):
        spec, mods = _sm_modify([
            {"action": "remove_element", "target": {"sourceState": "Red", "targetState": "Green"}},
        ])
        assert mods == [{
            "action": "remove_transition",
            "target": {"sourceState": "Red", "targetState": "Green", "transitionId": "t-rg"},
            "changes": None,
        }]
        assert spec["message"] == "Removed transition **Red → Green**."

    def test_no_state_removal_without_a_state_name_is_emitted(self):
        _, mods = _sm_modify([
            {"action": "add_state", "target": {"stateName": "Blink"}, "changes": {"stateType": "regular"}},
            {"action": "remove_element", "target": {"sourceState": "(initial)", "targetState": "Red"}},
            {"action": "remove_element", "target": {"sourceState": "Amber"}},
        ])
        assert [m["action"] for m in mods] == ["add_state", "remove_transition"]
        assert mods[1]["target"]["transitionId"] == "t-init"
        assert not any(m["action"] == "remove_element" and not m["target"].get("stateName") for m in mods)

    def test_state_removal_is_untouched(self):
        _, mods = _sm_modify([{"action": "remove_element", "target": {"stateName": "Amber"}}])
        assert mods == [{"action": "remove_element", "target": {
            "stateName": "Amber", "sourceState": None, "targetState": None}, "changes": None}]

    def test_unresolved_transition_never_falls_back_to_remove_element(self):
        _, mods = _sm_modify([
            {"action": "remove_element", "target": {"sourceState": "Red", "targetState": "Amber"}},
        ])
        assert mods == [{"action": "remove_transition",
                         "target": {"sourceState": "Red", "targetState": "Amber"}, "changes": None}]


class TestStateMachineTransitionAdd:
    """v3 ``add_transition`` reads ``changes.source`` / ``changes.target``; with
    only ``target.sourceState`` set it drew a self-loop on the initial node."""

    def test_endpoints_and_label_are_also_in_changes(self):
        spec, mods = _sm_modify([
            {"action": "add_transition", "target": {"sourceState": "Red", "targetState": "Amber"},
             "changes": {"trigger": "skip"}},
        ])
        assert mods[0]["target"] == {"sourceState": "Red", "targetState": "Amber", "stateName": None}
        changes = mods[0]["changes"]
        assert (changes["source"], changes["target"], changes["label"]) == ("Red", "Amber", "skip")
        assert changes["trigger"] == "skip"
        assert spec["message"] == "Added transition **Red → Amber**."

    def test_pseudostate_endpoint_is_empty_for_v3_and_a_keyword_for_v4(self):
        _, mods = _sm_modify([
            {"action": "add_transition", "target": {"sourceState": "(initial)", "targetState": "Green"},
             "changes": {}},
        ])
        assert mods[0]["target"]["sourceState"] == "initial"
        assert (mods[0]["changes"]["source"], mods[0]["changes"]["target"]) == ("", "Green")

    def test_star_endpoint_resolves_by_role_even_when_final_comes_first(self):
        model = _sm_model()
        model["elements"] = {"fin": {"id": "fin", "type": "StateFinalNode", "name": "", "owner": None},
                             **model["elements"]}
        _, mods = _sm_modify([
            {"action": "add_transition", "target": {"sourceState": "[*]", "targetState": "Green"}, "changes": {}},
            {"action": "add_transition", "target": {"sourceState": "Amber", "targetState": "[*]"}, "changes": {}},
        ], model)
        assert (mods[0]["target"]["sourceState"], mods[0]["changes"]["source"]) == ("initial", "")
        assert (mods[1]["target"]["targetState"], mods[1]["changes"]["target"]) == ("final", "")


def test_agent_posts_no_diagram_model_to_the_backend_validator():
    """``validate_diagram`` had no callers and posted the agent's internal v3
    model to ``/besser_api/validate-diagram``; the v4 backend reads only
    ``{nodes, edges}``."""
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src"
    offenders = [
        str(path.relative_to(src)) for path in src.rglob("*.py")
        if "validate-diagram" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_docs_list_the_outbound_spec_changes():
    """websocket_protocol.rst and CLAUDE.md said outbound specs were unchanged."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    protocol = (root / "docs" / "source" / "websocket_protocol.rst").read_text(encoding="utf-8")
    guide = (root / "CLAUDE.md").read_text(encoding="utf-8")
    assert "so they are unchanged" not in protocol
    assert "The action set is unchanged" not in protocol  # state machines now emit remove_transition
    assert "Outbound specs are unchanged" not in guide
    for token in ("remove_transition", "transitionId", "governanceDsl", "changes.source"):
        assert token in protocol


# ---------------------------------------------------------------------------
# BPMN governance: the v3 editor stores governanceDsl verbatim and BESSER parses
# it (swarm_bindings -> summarize_governance), so it must be a valid document.
# ---------------------------------------------------------------------------

_POLICIES = ("MajorityPolicy", "AbsoluteMajorityPolicy", "LeaderDrivenPolicy", "ConsensusPolicy")


def _parse_governance(text):
    """BESSER's govdsl_runtime._parse_policies, including its comment stripping."""
    import io

    pytest.importorskip("governancedsl")
    from antlr4 import CommonTokenStream, InputStream, ParseTreeWalker
    from governancedsl.grammar.govdslLexer import govdslLexer
    from governancedsl.grammar.govdslParser import govdslParser
    from governancedsl.grammar.PolicyCreationListener import PolicyCreationListener
    from governancedsl.grammar.govErrorListener import govErrorListener

    text = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("//"))
    lexer = govdslLexer(InputStream(text))
    lexer.removeErrorListeners()
    lexer_errors = govErrorListener(io.StringIO())
    lexer.addErrorListener(lexer_errors)
    parser = govdslParser(CommonTokenStream(lexer))
    parser.removeErrorListeners()
    parser_errors = govErrorListener(io.StringIO())
    parser.addErrorListener(parser_errors)
    tree = parser.governance()
    assert not (lexer_errors.error_message or parser_errors.error_message), \
        lexer_errors.error_message or parser_errors.error_message
    listener = PolicyCreationListener()
    ParseTreeWalker().walk(listener, tree)
    return listener.get_policies()


def _loan_model():
    """The retest's v3 loan-approval process: an agentic block in the RiskAgent lane."""
    def el(eid, kind, name="", owner=None, **extra):
        return {"id": eid, "type": kind, "name": name, "owner": owner, **extra}

    def flow(src, dst):
        return {"type": "BPMNFlow", "flowType": "sequence", "source": {"element": src}, "target": {"element": dst}}

    elements = [
        el("pool", "BPMNPool", "LoanProcess"),
        el("risk", "BPMNSwimlane", "RiskAgent", "pool", isAgentic=True, role="solution", trustScore=0),
        el("officer", "BPMNSwimlane", "Officer", "pool", isAgentic=False, role="solution", trustScore=0),
        el("start", "BPMNStartEvent", "Submitted"),
        el("credit", "BPMNTask", "Check Credit", "risk", isAgentic=True),
        el("income", "BPMNTask", "Verify Income", "risk", isAgentic=True),
        el("approve", "BPMNTask", "Approve Loan", "officer"),
        el("bpmn_v20xlni9r", "BPMNGateway", "", "risk", isAgentic=True, gatewayRole="diverging",
           gatewayType="parallel", trustScore=0),
        el("bpmn_merge_gw1", "BPMNGateway", "", None, isAgentic=True, gatewayRole="merging",
           gatewayType="parallel", trustScore=0),
    ]
    flows = [("start", "bpmn_v20xlni9r"), ("bpmn_v20xlni9r", "credit"), ("bpmn_v20xlni9r", "income"),
             ("credit", "bpmn_merge_gw1"), ("income", "bpmn_merge_gw1"), ("bpmn_merge_gw1", "approve")]
    return {"type": "BPMN", "elements": {e["id"]: e for e in elements},
            "relationships": {f"f{i}": flow(*f) for i, f in enumerate(flows)}}


def _bpmn_modify(mods, model):
    from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
    from schemas import BPMNModificationResponse

    parsed = BPMNModificationResponse.model_validate({"modifications": mods, "message": "ok"})
    with patch.object(BPMNDiagramHandler, "predict_structured", return_value=parsed):
        spec = BPMNDiagramHandler(None).generate_modification("add a governance policy", current_model=model)
    return spec.get("modifications") or [spec.get("modification")]


class TestGovernanceDslIsAValidDocument:
    def test_modify_node_sends_the_document_the_v4_editor_generates(self):
        # Byte-for-byte what the v4 editor's resolveGovernanceDsl wrote for the
        # retest case "Add a majority governance policy to the merging gateway".
        expected = "\n".join([
            '// Generated from agentic merging gateway ""',
            "// policyType=MajorityPolicy, trustScore=0",
            "Scopes:", "    Tasks:", "        MergeDecision_bpmn_mer",
            "Participants:", "    Individuals :", "        (Agent) RiskAgent { confidence : 0.00 }",
            "MajorityPolicy MergeDecision_bpmn_merPolicy {", "    Scope: MergeDecision_bpmn_mer",
            "    DecisionType as BooleanDecision", "    Participant list : RiskAgent",
            "    Parameters:", "        ratio : 0.5", "}",
        ])
        [mod] = _bpmn_modify([{"action": "modify_node", "target": {"nodeId": "bpmn_merge_gw1"},
                               "changes": {"governancePolicy": "MajorityPolicy"}}], _loan_model())
        assert mod["changes"]["governancePolicy"] == "MajorityPolicy"
        assert mod["changes"]["governanceDsl"] == expected

    def test_every_policy_parses_on_modify_and_generate(self):
        import copy
        import json
        from pathlib import Path

        from diagram_handlers.types.bpmn_diagram_handler import BPMNDiagramHandler
        from schemas import SystemBPMNSpec

        fixture = json.loads((Path(__file__).parent / "fixtures" / "contract" / "agentic_bpmn_spec.json")
                             .read_text(encoding="utf-8"))
        for policy in _POLICIES:
            [mod] = _bpmn_modify([{"action": "modify_node", "target": {"nodeId": "bpmn_merge_gw1"},
                                   "changes": {"governancePolicy": policy}}], _loan_model())
            [parsed] = _parse_governance(mod["changes"]["governanceDsl"])
            assert type(parsed).__name__ == policy

            raw = copy.deepcopy(fixture)
            next(n for n in raw["nodes"] if n["id"] == "merge")["governancePolicy"] = policy
            handler = BPMNDiagramHandler(None)
            spec = handler._validate_and_refine(SystemBPMNSpec.model_validate(raw).model_dump())
            handler._sanitize_agentic_refs(spec, {})
            merge = next(n for n in spec["nodes"] if n["id"] == "merge")
            [parsed] = _parse_governance(merge["governanceDsl"])
            assert type(parsed).__name__ == policy
            expected = {"Supervisor_Agent"} if policy == "LeaderDrivenPolicy" \
                else {"Writer_Agent", "Supervisor_Agent", "Reviewer_Agents"}
            assert {p.name for p in parsed.participants} == expected

    def test_non_agentic_gateway_gets_no_document(self):
        model = _loan_model()
        model["elements"]["bpmn_merge_gw1"]["isAgentic"] = False
        [mod] = _bpmn_modify([{"action": "modify_node", "target": {"nodeId": "bpmn_merge_gw1"},
                               "changes": {"governancePolicy": "ConsensusPolicy"}}], model)
        assert "governanceDsl" not in mod["changes"]
