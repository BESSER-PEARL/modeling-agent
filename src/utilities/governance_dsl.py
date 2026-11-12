"""Governance DSL text for an agentic merging gateway.

Port of the editor's ``generateGovernanceDsl`` (WME
``packages/library/lib/services/bpmnCollaborationBlocks.ts``), which the v4
editor runs when it expands a policy keyword. The production (v3) editor stores
``governanceDsl`` verbatim and BESSER parses it, so the agent sends the same
document. The walk reads a v4-like graph: ``nodes`` with ``id`` / ``type`` /
``parentId`` / ``data`` and ``edges`` with ``source`` / ``target`` / ``type``.
"""

import math
import re
from typing import Any, Dict, List, Optional

GOV_POLICY_TYPES = ("MajorityPolicy", "AbsoluteMajorityPolicy", "LeaderDrivenPolicy", "ConsensusPolicy")

_MAX_COLLAB_WALK_DEPTH = 50
_SEQUENCE_FLOW = "BPMNSequenceFlow"


def _data(node: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return (node or {}).get("data") or {}


def _is_agentic_gateway(node: Optional[Dict[str, Any]], role: str) -> bool:
    return (node or {}).get("type") == "bpmnGateway" and _data(node).get("isAgentic") is True \
        and _data(node).get("gatewayRole") == role


def _sequence_flows(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [e for e in graph.get("edges") or [] if e.get("type") == _SEQUENCE_FLOW]


def _upstream_diverging_id(element_id: str, flows: List[Dict[str, Any]], nodes: Dict[str, Dict[str, Any]],
                           visited: set) -> Optional[str]:
    frontier = [element_id]
    depth = 0
    while depth < _MAX_COLLAB_WALK_DEPTH and frontier:
        nxt: List[str] = []
        for node_id in frontier:
            if node_id in visited:
                continue
            visited.add(node_id)
            for flow in flows:
                if flow.get("target") != node_id:
                    continue
                src_id = flow.get("source")
                if src_id in visited:
                    continue
                src = nodes.get(src_id)
                if _is_agentic_gateway(src, "diverging"):
                    return src_id
                if _is_agentic_gateway(src, "merging"):
                    inner = _upstream_diverging_id(src_id, flows, nodes, visited)
                    if inner is not None and inner not in visited:
                        nxt.append(inner)
                        continue
                nxt.append(src_id)
        frontier = nxt
        depth += 1
    return None


def _sanitize_id(raw: Optional[str], fallback: str) -> str:
    base = (raw or "").strip()
    if not base:
        return fallback
    # JS replaces per UTF-16 code unit: an astral character becomes two '_'.
    s = "".join(
        ch if re.match(r"[a-zA-Z0-9_/-]", ch) else "_" * (len(ch.encode("utf-16-le")) // 2)
        for ch in base
    )
    return s if re.match(r"[a-zA-Z_]", s) else "_" + s


def _js_number(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _number(value: Any) -> Optional[float]:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def generate_governance_dsl(merging_gateway_id: str, graph: Dict[str, Any], policy_type: str) -> str:
    """The starter Governance DSL the editor writes for *policy_type* on this gateway."""
    nodes = {n.get("id"): n for n in graph.get("nodes") or []}
    flows = _sequence_flows(graph)
    gw = nodes.get(merging_gateway_id)
    gw_data = _data(gw)
    trust = _number(gw_data.get("trustScore"))
    trust = 0 if trust is None else trust
    ratio = 0.5 if policy_type in ("MajorityPolicy", "AbsoluteMajorityPolicy") else None

    scope_id = _sanitize_id(gw_data.get("name"), f"MergeDecision_{merging_gateway_id[:8]}")
    policy_id = f"{scope_id}Policy"

    lanes: Dict[str, Dict[str, Any]] = {}

    def add_lane_of(node: Optional[Dict[str, Any]]) -> None:
        lane = nodes.get(node.get("parentId")) if node and node.get("parentId") else None
        if lane and _data(lane).get("isAgentic"):
            lanes.setdefault(lane["id"], lane)

    diverging_id = _upstream_diverging_id(merging_gateway_id, flows, nodes, set())
    diverging = nodes.get(diverging_id) if diverging_id is not None else None
    if diverging:
        add_lane_of(diverging)
        add_lane_of(gw)
        visited = {diverging["id"]}
        queue = [diverging["id"]]
        while queue:
            current = queue.pop(0)
            for flow in flows:
                if flow.get("source") != current:
                    continue
                tgt_id = flow.get("target")
                if tgt_id in visited:
                    continue
                visited.add(tgt_id)
                tgt = nodes.get(tgt_id)
                if not tgt:
                    continue
                if _is_agentic_gateway(tgt, "merging"):
                    continue
                add_lane_of(tgt)
                queue.append(tgt_id)

    all_lanes = list(lanes.values())
    supervision = [lane for lane in all_lanes if _data(lane).get("role") in ("supervisor", "supervision")]
    participant_lanes = supervision if policy_type == "LeaderDrivenPolicy" and supervision else all_lanes

    agents: List[Dict[str, str]] = []
    used: set = set()
    for lane in participant_lanes:
        base = _sanitize_id(_data(lane).get("name"), f"Agent_{lane['id'][:8]}")
        agent_id, i = base, 2
        while agent_id in used:
            agent_id, i = f"{base}_{i}", i + 1
        used.add(agent_id)
        lane_trust = _number(_data(lane).get("trustScore")) or 0
        agents.append({"id": agent_id, "confidence": f"{lane_trust / 100:.2f}"})
    if not agents:
        agents.append({"id": "Participant", "confidence": "0.00"})

    title = gw_data.get("name")
    title = re.sub(r"[\r\n]+", " ", str(merging_gateway_id if title is None else title))
    lines = [
        f'// Generated from agentic merging gateway "{title}"',
        f"// policyType={policy_type}, trustScore={_js_number(trust)}",
        "Scopes:",
        "    Tasks:",
        f"        {scope_id}",
        "Participants:",
    ]
    if not participant_lanes:
        lines.append("    // No agentic lanes were found in the block: replace the placeholder agent.")
    lines.append("    Individuals :")
    for i, agent in enumerate(agents):
        comma = "," if i < len(agents) - 1 else ""
        lines.append(f"        (Agent) {agent['id']} {{ confidence : {agent['confidence']} }}{comma}")
    lines += [
        f"{policy_type} {policy_id} {{",
        f"    Scope: {scope_id}",
        "    DecisionType as BooleanDecision",
        f"    Participant list : {', '.join(a['id'] for a in agents)}",
    ]
    if ratio is not None:
        lines += ["    Parameters:", f"        ratio : {_js_number(ratio)}"]
    lines.append("}")
    return "\n".join(lines)


def clamp_trust_score(value: Any) -> int:
    """The editor's ``clampTrustScore``: a rounded 0-100 percentage (0 when unset)."""
    number = _number(value)
    if number is None:
        return 0
    return int(min(100, max(0, math.floor(number + 0.5))))  # JS Math.round


def graph_from_v3_model(model: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """A v3 BPMN model (``elements`` / ``relationships``) as the graph the walk reads."""
    model = model if isinstance(model, dict) else {}
    nodes = []
    for element_id, el in (model.get("elements") or {}).items():
        if not isinstance(el, dict):
            continue
        el_type = str(el.get("type") or "")
        nodes.append({
            "id": element_id,
            "type": "bpmn" + el_type[4:] if el_type.startswith("BPMN") else el_type,
            "parentId": el.get("owner"),
            "data": el,
        })
    edges = [
        {"source": (rel.get("source") or {}).get("element"), "target": (rel.get("target") or {}).get("element"),
         "type": _SEQUENCE_FLOW if rel.get("flowType", "sequence") == "sequence" else rel.get("flowType")}
        for rel in (model.get("relationships") or {}).values()
        if isinstance(rel, dict) and rel.get("type") == "BPMNFlow"
    ]
    return {"nodes": nodes, "edges": edges}


def graph_from_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """A generated BPMN spec as the graph the v4 editor builds from it.

    Agentic fields follow the editor's builders (only ``isAgentic: true`` turns
    them on; a gateway's role defaults to diverging); a flow between two pools is
    a message flow.
    """
    nodes: List[Dict[str, Any]] = []
    pool_of: Dict[str, Any] = {}
    for pool in spec.get("pools") or []:
        for lane in pool.get("lanes") or []:
            agentic = lane.get("isAgentic") is True
            nodes.append({"id": lane.get("id"), "type": "bpmnSwimlane", "parentId": None, "data": {
                "name": lane.get("name") or "", "isAgentic": agentic,
                **({"role": lane.get("role"), "trustScore": clamp_trust_score(lane.get("trustScore"))}
                   if agentic else {}),
            }})
            pool_of[lane.get("id")] = pool.get("id")
    for node in spec.get("nodes") or []:
        kind = str(node.get("type") or "")
        data: Dict[str, Any] = {"name": node.get("name") or ""}
        if node.get("isAgentic") is True:
            data["isAgentic"] = True
            data["trustScore"] = clamp_trust_score(node.get("trustScore"))
            if kind == "gateway":
                data["gatewayRole"] = node.get("gatewayRole") if isinstance(node.get("gatewayRole"), str) else "diverging"
        lane_id = node.get("laneId")
        nodes.append({"id": node.get("id"), "type": "bpmn" + kind[:1].upper() + kind[1:],
                      "parentId": lane_id if lane_id in pool_of else None, "data": data})
        pool_of[node.get("id")] = node.get("poolId") or pool_of.get(lane_id)
    edges = []
    for flow in spec.get("flows") or []:
        s_pool, t_pool = pool_of.get(flow.get("source")), pool_of.get(flow.get("target"))
        cross = s_pool is not None and t_pool is not None and s_pool != t_pool
        edges.append({"source": flow.get("source"), "target": flow.get("target"),
                      "type": "BPMNMessageFlow" if cross else _SEQUENCE_FLOW})
    return {"nodes": nodes, "edges": edges}
