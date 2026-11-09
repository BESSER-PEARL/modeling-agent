"""
BPMN Diagram Handler
Handles generation and modification of BPMN process diagrams, including
multi-participant collaboration diagrams and Agentic BPMN.

Emits a process (start/end events, tasks, exclusive/parallel/inclusive
gateways, flows, data objects) optionally grouped into pools (participants) and lanes
(roles within a pool). Agentic mode — selected only on explicit agentic intent
or an existing agentic lane (``_is_agentic_bpmn_request``) — adds the editor's
Agentic BPMN attributes (agentic lanes with a role, trust score and swarm
multiplicity; reflection modes; governed merging gateways) on top of the base
rules, and runs the same deterministic repair pass. Positions are NOT
generated here: the WME injector lays the process (and any pools/lanes) out
and the editor's layouter routes the flows. Message vs. sequence flow type is
also derived on the WME side from pool membership — the agent never sets a
flow type.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ..core.base_handler import BaseDiagramHandler
from prompts import bpmn as bpmn_prompts
from schemas import SystemBPMNSpec, BPMNModificationResponse
from utilities.governance_dsl import (
    GOV_POLICY_TYPES, clamp_trust_score, generate_governance_dsl, graph_from_spec, graph_from_v3_model,
)
from utilities.model_context import detailed_model_summary
from protocol.actions import assistant_message, inject_complete_system

logger = logging.getLogger(__name__)


# Explicit agentic intent. Word-boundary matched: substrings such as "pool"
# (carpool), "participant" or "orchestrat" describe ordinary collaboration
# processes, not agent swarms.
_AGENTIC_INTENT_RE = re.compile(
    r"\b(?:agentic"
    r"|multi[- ]?agents?"
    r"|agent[- ]?swarms?|swarms?\s+of\s+(?:ai\s+)?agents"
    r"|(?:ai[- ])?agent\s+lanes?"
    r"|(?:solution|supervision)\s+(?:agents?|lanes?|roles?)"
    r"|lane\s+roles?"
    r"|trust\s+scores?"
    r"|reflection\s+modes?"
    r"|governance\s+dsl)\b",
    re.IGNORECASE,
)


def format_agent_diagram_refs(agent_diagram_refs: Optional[Dict[str, str]]) -> str:
    """The per-request list of AgentDiagram ids the LLM may use as agentDiagramRef."""
    if not agent_diagram_refs:
        return "Agent diagrams in this project: none — leave every agentDiagramRef null."
    lines = [f"- {diagram_id}: {title or 'untitled'}" for diagram_id, title in agent_diagram_refs.items()]
    return "Agent diagrams in this project (the only valid agentDiagramRef values):\n" + "\n".join(lines)


# The production (v3) editor stores only ``governanceDsl`` and BESSER parses it,
# so a requested governancePolicy also goes out as the DSL document the v4
# editor generates from it. Like the v4 editor, only an agentic gateway gets one.

def _spec_governance_dsl(spec: Dict[str, Any]) -> None:
    """Write ``governanceDsl`` on generated agentic gateways that name a policy (mutates *spec*)."""
    graph = None
    for node in spec.get("nodes") or []:
        policy = node.get("governancePolicy")
        if policy in GOV_POLICY_TYPES and node.get("type") == "gateway" and node.get("isAgentic") is True:
            graph = graph or graph_from_spec(spec)
            node["governanceDsl"] = generate_governance_dsl(str(node.get("id")), graph, policy)


def _find_gateway(elements: Dict[str, Any], target: Dict[str, Any]) -> Optional[str]:
    node_id = target.get("nodeId")
    if node_id in elements and (elements[node_id] or {}).get("type") == "BPMNGateway":
        return node_id
    name = target.get("nodeName")
    if not isinstance(name, str) or not name.strip():
        return None
    gateways = [(eid, el) for eid, el in elements.items() if isinstance(el, dict) and el.get("type") == "BPMNGateway"]
    return next((eid for eid, el in gateways if el.get("name") == name), None) or next(
        (eid for eid, el in gateways if (el.get("name") or "").strip().lower() == name.strip().lower()), None)


def _modification_governance_dsl(mod_list: List[Dict[str, Any]], current_model: Optional[Dict[str, Any]]) -> None:
    """Write ``governanceDsl`` next to a gateway's requested governancePolicy (mutates the changes).

    The gateway is read as the editor sees it after this step: its name, agentic
    switch, role and trust score from the same changes.
    """
    model = current_model if isinstance(current_model, dict) else {}
    elements = model.get("elements") if isinstance(model.get("elements"), dict) else {}
    for mod in mod_list:
        changes = mod.get("changes") if isinstance(mod, dict) else None
        if not isinstance(changes, dict) or changes.get("governancePolicy") not in GOV_POLICY_TYPES:
            continue
        target = mod.get("target") if isinstance(mod.get("target"), dict) else {}
        if mod.get("action") == "modify_node":
            gateway_id = _find_gateway(elements, target)
            if gateway_id is None:
                continue
            current = elements[gateway_id]
            was_agentic = current.get("isAgentic") is True
            if not (changes["isAgentic"] if isinstance(changes.get("isAgentic"), bool) else was_agentic):
                continue
            graph = graph_from_v3_model(model)
            data = dict(current, isAgentic=True)
            if not was_agentic:
                data.update(gatewayRole="diverging", trustScore=0)
            if changes.get("name"):
                data["name"] = changes["name"]
            if isinstance(changes.get("gatewayRole"), str):
                data["gatewayRole"] = changes["gatewayRole"]
            if isinstance(changes.get("trustScore"), (int, float)) and not isinstance(changes["trustScore"], bool):
                data["trustScore"] = clamp_trust_score(changes["trustScore"])
            for node in graph["nodes"]:
                if node["id"] == gateway_id:
                    node["data"] = data
        elif mod.get("action") == "add_gateway" and changes.get("isAgentic") is True:
            gateway_id = target.get("nodeId") or "gateway"
            graph = {"nodes": [{"id": gateway_id, "type": "bpmnGateway", "parentId": None, "data": {
                "name": target.get("nodeName") or changes.get("name") or "", "isAgentic": True,
                "gatewayRole": changes.get("gatewayRole") or "diverging",
                "trustScore": clamp_trust_score(changes.get("trustScore")),
            }}], "edges": []}
        else:
            continue
        changes["governanceDsl"] = generate_governance_dsl(gateway_id, graph, changes["governancePolicy"])


class BPMNDiagramHandler(BaseDiagramHandler):
    """Handler for base BPMN process generation and modification."""

    _SYSTEM_FAILURE_MESSAGE = "I couldn't generate that process. Please try again or rephrase your request."
    _LLM_FAILURE_CODE = "llm_failure"

    def get_diagram_type(self) -> str:
        # The WME storage-bucket token (NOT the Apollon model.type
        # "BPMNDiagram"); the WME converter sets model.type itself.
        return "BPMN"

    def get_system_prompt(self) -> str:
        return bpmn_prompts.SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # Complete system (the primary generation path)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_agentic_bpmn_request(user_request: str, current_model: Dict[str, Any] = None) -> bool:
        """True only on explicit agentic intent or an existing agentic lane.

        Pools, participants or orchestration alone describe an ordinary
        collaboration process (e.g. the WME Pizza Store / Car Wash templates).
        """
        if _AGENTIC_INTENT_RE.search(user_request or ""):
            return True
        elements = current_model.get("elements") if isinstance(current_model, dict) else None
        if isinstance(elements, dict):
            return any(
                isinstance(el, dict) and el.get("type") == "BPMNSwimlane" and el.get("isAgentic") is True
                for el in elements.values()
            )
        return False

    def _prepare_system(
        self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs,
    ) -> Tuple[bool, str, str, str]:
        """``(agentic, system_prompt, request_prompt, reasoning_prompt)``."""
        # Detect on the user's own words: the enriched request also carries
        # conversation history and workspace context.
        if self._is_agentic_bpmn_request(kwargs.get("raw_request") or user_request, existing_model):
            # The base rules plus the agentic ones; same repair pass after.
            system_prompt = f"{self.get_system_prompt()}\n\n{bpmn_prompts.AGENTIC_DESIGN_RULES}"
            refs_block = format_agent_diagram_refs(kwargs.get("agent_diagram_refs") or {})
            reasoning_prompt = bpmn_prompts.agentic_reasoning_prompt(user_request)
            return True, system_prompt, f"{user_request}\n\n{refs_block}", reasoning_prompt
        return False, self.get_system_prompt(), user_request, bpmn_prompts.reasoning_prompt(user_request)

    def _build_system(
        self, user_request: str, existing_model: Dict[str, Any] = None, *, prepared=None, **kwargs,
    ) -> Dict[str, Any]:
        agentic, system_prompt, request_prompt, reasoning_prompt = prepared
        parsed = self.predict_two_pass_structured(
            user_request=request_prompt,
            system_prompt=system_prompt,
            reasoning_prompt=reasoning_prompt,
            response_schema=SystemBPMNSpec,
        )
        system_spec = parsed.model_dump()
        system_spec = self._validate_and_refine(system_spec)
        self._sanitize_agentic_refs(system_spec, kwargs.get("agent_diagram_refs") or {})

        return inject_complete_system(
            system_spec,
            self.get_diagram_type(),
            self._build_agentic_message(system_spec) if agentic else self._build_system_message(system_spec),
        )

    def _system_failure_reply(self, exc, user_request, existing_model, *, prepared=None, **kwargs) -> Dict[str, Any]:
        if prepared[0]:
            return self._error_response(
                "I couldn't generate that agentic process. Please try again or rephrase your request.",
                code=self._LLM_FAILURE_CODE,
            )
        return super()._system_failure_reply(exc, user_request, existing_model, prepared=prepared, **kwargs)

    @staticmethod
    def _sanitize_agentic_refs(spec: Dict[str, Any], agent_diagram_refs: Dict[str, str]) -> None:
        """Drop references the editor could not resolve (mutates *spec*).

        - agentDiagramRef is a project AgentDiagram UUID in the editor: keep it
          only when it is one of this project's AgentDiagram ids.
        - reflectionReviewerLaneId must name a declared lane, and only on a
          task whose reflectionMode is 'cross'.
        - An agentic gateway's governancePolicy also gets its governanceDsl.
        """
        valid_refs = set(agent_diagram_refs or {})
        lane_ids = set()
        for pool in spec.get("pools") or []:
            for lane in pool.get("lanes") or []:
                lane_ids.add(lane.get("id"))
                if lane.get("agentDiagramRef") and lane["agentDiagramRef"] not in valid_refs:
                    logger.info(f"[BPMN] Dropped invented lane agentDiagramRef {lane['agentDiagramRef']!r}")
                    lane["agentDiagramRef"] = None
        for node in spec.get("nodes") or []:
            if node.get("agentDiagramRef") and node["agentDiagramRef"] not in valid_refs:
                logger.info(f"[BPMN] Dropped invented node agentDiagramRef {node['agentDiagramRef']!r}")
                node["agentDiagramRef"] = None
            reviewer = node.get("reflectionReviewerLaneId")
            if reviewer and (node.get("reflectionMode") != "cross" or reviewer not in lane_ids):
                node["reflectionReviewerLaneId"] = None
        _spec_governance_dsl(spec)

    def _build_agentic_message(self, spec: Dict[str, Any]) -> str:
        name = spec.get("systemName") or "process"
        pools = spec.get("pools", [])
        nodes = spec.get("nodes", [])
        total_lanes = sum(len(p.get("lanes", [])) for p in pools)
        tasks = [n.get("name", "?") for n in nodes if n.get("type") == "task"][:5]
        msg = f"Built the **{name}** agentic process with {len(pools)} pool(s) and {total_lanes} lane(s)"
        if tasks:
            msg += f": {', '.join(f'**{t}**' for t in tasks)}"
        msg += ". Ask me to add agents, modify roles, or adjust the flow!"
        return msg

    # ------------------------------------------------------------------
    # Validation / light repair (no LLM round-trip)
    # ------------------------------------------------------------------

    def _validate_and_refine(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure a start event, an end event, valid flow refs, basic connectivity."""
        nodes: List[Dict[str, Any]] = spec.get("nodes", []) or []
        flows: List[Dict[str, Any]] = spec.get("flows", []) or []
        if not nodes:
            return spec

        ids = {n.get("id") for n in nodes if n.get("id")}
        flows = [
            f for f in flows
            if f.get("source") in ids and f.get("target") in ids and f.get("source") != f.get("target")
        ]

        has_start = any(n.get("type") == "startEvent" for n in nodes)
        has_end = any(n.get("type") == "endEvent" for n in nodes)
        sources = {f.get("source") for f in flows}
        targets = {f.get("target") for f in flows}

        if not has_start:
            start_id = self._unique_id("start", ids)
            nodes.insert(0, {"id": start_id, "name": "Start", "type": "startEvent"})
            ids.add(start_id)
            first = next(
                (n.get("id") for n in nodes
                 if n.get("type") not in ("startEvent", "endEvent") and n.get("id") not in targets),
                None,
            )
            if first:
                flows.insert(0, {"source": start_id, "target": first, "name": ""})
            logger.info("[BPMN] Validation: added missing start event")

        if not has_end:
            end_id = self._unique_id("end", ids)
            nodes.append({"id": end_id, "name": "End", "type": "endEvent"})
            ids.add(end_id)
            last = next(
                (n.get("id") for n in reversed(nodes)
                 if n.get("type") not in ("startEvent", "endEvent") and n.get("id") not in sources),
                None,
            )
            if last:
                flows.append({"source": last, "target": end_id, "name": ""})
            logger.info("[BPMN] Validation: added missing end event")

        self._connect_orphaned_nodes(nodes, flows)
        self._normalize_pool_refs(spec, nodes)
        self._place_nodes_in_pools(spec, nodes, flows)
        self._keep_sequence_flows_in_one_pool(spec, nodes, flows)
        self._normalize_data_objects(spec, ids)

        spec["nodes"] = nodes
        spec["flows"] = flows
        return spec

    @classmethod
    def _normalize_data_objects(cls, spec: Dict[str, Any], node_ids: set) -> None:
        """Keep data objects and their associations consistent (mutates *spec*).

        A data object whose id collides with a node or another data object gets
        a unique id; associations to an unknown data object or node are dropped,
        as are duplicates.
        """
        taken = set(node_ids)
        data_ids = set()
        data_objects = []
        renamed: Dict[str, str] = {}
        for obj in spec.get("dataObjects") or []:
            obj_id = (obj.get("id") or "").strip()
            if not obj_id:
                continue
            unique_id = cls._unique_id(obj_id, taken)
            if unique_id != obj_id:
                renamed.setdefault(obj_id, unique_id)
            taken.add(unique_id)
            data_ids.add(unique_id)
            data_objects.append({**obj, "id": unique_id})

        associations = []
        seen = set()
        for assoc in spec.get("dataAssociations") or []:
            data_id = renamed.get(assoc.get("dataObject"), assoc.get("dataObject"))
            key = (data_id, assoc.get("node"), assoc.get("direction"))
            if data_id not in data_ids or assoc.get("node") not in node_ids or key in seen:
                continue
            seen.add(key)
            associations.append({**assoc, "dataObject": data_id})

        spec["dataObjects"] = data_objects
        spec["dataAssociations"] = associations

    @staticmethod
    def _normalize_pool_refs(spec: Dict[str, Any], nodes: List[Dict[str, Any]]) -> None:
        """Drop dangling poolId/laneId references so a malformed pools[] entry
        (or a typo'd id) never breaks the WME converter's pool/lane layout.
        Mutates `nodes` in place; also drops lanes with duplicate/empty ids."""
        pools: List[Dict[str, Any]] = spec.get("pools") or []
        if not pools:
            for node in nodes:
                node["poolId"] = None
                node["laneId"] = None
                node["owner"] = None
            spec["pools"] = []
            return

        valid_pool_ids = set()
        lane_ids_by_pool: Dict[str, set] = {}
        cleaned_pools: List[Dict[str, Any]] = []
        for pool in pools:
            pool_id = pool.get("id")
            if not pool_id or pool_id in valid_pool_ids:
                continue
            valid_pool_ids.add(pool_id)
            lanes = pool.get("lanes") or []
            seen_lane_ids: set = set()
            cleaned_lanes = []
            for lane in lanes:
                lane_id = lane.get("id")
                if not lane_id or lane_id in seen_lane_ids:
                    continue
                seen_lane_ids.add(lane_id)
                cleaned_lanes.append(lane)
            lane_ids_by_pool[pool_id] = seen_lane_ids
            cleaned_pools.append({**pool, "lanes": cleaned_lanes})
        spec["pools"] = cleaned_pools

        for node in nodes:
            pool_id = node.get("poolId")
            if pool_id not in valid_pool_ids:
                node["poolId"] = None
                node["laneId"] = None
                node["owner"] = None
                continue
            lane_id = node.get("laneId")
            if lane_id and lane_id not in lane_ids_by_pool.get(pool_id, set()):
                node["laneId"] = None
                node["owner"] = None
                continue
            node["owner"] = lane_id if lane_id else None

    @staticmethod
    def _place_nodes_in_pools(
        spec: Dict[str, Any], nodes: List[Dict[str, Any]], flows: List[Dict[str, Any]]
    ) -> None:
        """Put every node of a pooled process in a pool, and in a lane when that pool has lanes.

        A node left outside the pools lands in another BPMN process than its
        neighbours, so each sequence flow to it crosses a process boundary.
        An unplaced node takes the pool/lane of a placed neighbour (predecessors
        first), else its pool's (or the first pool's) first lane. Runs after
        ``_normalize_pool_refs``, so every remaining ref is valid.
        """
        pools: List[Dict[str, Any]] = spec.get("pools") or []
        if not pools:
            return
        lanes_of = {pool["id"]: [lane["id"] for lane in pool.get("lanes") or []] for pool in pools}
        by_id = {node["id"]: node for node in nodes if node.get("id")}
        neighbours: Dict[str, List[str]] = {}
        for flow in flows:
            neighbours.setdefault(flow.get("target"), []).append(flow.get("source"))
        for flow in flows:
            neighbours.setdefault(flow.get("source"), []).append(flow.get("target"))

        def placed(node: Dict[str, Any]) -> bool:
            lanes = lanes_of.get(node.get("poolId"))
            return lanes is not None and (not lanes or node.get("laneId") in lanes)

        changed = True
        while changed:
            changed = False
            for node in nodes:
                if placed(node):
                    continue
                for neighbour in (by_id.get(nid) for nid in neighbours.get(node.get("id"), [])):
                    if neighbour is None or not placed(neighbour):
                        continue
                    if node.get("poolId") in lanes_of and neighbour["poolId"] != node["poolId"]:
                        continue
                    node["poolId"], node["laneId"] = neighbour["poolId"], neighbour.get("laneId")
                    changed = True
                    break

        for node in nodes:
            if not placed(node):
                pool_id = node.get("poolId") if node.get("poolId") in lanes_of else pools[0]["id"]
                node["poolId"] = pool_id
                node["laneId"] = lanes_of[pool_id][0] if lanes_of[pool_id] else None
                logger.info(f"[BPMN] Validation: placed {node.get('id')!r} in pool {pool_id!r} (no placed neighbour)")
            node.setdefault("laneId", None)
            node["owner"] = node["laneId"]

    # Message-flow ends (BPMN 2.0 §9.3): a gateway carries none, a start event
    # only receives one and an end event only sends one.
    _MESSAGE_SOURCE_TYPES = frozenset({"task", "endEvent", "intermediateEvent"})
    _MESSAGE_TARGET_TYPES = frozenset({"task", "startEvent", "intermediateEvent"})

    @classmethod
    def _keep_sequence_flows_in_one_pool(
        cls, spec: Dict[str, Any], nodes: List[Dict[str, Any]], flows: List[Dict[str, Any]]
    ) -> None:
        """Keep each flow that cannot be a message flow inside one pool (mutates *spec* and *nodes*).

        Both editors turn a flow between two pools into a message flow, and the
        backend drops one whose end cannot carry it. The offending end moves to
        the other pool when all its neighbours are there; otherwise the two
        pools merge, keeping every lane. Task-to-task and end-to-start event
        flows stay message flows between their pools.
        """
        if len(spec.get("pools") or []) < 2:
            return
        by_id = {node["id"]: node for node in nodes if node.get("id")}
        neighbours: Dict[str, List[str]] = {}
        for flow in flows:
            neighbours.setdefault(flow.get("source"), []).append(flow.get("target"))
            neighbours.setdefault(flow.get("target"), []).append(flow.get("source"))

        def ends(flow: Dict[str, Any]):
            source, target = by_id.get(flow.get("source")), by_id.get(flow.get("target"))
            if not source or not target or source["poolId"] == target["poolId"]:
                return None
            bad = [node for node, ok in ((source, source["type"] in cls._MESSAGE_SOURCE_TYPES),
                                         (target, target["type"] in cls._MESSAGE_TARGET_TYPES)) if not ok]
            return (source, target, bad) if bad else None

        for flow in flows:
            found = ends(flow)
            if not found:
                continue
            source, target, bad = found
            for node in bad:
                other = target if node is source else source
                pool_ids = {by_id[nid]["poolId"] for nid in neighbours.get(node["id"], []) if nid in by_id}
                if pool_ids == {other["poolId"]}:
                    node["poolId"], node["laneId"] = other["poolId"], other.get("laneId")
                    node["owner"] = node["laneId"]
                    logger.info(f"[BPMN] Validation: moved {node['id']!r} into pool {other['poolId']!r}")
                    break

        for flow in flows:
            found = ends(flow)
            if found:
                cls._merge_pools(spec, nodes, found[0]["poolId"], found[1]["poolId"])

    @classmethod
    def _merge_pools(cls, spec: Dict[str, Any], nodes: List[Dict[str, Any]], first: str, second: str) -> None:
        """Fold the later of two pools into the earlier one as extra lanes (mutates *spec* and *nodes*)."""
        pools = spec["pools"]
        keep, gone = sorted((p for p in pools if p["id"] in (first, second)), key=pools.index)
        members = {pool["id"]: [n for n in nodes if n.get("poolId") == pool["id"]] for pool in (keep, gone)}
        if keep["lanes"] or gone["lanes"]:
            for pool in (keep, gone):
                if not pool["lanes"]:
                    pool["lanes"] = [{"id": pool["id"], "name": pool.get("name") or "", "isAgentic": False}]
                    for node in members[pool["id"]]:
                        node["laneId"] = pool["id"]
            taken = {lane["id"] for lane in keep["lanes"]}
            for lane in gone["lanes"]:
                if lane["id"] in taken:
                    new_id = cls._unique_id(lane["id"], taken)
                    for node in members[gone["id"]]:
                        if node.get("laneId") == lane["id"]:
                            node["laneId"] = new_id
                    lane["id"] = new_id
                taken.add(lane["id"])
            keep["lanes"] = keep["lanes"] + gone["lanes"]
        for node in members[keep["id"]] + members[gone["id"]]:
            node["poolId"] = keep["id"]
            node["owner"] = node.get("laneId")
        pools.remove(gone)
        logger.info(f"[BPMN] Validation: merged pool {gone['id']!r} into {keep['id']!r} (a flow between them cannot be a message flow)")

    @staticmethod
    def _connect_orphaned_nodes(nodes: List[Dict[str, Any]], flows: List[Dict[str, Any]]) -> None:
        """Give every non-start node at least one incoming flow (design rule 5 /
        BPMNFlowSpec's "every node except the start has an incoming flow").

        The model most commonly drops the flow into a branch target it clearly
        intended — e.g. a "no" branch off an exclusive gateway that ends up with
        only one outgoing flow while the matching end event ("Order Cancelled")
        sits unconnected. Prefer reconnecting from that kind of under-connected
        gateway (and infer the opposite yes/no label when the gateway's existing
        branch has one) before falling back to the previous node in generation
        order, so the graph is always fully reachable. Mutates `flows` in place.
        """
        start_id = next((n.get("id") for n in nodes if n.get("type") == "startEvent"), None)
        targets = {f.get("target") for f in flows}

        outgoing_count: Dict[str, int] = {}
        outgoing_labels: Dict[str, List[str]] = {}
        for f in flows:
            src = f.get("source")
            outgoing_count[src] = outgoing_count.get(src, 0) + 1
            outgoing_labels.setdefault(src, []).append((f.get("name") or "").strip().lower())

        gateway_ids = [n.get("id") for n in nodes if n.get("type") == "gateway"]
        under_connected_gateways = [gid for gid in gateway_ids if outgoing_count.get(gid, 0) < 2]

        for index, node in enumerate(nodes):
            node_id = node.get("id")
            if node_id is None or node_id == start_id or node_id in targets:
                continue

            source_id = None
            label = ""
            if under_connected_gateways:
                source_id = under_connected_gateways.pop(0)
                existing_labels = outgoing_labels.get(source_id, [])
                if existing_labels == ["yes"]:
                    label = "no"
                elif existing_labels == ["no"]:
                    label = "yes"
            elif index > 0:
                source_id = nodes[index - 1].get("id")

            if source_id and source_id != node_id:
                flows.append({"source": source_id, "target": node_id, "name": label})
                targets.add(node_id)
                outgoing_count[source_id] = outgoing_count.get(source_id, 0) + 1
                logger.info(
                    f"[BPMN] Validation: connected orphaned node {node_id!r} from {source_id!r}"
                    + (f" (label={label!r})" if label else "")
                )

    @staticmethod
    def _unique_id(base: str, existing: set) -> str:
        if base not in existing:
            return base
        i = 1
        while f"{base}_{i}" in existing:
            i += 1
        return f"{base}_{i}"

    # ------------------------------------------------------------------
    # Modification
    # ------------------------------------------------------------------

    def _prepare_modification(
        self, user_request: str, current_model: Dict[str, Any] = None, **kwargs,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """``(system_prompt, user_prompt, elements)``."""
        agentic = self._is_agentic_bpmn_request(kwargs.get("raw_request") or user_request, current_model)
        system_prompt = bpmn_prompts.MODIFY_SYSTEM_PROMPT_AGENTIC_BPMN if agentic else bpmn_prompts.MODIFY_SYSTEM_PROMPT_BPMN
        agent_diagram_refs = kwargs.get("agent_diagram_refs") or {}

        # The element map is passed explicitly to every helper: handlers are
        # singletons shared by all sessions, so per-request state on `self`
        # would race between concurrent requests.
        elements: Dict[str, Any] = {}
        if current_model and isinstance(current_model, dict):
            raw = current_model.get("elements")
            if isinstance(raw, dict):
                elements = raw

        context_block = ""
        if current_model and isinstance(current_model, dict):
            summary = detailed_model_summary(current_model, "BPMN")
            if summary:
                context_block = f"\n\n{summary}"

        if agentic:
            context_block += f"\n\n{format_agent_diagram_refs(agent_diagram_refs)}"

        return system_prompt, f"Modify the BPMN process: {user_request}{context_block}", elements

    def _build_modification(
        self, user_request: str, current_model: Dict[str, Any] = None, *, prepared=None, **kwargs,
    ) -> Dict[str, Any]:
        system_prompt, user_prompt, elements = prepared
        agent_diagram_refs = kwargs.get("agent_diagram_refs") or {}

        def _normalize_bpmn_mods(mod_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Normalize common malformed BPMN batches from the LLM.

            - Convert synthetic ids for newly-added nodes into stable names for
              same-batch flow references.
            - Expand add_task/add_gateway/add_event entries that incorrectly
              embed source/target refs into explicit add_flow actions.
            - Prefer clean node labels over leaked UI/type suffixes such as
              ``"Record video demo 1 (Task)"``.
            """
            alias_to_name: Dict[str, str] = {}
            node_add_actions = {"add_task", "add_gateway", "add_event", "add_data_object"}

            def _clean_added_name(action: str, target: Dict[str, Any], changes: Dict[str, Any]) -> Optional[str]:
                target_name = (target.get("nodeName") or "").strip()
                change_name = (changes.get("name") or "").strip()
                if change_name:
                    typed_suffixes = {
                        "add_task": " (Task)",
                        "add_gateway": " (Gateway)",
                        "add_event": " (Event)",
                        "add_data_object": " (DataObject)",
                    }
                    suffix = typed_suffixes.get(action)
                    if suffix and target_name == f"{change_name}{suffix}":
                        return change_name
                return target_name or change_name or None

            def _register_alias(name: Optional[str], alias: Optional[str]) -> None:
                if alias and name:
                    alias_to_name[alias] = name

            for mod in mod_list:
                if not isinstance(mod, dict):
                    continue
                action = mod.get("action", "")
                if action not in node_add_actions:
                    continue
                target = mod.get("target") or {}
                changes = mod.get("changes") or {}
                clean_name = _clean_added_name(action, target, changes)
                if not clean_name:
                    continue
                _register_alias(clean_name, clean_name)
                _register_alias(clean_name, target.get("nodeId"))
                _register_alias(clean_name, target.get("nodeName"))
                _register_alias(clean_name, changes.get("name"))

            normalized: List[Dict[str, Any]] = []
            expanded_flows = 0
            self._sanitize_modification_refs(mod_list, elements, agent_diagram_refs)
            _modification_governance_dsl(mod_list, current_model)

            for mod in mod_list:
                if not isinstance(mod, dict):
                    normalized.append(mod)
                    continue

                action = mod.get("action", "")
                target = dict(mod.get("target") or {})
                changes = dict(mod.get("changes") or {})

                if action in node_add_actions:
                    clean_name = _clean_added_name(action, target, changes)
                    if clean_name:
                        target["nodeName"] = clean_name
                        if changes.get("name") is not None:
                            changes["name"] = clean_name

                    raw_embedded_source = changes.pop("source", None)
                    raw_embedded_target = changes.pop("target", None)
                    embedded_source = alias_to_name.get(raw_embedded_source, raw_embedded_source)
                    embedded_target = alias_to_name.get(raw_embedded_target, raw_embedded_target)
                    embedded_label = changes.pop("label", None)

                    updated_mod = dict(mod)
                    updated_mod["target"] = target
                    updated_mod["changes"] = changes
                    normalized.append(updated_mod)

                    if embedded_source and embedded_target:
                        normalized.append(
                            {
                                "action": "add_flow",
                                "target": {},
                                "changes": {
                                    "source": embedded_source,
                                    "target": embedded_target,
                                    "label": embedded_label,
                                },
                            }
                        )
                        expanded_flows += 1
                    continue

                if action in ("add_flow", "remove_flow"):
                    if changes.get("source") in alias_to_name:
                        changes["source"] = alias_to_name[changes["source"]]
                    if changes.get("target") in alias_to_name:
                        changes["target"] = alias_to_name[changes["target"]]
                    updated_mod = dict(mod)
                    updated_mod["target"] = target
                    updated_mod["changes"] = changes
                    normalized.append(updated_mod)
                    continue

                # Both editors' modifiers read changes.<field> unguarded.
                normalized.append({**mod, "changes": mod.get("changes") or {}})

            if expanded_flows:
                logger.info(
                    f"[BPMN] Normalized {expanded_flows} embedded node-connection(s) into explicit add_flow action(s)"
                )
            self._place_added_nodes(normalized, elements)
            return normalized

        result = self._execute_modification(
            user_prompt, system_prompt, BPMNModificationResponse,
            post_processor=_normalize_bpmn_mods,
            elements=elements,
        )
        return self._validate_mod_refs(result, elements)

    def generate_fallback_modification(self, request: str) -> Dict[str, Any]:
        return assistant_message(
            "I couldn't apply that modification automatically. Could you rephrase it? "
            "For example: *'add a Send Invoice task after Ship Order'* or "
            "*'rename Check Inventory to Verify Stock'*.",
        )

    @classmethod
    def _place_added_nodes(cls, mod_list: List[Dict[str, Any]], elements: Dict[str, Any]) -> None:
        """Give each added task/gateway/event without ``changes.owner`` a lane (mutates *mod_list*).

        Both editors leave a node without an owner outside every lane, i.e. in
        another process than the laned nodes it connects to. The lane comes from
        a node it is connected to in this batch; failing that, the first lane,
        but only when every pool has lanes (a node of a lane-less pool has none).
        """
        lanes = [eid for eid, el in elements.items() if isinstance(el, dict) and el.get("type") == "BPMNSwimlane"]
        if not lanes:
            return
        pending = {
            (mod["target"].get("nodeName") or "").strip().lower(): mod
            for mod in mod_list
            if isinstance(mod, dict) and mod.get("action") in ("add_task", "add_gateway", "add_event")
            and isinstance(mod.get("target"), dict) and (mod["target"].get("nodeName") or "").strip()
            and not (mod.get("changes") or {}).get("owner")
        }
        if not pending:
            return
        edges = [
            ((mod.get("changes") or {}).get("source"), (mod.get("changes") or {}).get("target"))
            for mod in mod_list if isinstance(mod, dict) and mod.get("action") == "add_flow"
        ]
        added: Dict[str, Optional[str]] = {}  # new node name -> lane id, or None when not in a lane
        unknown = object()

        def lane_of(ref: Any) -> Any:
            """The lane id of *ref*, None when it is in no lane, ``unknown`` when not (yet) known."""
            key = ref.strip().lower() if isinstance(ref, str) else ""
            if key in added:
                return added[key]
            element = cls._bpmn_resolve(ref, elements) if key and key not in pending else None
            if element is None:
                return unknown
            return element.get("owner") if element.get("owner") in lanes else None

        changed = True
        while changed:
            changed = False
            for name in [n for n in pending if n not in added]:
                for source, target in edges:
                    for end, other in ((source, target), (target, source)):
                        if isinstance(end, str) and end.strip().lower() == name:
                            lane = lane_of(other)
                            if lane is not unknown:
                                added[name] = lane
                                changed = True
                    if name in added:
                        break

        every_pool_has_lanes = all(
            any(elements[lane].get("owner") == eid for lane in lanes)
            for eid, el in elements.items() if isinstance(el, dict) and el.get("type") == "BPMNPool"
        )
        for name, mod in pending.items():
            lane = added.get(name, lanes[0] if every_pool_has_lanes else None)
            if lane:
                mod["changes"] = {**(mod.get("changes") or {}), "owner": lane}

    @staticmethod
    def _sanitize_modification_refs(
        mod_list: List[Dict[str, Any]], elements: Dict[str, Any], agent_diagram_refs: Dict[str, str],
    ) -> None:
        """Clear agentDiagramRef / reflectionReviewerLaneId values the editor could not resolve.

        agentDiagramRef must be one of the project's AgentDiagram ids; a
        reflectionReviewerLaneId must resolve (by [id] or name) to an existing
        lane and is normalized to that lane's id, or name a lane an earlier
        add_swimlane of the same batch creates (the editor applies the batch in
        order and resolves it by name). Mutates the changes in place.
        """
        valid_refs = set(agent_diagram_refs or {})
        lanes = {
            eid: el for eid, el in (elements or {}).items()
            if isinstance(el, dict) and el.get("type") == "BPMNSwimlane"
        }
        new_lanes: Dict[str, str] = {}
        for mod in mod_list:
            if not isinstance(mod, dict):
                continue
            new_lane = (mod.get("target") or {}).get("nodeName") if mod.get("action") == "add_swimlane" else None
            if isinstance(new_lane, str) and new_lane.strip():
                new_lanes[new_lane.strip().lower()] = new_lane.strip()
            changes = mod.get("changes")
            if not isinstance(changes, dict):
                continue
            ref = changes.get("agentDiagramRef")
            if ref and ref not in valid_refs:
                logger.info(f"[BPMN] Dropped invented agentDiagramRef {ref!r}")
                changes["agentDiagramRef"] = None
            reviewer = changes.get("reflectionReviewerLaneId")
            if reviewer:
                lane_id = reviewer if reviewer in lanes else next(
                    (eid for eid, el in lanes.items()
                     if (el.get("name") or "").strip().lower() == reviewer.strip().lower()),
                    None,
                )
                changes["reflectionReviewerLaneId"] = lane_id or new_lanes.get(reviewer.strip().lower())

    # ------------------------------------------------------------------
    # Single element + fallbacks (required by BaseDiagramHandler)
    # ------------------------------------------------------------------

    # v1 has no append-one-node BPMN path on the WME side: a single-element
    # request becomes a one-task starter process (no LLM call).
    def _prepare_single_element(
        self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs,
    ) -> str:
        return self.extract_name_from_request(user_request, "Task")

    def _build_single_element(
        self, user_request: str, existing_model: Dict[str, Any] = None, *, prepared=None, **kwargs,
    ) -> Dict[str, Any]:
        return self._starter_process(prepared)

    def generate_fallback_element(self, request: str) -> Dict[str, Any]:
        return self._starter_process(self.extract_name_from_request(request, "Task"))

    def _starter_process(self, name: str) -> Dict[str, Any]:
        return inject_complete_system(
            {
                "systemName": name,
                "nodes": [
                    {"id": "start", "name": "Start", "type": "startEvent"},
                    {"id": "task1", "name": name, "type": "task", "taskType": "default"},
                    {"id": "end", "name": "End", "type": "endEvent"},
                ],
                "flows": [
                    {"source": "start", "target": "task1", "name": ""},
                    {"source": "task1", "target": "end", "name": ""},
                ],
            },
            self.get_diagram_type(),
            f"I created a starter process with a **{name}** task. Describe the full flow and I'll build it out!",
        )

    def generate_fallback_system(self) -> Dict[str, Any]:
        fallback = {
            "systemName": "BasicProcess",
            "nodes": [
                {"id": "start", "name": "Start", "type": "startEvent"},
                {"id": "task1", "name": "Do Work", "type": "task", "taskType": "default"},
                {"id": "end", "name": "End", "type": "endEvent"},
            ],
            "flows": [
                {"source": "start", "target": "task1", "name": ""},
                {"source": "task1", "target": "end", "name": ""},
            ],
        }
        return inject_complete_system(
            fallback,
            self.get_diagram_type(),
            (
                "I created a starter process. Describe your workflow "
                "(e.g. *'an order process: receive order, check stock, then ship "
                "or back-order'*) and I'll build a richer model!"
            ),
        )

    # ------------------------------------------------------------------
    # Message builder
    # ------------------------------------------------------------------

    def _build_system_message(self, spec: Dict[str, Any]) -> str:
        name = spec.get("systemName") or "process"
        nodes = spec.get("nodes", [])
        flows = spec.get("flows", [])
        pools = spec.get("pools") or []
        tasks = [n.get("name", "?") for n in nodes if n.get("type") == "task"][:6]
        msg = f"Built the **{name}** process with {len(nodes)} node(s)"
        if tasks:
            msg += f": {', '.join(f'**{t}**' for t in tasks)}"
        if flows:
            msg += f", connected by {len(flows)} flow(s)"
        if pools:
            pool_names = ", ".join(f"**{p.get('name') or p.get('id')}**" for p in pools)
            msg += f", across {len(pools)} participant(s) ({pool_names})"
        data_objects = spec.get("dataObjects") or []
        if data_objects:
            msg += f", with {len(data_objects)} data object(s)"
        msg += ". Ask me to add steps, rename nodes, or regenerate any time!"
        return msg

    # ------------------------------------------------------------------
    # BPMN-specific element resolution helpers
    # ------------------------------------------------------------------

    _GATEWAY_TYPE_LABELS = {
        "exclusive": "Exclusive Gateway",
        "parallel": "Parallel Gateway",
        "inclusive": "Inclusive Gateway",
        "event-based": "Event-Based Gateway",
        "complex": "Complex Gateway",
    }
    _TASK_TYPE_LABELS = {
        "user": "User Task", "service": "Service Task",
        "send": "Send Task", "receive": "Receive Task",
        "manual": "Manual Task", "business-rule": "Business Rule Task",
        "script": "Script Task",
    }
    _EVENT_KIND_LABELS = {
        "start": "Start Event", "end": "End Event", "intermediate": "Intermediate Event",
    }
    _APOLLON_TYPE_LABELS = {
        "BPMNStartEvent": "Start Event",
        "BPMNEndEvent": "End Event",
        "BPMNIntermediateEvent": "Intermediate Event",
        "BPMNCallActivity": "Call Activity",
        "BPMNDataObject": "Data Object",
        "BPMNDataStore": "Data Store",
    }

    @classmethod
    def _bpmn_el_type_label(cls, el: Dict[str, Any]) -> str:
        """Human-readable type label including gateway/task subtype."""
        el_type = el.get("type", "")
        static = cls._APOLLON_TYPE_LABELS.get(el_type)
        if static:
            return static
        if el_type == "BPMNGateway":
            return cls._GATEWAY_TYPE_LABELS.get(el.get("gatewayType", "exclusive"), "Gateway")
        if el_type == "BPMNTask":
            return cls._TASK_TYPE_LABELS.get(el.get("taskType", "default"), "Task")
        return "Element"

    @staticmethod
    def _bpmn_resolve(ref: Optional[str], elements: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Look up a BPMN element by id, exact name, or unique unnamed type label."""
        if not ref or not isinstance(elements, dict):
            return None
        el = elements.get(ref)
        if isinstance(el, dict):
            return el
        lower = ref.lower()
        for el in elements.values():
            if isinstance(el, dict) and (el.get("name") or "").lower() == lower:
                return el
        unnamed_matches = [
            el for el in elements.values()
            if isinstance(el, dict)
            and not (el.get("name") or "").strip()
            and BPMNDiagramHandler._bpmn_el_type_label(el).lower() == lower
        ]
        if len(unnamed_matches) == 1:
            return unnamed_matches[0]
        return None

    # ------------------------------------------------------------------
    # Base-class extension: BPMN-aware target name resolution
    # ------------------------------------------------------------------

    def _build_mod_target_name(
        self, action: str, target: dict, mod: dict = None, elements: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Extend base name resolution for BPMN-specific operations.

        - Flow operations (add_flow/remove_flow) display endpoint names joined
          by an arrow, resolved from the request's ``elements`` map.
        - Node operations on unnamed elements fall back to the type label
          (e.g. "Parallel Gateway") instead of the raw Apollon UUID.
        """
        elements = elements or {}

        if action in ("add_flow", "remove_flow"):
            changes = (mod or {}).get("changes") or {}
            src_ref = changes.get("source", "")
            tgt_ref = changes.get("target", "")
            src_el = self._bpmn_resolve(src_ref, elements)
            tgt_el = self._bpmn_resolve(tgt_ref, elements)
            src_name = (src_el.get("name") if src_el else None) or (
                self._bpmn_el_type_label(src_el) if src_el else src_ref or "element"
            )
            tgt_name = (tgt_el.get("name") if tgt_el else None) or (
                self._bpmn_el_type_label(tgt_el) if tgt_el else tgt_ref or "element"
            )
            return f"{src_name} → {tgt_name}"

        node_ref = target.get("nodeId") or target.get("nodeName")
        if node_ref and elements:
            el = self._bpmn_resolve(node_ref, elements)
            if el is not None:
                return el.get("name") or self._bpmn_el_type_label(el)

        return super()._build_mod_target_name(action, target, mod, elements=elements)

    # ------------------------------------------------------------------
    # Server-side reference guardrail (BaseDiagramHandler._validate_mod_refs)
    # ------------------------------------------------------------------

    _REF_ID_KEY = "nodeId"
    _REF_NAME_KEY = "nodeName"
    _REF_TARGET_ACTIONS = frozenset({"remove_element", "modify_node"})
    _REF_ENDPOINT_ACTIONS = frozenset({"add_flow", "remove_flow"})
    _REF_ADD_ACTIONS = {
        "add_task": "BPMNTask", "add_gateway": "BPMNGateway", "add_event": "BPMNEvent",
        "add_data_object": "BPMNDataObject",
    }
    _REF_RENAME_ACTIONS = frozenset({"modify_node"})
    _REF_REMOVE_ACTIONS = frozenset({"remove_element"})

    def _resolve_element_ref(self, ref: Optional[str], elements: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self._bpmn_resolve(ref, elements)

    def _preview_added_element(self, mod: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Carry the task/gateway/event subtype so type-label lookups still work."""
        element = super()._preview_added_element(mod)
        if element is None:
            return None
        changes = mod.get("changes") or {}
        action = mod.get("action")
        if action == "add_task":
            element["taskType"] = changes.get("taskType") or "default"
        elif action == "add_gateway":
            element["gatewayType"] = changes.get("gatewayType") or "exclusive"
        elif action == "add_event":
            element["type"] = f"BPMN{(changes.get('eventKind') or 'intermediate').capitalize()}Event"
        return element
