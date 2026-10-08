"""
BPMN Diagram Handler
Handles generation and modification of BPMN process diagrams, including
multi-participant collaboration diagrams and Agentic BPMN.

Emits a process (start/end events, tasks, exclusive/parallel/inclusive
gateways, flows) optionally grouped into pools (participants) and lanes
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
from typing import Any, Dict, List, Optional

from ..core.base_handler import BaseDiagramHandler, LLMPredictionError
from ..core.prompt_fragments import EXACT_NAMES_RULE, POSITION_DISCLAIMER, REMOVE_ELEMENT_RULE
from schemas import SystemBPMNSpec, BPMNModificationResponse
from utilities.model_context import detailed_model_summary

logger = logging.getLogger(__name__)


MODIFY_SYSTEM_PROMPT_BPMN = f"""You are a BPMN modeling expert. The user wants to modify a BPMN process diagram.

READING THE CONTEXT:
Each node appears as:  [id] Name (type)   ← named node
                       [id] (type)         ← unnamed node — MUST reference by id
Each flow appears as:  Flow: [src-id] Name -> [tgt-id] Name

MODIFICATION RULES:
1. Actions available: "add_task", "add_gateway", "add_event", "add_flow", "modify_node", "remove_flow", "remove_element"
2. add_task: set target.nodeName to the task name only. Do NOT append UI/type suffixes like "(Task)". Optional changes.taskType (default/user/service/send/receive/manual/business-rule/script).
3. add_gateway: set target.nodeName to the gateway label/question only. Do NOT append "(Gateway)". Optional changes.gatewayType (exclusive/parallel/inclusive). Default exclusive.
4. add_event: set target.nodeName and changes.eventKind to "start", "end", or "intermediate". Do NOT append "(Event)".
5. add_flow: set changes.source and changes.target to the node ID (exact [id] from context) or name. Use the id for unnamed nodes.
6. Never put flow endpoints inside add_task/add_gateway/add_event. Connections must be emitted as separate add_flow actions.
7. modify_node: {EXACT_NAMES_RULE} For unnamed nodes set target.nodeId to the exact [id] from the context. Put the new name in changes.name (and/or changes.taskType / changes.gatewayType).
8. {REMOVE_ELEMENT_RULE} For remove_element: use target.nodeName for named nodes; for UNNAMED nodes set target.nodeId to the exact [id] from the context. Connected flows are removed automatically.
9. remove_flow: set changes.source and changes.target to the node IDs or names of the flow endpoints.
10. For NAMED nodes you may use the display name. For UNNAMED nodes (no name shown before the type) you MUST use the exact id from [id].
11. If the user asks for "a second", "another", or "one more" task, add exactly ONE new task unless they explicitly ask for two or more.

When the user asks to remove or modify an element, always verify the element exists in the current context listing before emitting any remove_element or
modify_node action. If no entry in the listing matches the user's description (by name or id):
- Set elementFound: false
- Set modifications: [] (empty — do NOT substitute a different element)
- Set message to explain what was not found, e.g.: "I couldn't find an element named 'Buy Groceries' in this diagram. Current nodes are: Document Review Started, Review by Reviewer 1, …"
Partial matches are valid (e.g. "Reviewer 1" matching "Review by Reviewer 1"). Only set elementFound: false when there is genuinely no match.

If the user says 'undo', 'undo that', 'revert', or similar, do not emit any modifications. Reply with modifications: [], elementFound: false,
and set message to: 'To undo, use Ctrl+Z or the undo button in the editor toolbar.'"""

AGENT_DIAGRAM_REF_RULE = (
    "agentDiagramRef links an agent to an EXISTING AgentDiagram of the project. Use ONLY an id "
    "from the 'Agent diagrams in this project' list in the request; when the list is empty or "
    "none fits, leave agentDiagramRef null. Never invent an id."
)

AGENTIC_MODIFY_RULES = f"""AGENTIC BPMN RULES (the process has agentic lanes or the user asked for agentic changes):
The context also lists pools and lanes:
  Pool: [id] Name
  Lane: [id] Name (role=..., agentic=..., multiplicity=...)
and flow nodes may show lane=..., agentic=true, trust=..., reflection=..., role=..., governanceDsl=set/unset.
A1. Additional actions: "add_pool", "add_swimlane", "modify_swimlane", "remove_swimlane", "remove_pool". The *_swimlane names are the editor's API names for LANES; call them lanes in your message.
A2. add_pool: target.nodeName = the new pool's name. add_swimlane: target.nodeName = the new lane's name and changes.poolName = the existing pool's name or [id]; optional changes.isAgentic, changes.role, changes.trustScore, changes.multiplicity.
A3. modify_swimlane / remove_swimlane: target.swimlaneName = the lane's name or [id]. remove_pool: target.poolName = the pool's name or [id].
A4. Lane role is exactly 'solution' (an agent that does the work) or 'supervision' (an agent that oversees other agents). multiplicity is the swarm size, an integer >= 1.
A5. To place a NEW node in an existing lane, set changes.owner to the lane's name or [id].
A6. modify_node on a task may set isAgentic, reflectionMode ('none', 'self', 'cross', 'human'), reflectionReviewerLaneId (the reviewing lane's [id], only with reflectionMode 'cross'), trustScore (0-100) and agentDiagramRef; on a gateway: isAgentic, gatewayRole ('diverging' or 'merging'), trustScore and governanceDsl. An agentic gateway must be parallel or inclusive; governanceDsl belongs only on a merging gateway.
A7. {AGENT_DIAGRAM_REF_RULE}
A8. Flows contain only endpoints and labels — never agentic message-flow notation, collaborationMode, or mergingStrategy."""

# The agentic prompt EXTENDS the base prompt: unnamed-node [id] references,
# partial matching, "add exactly ONE" and the flow-endpoint rules still apply.
MODIFY_SYSTEM_PROMPT_AGENTIC_BPMN = f"{MODIFY_SYSTEM_PROMPT_BPMN}\n\n{AGENTIC_MODIFY_RULES}"

AGENTIC_DESIGN_RULES = f"""AGENTIC BPMN RULES (the user asked for an agentic process / agent swarm; they extend the rules above):
A1. Put the agents in a pool and give each agent role its own lane with isAgentic=true. Use pools[].lanes, never pools[].swimlanes. Human or system participants stay ordinary (isAgentic=false) lanes or pools.
A2. Lane role is exactly 'solution' (an agent that does the work) or 'supervision' (an agent that oversees, approves or reviews other agents).
A3. multiplicity is the swarm size of an agentic lane (e.g. three reviewer agents → 3); leave it null when the request gives none. Set trustScore (0-100) only when the request states or clearly implies it.
A4. Every node in a pool sets poolId and every node in a lane sets laneId (owner may repeat laneId). Never use a generated WME/Apollon element id.
A5. A task performed by an agent sets isAgentic=true and may set reflectionMode: 'self', 'human', or 'cross' with reflectionReviewerLaneId = the id of the reviewing agent lane.
A6. A gateway that fans work out to agents or merges their results is agentic (isAgentic=true), must be parallel or inclusive, and sets gatewayRole 'diverging' or 'merging'. A merging agentic gateway that decides how the results are accepted carries a non-empty governanceDsl (e.g. 'Policy: MajorityPolicy').
A7. {AGENT_DIAGRAM_REF_RULE}
A8. Flows contain only source, target, and name. Never emit flowType, agentic flow fields, collaborationMode, or mergingStrategy."""

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


class BPMNDiagramHandler(BaseDiagramHandler):
    """Handler for base BPMN process generation and modification."""

    def get_diagram_type(self) -> str:
        # The WME storage-bucket token (NOT the Apollon model.type
        # "BPMNDiagram"); the WME converter sets model.type itself.
        return "BPMN"

    def get_system_prompt(self) -> str:
        return f"""You are a business-process modeling expert. Create a base BPMN process from the user's request.

DESIGN RULES:
1. Exactly ONE start event; at least one end event.
2. Use tasks for activities/steps with clear verb-phrase names ('Check Inventory', 'Ship Order'). Set taskType based on WHO/WHAT performs it: 'user' for a person acting (customer, staff member typing/clicking), 'manual' for a person doing a PHYSICAL action with no system involved (packing a box), 'service' for an automated system check or call, 'send'/'receive' for a message to/from another party. Do not leave every task as the 'default' type — pick the closest fit.
3. Use an exclusive gateway for EACH distinct either/or decision the request describes; name it as a question ('In stock?') and label its outgoing flows with the condition ('yes' / 'no'). If the request describes N separate checks (e.g. "validate payment" AND "check stock" are two different checks), you MUST emit N separate gateways — do not merge multiple checks into one gateway or fold a described decision into a plain task.
4. Use a parallel gateway to split into CONCURRENT work and another to JOIN it back. A parallel split MUST have ≥2 outgoing flows to DIFFERENT target nodes; a parallel join MUST have ≥2 incoming flows from different sources. NEVER chain parallel tasks linearly — always fan them out from the split gateway and fan them back into the join gateway.
5. Connect everything with sequence flows. Every node except the start has an incoming flow; every node except end events has an outgoing flow. Every end event must be reachable — if a decision branch leads to a distinct outcome (e.g. "order cannot be completed"), route that branch's flow explicitly to the end event that represents it; never leave an end event with no incoming flow.
6. Keep it focused (typically 4-10 nodes, more if the request genuinely describes more distinct steps/decisions — do not compress described steps just to stay under 10).
7. Use POOLS only when the request describes two or more distinct participants — separate organizations, companies, or systems interacting (e.g. "customer" and "vendor", "shop" and "supplier", "our system" and "the payment gateway"). Give each participant its own entry in `pools` and set every node it owns to that pool's id via `poolId`. Communication between two pools MUST be a flow from a node in one pool to a node in another — the WME renders these as message flows automatically, you never set a flow type. Use LANES inside a single pool ONLY when the request names distinct roles or departments performing steps within ONE organization (e.g. "clerk", "chef", "delivery driver"); declare them under that pool's `lanes` and set each node's `laneId`. If the request describes a single actor doing everything, leave `pools` empty and every node's `poolId`/`laneId` null — do not invent participants that were not described.
8. {POSITION_DISCLAIMER}

Node ids are short lowercase slugs ('check_stock') referenced by flows. Pool and lane ids follow the same convention ('customer', 'chef')."""

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

    def generate_complete_system(
        self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs,
    ) -> Dict[str, Any]:
        logger.info(f"[BPMN] generate_complete_system called with: {user_request!r}")
        agent_diagram_refs = kwargs.get("agent_diagram_refs") or {}

        # Detect on the user's own words: the enriched request also carries
        # conversation history and workspace context.
        if self._is_agentic_bpmn_request(kwargs.get("raw_request") or user_request, existing_model):
            return self._generate_agentic_complete_system(user_request, agent_diagram_refs)

        system_prompt = self.get_system_prompt()

        reasoning_prompt = (
            "You are a BPMN process-design expert. Think step by step about the "
            "following process request and plan it before producing JSON.\n\n"
            f"User Request: {user_request}\n\n"
            "Analyze:\n"
            "0. Does this involve two or more distinct participants (separate organizations, "
            "companies, or systems) communicating, or distinct roles/departments within ONE "
            "organization? If yes, plan the pools (one per participant) and, for role-based "
            "processes, the lanes inside the relevant pool BEFORE listing nodes — every node "
            "you write below must then declare which pool/lane it belongs to. If the request "
            "describes a single actor doing everything, skip pools/lanes entirely.\n"
            "1. What is the trigger (start event)?\n"
            "2. What are the activities (tasks) and their order? For each, who/what "
            "performs it (a person = user/manual task, a system = service/script task, "
            "a notification = send/receive task)?\n"
            "3. Where are the decisions (exclusive gateways) and what are the conditions? "
            "List EVERY distinct check the request describes as its own gateway — do not "
            "merge two different checks (e.g. 'validate payment' and 'check stock' are "
            "TWO gateways, not one) and do not fold a described decision into a plain task.\n"
            "4. Is any work concurrent (parallel gateways)?\n"
            "5. What are the possible outcomes (end events)? For each negative/exception "
            "outcome a gateway branch leads to (e.g. 'order cannot be completed'), make sure "
            "that branch's flow actually reaches the matching end event.\n\n"
            "Focus on the SEQUENCE FLOWS — they are the most commonly under-specified part. "
            "Before finalizing, re-read the request once more and check you have not silently "
            "dropped or merged any step or decision point it mentioned, and that every node "
            "belonging to a pool/lane you planned actually has that poolId/laneId set."
        )

        try:
            parsed = self.predict_two_pass_structured(
                user_request=user_request,
                system_prompt=system_prompt,
                reasoning_prompt=reasoning_prompt,
                response_schema=SystemBPMNSpec,
            )
            system_spec = parsed.model_dump()
            system_spec = self._validate_and_refine(system_spec)
            self._sanitize_agentic_refs(system_spec, agent_diagram_refs)

            return {
                "action": "inject_complete_system",
                "systemSpec": system_spec,
                "diagramType": self.get_diagram_type(),
                "message": self._build_system_message(system_spec),
            }

        except LLMPredictionError as exc:
            logger.error(f"[BPMN] generate_complete_system LLM FAILED: {exc}")
            return self._error_response(
                "I couldn't generate that process. Please try again or rephrase your request.",
                code="llm_failure",
            )
        except Exception as exc:
            logger.error(f"[BPMN] generate_complete_system FAILED: {exc}", exc_info=True)
            return self.generate_fallback_system()

    def _generate_agentic_complete_system(
        self, user_request: str, agent_diagram_refs: Dict[str, str],
    ) -> Dict[str, Any]:
        """Generate an Agentic BPMN process: the base rules plus the agentic ones,
        followed by the same deterministic repair pass as the base path."""
        system_prompt = f"{self.get_system_prompt()}\n\n{AGENTIC_DESIGN_RULES}"
        refs_block = format_agent_diagram_refs(agent_diagram_refs)

        reasoning_prompt = (
            "You are an agentic process-design expert. Think step by step about the "
            "following agent-swarm request and plan it before producing JSON.\n\n"
            f"User Request: {user_request}\n\n"
            "Analyze:\n"
            "1. Which agent roles are there, and is each a 'solution' agent (does the work) or a "
            "'supervision' agent (oversees other agents)? Which participants are humans or systems?\n"
            "2. How many instances of each agent are needed (multiplicity)?\n"
            "3. What is the trigger (start event), and what tasks does each lane perform?\n"
            "4. Where does work fan out to agents or merge back (agentic parallel/inclusive gateways), "
            "and which merge needs a Governance DSL policy?\n"
            "5. Which agent outputs are reviewed (reflection: self, cross with a reviewer lane, or human)?\n"
            "6. What are the possible outcomes (end events), and does every branch reach one?\n\n"
            "Focus on correct lane ownership — every task must be owned by a lane — and on the "
            "SEQUENCE FLOWS between them."
        )

        try:
            parsed = self.predict_two_pass_structured(
                user_request=f"{user_request}\n\n{refs_block}",
                system_prompt=system_prompt,
                reasoning_prompt=reasoning_prompt,
                response_schema=SystemBPMNSpec,
            )
            system_spec = parsed.model_dump()
            system_spec = self._validate_and_refine(system_spec)
            self._sanitize_agentic_refs(system_spec, agent_diagram_refs)
            return {
                "action": "inject_complete_system",
                "systemSpec": system_spec,
                "diagramType": self.get_diagram_type(),
                "message": self._build_agentic_message(system_spec),
            }
        except LLMPredictionError as exc:
            logger.error(f"[BPMN] _generate_agentic_complete_system LLM FAILED: {exc}")
            return self._error_response(
                "I couldn't generate that agentic process. Please try again or rephrase your request.",
                code="llm_failure",
            )
        except Exception as exc:
            logger.error(f"[BPMN] _generate_agentic_complete_system FAILED: {exc}", exc_info=True)
            return self.generate_fallback_system()

    @staticmethod
    def _sanitize_agentic_refs(spec: Dict[str, Any], agent_diagram_refs: Dict[str, str]) -> None:
        """Drop references the editor could not resolve (mutates *spec*).

        - agentDiagramRef is a project AgentDiagram UUID in the editor: keep it
          only when it is one of this project's AgentDiagram ids.
        - reflectionReviewerLaneId must name a declared lane, and only on a
          task whose reflectionMode is 'cross'.
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
        self._infer_missing_lane_owners(spec, nodes, flows)

        spec["nodes"] = nodes
        spec["flows"] = flows
        return spec

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
    def _infer_missing_lane_owners(
        spec: Dict[str, Any], nodes: List[Dict[str, Any]], flows: List[Dict[str, Any]]
    ) -> None:
        """Backfill laneId/owner for pool-contained nodes when lane membership can
        be inferred from the validated pool structure and adjacent sequence flows."""
        pools: List[Dict[str, Any]] = spec.get("pools") or []
        if not pools:
            return

        lane_ids_by_pool: Dict[str, set[str]] = {
            pool["id"]: {
                lane["id"]
                for lane in (pool.get("lanes") or [])
                if lane.get("id")
            }
            for pool in pools
            if pool.get("id")
        }
        if not lane_ids_by_pool:
            return

        nodes_by_id: Dict[str, Dict[str, Any]] = {
            node["id"]: node
            for node in nodes
            if node.get("id")
        }
        incoming: Dict[str, List[str]] = {}
        outgoing: Dict[str, List[str]] = {}

        for flow in flows:
            source = flow.get("source")
            target = flow.get("target")
            if source and target:
                outgoing.setdefault(source, []).append(target)
                incoming.setdefault(target, []).append(source)

        for node in nodes:
            pool_id = node.get("poolId")
            if not pool_id:
                node["owner"] = None
                continue

            valid_lanes = lane_ids_by_pool.get(pool_id, set())
            if not valid_lanes:
                node["owner"] = None
                continue

            lane_id = node.get("laneId")
            if lane_id in valid_lanes:
                node["owner"] = lane_id
                continue
            if node.get("type") == "task":
                node["laneId"] = None
                node["owner"] = None
                continue

            inferred_lane_ids: set[str] = set()
            for neighbor_id in incoming.get(node["id"], []) + outgoing.get(node["id"], []):
                neighbor = nodes_by_id.get(neighbor_id)
                if not neighbor or neighbor.get("poolId") != pool_id:
                    continue
                neighbor_lane_id = neighbor.get("laneId")
                if neighbor_lane_id in valid_lanes:
                    inferred_lane_ids.add(neighbor_lane_id)

            inferred_lane_id = None
            if len(valid_lanes) == 1:
                inferred_lane_id = next(iter(valid_lanes))
            elif len(inferred_lane_ids) == 1:
                inferred_lane_id = next(iter(inferred_lane_ids))

            if inferred_lane_id:
                node["laneId"] = inferred_lane_id
                node["owner"] = inferred_lane_id
            else:
                node["laneId"] = None
                node["owner"] = None

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

    def generate_modification(
        self, user_request: str, current_model: Dict[str, Any] = None, **kwargs,
    ) -> Dict[str, Any]:
        agentic = self._is_agentic_bpmn_request(kwargs.get("raw_request") or user_request, current_model)
        system_prompt = MODIFY_SYSTEM_PROMPT_AGENTIC_BPMN if agentic else MODIFY_SYSTEM_PROMPT_BPMN
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

        user_prompt = f"Modify the BPMN process: {user_request}{context_block}"
        logger.info(f"[BPMN] generate_modification called with: {user_request!r}")

        try:
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
                node_add_actions = {"add_task", "add_gateway", "add_event"}

                def _clean_added_name(action: str, target: Dict[str, Any], changes: Dict[str, Any]) -> Optional[str]:
                    target_name = (target.get("nodeName") or "").strip()
                    change_name = (changes.get("name") or "").strip()
                    if change_name:
                        typed_suffixes = {
                            "add_task": " (Task)",
                            "add_gateway": " (Gateway)",
                            "add_event": " (Event)",
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
                        updated_mod["changes"] = changes or None
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

                    normalized.append(mod)

                if expanded_flows:
                    logger.info(
                        f"[BPMN] Normalized {expanded_flows} embedded node-connection(s) into explicit add_flow action(s)"
                    )
                return normalized

            result = self._execute_modification(
                user_prompt, system_prompt, BPMNModificationResponse,
                post_processor=_normalize_bpmn_mods,
                elements=elements,
            )
            return self._validate_mod_refs(result, elements)
        except LLMPredictionError as exc:
            logger.error(f"[BPMN] generate_modification LLM FAILED: {exc}")
            return self._error_response(
                "I couldn't process that modification. Please try again or rephrase your request.",
            )
        except Exception as exc:
            logger.error(f"[BPMN] generate_modification FAILED: {exc}", exc_info=True)
            return {
                "action": "assistant_message",
                "message": (
                    "I couldn't apply that modification automatically. Could you rephrase it? "
                    "For example: *'add a Send Invoice task after Ship Order'* or "
                    "*'rename Check Inventory to Verify Stock'*."
                ),
            }

    @staticmethod
    def _sanitize_modification_refs(
        mod_list: List[Dict[str, Any]], elements: Dict[str, Any], agent_diagram_refs: Dict[str, str],
    ) -> None:
        """Clear agentDiagramRef / reflectionReviewerLaneId values the editor could not resolve.

        agentDiagramRef must be one of the project's AgentDiagram ids; a
        reflectionReviewerLaneId must resolve (by [id] or name) to an existing
        lane and is normalized to that lane's id. Mutates the changes in place.
        """
        valid_refs = set(agent_diagram_refs or {})
        lanes = {
            eid: el for eid, el in (elements or {}).items()
            if isinstance(el, dict) and el.get("type") == "BPMNSwimlane"
        }
        for mod in mod_list:
            changes = mod.get("changes") if isinstance(mod, dict) else None
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
                changes["reflectionReviewerLaneId"] = lane_id

    # ------------------------------------------------------------------
    # Single element + fallbacks (required by BaseDiagramHandler)
    # ------------------------------------------------------------------

    def generate_single_element(
        self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs,
    ) -> Dict[str, Any]:
        """v1 has no append-one-node BPMN path on the WME side — funnel single-
        element requests into a one-task starter process so the contract holds."""
        name = self.extract_name_from_request(user_request, "Task")
        return {
            "action": "inject_complete_system",
            "systemSpec": {
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
            "diagramType": self.get_diagram_type(),
            "message": f"I created a starter process with a **{name}** task. Describe the full flow and I'll build it out!",
        }

    def generate_fallback_element(self, request: str) -> Dict[str, Any]:
        return self.generate_single_element(request)

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
        return {
            "action": "inject_complete_system",
            "systemSpec": fallback,
            "diagramType": self.get_diagram_type(),
            "message": (
                "I created a starter process. Describe your workflow "
                "(e.g. *'an order process: receive order, check stock, then ship "
                "or back-order'*) and I'll build a richer model!"
            ),
        }

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
    _REF_ADD_ACTIONS = {"add_task": "BPMNTask", "add_gateway": "BPMNGateway", "add_event": "BPMNEvent"}
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
