"""
BPMN Diagram Handler
Handles generation and modification of BPMN process diagrams, including
multi-participant collaboration diagrams.

Emits a process (start/end events, tasks, exclusive/parallel/inclusive
gateways, flows) optionally grouped into pools (participants) and lanes
(roles within a pool). No other agentic concepts (governance, trust, etc).
Positions are NOT generated here: the WME injector lays the process (and any
pools/lanes) out and the editor's layouter routes the flows. Message vs.
sequence flow type is also derived on the WME side from pool membership —
the agent never sets a flow type. Pools/lanes are generation-only for now;
the modification path (generate_modification) does not yet support
add_pool/add_lane actions.
"""

import logging
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
2. add_task: set target.nodeName to the task name. Optional changes.taskType (default/user/service/send/receive/manual/business-rule/script).
3. add_gateway: set target.nodeName to the gateway label/question. Optional changes.gatewayType (exclusive/parallel/inclusive). Default exclusive.
4. add_event: set target.nodeName and changes.eventKind to "start", "end", or "intermediate".
5. add_flow: set changes.source and changes.target to the node ID (exact [id] from context) or name. Use the id for unnamed nodes.
6. modify_node: {EXACT_NAMES_RULE} For unnamed nodes set target.nodeId to the exact [id] from the context. Put the new name in changes.name (and/or changes.taskType / changes.gatewayType).
7. {REMOVE_ELEMENT_RULE} For remove_element: use target.nodeName for named nodes; for UNNAMED nodes set target.nodeId to the exact [id] from the context. Connected flows are removed automatically.
8. remove_flow: set changes.source and changes.target to the node IDs or names of the flow endpoints.
9. For NAMED nodes you may use the display name. For UNNAMED nodes (no name shown before the type) you MUST use the exact id from [id].

When the user asks to remove or modify an element, always verify the element exists in the current context listing before emitting any remove_element or
modify_node action. If no entry in the listing matches the user's description (by name or id):
- Set elementFound: false
- Set modifications: [] (empty — do NOT substitute a different element)
- Set message to explain what was not found, e.g.: "I couldn't find an element named 'Buy Groceries' in this diagram. Current nodes are: Document Review Started, Review by Reviewer 1, …"
Partial matches are valid (e.g. "Reviewer 1" matching "Review by Reviewer 1"). Only set elementFound: false when there is genuinely no match.

If the user says 'undo', 'undo that', 'revert', or similar, do not emit any modifications. Reply with modifications: [], elementFound: false, 
and set message to: 'To undo, use Ctrl+Z or the undo button in the editor toolbar.'"""


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

    def generate_complete_system(
        self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs,
    ) -> Dict[str, Any]:
        system_prompt = self.get_system_prompt()
        logger.info(f"[BPMN] generate_complete_system called with: {user_request!r}")

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
        system_prompt = MODIFY_SYSTEM_PROMPT_BPMN

        # Store elements on the instance so _build_mod_target_name can resolve
        # element names without needing a separate parameter thread.
        self._elements: Dict[str, Any] = {}
        if current_model and isinstance(current_model, dict):
            raw = current_model.get("elements")
            if isinstance(raw, dict):
                self._elements = raw

        context_block = ""
        if current_model and isinstance(current_model, dict):
            summary = detailed_model_summary(current_model, "BPMN")
            if summary:
                context_block = f"\n\n{summary}"

        user_prompt = f"Modify the BPMN process: {user_request}{context_block}"
        logger.info(f"[BPMN] generate_modification called with: {user_request!r}")

        try:
            result = self._execute_modification(
                user_prompt, system_prompt, BPMNModificationResponse,
            )
            return self._validate_mod_refs(result)
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
        """Look up a BPMN element by Apollon id (exact key) then by name (case-insensitive)."""
        if not ref or not isinstance(elements, dict):
            return None
        el = elements.get(ref)
        if isinstance(el, dict):
            return el
        lower = ref.lower()
        for el in elements.values():
            if isinstance(el, dict) and (el.get("name") or "").lower() == lower:
                return el
        return None

    # ------------------------------------------------------------------
    # Base-class extension: BPMN-aware target name resolution
    # ------------------------------------------------------------------

    def _build_mod_target_name(self, action: str, target: dict, mod: dict = None) -> str:
        """Extend base name resolution for BPMN-specific operations.

        - Flow operations (add_flow/remove_flow) display endpoint names joined
          by an arrow, resolved from self._elements when available.
        - Node operations on unnamed elements fall back to the type label
          (e.g. "Parallel Gateway") instead of the raw Apollon UUID.
        """
        elements = getattr(self, "_elements", {})

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

        return super()._build_mod_target_name(action, target, mod)

    # ------------------------------------------------------------------
    # Server-side ref guardrail (item 1)
    # ------------------------------------------------------------------

    def _ref_exists(self, mod: Dict[str, Any], elements: Dict[str, Any]) -> bool:
        """Return True if every element ref in this modification exists in the model."""
        action = mod.get("action", "")
        if action in ("remove_element", "modify_node"):
            ref = (mod.get("target") or {}).get("nodeId") or (mod.get("target") or {}).get("nodeName")
            return ref is None or self._bpmn_resolve(ref, elements) is not None
        if action in ("add_flow", "remove_flow"):
            changes = mod.get("changes") or {}
            src, tgt = changes.get("source"), changes.get("target")
            src_ok = src is None or self._bpmn_resolve(src, elements) is not None
            tgt_ok = tgt is None or self._bpmn_resolve(tgt, elements) is not None
            return src_ok and tgt_ok
        return True

    def _validate_mod_refs(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Drop modifications whose element refs cannot be resolved in the current model.

        If all modifications are dropped, converts the result to an assistant_message
        so the user gets a clear explanation rather than a silent no-op.
        """
        elements = self._elements
        if not elements or result.get("action") != "modify_model":
            return result

        if "modifications" in result:
            mods = result["modifications"]
            valid = [m for m in mods if self._ref_exists(m, elements)]
            dropped = len(mods) - len(valid)
            if dropped:
                logger.info(f"[BPMN] Dropped {dropped} modification(s) with unresolved element ref(s)")
            if not valid:
                return {
                    "action": "assistant_message",
                    "message": (
                        "I couldn't find the element(s) you described in the current diagram. "
                        "Please check the names and try again."
                    ),
                }
            result = dict(result)
            result["modifications"] = valid
            return result

        if "modification" in result:
            if not self._ref_exists(result["modification"], elements):
                logger.info("[BPMN] Dropped modification with unresolved element ref")
                return {
                    "action": "assistant_message",
                    "message": (
                        "I couldn't find that element in the current diagram. "
                        "Please check the name and try again."
                    ),
                }

        return result
