"""
BPMN Diagram Handler
Handles generation and modification of base BPMN process diagrams.

Emits a flat process (start/end events, tasks, exclusive/parallel/inclusive
gateways, sequence flows).  No pools/lanes and no agentic concepts (roles,
governance, collaboration, trust).  Positions are NOT generated here: the WME
injector lays the process out left-to-right and the editor's layouter routes
the flows.
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

MODIFY_SYSTEM_PROMPT_AGENTIC_BPMN = f"""You are an Agentic BPMN modeling expert.

READING THE CONTEXT:
Each pool appears as: Pool: [id] Name
Each lane appears as: Lane: [id] Name (role=..., agentic=..., multiplicity=...)
Each node appears as: [id] Name (type) [in lane: LaneName]
Each flow appears as: Flow: [src-id] Name -> [tgt-id] Name

MODIFICATION RULES:
1. Actions are: "add_task", "add_gateway", "add_event", "add_flow",
   "modify_node", "remove_flow", "remove_element", "add_pool",
   "add_swimlane", "modify_swimlane", "remove_swimlane", "remove_pool".
2. The `*_swimlane` action names are WME API names. Call the modeled
   constructs lanes in explanations and never generate pools[].swimlanes.
3. add_swimlane: set target.nodeName and changes.poolName. Allowed role values:
   solution, supervision, collaboration, consensus.
4. Use changes.owner only to place a newly added node in an existing lane.
5. Use modify_node for task fields: isAgentic, reflectionMode, trustScore,
   agentDiagramRef; or gateway fields: isAgentic, gatewayRole, trustScore,
   governanceDsl.
6. An agentic gateway must be parallel or inclusive. governanceDsl is only
   meaningful for a merging gateway.
7. Flows contain only normal endpoints and labels. Do not use agentic
   message-flow notation, collaborationMode, or mergingStrategy.
8. {REMOVE_ELEMENT_RULE}
9. {EXACT_NAMES_RULE}

If an element is not present in context, set elementFound: false and emit
no modifications."""

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
2. Use tasks for activities/steps with clear verb-phrase names ('Check Inventory', 'Ship Order').
3. Use an exclusive gateway for an either/or decision; name it as a question ('In stock?') and label its outgoing flows with the condition ('yes' / 'no').
4. Use a parallel gateway to split into CONCURRENT work and another to JOIN it back. A parallel split MUST have ≥2 outgoing flows to DIFFERENT target nodes; a parallel join MUST have ≥2 incoming flows from different sources. NEVER chain parallel tasks linearly — always fan them out from the split gateway and fan them back into the join gateway.
5. Connect everything with sequence flows. Every node except the start has an incoming flow; every node except end events has an outgoing flow.
6. Keep it focused (typically 4-10 nodes). Base BPMN only — no pools, lanes, message flows, or sub-processes.
7. {POSITION_DISCLAIMER}

Node ids are short lowercase slugs ('check_stock') referenced by flows."""

    # ------------------------------------------------------------------
    # Complete system (the primary generation path)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_agentic_bpmn_request(user_request: str, current_model: Dict[str, Any] = None) -> bool:
        """Detect if the user wants an agentic BPMN (with pools/swimlanes)."""
        lower = (user_request or "").lower()
        agentic_keywords = [
            "pool", "swimlane", "participant", "multi-agent", "multiagent",
            "swarm", "agentic bpmn", "agentic process", "orchestrat",
        ]
        if any(kw in lower for kw in agentic_keywords):
            return True
        if isinstance(current_model, dict):
            elements = current_model.get("elements", {})
            if isinstance(elements, dict):
                return any(
                    isinstance(el, dict) and el.get("type") in ("BPMNPool", "BPMNSwimlane")
                    for el in elements.values()
                )
        return False

    def generate_complete_system(
        self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs,
    ) -> Dict[str, Any]:
        logger.info(f"[BPMN] generate_complete_system called with: {user_request!r}")

        if self._is_agentic_bpmn_request(user_request, existing_model):
            return self._generate_agentic_complete_system(user_request)

        system_prompt = self.get_system_prompt()

        reasoning_prompt = (
            "You are a BPMN process-design expert. Think step by step about the "
            "following process request and plan it before producing JSON.\n\n"
            f"User Request: {user_request}\n\n"
            "Analyze:\n"
            "1. What is the trigger (start event)?\n"
            "2. What are the activities (tasks) and their order?\n"
            "3. Where are the decisions (exclusive gateways) and what are the conditions?\n"
            "4. Is any work concurrent (parallel gateways)?\n"
            "5. What are the possible outcomes (end events)?\n\n"
            "Focus on the SEQUENCE FLOWS — they are the most commonly under-specified part."
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

    def _generate_agentic_complete_system(self, user_request: str) -> Dict[str, Any]:
        """Generate a BPMN process with pools and swimlanes (agentic BPMN)."""
        system_prompt = """You are an agentic BPMN modeling expert. Create an
AgenticSwarm BPMN process from the user's request.

DESIGN RULES:
1. Emit pools[].lanes. Never emit pools[].swimlanes.
2. Use pools for distinct participants and lanes for distinct roles inside a pool.
3. Agentic lane roles are exactly: solution, supervision, collaboration, consensus.
4. Preserve lane metadata where relevant: isAgentic, role, trustScore,
multiplicity, agentDiagramRef.
5. Every node in a pool must set poolId. Every node in a lane must set laneId.
owner may match laneId for backend normalization; never use a generated
WME/Apollon element id.
6. Agentic tasks may use isAgentic, reflectionMode, trustScore, and
agentDiagramRef.
7. Agentic gateways must be parallel or inclusive. They may use isAgentic,
gatewayRole, trustScore, and governanceDsl. A governed merge must have
gatewayRole="merging" and a non-empty governanceDsl.
8. Flows contain only source, target, and name. Never emit flowType, agentic
flow fields, collaborationMode, or mergingStrategy.
9. Use one start event and at least one end event. Do not emit positions."""

        reasoning_prompt = (
            "You are an agentic process-design expert. Think step by step about the "
            "following collaboration request and plan it before producing JSON.\n\n"
            f"User Request: {user_request}\n\n"
            "Analyze:\n"
            "1. Who are the participants and which lane role fits each: solution, supervision, collaboration, or consensus?\n"
            "2. How many instances of each participant are needed (multiplicity)?\n"
            "3. What tasks does each participant perform?\n"
            "4. How do they coordinate (what sequence flows cross lanes)?\n"
            "5. What is the trigger and the completion conditions?\n\n"
            "Focus on correct lane ownership — every task must be owned by a swimlane."
        )

        try:
            parsed = self.predict_two_pass_structured(
                user_request=user_request,
                system_prompt=system_prompt,
                reasoning_prompt=reasoning_prompt,
                response_schema=SystemBPMNSpec,
            )
            system_spec = parsed.model_dump()
            return {
                "action": "inject_complete_system",
                "systemSpec": system_spec,
                "diagramType": self.get_diagram_type(),
                "message": self._build_agentic_message(system_spec),
            }
        except LLMPredictionError as exc:
            logger.error(f"[BPMN] _generate_agentic_complete_system LLM FAILED: {exc}")
            return self._error_response("I couldn't generate that agentic process. Please try again.")
        except Exception as exc:
            logger.error(f"[BPMN] _generate_agentic_complete_system FAILED: {exc}", exc_info=True)
            return self.generate_fallback_system()

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

        spec["nodes"] = nodes
        spec["flows"] = flows
        return spec

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
        system_prompt = (
            MODIFY_SYSTEM_PROMPT_AGENTIC_BPMN
            if self._is_agentic_bpmn_request(user_request, current_model)
            else MODIFY_SYSTEM_PROMPT_BPMN
        )

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
        tasks = [n.get("name", "?") for n in nodes if n.get("type") == "task"][:6]
        msg = f"Built the **{name}** process with {len(nodes)} node(s)"
        if tasks:
            msg += f": {', '.join(f'**{t}**' for t in tasks)}"
        if flows:
            msg += f", connected by {len(flows)} sequence flow(s)"
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

    @staticmethod
    def _preview_register_element(
        preview: Dict[str, Any], element: Dict[str, Any], *aliases: Optional[str],
    ) -> Dict[str, Any]:
        for alias in aliases:
            if alias:
                preview[alias] = element
        return preview

    @staticmethod
    def _preview_remove_element(preview: Dict[str, Any], element: Dict[str, Any]) -> Dict[str, Any]:
        keys_to_remove = [key for key, candidate in preview.items() if candidate is element]
        for key in keys_to_remove:
            preview.pop(key, None)
        return preview

    def _apply_preview_mod(self, mod: Dict[str, Any], elements: Dict[str, Any]) -> Dict[str, Any]:
        """Return a preview element map after applying the modification.

        This lets later modifications in the same batch resolve refs to nodes
        added or renamed earlier in the response, while preserving the existing
        guardrail against references to elements that never existed.
        """
        if not isinstance(elements, dict):
            return {}

        preview = dict(elements)
        action = mod.get("action", "")
        target = mod.get("target") or {}
        changes = mod.get("changes") or {}

        if action == "add_task":
            name = target.get("nodeName") or changes.get("name")
            if name:
                element = {"type": "BPMNTask", "name": name, "taskType": changes.get("taskType", "default")}
                preview = self._preview_register_element(
                    preview, element, target.get("nodeId"), target.get("nodeName"), changes.get("name"),
                )
            return preview

        if action == "add_gateway":
            name = target.get("nodeName") or changes.get("name")
            if name:
                element = {
                    "type": "BPMNGateway", "name": name, "gatewayType": changes.get("gatewayType", "exclusive"),
                }
                preview = self._preview_register_element(
                    preview, element, target.get("nodeId"), target.get("nodeName"), changes.get("name"),
                )
            return preview

        if action == "add_event":
            name = target.get("nodeName") or changes.get("name")
            event_kind = changes.get("eventKind", "intermediate")
            if name:
                element = {"type": f"BPMN{event_kind.capitalize()}Event", "name": name}
                preview = self._preview_register_element(
                    preview, element, target.get("nodeId"), target.get("nodeName"), changes.get("name"),
                )
            return preview

        if action == "modify_node":
            ref = target.get("nodeId") or target.get("nodeName")
            element = self._bpmn_resolve(ref, preview)
            if element is None:
                return preview
            new_name = changes.get("name")
            if new_name and new_name != element.get("name"):
                updated = dict(element)
                updated["name"] = new_name

                matched_key = None
                for key, candidate in preview.items():
                    if candidate is element:
                        matched_key = key
                        break

                if matched_key is not None:
                    for key, candidate in list(preview.items()):
                        if candidate is element:
                            preview[key] = updated
                    preview.setdefault(new_name, updated)
                return preview

            return preview

        if action == "remove_element":
            ref = target.get("nodeId") or target.get("nodeName")
            matched_key = None
            matched_element = None
            for key, candidate in preview.items():
                if key == ref:
                    matched_key = key
                    matched_element = candidate
                    break
                if isinstance(candidate, dict) and (candidate.get("name") or "").lower() == (ref or "").lower():
                    matched_key = key
                    matched_element = candidate
                    break

            if matched_key is not None:
                preview = self._preview_remove_element(preview, matched_element)
            return preview

        return preview

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
            preview_elements = dict(elements)
            valid = []
            for mod in mods:
                if self._ref_exists(mod, preview_elements):
                    valid.append(mod)
                    preview_elements = self._apply_preview_mod(mod, preview_elements)
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
