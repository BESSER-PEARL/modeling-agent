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
