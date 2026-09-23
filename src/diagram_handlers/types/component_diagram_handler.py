"""
Component Diagram Handler
Handles generation and modification of UML Component Diagrams.

Elements: Subsystem (container), Component.
Relationships: ComponentDependency.
Positions are NOT generated here — the WME layout engine handles placement.
"""

import logging
from typing import Any, Dict, List, Optional

from ..core.base_handler import BaseDiagramHandler, LLMPredictionError
from ..core.prompt_fragments import EXACT_NAMES_RULE, POSITION_DISCLAIMER, REMOVE_ELEMENT_RULE
from schemas import SystemComponentSpec, ComponentModificationResponse
from utilities.model_context import detailed_model_summary

logger = logging.getLogger(__name__)


MODIFY_SYSTEM_PROMPT_COMPONENT = """You are a component diagram modeling expert. The user wants to modify a UML Component Diagram.

READING THE CONTEXT:
Each element appears as: [id] Name (type/stereotype)
Each dependency appears as: Dependency: [src-id] Name --stereotype--> [tgt-id] Name

MODIFICATION RULES:
1. Actions: "add_component", "add_subsystem", "add_dependency", "modify_element", "remove_element", "remove_dependency"
2. add_component: set target.elementName to the component name. Optional changes.stereotype and changes.owner (subsystem name/id).
3. add_subsystem: set target.elementName to the subsystem name. Optional changes.owner for nesting.
4. add_dependency: set changes.source and changes.target to element name or id. Optional changes.dependencyStereotype.
5. modify_element: set target.elementId or target.elementName. Put new name in changes.name and/or new stereotype in changes.stereotype.
6. remove_element: set target.elementId or target.elementName. Connected dependencies are removed automatically.
7. remove_dependency: set changes.source and changes.target to endpoint names/ids.

When the user asks to remove or modify an element, verify it exists first. If not found:
- Set elementFound: false, modifications: [], and explain in message.

If the user says 'undo', reply with modifications: [], elementFound: false, message: 'To undo, use Ctrl+Z or the undo button in the editor toolbar.'"""


class ComponentDiagramHandler(BaseDiagramHandler):
    """Handler for UML Component Diagram generation and modification."""

    def get_diagram_type(self) -> str:
        return "ComponentDiagram"

    def get_system_prompt(self) -> str:
        return f"""You are a software architecture expert. Create a UML Component Diagram from the user's request.

DESIGN RULES:
1. Use Subsystems to group related components (e.g., 'Backend', 'Frontend', 'AI Layer', 'Database').
2. Use Components for software units: services, agents, models, databases, tools. Clear noun names ('UserService', 'LLM', 'ProductDB').
3. Component stereotypes: 'solution' (general component/agent), 'llm' (language model), 'db' (database/storage), 'rag' (retrieval-augmented), 'tool' (utility/function), 'skill' (capability).
4. Use ComponentDependency to show interactions. Stereotypes: 'uses' (service call/library use), 'supervises' (supervision->solution), 'collaborates' (peer exchange), 'revises' (feedback/revision loop), 'delegates' (task handoff).
5. Components can be owned by subsystems (set owner to subsystem id). Subsystems can be top-level (owner: null).
6. Keep focused (typically 3-10 components). Do NOT add positions — the editor handles layout.

Component ids are short lowercase slugs (e.g. 'user_service', 'llm_core') referenced by dependencies."""

    # ------------------------------------------------------------------
    # Complete system (the primary generation path)
    # ------------------------------------------------------------------

    def generate_complete_system(
        self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs,
    ) -> Dict[str, Any]:
        system_prompt = self.get_system_prompt()
        logger.info(f"[ComponentDiagram] generate_complete_system called with: {user_request!r}")

        reasoning_prompt = (
            "You are a software architecture expert. Think step by step about the "
            "following component architecture request and plan it before producing JSON.\n\n"
            f"User Request: {user_request}\n\n"
            "Analyze:\n"
            "1. What subsystems are needed to group related concerns?\n"
            "2. What components go in each subsystem (services, agents, models, DBs)?\n"
            "3. What dependencies exist between components?\n"
            "4. What stereotypes best describe each component and dependency?\n"
            "5. Are there any cross-subsystem dependencies to model?\n\n"
            "Focus on clear component boundaries and meaningful dependency stereotypes."
        )

        try:
            parsed = self.predict_two_pass_structured(
                user_request=user_request,
                system_prompt=system_prompt,
                reasoning_prompt=reasoning_prompt,
                response_schema=SystemComponentSpec,
            )
            system_spec = parsed.model_dump()

            return {
                "action": "inject_complete_system",
                "systemSpec": system_spec,
                "diagramType": self.get_diagram_type(),
                "message": self._build_system_message(system_spec),
            }

        except LLMPredictionError as exc:
            logger.error(f"[ComponentDiagram] generate_complete_system LLM FAILED: {exc}")
            return self._error_response(
                "I couldn't generate that component diagram. Please try again or rephrase your request.",
                code="llm_failure",
            )
        except Exception as exc:
            logger.error(f"[ComponentDiagram] generate_complete_system FAILED: {exc}", exc_info=True)
            return self.generate_fallback_system()

    # ------------------------------------------------------------------
    # Modification
    # ------------------------------------------------------------------

    def generate_modification(
        self, user_request: str, current_model: Dict[str, Any] = None, **kwargs,
    ) -> Dict[str, Any]:
        system_prompt = MODIFY_SYSTEM_PROMPT_COMPONENT

        # Store elements on the instance for ref validation
        self._elements: Dict[str, Any] = {}
        if current_model and isinstance(current_model, dict):
            raw = current_model.get("elements")
            if isinstance(raw, dict):
                self._elements = raw

        context_block = ""
        if current_model and isinstance(current_model, dict):
            summary = detailed_model_summary(current_model, self.get_diagram_type())
            if summary:
                context_block = f"\n\n{summary}"

        user_prompt = f"Modify the {self.get_diagram_type()} diagram: {user_request}{context_block}"
        logger.info(f"[ComponentDiagram] generate_modification called with: {user_request!r}")

        try:
            result = self._execute_modification(
                user_prompt, system_prompt, ComponentModificationResponse,
            )
            return self._validate_mod_refs(result)
        except LLMPredictionError as exc:
            logger.error(f"[ComponentDiagram] generate_modification LLM FAILED: {exc}")
            return self._error_response(
                "I couldn't process that modification. Please try again or rephrase your request.",
            )
        except Exception as exc:
            logger.error(f"[ComponentDiagram] generate_modification FAILED: {exc}", exc_info=True)
            return {
                "action": "assistant_message",
                "message": (
                    "I couldn't apply that modification automatically. Could you rephrase it? "
                    "For example: *'add a UserService component to the Backend subsystem'* or "
                    "*'rename LLM to GPT-4o'*."
                ),
            }

    # ------------------------------------------------------------------
    # Single element + fallbacks
    # ------------------------------------------------------------------

    def generate_single_element(
        self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs,
    ) -> Dict[str, Any]:
        name = self.extract_name_from_request(user_request, "Component")
        return {
            "action": "inject_complete_system",
            "systemSpec": {
                "systemName": name,
                "subsystems": [],
                "components": [{"id": "comp1", "name": name, "owner": None, "stereotype": "solution"}],
                "dependencies": [],
            },
            "diagramType": self.get_diagram_type(),
            "message": f"Created a starter **{name}** component. Describe the full architecture and I'll build it out!",
        }

    def generate_fallback_element(self, request: str) -> Dict[str, Any]:
        return self.generate_single_element(request)

    def generate_fallback_system(self) -> Dict[str, Any]:
        fallback = {
            "systemName": "BasicSystem",
            "subsystems": [
                {"id": "backend", "name": "Backend", "owner": None, "stereotype": "subsystem"},
                {"id": "frontend", "name": "Frontend", "owner": None, "stereotype": "subsystem"},
            ],
            "components": [
                {"id": "api", "name": "API", "owner": "backend", "stereotype": "solution"},
                {"id": "ui", "name": "UI", "owner": "frontend", "stereotype": "solution"},
            ],
            "dependencies": [
                {"source": "ui", "target": "api", "stereotype": "uses"},
            ],
        }
        return {
            "action": "inject_complete_system",
            "systemSpec": fallback,
            "diagramType": self.get_diagram_type(),
            "message": (
                "I created a starter component diagram. Describe your architecture "
                "(e.g. *'a microservices system with user service, order service, and a database'*) "
                "and I'll build a richer model!"
            ),
        }

    # ------------------------------------------------------------------
    # Message builder
    # ------------------------------------------------------------------

    def _build_system_message(self, spec: Dict[str, Any]) -> str:
        name = spec.get("systemName") or "architecture"
        components = spec.get("components", [])
        subsystems = spec.get("subsystems", [])
        comp_names = [c.get("name", "?") for c in components][:5]
        msg = f"Built the **{name}** component diagram with {len(subsystems)} subsystem(s) and {len(components)} component(s)"
        if comp_names:
            msg += f": {', '.join(f'**{c}**' for c in comp_names)}"
        msg += ". Ask me to add components, define subsystems, or adjust dependencies!"
        return msg

    # ------------------------------------------------------------------
    # Element resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_element(ref: Optional[str], elements: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Look up a Component/Subsystem element by id (exact key) then by name (case-insensitive)."""
        if not ref or not isinstance(elements, dict):
            return None
        el = elements.get(ref)
        if isinstance(el, dict):
            return el
        lower = ref.lower()
        for el in elements.values():
            if isinstance(el, dict) and el.get("type") in ("Component", "Subsystem"):
                if (el.get("name") or "").lower() == lower:
                    return el
        return None

    # ------------------------------------------------------------------
    # Server-side ref guardrail
    # ------------------------------------------------------------------

    def _ref_exists(self, mod: Dict[str, Any], elements: Dict[str, Any]) -> bool:
        """Return True if every element ref in this modification exists in the model."""
        action = mod.get("action", "")
        if action in ("modify_element", "remove_element"):
            target = mod.get("target") or {}
            ref = target.get("elementId") or target.get("elementName")
            return ref is None or self._resolve_element(ref, elements) is not None
        if action in ("add_dependency", "remove_dependency"):
            changes = mod.get("changes") or {}
            src = changes.get("source")
            tgt = changes.get("target")
            src_ok = src is None or self._resolve_element(src, elements) is not None
            tgt_ok = tgt is None or self._resolve_element(tgt, elements) is not None
            return src_ok and tgt_ok
        return True

    def _validate_mod_refs(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Drop modifications whose element refs cannot be resolved in the current model."""
        elements = self._elements
        if not elements or result.get("action") != "modify_model":
            return result

        if "modifications" in result:
            mods = result["modifications"]
            valid = [m for m in mods if self._ref_exists(m, elements)]
            dropped = len(mods) - len(valid)
            if dropped:
                logger.info(f"[ComponentDiagram] Dropped {dropped} modification(s) with unresolved element ref(s)")
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
                logger.info("[ComponentDiagram] Dropped modification with unresolved element ref")
                return {
                    "action": "assistant_message",
                    "message": (
                        "I couldn't find that element in the current diagram. "
                        "Please check the name and try again."
                    ),
                }

        return result
