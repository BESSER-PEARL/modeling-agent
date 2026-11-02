"""
Component Diagram Handler
Handles generation and modification of UML Component Diagrams.

Elements: Subsystem (container), Component.
Relationships: ComponentDependency.
Stereotypes follow the editor's agentic-tokens.ts vocabulary (see
``schemas.component_diagram``). Positions are NOT generated here — the WME
converter places the elements.
"""

import logging
from typing import Any, Dict

from ..core.architecture_handler import ArchitectureDiagramHandler
from prompts import component_diagram as component_prompts
from schemas import ComponentModificationResponse, SystemComponentSpec
from protocol.actions import inject_complete_system

logger = logging.getLogger(__name__)


class ComponentDiagramHandler(ArchitectureDiagramHandler):
    """Handler for UML Component Diagram generation and modification."""

    _LABEL = "component diagram"
    _SYSTEM_FAILURE_MESSAGE = "I couldn't generate that component diagram. Please try again or rephrase your request."
    _SYSTEM_SCHEMA = SystemComponentSpec
    _MODIFICATION_SCHEMA = ComponentModificationResponse
    _MODIFY_SYSTEM_PROMPT = component_prompts.MODIFY_SYSTEM_PROMPT_COMPONENT
    _MODIFY_EXAMPLES = "*'add a UserService component to the Backend subsystem'* or *'rename LLM to GPT-4o'*"
    _REF_ADD_ACTIONS = {"add_component": "Component", "add_subsystem": "Subsystem"}
    _REF_NAMED_TYPES = frozenset({"Component", "Subsystem"})

    def get_diagram_type(self) -> str:
        return "ComponentDiagram"

    def get_system_prompt(self) -> str:
        return component_prompts.SYSTEM_PROMPT

    def _reasoning_prompt(self, request: str) -> str:
        return component_prompts.reasoning_prompt(request)

    # ------------------------------------------------------------------
    # Single element + fallbacks
    # ------------------------------------------------------------------

    def _build_single_element(
        self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs,
    ) -> Dict[str, Any]:
        name = self.extract_name_from_request(user_request, "Component")
        return inject_complete_system(
            {
                "systemName": name,
                "subsystems": [],
                "components": [{"id": "comp1", "name": name, "owner": None, "stereotype": "solution"}],
                "dependencies": [],
            },
            self.get_diagram_type(),
            f"Created a starter **{name}** component. Describe the full architecture and I'll build it out!",
        )

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
        return inject_complete_system(
            fallback,
            self.get_diagram_type(),
            (
                "I created a starter component diagram. Describe your architecture "
                "(e.g. *'a microservices system with user service, order service, and a database'*) "
                "and I'll build a richer model!"
            ),
        )

    def _build_system_message(self, spec: Dict[str, Any]) -> str:
        name = spec.get("systemName") or "architecture"
        components = spec.get("components", [])
        subsystems = spec.get("subsystems", [])
        comp_names = [c.get("name", "?") for c in components][:5]
        msg = (
            f"Built the **{name}** component diagram with {len(subsystems)} subsystem(s) "
            f"and {len(components)} component(s)"
        )
        if comp_names:
            msg += f": {', '.join(f'**{c}**' for c in comp_names)}"
        msg += ". Ask me to add components, define subsystems, or adjust dependencies!"
        return msg
