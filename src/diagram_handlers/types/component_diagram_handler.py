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

from ..core.architecture_handler import ARCHITECTURE_MODIFY_SHARED_RULES, ArchitectureDiagramHandler
from ..core.prompt_fragments import POSITION_DISCLAIMER
from schemas import ComponentModificationResponse, SystemComponentSpec
from schemas.component_diagram import COMPONENT_STEREOTYPE_DESCRIPTION, DEPENDENCY_STEREOTYPE_DESCRIPTION

logger = logging.getLogger(__name__)


MODIFY_SYSTEM_PROMPT_COMPONENT = f"""You are a component diagram modeling expert. The user wants to modify a UML Component Diagram.

READING THE CONTEXT:
Each element appears as: [id] Name (Subsystem:stereotype) or [id] Name (Component:stereotype [in SubsystemName])
Each dependency appears as: Dependency: [src-id] Name --stereotype--> [tgt-id] Name

MODIFICATION RULES:
1. Actions: "add_component", "add_subsystem", "add_dependency", "modify_element", "remove_element", "remove_dependency"
2. add_component: set target.elementName to the new component's name. Optional changes.stereotype and changes.owner (the containing subsystem's name or [id]).
3. add_subsystem: set target.elementName to the new subsystem's name. Optional changes.owner for nesting.
4. add_dependency: optional changes.dependencyStereotype.
5. modify_element: put the new name in changes.name and/or the new stereotype in changes.stereotype.
6. Component stereotypes — {COMPONENT_STEREOTYPE_DESCRIPTION}
7. Dependency stereotypes — {DEPENDENCY_STEREOTYPE_DESCRIPTION}

{ARCHITECTURE_MODIFY_SHARED_RULES}"""


class ComponentDiagramHandler(ArchitectureDiagramHandler):
    """Handler for UML Component Diagram generation and modification."""

    _LABEL = "component diagram"
    _SYSTEM_SCHEMA = SystemComponentSpec
    _MODIFICATION_SCHEMA = ComponentModificationResponse
    _MODIFY_SYSTEM_PROMPT = MODIFY_SYSTEM_PROMPT_COMPONENT
    _MODIFY_EXAMPLES = "*'add a UserService component to the Backend subsystem'* or *'rename LLM to GPT-4o'*"
    _REF_ADD_ACTIONS = {"add_component": "Component", "add_subsystem": "Subsystem"}
    _REF_NAMED_TYPES = frozenset({"Component", "Subsystem"})

    def get_diagram_type(self) -> str:
        return "ComponentDiagram"

    def get_system_prompt(self) -> str:
        return f"""You are a software architecture expert. Create a UML Component Diagram from the user's request.

DESIGN RULES:
1. Use Subsystems to group related components (e.g., 'Backend', 'Frontend', 'AI Layer', 'Data').
2. Use Components for software units: agents, services, models, databases, tools. Clear noun names ('UserService', 'LLM', 'ProductDB').
3. Component stereotypes — {COMPONENT_STEREOTYPE_DESCRIPTION}
4. Use ComponentDependency for interactions. Stereotypes — {DEPENDENCY_STEREOTYPE_DESCRIPTION}
5. Components can be owned by subsystems (set owner to the subsystem id). Subsystems can be top-level (owner: null).
6. Keep it focused (typically 3-10 components).
7. {POSITION_DISCLAIMER}

Component ids are short lowercase slugs (e.g. 'user_service', 'llm_core') referenced by dependencies."""

    def _reasoning_prompt(self, request: str) -> str:
        return (
            "You are a software architecture expert. Think step by step about the "
            "following component architecture request and plan it before producing JSON.\n\n"
            f"User Request: {request}\n\n"
            "Analyze:\n"
            "1. What subsystems are needed to group related concerns?\n"
            "2. What components go in each subsystem? Which are agents (solution / "
            "supervision) and which are capabilities (llm, db, rag, tool, skill)?\n"
            "3. What dependencies exist between components, and which kind is each "
            "(delegates, supervises, revises, collaborates, has, uses, granted, implements)?\n"
            "4. Are there any cross-subsystem dependencies to model?\n\n"
            "Focus on clear component boundaries and meaningful dependency stereotypes."
        )

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
