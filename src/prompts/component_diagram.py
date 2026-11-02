"""Component diagram prompts: design rules, reasoning and modification."""

from schemas.component_diagram import COMPONENT_STEREOTYPE_DESCRIPTION, DEPENDENCY_STEREOTYPE_DESCRIPTION

from .fragments import ARCHITECTURE_MODIFY_SHARED_RULES, POSITION_DISCLAIMER


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


SYSTEM_PROMPT = f"""You are a software architecture expert. Create a UML Component Diagram from the user's request.

DESIGN RULES:
1. Use Subsystems to group related components (e.g., 'Backend', 'Frontend', 'AI Layer', 'Data').
2. Use Components for software units: agents, services, models, databases, tools. Clear noun names ('UserService', 'LLM', 'ProductDB').
3. Component stereotypes — {COMPONENT_STEREOTYPE_DESCRIPTION}
4. Use ComponentDependency for interactions. Stereotypes — {DEPENDENCY_STEREOTYPE_DESCRIPTION}
5. Components can be owned by subsystems (set owner to the subsystem id). Subsystems can be top-level (owner: null).
6. Keep it focused (typically 3-10 components).
7. {POSITION_DISCLAIMER}

Component ids are short lowercase slugs (e.g. 'user_service', 'llm_core') referenced by dependencies."""


def reasoning_prompt(request: str) -> str:
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
