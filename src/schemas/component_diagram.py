"""Pydantic schemas for ComponentDiagram structured outputs.

Elements: Subsystem (container), Component.
Relationships: ComponentDependency.
Layout is handled on the WME side; the agent emits no positions.

The stereotype vocabularies mirror the editor's
``packages/editor/src/main/packages/common/agentic/agentic-tokens.ts``
(``AGENT_CATEGORY_TOKENS``, ``CAPABILITY_TOKENS``,
``AGENTIC_EDGE_KIND_TOKENS``), which in turn mirror the BESSER
``uml_component/agentic.py`` enums. Keep them in sync.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# AgentCategory — a component that IS an agent.
AGENT_CATEGORY_TOKENS = ("solution", "supervision")
# Capabilities an agent has or uses — NOT agents themselves.
CAPABILITY_TOKENS = ("skill", "tool", "llm", "db", "rag")
COMPONENT_STEREOTYPE_TOKENS = AGENT_CATEGORY_TOKENS + CAPABILITY_TOKENS
# AgenticEdgeKind — agent→agent, agent→capability, capability.
AGENTIC_EDGE_KIND_TOKENS = (
    "delegates", "supervises", "revises", "collaborates",
    "has", "uses", "granted",
    "implements",
)

_COMPONENT_STEREOTYPE = Literal["solution", "supervision", "skill", "tool", "llm", "db", "rag"]
_DEPENDENCY_STEREOTYPE = Literal[
    "delegates", "supervises", "revises", "collaborates",
    "has", "uses", "granted",
    "implements",
]
_MODIFICATION_STEREOTYPE = Literal[
    "solution", "supervision", "skill", "tool", "llm", "db", "rag", "subsystem",
]

_COMPONENT_STEREOTYPE_DESCRIPTION = (
    "Agents: 'solution' (an agent that does the work) or 'supervision' (an agent "
    "that oversees other agents). Capabilities (not agents): 'llm' (language "
    "model), 'db' (database/storage), 'rag' (retrieval-augmented knowledge "
    "source), 'tool' (callable utility), 'skill' (reusable capability)."
)
_DEPENDENCY_STEREOTYPE_DESCRIPTION = (
    "Agent→agent: 'delegates' (hands off a task), 'supervises' (supervision agent "
    "over a solution agent), 'revises' (feedback/revision loop), 'collaborates' "
    "(peer exchange). Agent→capability: 'has' (owns the capability), 'uses' "
    "(calls it), 'granted' (is given access to it). Capability: 'implements' "
    "(a capability realizes another)."
)


# -- Generation schemas --

class ComponentSubsystemSpec(BaseModel):
    id: str = Field(min_length=1, max_length=40, description="Short unique slug (e.g. 'backend'). Lowercase, no spaces.")
    name: str = Field(max_length=60, description="Human-readable name (e.g. 'Backend').")
    owner: Optional[str] = Field(default=None, description="Id of a parent subsystem if nested, or null for top-level.")
    stereotype: Literal["subsystem"] = Field(default="subsystem", description="Always 'subsystem'.")


class ComponentSpec(BaseModel):
    id: str = Field(min_length=1, max_length=40, description="Short unique slug (e.g. 'user_service').")
    name: str = Field(max_length=60, description="Human-readable component name (e.g. 'UserService').")
    owner: Optional[str] = Field(default=None, description="Id of the subsystem that contains this component, or null.")
    stereotype: _COMPONENT_STEREOTYPE = Field(default="solution", description=_COMPONENT_STEREOTYPE_DESCRIPTION)


class ComponentDependencySpec(BaseModel):
    source: str = Field(description="Source component/subsystem id.")
    target: str = Field(description="Target component/subsystem id.")
    stereotype: _DEPENDENCY_STEREOTYPE = Field(default="uses", description=_DEPENDENCY_STEREOTYPE_DESCRIPTION)


class SystemComponentSpec(BaseModel):
    """Schema for a complete ComponentDiagram."""
    systemName: str = Field(default="", description="Descriptive name (e.g. 'Order System Architecture').")
    subsystems: List[ComponentSubsystemSpec] = Field(default_factory=list, description="Subsystem containers grouping related components.")
    components: List[ComponentSpec] = Field(min_length=1, description="Components: agents, services, models, and their capabilities.")
    dependencies: List[ComponentDependencySpec] = Field(default_factory=list, description="Dependencies connecting components/subsystems.")


# -- Modification schemas --

class ComponentModificationTarget(BaseModel):
    elementId: Optional[str] = Field(default=None, description="Apollon element id (exact [id] from the context). Use for unnamed elements.")
    elementName: Optional[str] = Field(default=None, description="Existing element display name (case-insensitive), or the name of the element to add.")


class ComponentModificationChanges(BaseModel):
    name: Optional[str] = Field(default=None, max_length=60, description="New name for modify_element, or the name for add_component/add_subsystem.")
    stereotype: Optional[_MODIFICATION_STEREOTYPE] = Field(
        default=None,
        description="add_component / modify_element: a component stereotype (" + _COMPONENT_STEREOTYPE_DESCRIPTION
        + ") — or 'subsystem' for add_subsystem.",
    )
    owner: Optional[str] = Field(default=None, description="Parent subsystem name/id for add_component/add_subsystem (null for top-level).")
    source: Optional[str] = Field(default=None, description="Source element name/id for add_dependency/remove_dependency.")
    target: Optional[str] = Field(default=None, description="Target element name/id for add_dependency/remove_dependency.")
    dependencyStereotype: Optional[_DEPENDENCY_STEREOTYPE] = Field(
        default=None, description="add_dependency: " + _DEPENDENCY_STEREOTYPE_DESCRIPTION,
    )


class ComponentModification(BaseModel):
    action: Literal[
        "add_component", "add_subsystem", "add_dependency",
        "modify_element", "remove_element", "remove_dependency",
    ] = Field(description="Action to perform.")
    target: ComponentModificationTarget = Field(description="Identifies the element to act on.")
    changes: Optional[ComponentModificationChanges] = Field(default=None, description="Changes to apply.")


class ComponentModificationResponse(BaseModel):
    modifications: List[ComponentModification] = Field(default_factory=list, description="List of modifications. Empty when elementFound is false.")
    message: str = Field(description="Human-readable summary. When elementFound is false, explain what was not found.")
    elementFound: bool = Field(default=True, description="False when the referenced element cannot be found.")
