"""Pydantic schemas for ComponentDiagram structured outputs.

Elements: Subsystem (container), Component.
Relationships: ComponentDependency.
Layout is handled on the WME side; the agent emits no positions.
"""
from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

_COMPONENT_STEREOTYPE = Literal["solution", "llm", "db", "rag", "tool", "skill"]
_DEPENDENCY_STEREOTYPE = Literal["uses", "supervises", "collaborates", "revises", "delegates"]

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
    stereotype: _COMPONENT_STEREOTYPE = Field(default="solution", description="'solution' for general components/agents, 'llm' for language models, 'db' for databases, 'rag' for retrieval-augmented components, 'tool' for utilities, 'skill' for capabilities.")


class ComponentDependencySpec(BaseModel):
    source: str = Field(description="Source component/subsystem id.")
    target: str = Field(description="Target component/subsystem id.")
    stereotype: _DEPENDENCY_STEREOTYPE = Field(default="uses", description="'uses' for service/library dependency, 'supervises' for supervision→solution, 'collaborates' for peer, 'revises' for feedback loops, 'delegates' for task delegation.")


class SystemComponentSpec(BaseModel):
    """Schema for a complete ComponentDiagram."""
    systemName: str = Field(default="", description="Descriptive name (e.g. 'Order System Architecture').")
    subsystems: List[ComponentSubsystemSpec] = Field(default_factory=list, description="Subsystem containers grouping related components.")
    components: List[ComponentSpec] = Field(min_length=1, description="Components: software units, services, agents, or models.")
    dependencies: List[ComponentDependencySpec] = Field(default_factory=list, description="Dependencies connecting components/subsystems.")


# -- Modification schemas --

class ComponentModificationTarget(BaseModel):
    elementId: Optional[str] = Field(default=None, description="Apollon element id (exact value from context). Use for unnamed elements.")
    elementName: Optional[str] = Field(default=None, description="Element display name. Case-insensitive lookup.")


class ComponentModificationChanges(BaseModel):
    name: Optional[str] = Field(default=None, max_length=60, description="New name for modify_element, or the name for add_component/add_subsystem.")
    stereotype: Optional[str] = Field(default=None, description="Stereotype for add_component ('solution','llm','db','rag','tool','skill') or add_subsystem ('subsystem').")
    owner: Optional[str] = Field(default=None, description="Parent subsystem name/id for add_component/add_subsystem (null for top-level).")
    source: Optional[str] = Field(default=None, description="Source element name/id for add_dependency/remove_dependency.")
    target: Optional[str] = Field(default=None, description="Target element name/id for add_dependency/remove_dependency.")
    dependencyStereotype: Optional[str] = Field(default=None, description="Dependency stereotype: 'uses','supervises','collaborates','revises','delegates'.")


class ComponentModification(BaseModel):
    action: Literal["add_component","add_subsystem","add_dependency","modify_element","remove_element","remove_dependency"] = Field(description="Action to perform.")
    target: ComponentModificationTarget = Field(description="Identifies the element to act on.")
    changes: Optional[ComponentModificationChanges] = Field(default=None, description="Changes to apply.")


class ComponentModificationResponse(BaseModel):
    modifications: List[ComponentModification] = Field(default_factory=list, description="List of modifications. Empty when elementFound is false.")
    message: str = Field(description="Human-readable summary. When elementFound is false, explain what was not found.")
    elementFound: bool = Field(default=True, description="False when the referenced element cannot be found.")
