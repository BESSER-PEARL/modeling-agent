"""Pydantic schemas for DeploymentDiagram structured outputs.

Elements: DeploymentNode (execution environment), DeploymentArtifact (physical artifact),
DeploymentComponent (logical component).
Relationships: DeploymentDependency.
Layout is handled on the WME side; the agent emits no positions.

The editor's deployment node stereotype is free text (default ``node``); the
values below are the agent's preset vocabulary. A DeploymentComponent uses the
shared UML component popup, whose presets are the agentic component tokens, so
its stereotype reuses ``COMPONENT_STEREOTYPE_TOKENS``.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

NODE_STEREOTYPE_TOKENS = ("node", "device", "cloud", "server")

_NODE_STEREOTYPE = Literal["node", "device", "cloud", "server"]
_DEPLOY_COMPONENT_STEREOTYPE = Literal["solution", "supervision", "skill", "tool", "llm", "db", "rag"]
_MODIFICATION_STEREOTYPE = Literal[
    "node", "device", "cloud", "server",
    "solution", "supervision", "skill", "tool", "llm", "db", "rag",
]

_NODE_STEREOTYPE_DESCRIPTION = (
    "'node' (generic execution environment), 'device' (physical hardware), "
    "'cloud' (cloud environment / managed service), 'server' (a server or VM)."
)
_DEPLOY_COMPONENT_STEREOTYPE_DESCRIPTION = (
    "Agents: 'solution' or 'supervision'. Capabilities: 'llm', 'db', 'rag', "
    "'tool', 'skill'. Use 'solution' for an ordinary application component."
)


# -- Generation schemas --

class DeploymentNodeSpec(BaseModel):
    id: str = Field(min_length=1, max_length=40, description="Short unique slug (e.g. 'prod_server'). Lowercase, no spaces.")
    name: str = Field(max_length=60, description="Human-readable name (e.g. 'Production Server').")
    stereotype: _NODE_STEREOTYPE = Field(default="node", description=_NODE_STEREOTYPE_DESCRIPTION)


class DeploymentArtifactSpec(BaseModel):
    id: str = Field(min_length=1, max_length=40, description="Short unique slug (e.g. 'webapp_artifact').")
    name: str = Field(max_length=60, description="Artifact name (e.g. 'WebApp', 'APIServer').")
    owner: str = Field(description="Id of the DeploymentNode that hosts this artifact.")


class DeploymentComponentSpec(BaseModel):
    id: str = Field(min_length=1, max_length=40, description="Short unique slug (e.g. 'web_comp').")
    name: str = Field(max_length=60, description="Logical component name (e.g. 'WebApp').")
    stereotype: _DEPLOY_COMPONENT_STEREOTYPE = Field(default="solution", description=_DEPLOY_COMPONENT_STEREOTYPE_DESCRIPTION)
    manifestedBy: Optional[str] = Field(default=None, description="Id of the DeploymentArtifact that physically contains this component.")


class DeploymentDependencySpec(BaseModel):
    source: str = Field(description="Source element id (artifact or component).")
    target: str = Field(description="Target element id (artifact or component).")
    name: Optional[str] = Field(default="", max_length=40, description="Optional dependency label (e.g. 'HTTPS', 'JDBC').")


class SystemDeploymentSpec(BaseModel):
    """Schema for a complete DeploymentDiagram."""
    systemName: str = Field(default="", description="Descriptive name (e.g. 'Production Deployment').")
    nodes: List[DeploymentNodeSpec] = Field(min_length=1, description="Execution environment nodes (servers, cloud instances, Docker hosts).")
    artifacts: List[DeploymentArtifactSpec] = Field(default_factory=list, description="Physical deployment artifacts hosted inside nodes.")
    deployComponents: List[DeploymentComponentSpec] = Field(default_factory=list, description="Logical components. Each corresponds to an artifact via manifestedBy.")
    dependencies: List[DeploymentDependencySpec] = Field(default_factory=list, description="Dependencies between artifacts and components.")


# -- Modification schemas --

class DeploymentModificationTarget(BaseModel):
    elementId: Optional[str] = Field(default=None, description="Apollon element id (exact [id] from the context). Use for unnamed elements.")
    elementName: Optional[str] = Field(default=None, description="Existing element display name (case-insensitive), or the name of the element to add.")


class DeploymentModificationChanges(BaseModel):
    name: Optional[str] = Field(default=None, max_length=60, description="New name for modify_element, or the name for add operations.")
    stereotype: Optional[_MODIFICATION_STEREOTYPE] = Field(
        default=None,
        description=(
            "add_node / modify_element on a node: " + _NODE_STEREOTYPE_DESCRIPTION
            + " add_component / modify_element on a component: " + _DEPLOY_COMPONENT_STEREOTYPE_DESCRIPTION
        ),
    )
    owner: Optional[str] = Field(default=None, description="add_artifact: the existing node name/id that hosts the artifact.")
    source: Optional[str] = Field(default=None, description="Source element name/id for add_dependency/remove_dependency.")
    target: Optional[str] = Field(default=None, description="Target element name/id for add_dependency/remove_dependency.")
    label: Optional[str] = Field(default=None, max_length=40, description="Optional label for add_dependency.")


class DeploymentModification(BaseModel):
    action: Literal[
        "add_node", "add_artifact", "add_component", "add_dependency",
        "modify_element", "remove_element", "remove_dependency",
    ] = Field(description="Action to perform.")
    target: DeploymentModificationTarget = Field(description="Identifies the element to act on.")
    changes: Optional[DeploymentModificationChanges] = Field(default=None, description="Changes to apply.")


class DeploymentModificationResponse(BaseModel):
    modifications: List[DeploymentModification] = Field(default_factory=list, description="List of modifications. Empty when elementFound is false.")
    message: str = Field(description="Human-readable summary. When elementFound is false, explain what was not found.")
    elementFound: bool = Field(default=True, description="False when the referenced element cannot be found.")
