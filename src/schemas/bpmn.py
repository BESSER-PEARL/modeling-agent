"""Pydantic schemas for BPMN structured outputs.

Field descriptions are used by OpenAI Structured Outputs to guide generation.
Base BPMN plus collaboration diagrams — start/end events, tasks, gateways,
sequence flows, and optional pools/lanes for multi-participant processes.
No other agentic concepts (roles, governance, collaboration, trust).

Layout is handled on the WME side; the agent emits no positions. Message vs.
sequence flow type is also derived on the WME side from pool membership, not
emitted by the agent.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

_TASK_TYPE = Literal[
    "default", "user", "service", "send", "receive",
    "manual", "business-rule", "script",
]
_GATEWAY_TYPE = Literal["exclusive", "parallel", "inclusive", "event-based", "complex"]

_AGENT_ROLE = Literal[
    "solution", "supervision", "collaboration", "consensus",
]
_REFLECTION_MODE = Literal["none", "self", "cross", "human"]
_GATEWAY_ROLE = Literal["diverging", "merging"]


# -- Generation schemas --

class BPMNNodeSpec(BaseModel):
    id: str = Field(
        min_length=1,
        max_length=40,
        description=(
            "Short unique slug identifying this node within the process "
            "(e.g. 'check_stock'). Referenced by flows. Lowercase, no spaces."
        ),
    )
    name: str = Field(
        default="",
        max_length=60,
        description=(
            "Human-readable label. Verb phrase for tasks ('Check Inventory'), "
            "a question for gateways ('In stock?'), a short event name for "
            "events ('Order received'). May be empty for a gateway."
        ),
    )
    type: Literal["startEvent", "endEvent", "intermediateEvent", "task", "gateway"] = Field(
        description="BPMN flow-node kind."
    )
    taskType: Optional[_TASK_TYPE] = Field(default="default")
    gatewayType: Optional[_GATEWAY_TYPE] = Field(default="exclusive")

    # Keep upstream-main placement fields.
    poolId: Optional[str] = Field(
        default=None,
        description="Pool id from pools[].id, or null for a flat process.",
    )
    laneId: Optional[str] = Field(
        default=None,
        description="Lane id from the selected pool's lanes[].id, or null.",
    )
    owner: Optional[str] = Field(
        default=None,
        description=(
            "Lane token used by backend normalization. Never emit a generated "
            "WME/Apollon element id."
        ),
    )

    # Agentic task and gateway attributes. They are meaningful only for their
    # respective node types, but one shared schema avoids two incompatible
    # complete-system payloads.
    isAgentic: Optional[bool] = Field(default=None)
    reflectionMode: Optional[_REFLECTION_MODE] = Field(default=None)
    trustScore: Optional[int] = Field(default=None, ge=0, le=100)
    agentDiagramRef: Optional[str] = Field(default=None)
    gatewayRole: Optional[_GATEWAY_ROLE] = Field(default=None)
    governanceDsl: Optional[str] = Field(default=None)

class BPMNLaneSpec(BaseModel):
    id: str = Field(
        min_length=1,
        max_length=40,
        description=(
            "Short unique slug identifying this lane/role within its pool "
            "(e.g. 'reviewer'). Referenced by node laneId. Lowercase, no spaces."
        ),
    )
    name: str = Field(
        default="",
        max_length=60,
        description="Role/participant display name (e.g. 'Reviewer Agent').",
    )
    isAgentic: bool = Field(
        default=False,
        description="Whether this lane represents an AgenticSwarm role.",
    )
    role: Optional[_AGENT_ROLE] = Field(
        default=None,
        description="AgenticSwarm role when isAgentic is true.",
    )
    trustScore: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Optional trust score for an AgenticSwarm lane.",
    )
    multiplicity: Optional[str] = Field(
        default=None,
        max_length=60,
        description="Optional agent multiplicity, for example '1..*'.",
    )
    agentDiagramRef: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional reference to an Agent diagram.",
    )


class BPMNPoolSpec(BaseModel):
    id: str = Field(
        min_length=1,
        max_length=40,
        description=(
            "Short unique slug identifying this participant/pool "
            "(e.g. 'customer'). Lowercase, no spaces."
        ),
    )
    name: str = Field(
        default="",
        max_length=80,
        description="Pool/participant display name (e.g. 'Customer').",
    )
    lanes: List[BPMNLaneSpec] = Field(
        default_factory=list,
        description=(
            "Optional lanes/roles inside this pool. Nodes refer to them via laneId. "
            "Use `lanes`, never `swimlanes`."
        ),
    )


class BPMNFlowSpec(BaseModel):
    source: str = Field(description="Source node id.")
    target: str = Field(description="Target node id.")
    name: Optional[str] = Field(default="", max_length=40)


class SystemBPMNSpec(BaseModel):
    systemName: str = Field(
        default="BPMN Process",
        max_length=100,
        description="Concise process title.",
    )
    nodes: List[BPMNNodeSpec] = Field(
        default_factory=list,
        description="All activities, events, and gateways in the process.",
    )
    flows: List[BPMNFlowSpec] = Field(
        default_factory=list,
        description=(
            "Flows connecting the nodes by id. Every node except the start has "
            "an incoming flow; every node except end events has an outgoing "
            "flow. A flow between nodes in different pools is a message flow; "
            "the WME derives this automatically from poolId, do not set a flow "
            "type yourself."
        ),
    )
    pools: List[BPMNPoolSpec] = Field(
        default_factory=list,
        description=(
            "Participants/organizations — only when the request involves 2+ "
            "distinct actors communicating (e.g. customer/vendor, system A/system B) "
            "or explicit roles within one organization. Each node with a non-null "
            "poolId must reference one of these pool ids. Leave empty for a "
            "single-actor flat process (the common case)."
        ),
    )


# -- Modification schemas --

class BPMNModificationTarget(BaseModel):
    nodeId: Optional[str] = Field(
        default=None,
        description=(
            "Apollon element id — the exact value inside [id] shown in the process context. "
            "Required for UNNAMED nodes (shown as '[id] (type)' with no name). "
            "Resolved by the WME before falling back to nodeName."
        ),
    )
    nodeName: Optional[str] = Field(
        default=None,
        description=(
            "Existing node display name for modify_node / remove_element / add_* naming. "
            "For named nodes this is sufficient; for unnamed nodes use nodeId instead."
        ),
    )
    flowId: Optional[str] = Field(
        default=None,
        description="Id of a flow to remove (optional; remove_flow may use source/target instead).",
    )
    poolName: Optional[str] = Field(default=None, description="Pool name/id for add_swimlane or remove_pool.")
    swimlaneName: Optional[str] = Field(default=None, description="Swimlane name/id for modify_swimlane or remove_swimlane.")


class BPMNModificationChanges(BaseModel):
    name: Optional[str] = Field(
        default=None,
        max_length=60,
        description="New name for modify_node (rename), or the name for an added node.",
    )
    taskType: Optional[_TASK_TYPE] = Field(
        default=None,
        description="Task type for add_task / modify_node.",
    )
    gatewayType: Optional[_GATEWAY_TYPE] = Field(
        default=None,
        description="Gateway type for add_gateway / modify_node.",
    )
    eventKind: Optional[Literal["start", "end", "intermediate"]] = Field(
        default=None,
        description="Event kind for add_event.",
    )
    source: Optional[str] = Field(
        default=None,
        description=(
            "Source node id (exact [id] from context) or name for add_flow / remove_flow. "
            "Use the id for unnamed nodes."
        ),
    )
    target: Optional[str] = Field(
        default=None,
        description=(
            "Target node id (exact [id] from context) or name for add_flow / remove_flow. "
            "Use the id for unnamed nodes."
        ),
    )
    label: Optional[str] = Field(
        default=None,
        max_length=40,
        description="Optional flow label for add_flow (branch condition).",
    )
    # WME modifier actions retain their existing `*_swimlane` API spelling.
    # This does not change generated pool specifications, which use `lanes`.
    role: Optional[_AGENT_ROLE] = Field(
        default=None,
        description=(
            "Lane role: solution, supervision, collaboration, or consensus."
        ),
    )
    isAgentic: Optional[bool] = Field(default=None)
    trustScore: Optional[int] = Field(default=None, ge=0, le=100)
    multiplicity: Optional[int] = Field(default=None, ge=1)
    agentDiagramRef: Optional[str] = Field(default=None)
    reflectionMode: Optional[_REFLECTION_MODE] = Field(default=None)
    gatewayRole: Optional[_GATEWAY_ROLE] = Field(default=None)
    governanceDsl: Optional[str] = Field(default=None)

    poolName: Optional[str] = Field(
        default=None,
        description="Pool name/id for add_swimlane."
    )
    owner: Optional[str] = Field(
        default=None,
        description="Lane name/id for a newly added BPMN node."
    )

class BPMNModification(BaseModel):
    action: Literal[
        "add_task", "add_gateway", "add_event",
        "add_flow", "modify_node", "remove_flow", "remove_element",
        "add_pool", "add_swimlane", "modify_swimlane", "remove_swimlane", "remove_pool",
    ] = Field(description="Action to perform.")
    target: BPMNModificationTarget = Field(description="Identifies the element to act on.")
    changes: Optional[BPMNModificationChanges] = Field(
        default=None,
        description="Changes to apply. Required for all actions except remove_element.",
    )


class BPMNModificationResponse(BaseModel):
    # default_factory=list (not min_length=1) is intentional: when elementFound
    # is false the LLM returns an empty list, and Pydantic must accept that.
    modifications: List[BPMNModification] = Field(
        default_factory=list,
        description="List of modifications to apply to the process. Empty when elementFound is false.",
    )
    message: str = Field(
        description=(
            "Human-readable summary of the change. "
            "When elementFound is false, explain which element was not found and list the current nodes."
        ),
    )
    elementFound: bool = Field(
        default=True,
        description=(
            "Set to false when a remove_element or modify_node action cannot be matched "
            "to any element in the current context listing. "
            "When false, modifications must be empty."
        ),
    )
