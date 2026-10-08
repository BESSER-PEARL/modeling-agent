"""Pydantic schemas for BPMN structured outputs.

Field descriptions are used by OpenAI Structured Outputs to guide generation.
Base BPMN plus collaboration diagrams — start/end events, tasks, gateways,
sequence flows, and optional pools/lanes for multi-participant processes —
plus the editor's Agentic BPMN attributes (agentic lanes with a role, trust
score and swarm multiplicity; agentic tasks with a reflection mode; agentic
parallel/inclusive gateways with a diverging/merging role and a Governance
DSL). The agentic vocabulary mirrors the editor's
``packages/editor/src/main/packages/bpmn/common/types.ts``.

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

# Editor ``BPMNAgentProfile`` — the authorable lane-role presets.
AGENT_ROLE_VALUES = ("solution", "supervision")
_AGENT_ROLE = Literal["solution", "supervision"]
# Editor ``BPMNReflectionMode``.
REFLECTION_MODE_VALUES = ("none", "self", "cross", "human")
_REFLECTION_MODE = Literal["none", "self", "cross", "human"]
# Editor ``BPMNGatewayRole``.
GATEWAY_ROLE_VALUES = ("diverging", "merging")
_GATEWAY_ROLE = Literal["diverging", "merging"]

_TRUST_SCORE_DESCRIPTION = (
    "Agentic only: trust in this agent's output as a 0-100 percentage. "
    "Leave null unless the request states or clearly implies a trust level."
)
_AGENT_DIAGRAM_REF_DESCRIPTION = (
    "Agentic only: the id of an EXISTING AgentDiagram in the project that defines "
    "this agent's behavior. Use ONLY an id from the 'Agent diagrams in this "
    "project' list in the request; never invent one. Null when none is listed."
)


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
        description=(
            "BPMN flow-node kind: 'startEvent' (exactly one), 'endEvent' (one or "
            "more), 'task' (an activity/step), or 'gateway' (a branch/merge)."
        ),
    )
    taskType: Optional[_TASK_TYPE] = Field(
        default="default",
        description=(
            "For type='task' only. 'user' = performed by a person, 'service' = "
            "automated system call, 'send'/'receive' = message tasks, 'script' = "
            "automated script, 'business-rule' = decision rule. Default 'default'."
        ),
    )
    gatewayType: Optional[_GATEWAY_TYPE] = Field(
        default="exclusive",
        description=(
            "For type='gateway' only. 'exclusive' = one branch (XOR decision), "
            "'parallel' = all branches concurrently (AND), 'inclusive' = one or "
            "more (OR). Default 'exclusive'."
        ),
    )
    poolId: Optional[str] = Field(
        default=None,
        description=(
            "Id of the pool (participant) this node belongs to, matching one of the "
            "top-level pools[].id. Leave null for a flat process with no pools (the "
            "common case)."
        ),
    )
    laneId: Optional[str] = Field(
        default=None,
        description=(
            "Id of the lane (role/department) within poolId this node belongs to, "
            "matching one of that pool's lanes[].id. Only set when that pool declares "
            "lanes; leave null otherwise."
        ),
    )
    owner: Optional[str] = Field(
        default=None,
        description=(
            "WME ownership field for lane-contained nodes. When laneId is set and valid, "
            "the backend normalizes owner to the same lane id so the generated BPMN "
            "nodes move with their lane and export correctly."
        ),
    )
    # Agentic task and gateway attributes. Each is meaningful only for its
    # node type; one shared node schema keeps a single complete-system payload.
    isAgentic: Optional[bool] = Field(
        default=None,
        description=(
            "Task or gateway: true when an AI agent performs the task, or when the "
            "gateway splits/merges agent work. Null for an ordinary BPMN node."
        ),
    )
    reflectionMode: Optional[_REFLECTION_MODE] = Field(
        default=None,
        description=(
            "Agentic task only: how its output is reviewed. 'none', 'self' (the agent "
            "reviews itself), 'cross' (another agent lane reviews it, see "
            "reflectionReviewerLaneId) or 'human' (a person reviews it)."
        ),
    )
    reflectionReviewerLaneId: Optional[str] = Field(
        default=None,
        description=(
            "Agentic task with reflectionMode='cross' only: the lanes[].id of the "
            "reviewing agent lane. Must be a lane declared in pools; null otherwise."
        ),
    )
    trustScore: Optional[int] = Field(
        default=None, ge=0, le=100, description=_TRUST_SCORE_DESCRIPTION,
    )
    agentDiagramRef: Optional[str] = Field(
        default=None, description=_AGENT_DIAGRAM_REF_DESCRIPTION,
    )
    gatewayRole: Optional[_GATEWAY_ROLE] = Field(
        default=None,
        description=(
            "Agentic gateway only: 'diverging' when it fans work out to agents, "
            "'merging' when it combines their results."
        ),
    )
    governanceDsl: Optional[str] = Field(
        default=None,
        description=(
            "Agentic merging gateway only: the Governance DSL policy that decides how "
            "the merged agent results are accepted (e.g. 'Policy: MajorityPolicy'). "
            "Null on every other node."
        ),
    )


class BPMNLaneSpec(BaseModel):
    id: str = Field(
        min_length=1,
        max_length=40,
        description=(
            "Short unique slug identifying this lane/role within its pool "
            "(e.g. 'chef'). Referenced by node laneId. Lowercase, no spaces."
        ),
    )
    name: str = Field(
        default="",
        max_length=60,
        description="Role/department display name (e.g. 'Pizza Chef').",
    )
    isAgentic: bool = Field(
        default=False,
        description=(
            "True when this lane is an AI agent (an Agentic BPMN swarm member) rather "
            "than a person or department."
        ),
    )
    role: Optional[_AGENT_ROLE] = Field(
        default=None,
        description=(
            "Agentic lane only: 'solution' for an agent that does the work, "
            "'supervision' for an agent that oversees or approves other agents."
        ),
    )
    trustScore: Optional[int] = Field(
        default=None, ge=0, le=100, description=_TRUST_SCORE_DESCRIPTION,
    )
    multiplicity: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Agentic lane only: swarm size, i.e. how many identical copies of this "
            "agent work in parallel (e.g. 3 reviewers). Integer >= 1."
        ),
    )
    agentDiagramRef: Optional[str] = Field(
        default=None, max_length=100, description=_AGENT_DIAGRAM_REF_DESCRIPTION,
    )


class BPMNPoolSpec(BaseModel):
    id: str = Field(
        min_length=1,
        max_length=40,
        description=(
            "Short unique slug identifying this pool/participant (e.g. 'customer'). "
            "Referenced by node poolId. Lowercase, no spaces."
        ),
    )
    name: str = Field(
        default="",
        max_length=60,
        description="Participant/organization display name (e.g. 'Customer', 'Warehouse System').",
    )
    lanes: List[BPMNLaneSpec] = Field(
        default_factory=list,
        description=(
            "Optional role/department lanes inside this pool. Leave empty if the "
            "pool is a single undivided participant. Use `lanes`, never `swimlanes`."
        ),
    )


class BPMNFlowSpec(BaseModel):
    source: str = Field(description="Source node id.")
    target: str = Field(description="Target node id.")
    name: Optional[str] = Field(
        default="",
        max_length=40,
        description=(
            "Optional edge label. Use it on branches out of an exclusive/"
            "inclusive gateway to name the condition (e.g. 'yes', 'no', "
            "'amount > 1000'). Leave empty for ordinary flows."
        ),
    )


class SystemBPMNSpec(BaseModel):
    """Schema for a complete BPMN process (base or agentic)."""

    systemName: str = Field(
        default="",
        description="Descriptive name for the process (e.g. 'Order Handling').",
    )
    nodes: List[BPMNNodeSpec] = Field(
        min_length=1,
        description=(
            "All flow nodes. Include exactly one startEvent, at least one "
            "endEvent, tasks for the activities, and gateways for decisions/"
            "parallel splits."
        ),
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
            "For named nodes this is sufficient; for unnamed nodes use nodeId instead. "
            "Also the new pool name for add_pool and the new lane name for add_swimlane."
        ),
    )
    flowId: Optional[str] = Field(
        default=None,
        description="Id of a flow to remove (optional; remove_flow may use source/target instead).",
    )
    poolName: Optional[str] = Field(
        default=None,
        description="Existing pool name or [id] for remove_pool.",
    )
    swimlaneName: Optional[str] = Field(
        default=None,
        description="Existing lane name or [id] for modify_swimlane / remove_swimlane.",
    )


class BPMNModificationChanges(BaseModel):
    name: Optional[str] = Field(
        default=None,
        max_length=60,
        description="New name for modify_node / modify_swimlane (rename), or the name for an added node.",
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
    # Agentic attributes. The WME modifier keeps its `*_swimlane` action
    # spelling; generated pool specifications always use `lanes`.
    role: Optional[_AGENT_ROLE] = Field(
        default=None,
        description=(
            "Lane role for add_swimlane / modify_swimlane: 'solution' (does the work) "
            "or 'supervision' (oversees other agents)."
        ),
    )
    isAgentic: Optional[bool] = Field(
        default=None,
        description=(
            "add_swimlane / modify_swimlane / modify_node: mark the lane, task or "
            "gateway as agentic (true) or ordinary (false)."
        ),
    )
    trustScore: Optional[int] = Field(
        default=None, ge=0, le=100, description=_TRUST_SCORE_DESCRIPTION,
    )
    multiplicity: Optional[int] = Field(
        default=None,
        ge=1,
        description="add_swimlane / modify_swimlane: agent swarm size, an integer >= 1.",
    )
    agentDiagramRef: Optional[str] = Field(
        default=None, description=_AGENT_DIAGRAM_REF_DESCRIPTION,
    )
    reflectionMode: Optional[_REFLECTION_MODE] = Field(
        default=None,
        description="modify_node on an agentic task: 'none', 'self', 'cross' or 'human'.",
    )
    reflectionReviewerLaneId: Optional[str] = Field(
        default=None,
        description=(
            "modify_node on an agentic task with reflectionMode='cross': the [id] of "
            "the existing reviewer lane from the context."
        ),
    )
    gatewayRole: Optional[_GATEWAY_ROLE] = Field(
        default=None,
        description="modify_node on an agentic gateway: 'diverging' or 'merging'.",
    )
    governanceDsl: Optional[str] = Field(
        default=None,
        description="modify_node on an agentic MERGING gateway: the Governance DSL policy text.",
    )
    poolName: Optional[str] = Field(
        default=None,
        description="add_swimlane: the existing pool name or [id] the new lane goes into.",
    )
    owner: Optional[str] = Field(
        default=None,
        description="add_task / add_gateway / add_event: existing lane name or [id] to place the new node in.",
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
