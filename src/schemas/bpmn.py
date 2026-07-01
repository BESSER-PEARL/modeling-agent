"""Pydantic schemas for BPMN structured outputs.

Field descriptions are used by OpenAI Structured Outputs to guide generation.
Base BPMN only — start/end events, tasks, gateways, sequence flows.  No pools,
lanes, or agentic concepts (roles, governance, collaboration, trust).

Layout is handled on the WME side; the agent emits no positions.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

_TASK_TYPE = Literal[
    "default", "user", "service", "send", "receive",
    "manual", "business-rule", "script",
]
_GATEWAY_TYPE = Literal["exclusive", "parallel", "inclusive", "event-based", "complex"]


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
    """Schema for a complete base-BPMN process."""

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
            "Sequence flows connecting the nodes by id. Every node except the "
            "start has an incoming flow; every node except end events has an "
            "outgoing flow."
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


class BPMNModification(BaseModel):
    action: Literal[
        "add_task", "add_gateway", "add_event",
        "add_flow", "modify_node", "remove_flow", "remove_element",
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
