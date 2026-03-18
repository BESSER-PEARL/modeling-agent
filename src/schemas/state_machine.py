"""Pydantic schemas for State Machine structured outputs.

Field descriptions are used by OpenAI Structured Outputs to guide generation.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class StateSpec(BaseModel):
    stateName: str = Field(
        min_length=1,
        description=(
            "State name in PascalCase representing a real lifecycle stage. "
            "GOOD: PendingPayment, Shipped, UnderReview, Authenticated, InProgress. "
            "BAD: State1, Active, Process, Step2."
        ),
    )
    stateType: Literal["initial", "final", "regular"] = Field(
        default="regular",
        description="State type: 'initial' (exactly one per diagram), 'final' (exactly one per diagram), or 'regular'.",
    )
    entryAction: Optional[str] = Field(
        default=None,
        description=(
            "One-time action executed when entering the state. "
            "Examples: 'send confirmation email', 'lock account', 'start timer'. "
            "Leave null if none."
        ),
    )
    exitAction: Optional[str] = Field(
        default=None,
        description=(
            "One-time action executed when leaving the state. "
            "Examples: 'save progress', 'release lock', 'stop timer'. "
            "Leave null if none."
        ),
    )
    doActivity: Optional[str] = Field(
        default=None,
        description=(
            "Ongoing activity that continues while the state is active. "
            "Examples: 'await payment', 'monitor session', 'process data'. "
            "Leave null if none."
        ),
    )


class TransitionSpec(BaseModel):
    source: str = Field(
        description="Source state name. Must match an existing stateName exactly.",
    )
    target: str = Field(
        description="Target state name. Must match an existing stateName exactly.",
    )
    trigger: Optional[str] = Field(
        default=None,
        description=(
            "Event that causes this transition, as a camelCase verb or verb phrase. "
            "GOOD: submitPayment, approveRequest, sessionTimeout, deliveryConfirmed. "
            "BAD: go, next, transition1, move."
        ),
    )
    guard: Optional[str] = Field(
        default=None,
        description=(
            "Boolean condition that must be true for the transition to fire. "
            "Examples: 'payment valid', 'attempts < max', 'user authenticated'. "
            "Leave null if unconditional."
        ),
    )
    effect: Optional[str] = Field(
        default=None,
        description=(
            "Side-effect action executed when the transition fires. "
            "Examples: 'send notification', 'update inventory', 'log event'. "
            "Leave null if none."
        ),
    )


class SingleStateSpec(BaseModel):
    """Schema for a single state element."""
    stateName: str = Field(
        min_length=1,
        description=(
            "State name in PascalCase representing a real lifecycle stage. "
            "GOOD: PendingPayment, Shipped, UnderReview, Authenticated, InProgress. "
            "BAD: State1, Active, Process, Step2."
        ),
    )
    stateType: Literal["initial", "final", "regular"] = Field(
        default="regular",
        description="State type: 'initial', 'final', or 'regular'.",
    )
    entryAction: Optional[str] = Field(
        default=None,
        description=(
            "One-time action executed when entering the state. "
            "Examples: 'send confirmation email', 'lock account', 'start timer'. "
            "Leave null if none."
        ),
    )
    exitAction: Optional[str] = Field(
        default=None,
        description=(
            "One-time action executed when leaving the state. "
            "Examples: 'save progress', 'release lock', 'stop timer'. "
            "Leave null if none."
        ),
    )
    doActivity: Optional[str] = Field(
        default=None,
        description=(
            "Ongoing activity that continues while the state is active. "
            "Examples: 'await payment', 'monitor session', 'process data'. "
            "Leave null if none."
        ),
    )


class SystemStateMachineSpec(BaseModel):
    """Schema for a complete state machine system."""
    systemName: str = Field(
        default="",
        description="Descriptive name for the state machine (e.g., 'OrderProcessing', 'UserAuthentication').",
    )
    states: List[StateSpec] = Field(
        min_length=1,
        description=(
            "All states in the machine. Include exactly one 'initial' state, "
            "one 'final' state, and 4-8 'regular' states representing real lifecycle stages. "
            "Every regular state must have at least one incoming and one outgoing transition."
        ),
    )
    transitions: List[TransitionSpec] = Field(
        default_factory=list,
        description=(
            "Transitions connecting the states. Every state (except initial/final) should have "
            "both incoming and outgoing transitions. Include error/exception paths, not just the happy path. "
            "Self-transitions are valid for retry/refresh scenarios."
        ),
    )


# -- Modification schemas --

class StateMachineModificationTarget(BaseModel):
    stateName: Optional[str] = Field(
        default=None,
        description="Target state name for state modifications or removal.",
    )
    sourceState: Optional[str] = Field(
        default=None,
        description="Source state name for transition modifications.",
    )
    targetState: Optional[str] = Field(
        default=None,
        description="Target state name for transition modifications.",
    )


class StateMachineModificationChanges(BaseModel):
    name: Optional[str] = Field(
        default=None,
        description="New name for rename operations (PascalCase).",
    )
    entryAction: Optional[str] = Field(
        default=None,
        description="New entry action for the state.",
    )
    exitAction: Optional[str] = Field(
        default=None,
        description="New exit action for the state.",
    )
    doActivity: Optional[str] = Field(
        default=None,
        description="New ongoing activity for the state.",
    )
    trigger: Optional[str] = Field(
        default=None,
        description="New trigger event for a transition (camelCase verb phrase).",
    )
    guard: Optional[str] = Field(
        default=None,
        description="New guard condition for a transition.",
    )
    effect: Optional[str] = Field(
        default=None,
        description="New side-effect action for a transition.",
    )


class StateMachineModification(BaseModel):
    action: str = Field(
        description="Modification action: 'modify_state', 'add_transition', 'modify_transition', or 'remove_element'.",
    )
    target: StateMachineModificationTarget = Field(
        description="Identifies the element to modify. Use stateName for states, sourceState/targetState for transitions.",
    )
    changes: Optional[StateMachineModificationChanges] = Field(
        default=None,
        description="The changes to apply. Required for modify/add actions, omit for remove_element.",
    )


class StateMachineModificationResponse(BaseModel):
    modifications: List[StateMachineModification] = Field(
        min_length=1,
        description="List of modifications to apply to the state machine.",
    )
