"""Pydantic schemas for State Machine structured outputs.

Field descriptions are used by OpenAI Structured Outputs to guide generation.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class StateSpec(BaseModel):
    stateName: str = Field(
        min_length=1,
        max_length=30,
        description="State name in PascalCase representing a lifecycle stage.",
    )
    stateType: Literal["initial", "final", "regular"] = Field(
        default="regular",
        description="State type: initial, final, or regular.",
    )
    entryAction: Optional[str] = Field(
        default=None,
        description="Action executed once when entering this state.",
    )
    exitAction: Optional[str] = Field(
        default=None,
        description="Action executed once when leaving this state.",
    )
    doActivity: Optional[str] = Field(
        default=None,
        description="Ongoing activity while the state is active.",
    )


class TransitionSpec(BaseModel):
    source: str = Field(
        description="Source state name.",
    )
    target: str = Field(
        description="Target state name.",
    )
    trigger: Optional[str] = Field(
        default=None,
        description="Event that causes this transition, as a camelCase verb phrase.",
    )
    guard: Optional[str] = Field(
        default=None,
        description="Boolean condition that must be true for the transition to fire.",
    )
    effect: Optional[str] = Field(
        default=None,
        description="Side-effect action executed when the transition fires.",
    )


class SingleStateSpec(BaseModel):
    """Schema for a single state element."""
    stateName: str = Field(
        min_length=1,
        max_length=30,
        description="State name in PascalCase representing a lifecycle stage.",
    )
    stateType: Literal["initial", "final", "regular"] = Field(
        default="regular",
        description="State type: initial, final, or regular.",
    )
    entryAction: Optional[str] = Field(
        default=None,
        description="Action executed once when entering this state.",
    )
    exitAction: Optional[str] = Field(
        default=None,
        description="Action executed once when leaving this state.",
    )
    doActivity: Optional[str] = Field(
        default=None,
        description="Ongoing activity while the state is active.",
    )


class StateCodeBlockSpec(BaseModel):
    name: str = Field(description="Name/label for the code block")
    code: str = Field(description="Python code content")
    language: str = Field(default="python", description="Code language (always 'python')")


class SystemStateMachineSpec(BaseModel):
    """Schema for a complete state machine system."""
    systemName: str = Field(
        default="",
        description="Descriptive name for the state machine.",
    )
    states: List[StateSpec] = Field(
        min_length=1,
        description="All states in the machine, including one initial and one final state.",
    )
    transitions: List[TransitionSpec] = Field(
        default_factory=list,
        description="Transitions connecting the states.",
    )
    codeBlocks: List[StateCodeBlockSpec] = Field(default_factory=list, description="Python code blocks for state bodies.")


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
        max_length=30,
        description="New name for rename operations (PascalCase, ONE word only).",
    )
    stateType: Optional[Literal["initial", "final", "regular"]] = Field(
        default=None,
        description="State type for add_state: 'regular', 'initial', or 'final'.",
    )
    entryAction: Optional[str] = Field(
        default=None,
        description="Entry action for the state.",
    )
    exitAction: Optional[str] = Field(
        default=None,
        description="Exit action for the state.",
    )
    doActivity: Optional[str] = Field(
        default=None,
        description="Ongoing activity for the state.",
    )
    trigger: Optional[str] = Field(
        default=None,
        description="Trigger event for a transition (camelCase verb phrase).",
    )
    guard: Optional[str] = Field(
        default=None,
        description="Guard condition for a transition.",
    )
    effect: Optional[str] = Field(
        default=None,
        description="Side-effect action for a transition.",
    )
    code: Optional[str] = Field(default=None, description="Python code content for add_code_block")
    language: Optional[str] = Field(default=None, description="Code language (default: python)")


class StateMachineModification(BaseModel):
    action: Literal[
        "add_state", "modify_state",
        "add_transition", "modify_transition",
        "add_code_block",
        "remove_element",
    ] = Field(
        description="Action to perform.",
    )
    target: StateMachineModificationTarget = Field(
        description="Identifies the element to modify.",
    )
    changes: Optional[StateMachineModificationChanges] = Field(
        default=None,
        description="Changes to apply. Required for all actions except remove_element.",
    )


class StateMachineModificationResponse(BaseModel):
    modifications: List[StateMachineModification] = Field(
        min_length=1,
        description="List of modifications to apply to the state machine.",
    )
