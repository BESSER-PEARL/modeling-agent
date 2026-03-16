"""Pydantic schemas for State Machine structured outputs."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class StateSpec(BaseModel):
    stateName: str = Field(min_length=1)
    stateType: Literal["initial", "final", "regular"] = "regular"
    entryAction: Optional[str] = None
    exitAction: Optional[str] = None
    doActivity: Optional[str] = None


class TransitionSpec(BaseModel):
    source: str
    target: str
    trigger: Optional[str] = None
    guard: Optional[str] = None
    effect: Optional[str] = None


class SingleStateSpec(BaseModel):
    """Schema for a single state element."""
    stateName: str = Field(min_length=1)
    stateType: Literal["initial", "final", "regular"] = "regular"
    entryAction: Optional[str] = None
    exitAction: Optional[str] = None
    doActivity: Optional[str] = None


class SystemStateMachineSpec(BaseModel):
    """Schema for a complete state machine system."""
    systemName: str = ""
    states: List[StateSpec] = Field(min_length=1)
    transitions: List[TransitionSpec] = Field(default_factory=list)


# -- Modification schemas --

class StateMachineModificationTarget(BaseModel):
    stateName: Optional[str] = None
    sourceState: Optional[str] = None
    targetState: Optional[str] = None


class StateMachineModificationChanges(BaseModel):
    name: Optional[str] = None
    entryAction: Optional[str] = None
    exitAction: Optional[str] = None
    doActivity: Optional[str] = None
    trigger: Optional[str] = None
    guard: Optional[str] = None
    effect: Optional[str] = None


class StateMachineModification(BaseModel):
    action: str
    target: StateMachineModificationTarget
    changes: Optional[StateMachineModificationChanges] = None


class StateMachineModificationResponse(BaseModel):
    modifications: List[StateMachineModification] = Field(min_length=1)
