"""Pydantic schemas for Agent Diagram structured outputs."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AgentReplySpec(BaseModel):
    text: str
    replyType: Literal["text", "llm"] = "text"


class AgentStateSpec(BaseModel):
    type: Literal["state"] = "state"
    stateName: str = Field(min_length=1)
    replies: List[AgentReplySpec] = Field(default_factory=list)
    fallbackBodies: List[AgentReplySpec] = Field(default_factory=list)


class AgentIntentSpec(BaseModel):
    type: Literal["intent"] = "intent"
    intentName: str = Field(min_length=1)
    trainingPhrases: List[str] = Field(default_factory=list)


class AgentSingleElementSpec(BaseModel):
    """Schema for a single agent diagram element (state, intent, or initial node)."""
    type: Literal["state", "intent", "initial"] = "state"
    # State fields
    stateName: Optional[str] = None
    replies: List[AgentReplySpec] = Field(default_factory=list)
    fallbackBodies: List[AgentReplySpec] = Field(default_factory=list)
    # Intent fields
    intentName: Optional[str] = None
    trainingPhrases: List[str] = Field(default_factory=list)
    # Initial node fields
    description: Optional[str] = None


class AgentTransitionSpec(BaseModel):
    source: str
    target: str
    condition: Literal[
        "when_intent_matched", "when_no_intent_matched", "auto",
    ] = "when_intent_matched"
    conditionValue: Optional[str] = None
    label: Optional[str] = None
    sourceDirection: Optional[str] = None
    targetDirection: Optional[str] = None


class SystemAgentSpec(BaseModel):
    """Schema for a complete agent diagram system."""
    systemName: str = ""
    hasInitialNode: bool = True
    initialNode: Optional[Dict[str, Any]] = None
    intents: List[AgentIntentSpec] = Field(default_factory=list)
    states: List[AgentStateSpec] = Field(min_length=1)
    transitions: List[AgentTransitionSpec] = Field(default_factory=list)


# -- Modification schemas --

class AgentModificationTarget(BaseModel):
    stateName: Optional[str] = None
    intentName: Optional[str] = None
    sourceStateName: Optional[str] = None
    targetStateName: Optional[str] = None
    transitionId: Optional[str] = None

class AgentModificationChanges(BaseModel):
    name: Optional[str] = None
    intentName: Optional[str] = None
    condition: Optional[str] = None
    text: Optional[str] = None
    replyType: Optional[str] = None
    trainingPhrase: Optional[str] = None

class AgentModification(BaseModel):
    action: str
    target: AgentModificationTarget
    changes: Optional[AgentModificationChanges] = None

class AgentModificationResponse(BaseModel):
    modifications: List[AgentModification] = Field(min_length=1)
