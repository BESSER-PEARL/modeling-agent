"""Pydantic schemas for Agent Diagram structured outputs."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AgentReplySpec(BaseModel):
    text: str = Field(
        description="Reply text, LLM prompt, or Python code depending on replyType.",
    )
    replyType: Literal["text", "llm", "rag", "db_reply", "code"] = Field(
        default="text",
        description="Reply type: text, llm, rag, db_reply, or code.",
    )
    ragDatabaseName: Optional[str] = Field(
        default=None,
        description="RAG knowledge base name (required when replyType is rag).",
    )


class AgentStateSpec(BaseModel):
    type: Literal["state"] = Field(
        default="state",
        description="Node type discriminator; always 'state'.",
    )
    stateName: str = Field(
        min_length=1,
        max_length=30,
        description="Unique camelCase name for this state.",
    )
    replies: List[AgentReplySpec] = Field(
        default_factory=list,
        description="Replies the agent sends when entering this state.",
    )
    fallbackBodies: List[AgentReplySpec] = Field(
        default_factory=list,
        description="Fallback replies when no intent matches in this state.",
    )


class AgentIntentSpec(BaseModel):
    type: Literal["intent"] = Field(
        default="intent",
        description="Node type discriminator; always 'intent'.",
    )
    intentName: str = Field(
        min_length=1,
        max_length=30,
        description="Unique TitleCase name for this intent.",
    )
    trainingPhrases: List[str] = Field(
        default_factory=list,
        description="Example user utterances that trigger this intent.",
    )


class AgentSingleElementSpec(BaseModel):
    """Schema for a single agent diagram element (state, intent, or initial node)."""
    type: Literal["state", "intent", "initial"] = Field(
        default="state",
        description="Element kind: state, intent, or initial.",
    )
    # State fields
    stateName: Optional[str] = Field(
        default=None,
        max_length=30,
        description="Unique camelCase state name (required when type is state).",
    )
    replies: List[AgentReplySpec] = Field(
        default_factory=list,
        description="Replies the agent sends in this state.",
    )
    fallbackBodies: List[AgentReplySpec] = Field(
        default_factory=list,
        description="Fallback replies when no intent matches.",
    )
    # Intent fields
    intentName: Optional[str] = Field(
        default=None,
        max_length=30,
        description="Unique TitleCase intent name (required when type is intent).",
    )
    trainingPhrases: List[str] = Field(
        default_factory=list,
        description="Example phrases that trigger this intent.",
    )
    # Initial node fields
    description: Optional[str] = Field(
        default=None,
        description="Optional label for the initial entry-point node.",
    )


class AgentTransitionSpec(BaseModel):
    source: str = Field(
        description="Name of the source state, or 'initial' for the entry-point node.",
    )
    target: str = Field(
        description="Name of the target state this transition leads to.",
    )
    condition: Literal[
        "when_intent_matched", "when_no_intent_matched", "auto",
    ] = Field(
        default="when_intent_matched",
        description="Transition trigger: when_intent_matched, when_no_intent_matched, or auto.",
    )
    conditionValue: Optional[str] = Field(
        default=None,
        description="Intent name that triggers this transition (for when_intent_matched).",
    )
    label: Optional[str] = Field(
        default=None,
        description="Optional display label for the transition arrow.",
    )
    sourceDirection: Optional[str] = Field(
        default=None,
        description="Visual anchor direction on the source node (Right, Left, Top, Bottom).",
    )
    targetDirection: Optional[str] = Field(
        default=None,
        description="Visual anchor direction on the target node (Right, Left, Top, Bottom).",
    )


class AgentInitialNodeSpec(BaseModel):
    """Schema for the initial node in an agent diagram."""
    description: Optional[str] = Field(
        default=None,
        description="Optional note or label for the initial entry-point node.",
    )


class AgentRagSpec(BaseModel):
    name: str = Field(description="Name of the RAG knowledge base.")


class SystemAgentSpec(BaseModel):
    """Schema for a complete agent diagram system."""
    systemName: str = Field(
        default="",
        description="Display name for the agent system.",
    )
    hasInitialNode: bool = Field(
        default=True,
        description="Whether the diagram includes an initial entry-point node.",
    )
    initialNode: Optional[AgentInitialNodeSpec] = Field(
        default=None,
        description="Configuration for the initial entry-point node.",
    )
    intents: List[AgentIntentSpec] = Field(
        default_factory=list,
        description="All intent nodes in the agent diagram.",
    )
    states: List[AgentStateSpec] = Field(
        min_length=1,
        description="All state nodes in the agent diagram.",
    )
    transitions: List[AgentTransitionSpec] = Field(
        default_factory=list,
        description="Edges connecting states and intents.",
    )
    ragElements: List[AgentRagSpec] = Field(default_factory=list, description="RAG knowledge bases used by agent states.")


# -- Modification schemas --

class AgentModificationTarget(BaseModel):
    stateName: Optional[str] = Field(
        default=None,
        description="Name of the state to modify or remove.",
    )
    intentName: Optional[str] = Field(
        default=None,
        description="Name of the intent to modify or remove.",
    )
    sourceStateName: Optional[str] = Field(
        default=None,
        description="Source state name when adding or removing a transition.",
    )
    targetStateName: Optional[str] = Field(
        default=None,
        description="Target state name when adding or removing a transition.",
    )
    transitionId: Optional[str] = Field(
        default=None,
        description="Optional identifier for a specific transition to modify or remove.",
    )

class AgentModificationChanges(BaseModel):
    name: Optional[str] = Field(
        default=None,
        max_length=30,
        description="New name when renaming a state or intent.",
    )
    replies: Optional[List[AgentReplySpec]] = Field(
        default=None,
        description="Replies for add_state.",
    )
    trainingPhrases: Optional[List[str]] = Field(
        default=None,
        description="Training phrases for add_intent.",
    )
    intentName: Optional[str] = Field(
        default=None,
        description="Intent name for a transition.",
    )
    condition: Optional[Literal["when_intent_matched", "when_no_intent_matched", "auto"]] = Field(
        default=None,
        description="Transition condition: when_intent_matched, when_no_intent_matched, or auto.",
    )
    text: Optional[str] = Field(
        default=None,
        description="Reply text to add to a state.",
    )
    replyType: Optional[Literal["text", "llm", "rag", "db_reply", "code"]] = Field(
        default=None,
        description="Reply type: text, llm, rag, db_reply, or code.",
    )
    trainingPhrase: Optional[str] = Field(
        default=None,
        description="Single training phrase to add to an existing intent.",
    )

class AgentModification(BaseModel):
    action: Literal[
        "add_state", "modify_state",
        "add_intent", "modify_intent",
        "add_transition", "remove_transition",
        "add_state_body", "add_intent_training_phrase",
        "add_rag_element",
        "remove_element",
    ] = Field(
        description="Action to perform.",
    )
    target: AgentModificationTarget = Field(
        description="Identifies the element to modify.",
    )
    changes: Optional[AgentModificationChanges] = Field(
        default=None,
        description="Changes to apply. Required except for remove_element and remove_transition.",
    )

class AgentModificationResponse(BaseModel):
    modifications: List[AgentModification] = Field(
        min_length=1,
        description="One or more modification operations to apply to the agent diagram.",
    )
