"""Pydantic schemas for Agent Diagram structured outputs."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AgentReplySpec(BaseModel):
    text: str = Field(
        description="The reply text displayed to the user, used as an LLM prompt, or Python code to execute.",
    )
    replyType: Literal["text", "llm", "rag", "db_reply", "code"] = Field(
        default="text",
        description=(
            "'text' for a scripted reply, "
            "'llm' for a dynamically generated AI response, "
            "'rag' for a RAG-based reply (retrieval-augmented generation from a knowledge base), "
            "'db_reply' for a database query action, "
            "'code' for executing Python code."
        ),
    )
    ragDatabaseName: Optional[str] = Field(
        default=None,
        description="Name of the RAG knowledge base to query (required when replyType='rag').",
    )


class AgentStateSpec(BaseModel):
    type: Literal["state"] = Field(
        default="state",
        description="Node type discriminator; always 'state' for state nodes.",
    )
    stateName: str = Field(
        min_length=1,
        description="Unique name for this state in camelCase (e.g. 'welcomeGreeting').",
    )
    replies: List[AgentReplySpec] = Field(
        default_factory=list,
        description="Ordered list of 1-3 replies the agent sends when entering this state.",
    )
    fallbackBodies: List[AgentReplySpec] = Field(
        default_factory=list,
        description="Optional replies sent when no intent matches in this state (fallback / error handling).",
    )


class AgentIntentSpec(BaseModel):
    type: Literal["intent"] = Field(
        default="intent",
        description="Node type discriminator; always 'intent' for intent nodes.",
    )
    intentName: str = Field(
        min_length=1,
        description="Unique name for this intent in TitleCase (e.g. 'OrderPizza').",
    )
    trainingPhrases: List[str] = Field(
        default_factory=list,
        description="3-4 example user utterances that should trigger this intent.",
    )


class AgentSingleElementSpec(BaseModel):
    """Schema for a single agent diagram element (state, intent, or initial node)."""
    type: Literal["state", "intent", "initial"] = Field(
        default="state",
        description="Element kind: 'state' for a conversation state, 'intent' for a user intent, 'initial' for the entry point node.",
    )
    # State fields
    stateName: Optional[str] = Field(
        default=None,
        description="Required when type='state'. Unique camelCase name for the state.",
    )
    replies: List[AgentReplySpec] = Field(
        default_factory=list,
        description="Replies the agent sends in this state (1-3 items). Each has 'text' and 'replyType'.",
    )
    fallbackBodies: List[AgentReplySpec] = Field(
        default_factory=list,
        description="Fallback replies when no intent is matched. Include only for error-handling scenarios.",
    )
    # Intent fields
    intentName: Optional[str] = Field(
        default=None,
        description="Required when type='intent'. Unique TitleCase name for the intent.",
    )
    trainingPhrases: List[str] = Field(
        default_factory=list,
        description="3-4 example phrases a user would say to trigger this intent.",
    )
    # Initial node fields
    description: Optional[str] = Field(
        default=None,
        description="Optional note or label for the initial entry-point node.",
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
        description=(
            "Transition trigger: 'when_intent_matched' fires when conditionValue intent is recognised, "
            "'when_no_intent_matched' is the fallback, 'auto' fires immediately without user input."
        ),
    )
    conditionValue: Optional[str] = Field(
        default=None,
        description="The intent name that triggers this transition (required when condition='when_intent_matched', empty otherwise).",
    )
    label: Optional[str] = Field(
        default=None,
        description="Optional display label for the transition arrow.",
    )
    sourceDirection: Optional[str] = Field(
        default=None,
        description="Visual anchor direction on the source node (e.g. 'Right', 'Left', 'Top', 'Bottom').",
    )
    targetDirection: Optional[str] = Field(
        default=None,
        description="Visual anchor direction on the target node (e.g. 'Right', 'Left', 'Top', 'Bottom').",
    )


class AgentInitialNodeSpec(BaseModel):
    """Schema for the initial node in an agent diagram."""
    description: Optional[str] = Field(
        default=None,
        description="Optional note or label for the initial entry-point node.",
    )


class AgentRagSpec(BaseModel):
    name: str = Field(description="Name of the RAG knowledge base (e.g., 'CustomerKB', 'ProductDocs')")


class SystemAgentSpec(BaseModel):
    """Schema for a complete agent diagram system."""
    systemName: str = Field(
        default="",
        description="Display name for the overall agent system (e.g. 'CustomerSupportAgent').",
    )
    hasInitialNode: bool = Field(
        default=True,
        description="Whether the diagram includes an initial entry-point node. Almost always True.",
    )
    initialNode: Optional[AgentInitialNodeSpec] = Field(
        default=None,
        description="Configuration for the initial entry-point node.",
    )
    intents: List[AgentIntentSpec] = Field(
        default_factory=list,
        description="All intent nodes in the agent diagram. Each has a TitleCase name and 3-4 training phrases.",
    )
    states: List[AgentStateSpec] = Field(
        min_length=1,
        description="All state nodes (at least one). Each has a camelCase name and 1-3 replies.",
    )
    transitions: List[AgentTransitionSpec] = Field(
        default_factory=list,
        description=(
            "Edges connecting states and intents. Always include an initial transition from 'initial' to the first state. "
            "Every state must have at least one exit path to avoid dead-ends."
        ),
    )
    ragElements: List[AgentRagSpec] = Field(default_factory=list, description="RAG knowledge bases used by agent states with replyType='rag'")


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
        description="New name when renaming a state or intent.",
    )
    replies: Optional[List[AgentReplySpec]] = Field(
        default=None,
        description="Replies for add_state. Each reply has text and replyType ('text' or 'llm').",
    )
    trainingPhrases: Optional[List[str]] = Field(
        default=None,
        description="Training phrases for add_intent (3-5 example phrases).",
    )
    intentName: Optional[str] = Field(
        default=None,
        description="Intent name to associate with a transition (used with add_transition).",
    )
    condition: Optional[str] = Field(
        default=None,
        description="Transition condition: 'when_intent_matched', 'when_no_intent_matched', or 'auto'.",
    )
    text: Optional[str] = Field(
        default=None,
        description="Reply text to add to a state (used with add_state_body).",
    )
    replyType: Optional[str] = Field(
        default=None,
        description="Type of reply: 'text' for scripted, 'llm' for AI-generated (used with add_state_body).",
    )
    trainingPhrase: Optional[str] = Field(
        default=None,
        description="Single training phrase to add to an existing intent (used with add_intent_training_phrase).",
    )

class AgentModification(BaseModel):
    action: str = Field(
        description=(
            "The modification operation: 'add_state', 'add_intent', 'modify_state', 'modify_intent', "
            "'add_transition', 'remove_transition', 'add_state_body', 'add_intent_training_phrase', "
            "'add_rag_element', or 'remove_element'."
        ),
    )
    target: AgentModificationTarget = Field(
        description="Identifies the element(s) to modify. Populate the relevant fields based on the action.",
    )
    changes: Optional[AgentModificationChanges] = Field(
        default=None,
        description="The new values to apply. Required for all actions except 'remove_element' and 'remove_transition'.",
    )

class AgentModificationResponse(BaseModel):
    modifications: List[AgentModification] = Field(
        min_length=1,
        description="One or more modification operations to apply to the agent diagram.",
    )
