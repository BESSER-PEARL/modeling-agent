"""Pydantic schemas for Agent Diagram structured outputs."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AgentReplySpec(BaseModel):
    text: str = Field(
        description=(
            "Reply text, LLM prompt, or Python code depending on replyType. "
            "When replyType is 'code', this MUST be a complete function definition "
            "starting with 'def <name>(session):' — never bare statements."
        ),
    )
    replyType: Literal[
        "text", "llm", "llm_chat", "rag", "db_reply", "code",
        "web_crawl_llm",
        "ws_markdown", "ws_html", "ws_speech", "ws_options",
        "ws_location", "ws_file", "ws_image", "ws_dataframe", "ws_plotly",
        "gui_reply",
    ] = Field(
        default="text",
        description=(
            "Action type for this body element:\n"
            "  text          – scripted text reply (most common)\n"
            "  llm           – LLM-generated reply (set system_message / llm_name)\n"
            "  llm_chat      – LLM chat reply with conversation history\n"
            "  rag           – RAG knowledge-base lookup (set ragDatabaseName)\n"
            "  db_reply      – SQL / LLM database query (set dbSelectionType etc.)\n"
            "  code          – custom Python function (text must be a full def)\n"
            "  web_crawl_llm – crawl a URL then reply via LLM (set initial_url)\n"
            "  ws_markdown   – WebSocket reply as Markdown (set ws_message)\n"
            "  ws_html       – WebSocket reply as HTML (set ws_message)\n"
            "  ws_speech     – WebSocket text-to-speech reply (set ws_message)\n"
            "  ws_options    – WebSocket options/buttons (set ws_options)\n"
            "  ws_location   – WebSocket GPS location (set ws_latitude/ws_longitude)\n"
            "  ws_file       – WebSocket file transfer\n"
            "  ws_image      – WebSocket image transfer\n"
            "  ws_dataframe  – WebSocket dataframe\n"
            "  ws_plotly     – WebSocket Plotly chart\n"
            "  gui_reply     – show a GUI page (set guiId to the AgentGUI component id)"
        ),
    )

    # --- RAG fields ---
    ragDatabaseName: Optional[str] = Field(
        default=None,
        description="RAG knowledge base name (required when replyType is 'rag').",
    )

    # --- LLM / LLMChat fields ---
    system_message: Optional[str] = Field(
        default=None,
        description="System prompt / instruction for the LLM (for llm / llm_chat).",
    )
    llm_name: Optional[str] = Field(
        default=None,
        description="Name of the AgentLLM component to use (for llm / llm_chat / rag / db_reply).",
    )
    inputPromptMode: Optional[Literal["last_user_message", "custom"]] = Field(
        default=None,
        description="How to feed the user's message to the LLM: 'last_user_message' (default) or 'custom'.",
    )
    customInputPrompt: Optional[str] = Field(
        default=None,
        description="Custom input prompt when inputPromptMode is 'custom'.",
    )
    storeInSession: Optional[str] = Field(
        default=None,
        description="Session variable name to store the LLM output in.",
    )
    sendReply: Optional[bool] = Field(
        default=None,
        description="Whether to send the LLM output as a reply to the user (default true).",
    )

    # --- DB fields ---
    dbSelectionType: Optional[Literal["default", "custom"]] = Field(
        default=None,
        description="Which database to query: 'default' (agent's default DB) or 'custom' (set dbCustomName).",
    )
    dbCustomName: Optional[str] = Field(
        default=None,
        description="Custom database name when dbSelectionType is 'custom'.",
    )
    dbQueryMode: Optional[Literal["llm_query", "sql"]] = Field(
        default=None,
        description="Query mode: 'llm_query' (LLM writes the query) or 'sql' (raw SQL in dbSqlQuery).",
    )
    dbOperation: Optional[Literal["any", "create", "read", "update", "delete"]] = Field(
        default=None,
        description="DB operation constraint (default 'any').",
    )
    dbSqlQuery: Optional[str] = Field(
        default=None,
        description="Raw SQL query string (when dbQueryMode is 'sql').",
    )

    # --- WebCrawlLLM fields ---
    initial_url: Optional[str] = Field(
        default=None,
        description="Starting URL to crawl (required for web_crawl_llm).",
    )

    # --- WebSocket message fields ---
    ws_message: Optional[str] = Field(
        default=None,
        description="Message content for ws_markdown / ws_html / ws_speech.",
    )
    ws_options: Optional[str] = Field(
        default=None,
        description="Newline-separated list of options for ws_options.",
    )
    ws_latitude: Optional[float] = Field(
        default=None,
        description="Latitude for ws_location.",
    )
    ws_longitude: Optional[float] = Field(
        default=None,
        description="Longitude for ws_location.",
    )

    # --- GUI reply fields ---
    guiId: Optional[str] = Field(
        default=None,
        description="gui_id of the AgentGUI component to display (required for gui_reply).",
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
        description="Actions the agent executes when entering this state.",
    )
    fallbackBodies: List[AgentReplySpec] = Field(
        default_factory=list,
        description="Actions executed when no intent matches in this state.",
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
    intentDescription: Optional[str] = Field(
        default=None,
        description="Short description of what this intent represents.",
    )
    trainingPhrases: List[str] = Field(
        default_factory=list,
        description="Example user utterances that trigger this intent.",
    )


class AgentLLMSpec(BaseModel):
    name: str = Field(
        min_length=1,
        description="Unique name for this LLM configuration.",
    )
    provider: str = Field(
        default="openai",
        description="LLM provider (openai, anthropic, ollama, huggingface, mistral, google, etc.).",
    )
    num_previous_messages: int = Field(
        default=1,
        description="Number of previous conversation messages to include as context.",
    )
    global_context: Optional[str] = Field(
        default=None,
        description="Global system context applied to every LLM call using this model.",
    )


class AgentToolSpec(BaseModel):
    name: str = Field(min_length=1, description="Unique name for this tool.")
    description: str = Field(
        default="",
        description="What this tool does — used by the LLM to decide when to call it.",
    )
    code: str = Field(
        default="",
        description="Python function source code that implements the tool.",
    )


class AgentSkillSpec(BaseModel):
    name: str = Field(min_length=1, description="Unique name for this skill.")
    content: str = Field(
        default="",
        description="Skill instructions or content that the agent can draw on.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Short description of this skill.",
    )


class AgentWorkspaceSpec(BaseModel):
    name: str = Field(min_length=1, description="Unique name for this workspace.")
    path: str = Field(default="", description="Filesystem path for this workspace directory.")
    description: Optional[str] = Field(default=None, description="Short description of this workspace.")
    writable: bool = Field(default=True, description="Whether the agent can write to this workspace.")
    max_read_bytes: int = Field(default=200_000, description="Maximum bytes the agent may read at once.")


class AgentGUISpec(BaseModel):
    gui_id: str = Field(
        min_length=1,
        description="Unique identifier for this GUI page (matches the id in the GUI diagram).",
    )
    persist: bool = Field(
        default=True,
        description="Whether the GUI page persists between messages.",
    )
    width: Optional[str] = Field(
        default=None,
        description="CSS width for the GUI panel (e.g. '400px').",
    )
    is_form: bool = Field(
        default=False,
        description="Whether the GUI acts as a form that submits data back to the agent.",
    )


class AgentRagSpec(BaseModel):
    name: str = Field(description="Unique name for this RAG knowledge base.")
    llm_name: Optional[str] = Field(
        default=None,
        description="Name of the AgentLLM to use for RAG answer generation.",
    )
    llm_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt template for RAG answer generation.",
    )
    k: int = Field(
        default=4,
        description="Number of document chunks to retrieve per query.",
    )
    embedding_provider: str = Field(
        default="openai",
        description="Embedding model provider (openai, ollama).",
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
        description="Actions the agent executes in this state.",
    )
    fallbackBodies: List[AgentReplySpec] = Field(
        default_factory=list,
        description="Actions executed when no intent matches.",
    )
    # Intent fields
    intentName: Optional[str] = Field(
        default=None,
        max_length=30,
        description="Unique TitleCase intent name (required when type is intent).",
    )
    intentDescription: Optional[str] = Field(
        default=None,
        description="Short description of what this intent represents.",
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
        description="Intent components (go to the components section, no canvas bounds).",
    )
    states: List[AgentStateSpec] = Field(
        min_length=1,
        description="All state nodes in the agent diagram.",
    )
    transitions: List[AgentTransitionSpec] = Field(
        default_factory=list,
        description="Edges connecting states and intents.",
    )
    # Agent components (go to the components section without canvas bounds)
    ragElements: List[AgentRagSpec] = Field(
        default_factory=list,
        description="RAG knowledge-base components.",
    )
    llms: List[AgentLLMSpec] = Field(
        default_factory=list,
        description="LLM configuration components.",
    )
    tools: List[AgentToolSpec] = Field(
        default_factory=list,
        description="Tool components for reasoning states.",
    )
    skills: List[AgentSkillSpec] = Field(
        default_factory=list,
        description="Skill components for reasoning states.",
    )
    workspaces: List[AgentWorkspaceSpec] = Field(
        default_factory=list,
        description="Workspace (filesystem access) components.",
    )
    guis: List[AgentGUISpec] = Field(
        default_factory=list,
        description="GUI page components referenced by gui_reply actions.",
    )


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
    name: Optional[str] = Field(
        default=None,
        description="Component name for add_llm / add_tool / add_skill / add_workspace / add_gui / add_rag_element.",
    )


class AgentModificationChanges(BaseModel):
    name: Optional[str] = Field(
        default=None,
        max_length=60,
        description="New name when renaming a state, intent, or component.",
    )
    replies: Optional[List[AgentReplySpec]] = Field(
        default=None,
        description="Replies/actions for add_state.",
    )
    trainingPhrases: Optional[List[str]] = Field(
        default=None,
        description="Training phrases for add_intent.",
    )
    intentDescription: Optional[str] = Field(
        default=None,
        description="Intent description for add_intent.",
    )
    intentName: Optional[str] = Field(
        default=None,
        description="Intent name for a transition condition.",
    )
    condition: Optional[Literal["when_intent_matched", "when_no_intent_matched", "auto"]] = Field(
        default=None,
        description="Transition condition.",
    )

    # --- add_state_body fields ---
    text: Optional[str] = Field(
        default=None,
        description=(
            "Reply text / prompt / code for add_state_body. "
            "When replyType is 'code', MUST be a complete 'def <name>(session):' function."
        ),
    )
    replyType: Optional[Literal[
        "text", "llm", "llm_chat", "rag", "db_reply", "code",
        "web_crawl_llm",
        "ws_markdown", "ws_html", "ws_speech", "ws_options",
        "ws_location", "ws_file", "ws_image", "ws_dataframe", "ws_plotly",
        "gui_reply",
    ]] = Field(
        default=None,
        description="Action type for add_state_body.",
    )
    ragDatabaseName: Optional[str] = Field(default=None, description="RAG KB name (rag).")
    system_message: Optional[str] = Field(default=None, description="LLM system prompt (llm/llm_chat).")
    llm_name: Optional[str] = Field(default=None, description="LLM component name.")
    inputPromptMode: Optional[str] = Field(default=None, description="LLM input mode.")
    storeInSession: Optional[str] = Field(default=None, description="Session variable to store output.")
    sendReply: Optional[bool] = Field(default=None, description="Whether to send LLM output as reply.")
    dbSelectionType: Optional[str] = Field(default=None, description="DB selection (db_reply).")
    dbCustomName: Optional[str] = Field(default=None, description="Custom DB name (db_reply).")
    dbQueryMode: Optional[str] = Field(default=None, description="DB query mode (db_reply).")
    dbOperation: Optional[str] = Field(default=None, description="DB operation (db_reply).")
    dbSqlQuery: Optional[str] = Field(default=None, description="Raw SQL query (db_reply).")
    initial_url: Optional[str] = Field(default=None, description="Crawl start URL (web_crawl_llm).")
    ws_message: Optional[str] = Field(default=None, description="WS message content.")
    ws_options: Optional[str] = Field(default=None, description="WS options (newline-separated).")
    ws_latitude: Optional[float] = Field(default=None, description="WS location latitude.")
    ws_longitude: Optional[float] = Field(default=None, description="WS location longitude.")
    guiId: Optional[str] = Field(default=None, description="GUI component id (gui_reply).")

    # --- add_intent_training_phrase ---
    trainingPhrase: Optional[str] = Field(
        default=None,
        description="Single training phrase to add to an existing intent.",
    )

    # --- add_llm fields ---
    provider: Optional[str] = Field(default=None, description="LLM provider (for add_llm).")
    num_previous_messages: Optional[int] = Field(default=None, description="LLM context window in messages.")
    global_context: Optional[str] = Field(default=None, description="Global system context for the LLM.")

    # --- add_tool fields ---
    description: Optional[str] = Field(default=None, description="Tool / skill / workspace description.")
    code: Optional[str] = Field(default=None, description="Tool Python source code (for add_tool).")

    # --- add_skill fields ---
    content: Optional[str] = Field(default=None, description="Skill instructions / content (for add_skill).")

    # --- add_workspace fields ---
    path: Optional[str] = Field(default=None, description="Workspace filesystem path (for add_workspace).")
    writable: Optional[bool] = Field(default=None, description="Whether the workspace is writable.")

    # --- add_rag_element fields ---
    llm_prompt: Optional[str] = Field(default=None, description="RAG answer generation prompt.")
    k: Optional[int] = Field(default=None, description="Number of RAG chunks to retrieve.")
    embedding_provider: Optional[str] = Field(default=None, description="RAG embedding provider.")

    # --- add_gui fields ---
    gui_id: Optional[str] = Field(default=None, description="GUI page id (for add_gui).")
    persist: Optional[bool] = Field(default=None, description="GUI persist setting.")
    is_form: Optional[bool] = Field(default=None, description="Whether the GUI is a form.")
    width: Optional[str] = Field(default=None, description="GUI panel CSS width.")


class AgentModification(BaseModel):
    action: Literal[
        "add_state", "modify_state",
        "add_intent", "modify_intent",
        "add_transition", "remove_transition",
        "add_state_body", "add_intent_training_phrase",
        "add_rag_element", "add_llm", "add_tool", "add_skill", "add_workspace", "add_gui",
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
