"""Agent diagram prompts: single element, complete system and modification."""

from schemas import reply_type_help

from .fragments import EXACT_NAMES_RULE, MULTI_MOD_ARRAY_RULE, POSITION_DISCLAIMER


# replyType bullet lists are generated from the schema's ReplyType alias so the
# prompts can never drift from what the structured output accepts.
REPLY_TYPES_PROMPT = reply_type_help("   ")


AGENT_ACTIONS_BLOCK = """AVAILABLE ACTIONS:
- add_state: Create a new state. Set target.stateName, put replies [{text, replyType, ...}] in changes.
- add_intent: Create a new intent. Set target.intentName, put trainingPhrases ["phrase1","phrase2","phrase3"] in changes.
- modify_state / modify_intent: Rename elements (set changes.name).
- add_transition: Connect two states. Set target.sourceStateName (source state, or "initial") and
  target.targetStateName (target state); set changes.condition and, for "when_intent_matched",
  changes.intentName (the triggering intent).
- remove_transition: Disconnect two states. Set target.sourceStateName and target.targetStateName.
- add_state_body: Add an action to a state (changes.text, changes.replyType, plus type-specific fields).
- add_intent_training_phrase: Add example phrase to intent (changes.trainingPhrase).
- remove_element: Delete a state or intent.
- add_rag_element: Create a RAG knowledge base component. Set target.name to the KB name;
  optionally set changes.llm_name, changes.k, changes.embedding_provider.
- add_llm: Add an LLM configuration component. Set target.name; optionally changes.provider,
  changes.num_previous_messages, changes.global_context.
- add_tool: Add a tool component. Set target.name; set changes.description, changes.code (Python function source).
- add_skill: Add a skill component. Set target.name; set changes.content, optionally changes.description.
- add_workspace: Add a workspace component. Set target.name; set changes.path, optionally changes.writable.
- add_gui: Add a GUI page component. Set target.name; set changes.gui_id, optionally changes.persist,
  changes.is_form, changes.width."""


AGENT_RULES_BLOCK = f"""RULES:
1. For add_transition / remove_transition, always name the states with target.sourceStateName and
   target.targetStateName. For add_transition, changes.condition is usually "when_intent_matched"
   with changes.intentName set to the triggering intent.
2. {EXACT_NAMES_RULE} NEVER invent
   states or intents that are not listed in the current model.
3. {MULTI_MOD_ARRAY_RULE}
4. replyType options for state bodies. Type-specific fields sit next to text/replyType:
   for add_state_body they go directly in changes (e.g. changes.system_message); for add_state
   they go on each changes.replies[i] (e.g. changes.replies[0].system_message).
{REPLY_TYPES_PROMPT}
5. Example: "add a welcome state" → add_state with target.stateName="welcomeState", changes.replies=[{{text:"Welcome!", replyType:"text"}}]
6. Example: "add a greeting intent" → add_intent with target.intentName="GreetingIntent", changes.trainingPhrases=["hello","hi","hey there"]
7. CRITICAL for replyType="code": the "text" MUST be a complete Python function
   definition starting with "def <name>(session):" — never bare statements. The
   generated code is written verbatim into the agent's source file and the
   function name is extracted from the "def" line to be called at runtime; code
   without a "def" produces a broken agent.
   Example: "add a function that logs the user's message" → add_state_body with
   target.stateName="logState", changes.replyType="code",
   changes.text="def log_message(session):\\n    print(session.event.message)\"
8. NEVER remove or rename an element unless the user explicitly asks for it AND
   that element appears in the current model. If the request does not clearly
   map to one of the actions above (e.g. it is a single stray word or is
   ambiguous), return an empty modifications list rather than guessing.
9. Only ADD elements the user explicitly asked for. Do not delete existing
   elements as a side effect of an addition.
10. When you ADD a new state, ALSO emit an add_transition that connects it into
   the existing flow so it is never left orphaned: set target.sourceStateName
   (an existing state, or "initial") and target.targetStateName (the new
   state), and — if the transition is intent-triggered — changes.intentName.
   Only skip the transition if the user explicitly asked for a disconnected
   state."""


MODIFY_SYSTEM_PROMPT_AGENT = "\n\n".join([
    "You are a conversational agent modeling expert. The user wants to modify an agent diagram.",
    AGENT_ACTIONS_BLOCK,
    AGENT_RULES_BLOCK,
])


SYSTEM_PROMPT = f"""You are a conversational agent modeling expert. Create a SINGLE agent diagram element specification.

IMPORTANT RULES:
1. Provide the "type" field (state, intent, or initial) based on the user request.
2. For states include 1-3 "replies". Each reply has a "replyType" (default to "text"; use
   "code" only if the user explicitly asks for custom Python logic):
{REPLY_TYPES_PROMPT}
3. Add "fallbackBodies" only when the request mentions fallbacks or error handling.
4. For intents include 3-4 "trainingPhrases" that reflect how a user would trigger the intent.
5. Keep names concise (camelCase for states, TitleCase for intents).
6. {POSITION_DISCLAIMER}"""


COMPLETE_SYSTEM_PROMPT = f"""You are a conversational agent modeling expert. Create a COMPLETE agent diagram specification.

Before generating, think through:
- What conversation states does this agent need?
- What user intents trigger transitions between states?
- What does the agent reply in each state?
- Are there fallback paths for unrecognized input?
- Is every state reachable and does every state have an exit?
- What LLMs, RAG databases, tools, skills, workspaces, or GUI pages does the agent need?

IMPORTANT RULES:
1. SCOPE: match the agent's size to the request. A plain request gets a focused conversation flow (4-8 states); only build a bigger flow when the user explicitly asks for a comprehensive agent or lists many scenarios themselves.
2. Each state can have MULTIPLE replies/actions via "replyType" (use "code" ONLY when the
   user explicitly asks for custom Python logic):
{REPLY_TYPES_PROMPT}
3. AVOID DEAD-ENDS: Every state MUST have at least one exit path.
4. States can have MULTIPLE transitions.
5. Transition types:
   - "when_intent_matched" with "conditionValue" = intent name
   - "when_no_intent_matched" as fallback (conditionValue = "")
   - "auto" for immediate continuation without waiting (conditionValue = "")
6. Always include an initial transition from "initial" to the first state.
7. Keep names consistent (camelCase for states, TitleCase for intents).
8. Include "sourceDirection" and "targetDirection" for visual flow.
9. FallbackBodies are optional.
10. {POSITION_DISCLAIMER}
11. AGENT COMPONENTS (add to the corresponding lists when the agent needs them):
    - llms: LLM configurations (name, provider, num_previous_messages, global_context)
    - ragElements: RAG knowledge bases (name, llm_name, k, embedding_provider)
    - tools: callable tools for reasoning states (name, description, code)
    - skills: knowledge/instructions for reasoning states (name, content, description)
    - workspaces: filesystem access (name, path, writable)
    - guis: GUI pages referenced by gui_reply actions (gui_id, persist, is_form, width)"""
