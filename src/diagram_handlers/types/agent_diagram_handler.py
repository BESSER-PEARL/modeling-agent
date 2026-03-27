"""
Agent Diagram Handler
Handles generation of UML Agent Diagrams (multi-agent conversational flows)
"""

from typing import Dict, Any, List, Optional
import logging

from ..core.base_handler import BaseDiagramHandler, LLMPredictionError
from ..core.prompt_fragments import POSITION_DISCLAIMER
from schemas import AgentSingleElementSpec, SystemAgentSpec, AgentModificationResponse
from utilities.model_context import detailed_model_summary

# Get logger
logger = logging.getLogger(__name__)


class AgentDiagramHandler(BaseDiagramHandler):
    """Handler for Agent Diagram generation"""

    def get_diagram_type(self) -> str:
        return "AgentDiagram"

    def get_system_prompt(self) -> str:
        return f"""You are a conversational agent modeling expert. Create a SINGLE agent diagram element specification.

IMPORTANT RULES:
1. Provide the "type" field (state, intent, or initial) based on the user request.
2. For states include 1-3 "replies" with both "text" and "replyType" (text or llm).
3. Add "fallbackBodies" only when the request mentions fallbacks or error handling.
4. For intents include 3-4 "trainingPhrases" that reflect how a user would trigger the intent.
5. Keep names concise (camelCase for states, TitleCase for intents).
6. {POSITION_DISCLAIMER}"""

    def generate_single_element(self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Generate a single agent diagram element with deterministic positioning."""

        system_prompt = self.get_system_prompt()
        user_prompt = f"Create an agent diagram element specification for: {user_request}"

        try:
            parsed = self.predict_structured(user_prompt, AgentSingleElementSpec, system_prompt=system_prompt)
            agent_spec = parsed.model_dump()

            normalized_spec = self._normalize_single_element_spec(agent_spec, user_request)

            # Remove any hallucinated position and apply deterministic layout
            normalized_spec.pop("position", None)
            self.apply_single_layout(normalized_spec, existing_model)

            message = self._build_single_element_message(normalized_spec)

            return {
                "action": "inject_element",
                "element": normalized_spec,
                "diagramType": self.get_diagram_type(),
                "message": message
            }

        except LLMPredictionError:
            logger.error("[AgentDiagram] generate_single_element LLM FAILED", exc_info=True)
            return self._error_response("I couldn't generate that agent element. Please try again or rephrase your request.")
        except Exception:
            logger.error("[AgentDiagram] generate_single_element FAILED", exc_info=True)
            return self.generate_fallback_element(user_request)

    def generate_complete_system(self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Generate a complete agent conversation flow with deterministic positioning."""

        system_prompt = f"""You are a conversational agent modeling expert. Create a COMPLETE agent diagram specification.

Before generating, think through:
- What conversation states does this agent need?
- What user intents trigger transitions between states?
- What does the agent reply in each state?
- Are there fallback paths for unrecognized input?
- Is every state reachable and does every state have an exit?

IMPORTANT RULES:
1. Create AS MANY states and intents as needed for the conversation.
2. Each state can have MULTIPLE replies (text lines):
   - Use replyType="text" for scripted responses (most common)
   - Use replyType="llm" for AI-generated dynamic responses
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
10. {POSITION_DISCLAIMER}"""

        user_request_prompt = f"{user_request}"

        try:
            parsed = self.predict_structured(user_request_prompt, SystemAgentSpec, system_prompt=system_prompt)
            system_spec = parsed.model_dump()

            normalized_system = self._normalize_system_spec(system_spec, user_request)

            # Strip any hallucinated positions and apply deterministic layout
            for s in normalized_system.get("states", []):
                s.pop("position", None)
            for i in normalized_system.get("intents", []):
                i.pop("position", None)
            for n in normalized_system.get("initialNodes", []):
                if isinstance(n, dict):
                    n.pop("position", None)
            initial_node = normalized_system.get("initialNode")
            if isinstance(initial_node, dict):
                initial_node.pop("position", None)
            self.apply_system_layout(normalized_system, existing_model)

            states = normalized_system.get('states', [])
            intents = normalized_system.get('intents', [])
            sys_name = normalized_system.get('systemName', 'AgentSystem')
            state_names = [s.get('stateName', '?') for s in states[:5]]
            intent_names = [i.get('intentName', '?') for i in intents[:5]]
            parts = [f"Built the **{sys_name}** agent system"]
            if state_names:
                parts.append(f" with states: {', '.join(f'**{n}**' for n in state_names)}")
            if intent_names:
                parts.append(f" and intents: {', '.join(f'**{n}**' for n in intent_names)}")
            parts.append(". Feel free to ask me to add more conversation flows or modify existing ones!")
            message = "".join(parts)

            return {
                "action": "inject_complete_system",
                "systemSpec": normalized_system,
                "diagramType": self.get_diagram_type(),
                "message": message
            }

        except LLMPredictionError:
            logger.error("[AgentDiagram] generate_complete_system LLM FAILED", exc_info=True)
            return self._error_response("I couldn't generate that agent system. Please try again or rephrase your request.")
        except Exception:
            logger.error("[AgentDiagram] generate_complete_system FAILED", exc_info=True)
            return self.generate_fallback_system(user_request)

    def generate_fallback_element(self, request: str) -> Dict[str, Any]:
        """Generate a fallback agent element when AI generation fails"""
        state_name = self.extract_name_from_request(request, "support")
        fallback_spec = {
            "type": "state",
            "stateName": state_name.lower(),
            "replies": [
                {"text": f"How can I assist with {state_name}?", "replyType": "text"},
                {"text": "I'm here to help you move forward.", "replyType": "text"}
            ],
            "fallbackBodies": [
                {"text": "Let me know how else I can help.", "replyType": "text"}
            ]
        }

        # Apply deterministic positioning so the element doesn't render at 0,0
        self.apply_single_layout(fallback_spec, None)

        return {
            "action": "inject_element",
            "element": fallback_spec,
            "diagramType": self.get_diagram_type(),
            "message": f"I created a starter **{fallback_spec['stateName']}** agent state. Describe the conversation flow in more detail and I'll add intents, replies, and transitions!"
        }

    def generate_fallback_system(self, request: str = "Agent") -> Dict[str, Any]:
        """Generate a fallback agent system"""
        base_name = self.extract_name_from_request(request, "Assistant")
        system_spec = {
            "systemName": f"{base_name}AgentSystem",
            "hasInitialNode": True,
            "intents": [
                {
                    "type": "intent",
                    "intentName": "Greeting",
                    "trainingPhrases": ["hi", "hello", "hey"]
                },
                {
                    "type": "intent",
                    "intentName": "Support",
                    "trainingPhrases": ["I need help", "support please", "can you assist"]
                }
            ],
            "states": [
                {
                    "type": "state",
                    "stateName": "initialGreeting",
                    "replies": [
                        {"text": "Hi there!", "replyType": "text"},
                        {"text": "How are you doing today?", "replyType": "text"}
                    ],
                    "fallbackBodies": [
                        {"text": "If you need help just ask.", "replyType": "text"}
                    ]
                },
                {
                    "type": "state",
                    "stateName": "supportResponse",
                    "replies": [
                        {"text": "I'm sorry you're facing trouble.", "replyType": "text"},
                        {"text": "Let me gather some details to help.", "replyType": "text"}
                    ],
                    "fallbackBodies": [
                        {"text": "You can rephrase what you need help with.", "replyType": "text"}
                    ]
                }
            ],
            "transitions": [
                {
                    "source": "initial",
                    "target": "initialGreeting",
                    "condition": "when_intent_matched",
                    "conditionValue": "Greeting",
                    "label": ""
                },
                {
                    "source": "initialGreeting",
                    "target": "supportResponse",
                    "condition": "when_intent_matched",
                    "conditionValue": "Support",
                    "label": ""
                },
                {
                    "source": "supportResponse",
                    "target": "initialGreeting",
                    "condition": "auto",
                    "conditionValue": "",
                    "label": ""
                }
            ]
        }

        # Apply deterministic positioning so elements don't render at 0,0
        self.apply_system_layout(system_spec, None)

        return {
            "action": "inject_complete_system",
            "systemSpec": system_spec,
            "diagramType": self.get_diagram_type(),
            "message": "I created a starter chatbot agent with greeting and support flows. Describe your conversation scenarios in more detail (e.g. *'Create a pizza ordering bot with menu browsing and checkout'*) and I'll build a richer system!"
        }

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _normalize_single_element_spec(self, spec: Dict[str, Any], request: str) -> Dict[str, Any]:
        """Ensure single element spec matches converter expectations"""
        element_type = str(spec.get("type") or spec.get("elementType") or "").lower()

        if element_type == "intent":
            normalized_intent = self._normalize_intent_spec(spec, request)
            if not normalized_intent:
                raise ValueError("Intent specification requires at least one training phrase.")
            return normalized_intent

        if element_type in {"initial", "initialnode", "start"}:
            normalized_initial = {"type": "initial"}
            position = self._normalize_position(spec)
            if position:
                normalized_initial["position"] = position
            return normalized_initial

        # Default to state specification
        return self._normalize_state_spec(spec, request)

    def _normalize_state_spec(self, spec: Dict[str, Any], request: str) -> Dict[str, Any]:
        """Normalize a state specification"""
        state_name = spec.get("stateName") or spec.get("name")
        if not state_name:
            state_name = self.extract_name_from_request(request, "newState").lower()

        replies = self._normalize_reply_list(
            spec.get("replies") or spec.get("bodies") or spec.get("responses"),
            default_text=f"Response from {state_name} state"
        )
        fallback_bodies = self._normalize_reply_list(
            spec.get("fallbackBodies") or spec.get("fallbacks") or spec.get("fallbackReplies"),
            default_text=""
        )

        normalized_state = {
            "type": "state",
            "stateName": state_name,
            "replies": replies,
            "fallbackBodies": fallback_bodies
        }
        position = self._normalize_position(spec)
        if position:
            normalized_state["position"] = position
        return normalized_state

    def _normalize_intent_spec(self, spec: Dict[str, Any], request: str) -> Dict[str, Any]:
        """Normalize an intent specification. Returns None if no usable phrases."""
        intent_name = spec.get("intentName") or spec.get("name")
        if not intent_name:
            intent_name = self.extract_name_from_request(request, "Intent")

        raw_phrases = spec.get("trainingPhrases") or spec.get("intentBodies") or spec.get("examples") or []
        phrases: List[str] = []
        for entry in raw_phrases:
            if isinstance(entry, str):
                phrase = entry.strip()
                if phrase:
                    phrases.append(phrase)
            elif isinstance(entry, dict):
                text = (entry.get("text") or entry.get("phrase") or "").strip()
                if text:
                    phrases.append(text)

        if not phrases:
            return None

        normalized_intent = {
            "type": "intent",
            "intentName": intent_name,
            "trainingPhrases": phrases[:5]
        }
        position = self._normalize_position(spec)
        if position:
            normalized_intent["position"] = position
        return normalized_intent

    def _normalize_reply_list(self, replies: Any, default_text: str) -> List[Dict[str, str]]:
        """Normalize reply/fallback entries into structured dictionaries"""
        normalized: List[Dict[str, str]] = []
        if isinstance(replies, list):
            for entry in replies:
                if isinstance(entry, str):
                    text = entry.strip()
                    if text:
                        normalized.append({"text": text, "replyType": "text"})
                elif isinstance(entry, dict):
                    text = (
                        entry.get("text")
                        or entry.get("message")
                        or entry.get("name")
                        or ""
                    ).strip()
                    if not text:
                        continue
                    reply_type = entry.get("replyType") or entry.get("type") or "text"
                    normalized.append({"text": text, "replyType": reply_type})

        if not normalized and default_text:
            normalized.append({"text": default_text, "replyType": "text"})

        return normalized

    def _normalize_system_spec(self, spec: Dict[str, Any], request: str) -> Dict[str, Any]:
        """Normalize a complete agent system specification"""
        system_name = spec.get("systemName") or self.extract_name_from_request(request, "AgentSystem")

        intents = [
            normalized
            for intent_spec in spec.get("intents", [])
            for normalized in [self._normalize_intent_spec(intent_spec, request)]
            if normalized
        ]

        states = [
            self._normalize_state_spec(state_spec, request)
            for state_spec in spec.get("states", [])
        ]

        transitions: List[Dict[str, Any]] = []
        for transition in spec.get("transitions", []):
            if not transition:
                continue
            normalized = self._normalize_transition_spec(transition, states)
            if normalized:
                transitions.append(normalized)

        has_initial = bool(spec.get("hasInitialNode", True))
        if has_initial and states:
            has_initial_transition = any(t.get("source") == "initial" for t in transitions)
            if not has_initial_transition:
                transitions.insert(0, {
                    "source": "initial",
                    "target": states[0]["stateName"],
                    "condition": "auto",
                    "conditionValue": "",
                    "label": ""
                })

        normalized_system = {
            "systemName": system_name,
            "hasInitialNode": has_initial,
            "intents": intents,
            "states": states,
            "transitions": transitions
        }
        initial_position = self._normalize_position(spec.get("initialNode"))
        if not initial_position:
            initial_position = self._normalize_position({"position": spec.get("initialPosition")})
        if initial_position:
            normalized_system["initialNode"] = {"position": initial_position}
        return normalized_system

    def _normalize_position(self, spec: Any) -> Optional[Dict[str, int]]:
        """Normalize any x/y position payload into integer coordinates."""
        if not isinstance(spec, dict):
            return None

        position = spec.get("position")
        if isinstance(position, dict):
            x = position.get("x")
            y = position.get("y")
        else:
            x = spec.get("x")
            y = spec.get("y")

        try:
            if x is None or y is None:
                return None
            x_value = int(float(x))
            y_value = int(float(y))
            return {"x": x_value, "y": y_value}
        except (TypeError, ValueError):
            return None

    def _normalize_transition_spec(self, transition: Dict[str, Any], states: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Normalize transition specification"""
        if not states:
            return None

        state_names = {state["stateName"] for state in states}
        primary_state = next(iter(state_names), None)

        source = transition.get("source") or transition.get("from") or "initial"
        target = transition.get("target") or transition.get("to")
        if not target:
            target = primary_state or "initial"

        if source not in state_names and source != "initial":
            source = primary_state or "initial"

        if target not in state_names and target != "initial":
            target = primary_state

        if target is None or (target not in state_names and target != "initial"):
            return None

        condition = transition.get("condition") or transition.get("trigger") or "auto"
        condition_value = (
            transition.get("conditionValue")
            or transition.get("intent")
            or transition.get("triggerValue")
            or ""
        )
        label = transition.get("label") or transition.get("name") or ""

        normalized = {
            "source": source,
            "target": target,
            "condition": condition,
            "conditionValue": condition_value,
            "label": label
        }

        if transition.get("sourceDirection"):
            normalized["sourceDirection"] = transition["sourceDirection"]
        if transition.get("targetDirection"):
            normalized["targetDirection"] = transition["targetDirection"]

        return normalized

    def _build_single_element_message(self, spec: Dict[str, Any]) -> str:
        """Generate a friendly status message for a single element."""
        element_type = spec.get("type")

        if element_type == "intent":
            name = spec.get("intentName", "Intent")
            phrases = spec.get("trainingPhrases", [])
            phrase_preview = ", ".join(f'*"{p}"*' for p in phrases[:3])
            msg = f"Created the **{name}** intent"
            if phrase_preview:
                msg += f" with training phrases: {phrase_preview}"
                if len(phrases) > 3:
                    msg += f" (+{len(phrases) - 3} more)"
            msg += ". You can ask me to add more intents or connect them to states!"
            return msg

        if element_type == "initial":
            return "Created the **initial node** for the agent flow. Add states and intents to build out the conversation!"

        name = spec.get("stateName", "State")
        replies = spec.get("replies", [])
        msg = f"Created agent state **{name}**"
        if replies:
            msg += f" with {len(replies)} reply option(s)"
        msg += ". You can ask me to add transitions, intents, or more states!"
        return msg

    # ------------------------------------------------------------------
    # Modification Support (NEW)
    # ------------------------------------------------------------------

    def generate_modification(self, user_request: str, current_model: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Generate modifications for existing agent diagram elements"""
        
        system_prompt = """You are a conversational agent modeling expert. The user wants to modify an agent diagram.

AVAILABLE ACTIONS:
- add_state: Create a new state. Set target.stateName, put replies [{text, replyType}] in changes.
- add_intent: Create a new intent. Set target.intentName, put trainingPhrases ["phrase1","phrase2","phrase3"] in changes.
- modify_state / modify_intent: Rename elements (set changes.name).
- add_transition: Connect states (set target.sourceStateName, target.targetStateName, changes.condition, changes.intentName).
- remove_transition: Disconnect states.
- add_state_body: Add reply text to a state (changes.text, changes.replyType).
- add_intent_training_phrase: Add example phrase to intent (changes.trainingPhrase).
- remove_element: Delete a state or intent.
- add_rag_element: Create a RAG knowledge base element. Set target.name to the KB name.

RULES:
1. For transitions, "condition" is usually "when_intent_matched" with an "intentName".
2. For existing elements, use exact names from the current model.
3. Multiple changes → return multiple modification objects in the list.
4. replyType is "text" for scripted replies, "llm" for AI-generated.
5. Example: "add a welcome state" → add_state with target.stateName="welcomeState", changes.replies=[{text:"Welcome!", replyType:"text"}]
6. Example: "add a greeting intent" → add_intent with target.intentName="GreetingIntent", changes.trainingPhrases=["hello","hi","hey there"]"""

        # Build context from current model using centralized summariser
        context_block = ''
        if current_model and isinstance(current_model, dict):
            summary = detailed_model_summary(current_model, 'AgentDiagram')
            if summary and 'no model data' not in summary and 'no structured model' not in summary:
                context_block = '\n\n' + summary
        
        user_prompt = f"Modify the agent diagram: {user_request}{context_block}"
        
        try:
            return self._execute_modification(
                user_prompt, system_prompt, AgentModificationResponse,
            )

        except LLMPredictionError as e:
            logger.error(f"[AgentDiagram] generate_modification LLM FAILED: {e}")
            return self._error_response("I couldn't process that modification. Please try again or rephrase your request.")
        except Exception as e:
            logger.error(f"Error generating agent diagram modification: {e}")
            return self.generate_fallback_modification(user_request)
    
    def generate_fallback_modification(self, request: str) -> Dict[str, Any]:
        """Generate a fallback modification when AI generation fails"""
        return {
            "action": "modify_model",
            "modification": {
                "action": "modify_state",
                "target": {"stateName": "unknown"},
                "changes": {"name": "modifiedState"}
            },
            "diagramType": self.get_diagram_type(),
                "message": "I couldn't apply that modification automatically. Could you rephrase it? For example: *'Rename the greeting state to welcome'* or *'Add a new intent called OrderPizza'*."
        }

