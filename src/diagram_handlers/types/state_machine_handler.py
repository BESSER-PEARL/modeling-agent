"""
State Machine Diagram Handler
Handles generation of UML State Machine Diagrams.

Enhanced with:
- Two-pass generation (reasoning → structured JSON)
- Behavioral pattern injection for common domains
- Validation-feedback loop for state machine quality
- Richer prompts with transition design guidance
"""

import logging
from typing import Dict, Any, List, Optional
from ..core.base_handler import (
    BaseDiagramHandler,
    LLMPredictionError,
    SINGLE_STATE_REQUIRED,
    SINGLE_STATE_OPTIONAL,
    SYSTEM_STATE_REQUIRED,
    SYSTEM_STATE_OPTIONAL,
)
from schemas import SingleStateSpec as SingleStateSchema, SystemStateMachineSpec, StateMachineModificationResponse
from utilities.model_context import detailed_model_summary
from state_patterns import get_state_pattern_hint

logger = logging.getLogger(__name__)


class StateMachineHandler(BaseDiagramHandler):
    """Handler for State Machine Diagram generation"""

    def get_diagram_type(self) -> str:
        return "StateMachineDiagram"

    def get_system_prompt(self) -> str:
        return """You are a UML modeling expert. Create a state specification based on the user's request.

Return ONLY a JSON object with this structure:
{
  "stateName": "StateName",
  "stateType": "regular",
  "entryAction": "action on entry",
  "exitAction": "action on exit",
  "doActivity": "ongoing activity"
}

State Types: "initial", "final", "regular"

IMPORTANT RULES:
1. State names should be descriptive and represent real lifecycle stages (e.g., PendingPayment, Shipped, Authenticated — NOT generic names like State1, Active)
2. entryAction: one-time action when entering the state (e.g., "send confirmation email", "lock account")
3. exitAction: one-time action when leaving the state (e.g., "save progress", "release lock")
4. doActivity: ongoing activity while in the state (e.g., "await payment", "monitor session", "process data")
5. Use PascalCase for state names (e.g., PaymentProcessing, not payment_processing)
6. Keep it SIMPLE and focused
7. Do NOT include any "position" field - positioning is handled automatically
8. Return ONLY the JSON, no explanations

Examples:
- "create idle state" -> {"stateName": "Idle", "stateType": "regular", "entryAction": "", "exitAction": "", "doActivity": "await user input"}
- "create processing state" -> {"stateName": "Processing", "stateType": "regular", "entryAction": "start timer", "exitAction": "stop timer", "doActivity": "process data"}
- "create payment pending state" -> {"stateName": "PendingPayment", "stateType": "regular", "entryAction": "display payment form", "exitAction": "", "doActivity": "await payment confirmation"}

Return ONLY the JSON, no explanations."""

    def generate_single_element(self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Generate a single state with structured outputs and deterministic positioning."""

        system_prompt = self.get_system_prompt()
        user_prompt = f"Create a state specification for: {user_request}"

        try:
            parsed = self.predict_structured(
                user_prompt,
                SingleStateSchema,
                system_prompt=system_prompt,
            )
            state_spec = parsed.model_dump()

            # Remove any hallucinated position and apply deterministic layout
            state_spec.pop("position", None)
            self.apply_single_layout(state_spec, existing_model)

            return {
                "action": "inject_element",
                "element": state_spec,
                "diagramType": "StateMachineDiagram",
                "message": self._build_single_state_message(state_spec)
            }

        except LLMPredictionError as e:
            logger.error(f"[StateMachine] generate_single_element LLM FAILED: {e}")
            return self._error_response(
                "I couldn't generate that state. Please try again or rephrase your request.",
                code="llm_failure",
            )
        except Exception as e:
            logger.error(f"[StateMachine] generate_single_element FAILED: {e}", exc_info=True)
            return self.generate_fallback_element(user_request)

    def _get_system_generation_prompt(self) -> str:
        """Return the enhanced system prompt for complete state machine generation."""
        return """You are a UML modeling expert. Create a COMPLETE, well-structured state machine diagram.

Return ONLY a JSON object with this structure:
{
  "systemName": "StateMachineName",
  "states": [
    {
      "stateName": "StateName",
      "stateType": "regular",
      "entryAction": "action",
      "exitAction": "action",
      "doActivity": "activity"
    }
  ],
  "transitions": [
    {
      "source": "StateA",
      "target": "StateB",
      "trigger": "event",
      "guard": "condition",
      "effect": "action"
    }
  ]
}

State Types: "initial", "final", "regular"

IMPORTANT RULES:
1. Always start with exactly ONE "initial" state
2. Include 4-8 regular states that represent REAL lifecycle stages
3. End with ONE "final" state
4. EVERY regular state MUST have at least one incoming and one outgoing transition (no orphan states)
5. State names should be domain-specific and descriptive:
   - GOOD: PendingPayment, Shipped, UnderReview, Authenticated, InProgress
   - BAD: State1, Active, Process, Step2
6. Transitions MUST have meaningful triggers (events that cause the transition):
   - GOOD: submitPayment, approveRequest, sessionTimeout, deliveryConfirmed
   - BAD: go, next, transition1, move
7. Use guards to express conditions: [guard: "payment valid"], [guard: "attempts < max"]
8. Use effects for side-effects: [effect: "send notification"], [effect: "update inventory"]
9. Include error/exception paths — not just the happy path:
   - Payment failures, validation errors, timeouts, cancellations
   - Loop-back transitions for retry scenarios
10. entryAction: one-time action on state entry (e.g., "send email", "start timer")
11. doActivity: ongoing activity during the state (e.g., "await response", "monitor progress")
12. Do NOT include any "position" field - positioning is handled automatically

TRANSITION DESIGN GUIDELINES:
- Every state (except initial/final) should have both incoming AND outgoing transitions
- Include at least one alternative/error path (not just the happy flow)
- Self-transitions are valid for retry/refresh scenarios
- Guard conditions should be specific and testable
- Trigger names should be verbs or verb phrases in camelCase

Return ONLY the JSON, no explanations."""

    def generate_complete_system(self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Generate a complete state machine with two-pass structured outputs, pattern injection,
        validation-feedback loop, and deterministic layout."""

        system_prompt = self._get_system_generation_prompt()

        # Inject behavioral pattern reference if the request matches a known domain
        pattern_hint = get_state_pattern_hint(user_request)
        if pattern_hint:
            system_prompt += "\n\n" + pattern_hint

        logger.info(f"[StateMachine] generate_complete_system called with: {user_request!r}")

        try:
            # --- Two-pass structured: reason about behavior first, then produce validated model ---
            reasoning_prompt = (
                "You are a UML behavioral modeling expert. Think step by step about "
                "the following state machine request and plan the design.\n\n"
                f"User Request: {user_request}\n\n"
                "Analyze:\n"
                "1. What are the key lifecycle stages (states) of this process?\n"
                "2. What events (triggers) cause transitions between states?\n"
                "3. What conditions (guards) determine which path to take?\n"
                "4. What side effects (effects) occur during transitions?\n"
                "5. What error/exception paths exist? (timeouts, failures, cancellations)\n"
                "6. Are there any loop-back or retry scenarios?\n"
                "7. What entry/exit actions and ongoing activities does each state have?\n\n"
                "Provide a clear behavioral analysis. Focus on TRANSITIONS — they are "
                "the most commonly under-specified element."
            )

            parsed = self.predict_two_pass_structured(
                user_request=user_request,
                system_prompt=system_prompt,
                reasoning_prompt=reasoning_prompt,
                response_schema=SystemStateMachineSpec,
            )
            system_spec = parsed.model_dump()

            # --- Validation-feedback loop for state machine quality ---
            system_spec = self._validate_and_refine_state_machine(system_spec, user_request)

            # Strip any hallucinated positions and apply deterministic layout
            for s in system_spec.get("states", []):
                s.pop("position", None)
            self.apply_system_layout(system_spec, existing_model)

            return {
                "action": "inject_complete_system",
                "systemSpec": system_spec,
                "diagramType": "StateMachineDiagram",
                "message": self._build_system_message(system_spec)
            }

        except LLMPredictionError as e:
            logger.error(f"[StateMachine] generate_complete_system LLM FAILED: {e}")
            return self._error_response(
                "I couldn't generate that state machine. Please try again or rephrase your request.",
                code="llm_failure",
            )
        except Exception as e:
            logger.error(f"[StateMachine] generate_complete_system FAILED: {e}", exc_info=True)
            return self.generate_fallback_system()

    # ------------------------------------------------------------------
    # State Machine Validation & Refinement
    # ------------------------------------------------------------------

    def _validate_and_refine_state_machine(
        self, spec: Dict[str, Any], user_request: str,
    ) -> Dict[str, Any]:
        """Validate a generated state machine and fix common issues.

        Checks for:
        - Missing initial/final states
        - Orphan states (no incoming or outgoing transitions)
        - States with only generic names
        - Missing error/alternative paths
        """
        states = spec.get("states", [])
        transitions = spec.get("transitions", [])

        if not states:
            return spec

        # Check for orphan states (no transitions connecting them)
        state_names = {s.get("stateName") for s in states}
        sources = {t.get("source") for t in transitions}
        targets = {t.get("target") for t in transitions}
        connected = sources | targets

        orphans = []
        for s in states:
            name = s.get("stateName", "")
            stype = s.get("stateType", "regular")
            if stype == "initial" and name not in sources:
                orphans.append(name)
            elif stype == "final" and name not in targets:
                orphans.append(name)
            elif stype == "regular" and name not in connected:
                orphans.append(name)

        # Check for missing initial state
        has_initial = any(s.get("stateType") == "initial" for s in states)
        has_final = any(s.get("stateType") == "final" for s in states)

        if not has_initial:
            # Pick a name that doesn't collide with existing states
            initial_name = "Initial"
            if initial_name in state_names:
                initial_name = "InitialNode"
            states.insert(0, {
                "stateName": initial_name,
                "stateType": "initial",
                "entryAction": "",
                "exitAction": "",
                "doActivity": "",
            })
            first_regular = next(
                (s.get("stateName") for s in states if s.get("stateType") == "regular"),
                None,
            )
            if first_regular:
                transitions.insert(0, {
                    "source": initial_name,
                    "target": first_regular,
                    "trigger": "start",
                    "guard": "",
                    "effect": "",
                })
            logger.info("[StateMachine] Validation: added missing initial state")

        if not has_final:
            final_name = "Final"
            if final_name in state_names:
                final_name = "FinalNode"
            states.append({
                "stateName": final_name,
                "stateType": "final",
                "entryAction": "",
                "exitAction": "",
                "doActivity": "",
            })
            last_regular = None
            for s in reversed(states):
                if s.get("stateType") == "regular":
                    last_regular = s.get("stateName")
                    break
            if last_regular:
                transitions.append({
                    "source": last_regular,
                    "target": final_name,
                    "trigger": "complete",
                    "guard": "",
                    "effect": "",
                })
            logger.info("[StateMachine] Validation: added missing final state")

        if orphans:
            logger.info(f"[StateMachine] Validation: found {len(orphans)} orphan state(s): {orphans}")

        spec["states"] = states
        spec["transitions"] = transitions
        return spec
    
    def generate_fallback_element(self, request: str) -> Dict[str, Any]:
        """Generate a fallback state when AI generation fails"""
        state_name = self.extract_name_from_request(request, "NewState")
        
        fallback_spec = {
            "stateName": state_name,
            "stateType": "regular",
            "entryAction": "",
            "exitAction": "",
            "doActivity": ""
        }

        # Apply deterministic layout so the fallback doesn't render at 0,0
        self.apply_single_layout(fallback_spec)
        
        return {
            "action": "inject_element",
            "element": fallback_spec,
            "diagramType": "StateMachineDiagram",
            "message": f"I created a starter **{state_name}** state. Describe the state machine flow in more detail and I'll add transitions and guards!"
        }
    
    def generate_fallback_system(self) -> Dict[str, Any]:
        """Generate a fallback state machine"""
        fallback_system = {
            "systemName": "BasicStateMachine",
            "states": [
                {
                    "stateName": "Initial",
                    "stateType": "initial",
                    "entryAction": "",
                    "exitAction": "",
                    "doActivity": ""
                },
                {
                    "stateName": "Active",
                    "stateType": "regular",
                    "entryAction": "",
                    "exitAction": "",
                    "doActivity": ""
                },
                {
                    "stateName": "Final",
                    "stateType": "final",
                    "entryAction": "",
                    "exitAction": "",
                    "doActivity": ""
                }
            ],
            "transitions": [
                {
                    "source": "Initial",
                    "target": "Active",
                    "trigger": "start",
                    "guard": "",
                    "effect": ""
                },
                {
                    "source": "Active",
                    "target": "Final",
                    "trigger": "end",
                    "guard": "",
                    "effect": ""
                }
            ]
        }

        # Apply deterministic layout so the fallback doesn't render at 0,0
        self.apply_system_layout(fallback_system)

        return {
            "action": "inject_complete_system",
            "systemSpec": fallback_system,
            "diagramType": "StateMachineDiagram",
            "message": "I created a starter state machine with a basic state. Describe your workflow (e.g. *'Create an order processing flow with states: pending, confirmed, shipped, delivered'*) and I'll build a richer model!"
        }

    # ------------------------------------------------------------------
    # Message Builders
    # ------------------------------------------------------------------

    def _build_single_state_message(self, spec: Dict[str, Any]) -> str:
        """Build a descriptive message for a single state creation."""
        name = spec.get("stateName", "State")
        state_type = spec.get("stateType", "regular")
        entry = spec.get("entryAction", "")
        msg = f"Created a **{name}** state (type: {state_type})"
        if entry:
            msg += f" with entry action: *{entry}*"
        msg += ". You can ask me to add transitions, guards, or more states!"
        return msg

    def _build_system_message(self, spec: Dict[str, Any]) -> str:
        """Build a descriptive message for a complete state machine."""
        system_name = spec.get("systemName", "StateMachine")
        states = spec.get("states", [])
        transitions = spec.get("transitions", [])
        state_names = [s.get("stateName", "?") for s in states if s.get("stateType") == "regular"][:6]
        msg = f"Built the **{system_name}** state machine with {len(states)} state(s)"
        if state_names:
            msg += f": {', '.join(f'**{n}**' for n in state_names)}"
            if len(states) > len(state_names) + 2:  # account for initial/final
                msg += " and more"
        if transitions:
            msg += f", connected by {len(transitions)} transition(s)"
        msg += ". Feel free to ask me to add or modify states and transitions!"
        return msg

    # ------------------------------------------------------------------
    # Modification Support
    # ------------------------------------------------------------------

    def generate_modification(self, user_request: str, current_model: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Generate modifications for existing state machine elements."""

        system_prompt = """You are a UML modeling expert. The user wants to modify an existing state machine diagram.

Return ONLY a JSON object with one of these structures:

MODIFY STATE (rename or change properties)
{
  "action": "modify_model",
  "modification": {
    "action": "modify_state",
    "target": {
      "stateName": "CurrentStateName"
    },
    "changes": {
      "name": "NewStateName",
      "entryAction": "new entry action",
      "exitAction": "new exit action",
      "doActivity": "new activity"
    }
  }
}

ADD TRANSITION (connect two states)
{
  "action": "modify_model",
  "modification": {
    "action": "add_transition",
    "target": {
      "sourceState": "SourceState",
      "targetState": "TargetState"
    },
    "changes": {
      "trigger": "event",
      "guard": "condition",
      "effect": "action"
    }
  }
}

MODIFY TRANSITION (change existing transition properties)
{
  "action": "modify_model",
  "modification": {
    "action": "modify_transition",
    "target": {
      "sourceState": "SourceState",
      "targetState": "TargetState"
    },
    "changes": {
      "trigger": "newTrigger",
      "guard": "newGuard",
      "effect": "newEffect"
    }
  }
}

REMOVE ELEMENT (delete a state or transition)
{
  "action": "modify_model",
  "modification": {
    "action": "remove_element",
    "target": {
      "stateName": "StateToRemove"
    }
  }
}

OR for removing a transition:
{
  "action": "modify_model",
  "modification": {
    "action": "remove_element",
    "target": {
      "sourceState": "SourceState",
      "targetState": "TargetState"
    }
  }
}

IMPORTANT RULES:
1. Actions available: "modify_state", "add_transition", "modify_transition", "remove_element"
2. Always specify exact target names that exist in the current model
3. guard and effect are optional (can be empty strings)
4. For remove_element, only specify the target — no "changes" needed
5. When modifying, only include the fields that should change in "changes" object
6. When the user asks for MULTIPLE changes at once (e.g., "add states Idle, Running, and Done"), use the "modifications" array format:
   { "action": "modify_model", "modifications": [ { "action": "...", "target": {...}, "changes": {...} }, ... ] }
7. Use "modification" (singular) for a single change, "modifications" (plural array) for multiple changes
8. Return ONLY the JSON object — no explanations or markdown

Return ONLY the JSON object — no explanations"""

        # Build context from current model using centralized helper
        context_block = ''
        if current_model and isinstance(current_model, dict):
            summary = detailed_model_summary(current_model, 'StateMachineDiagram')
            if summary:
                context_block = f"\n\n{summary}"

        user_prompt = f"Modify the state machine: {user_request}{context_block}"

        logger.info(f"[StateMachine] generate_modification called with: {user_request!r}")

        try:
            parsed = self.predict_structured(user_prompt, StateMachineModificationResponse, system_prompt=system_prompt)
            mod_list = parsed.model_dump()["modifications"]
            if len(mod_list) == 1:
                modification_spec = {"action": "modify_model", "modification": mod_list[0]}
            else:
                modification_spec = {"action": "modify_model", "modifications": mod_list}

            self.validate_modification_spec(modification_spec)

            modification_spec.setdefault('action', 'modify_model')
            modification_spec.setdefault('diagramType', self.get_diagram_type())

            if 'message' not in modification_spec:
                if 'modifications' in modification_spec and isinstance(modification_spec['modifications'], list):
                    modification_spec['message'] = self._friendly_batch_message(modification_spec['modifications'])
                elif 'modification' in modification_spec and isinstance(modification_spec['modification'], dict):
                    mod = modification_spec['modification']
                    act = mod.get('action', 'modification')
                    target = mod.get('target', {})
                    name = (
                        target.get('stateName')
                        or f"{target.get('sourceState', '?')} \u2192 {target.get('targetState', '?')}"
                    )
                    modification_spec['message'] = self._friendly_mod_message(act, name)

            return modification_spec

        except LLMPredictionError as exc:
            logger.error(f"[StateMachine] generate_modification LLM FAILED: {exc}")
            return self._error_response("I couldn't process that modification. Please try again or rephrase your request.")
        except Exception as exc:
            logger.error(f"[StateMachine] generate_modification FAILED: {exc}", exc_info=True)
            return {
                "action": "modify_model",
                "modification": {
                    "action": "modify_state",
                    "target": {"stateName": "Unknown"},
                    "changes": {"name": "ModifiedState"}
                },
                "diagramType": self.get_diagram_type(),
                "message": "I couldn't apply that modification automatically. Could you rephrase it? For example: *'Add a transition from idle to active'* or *'Rename state X to Y'*."
            }
