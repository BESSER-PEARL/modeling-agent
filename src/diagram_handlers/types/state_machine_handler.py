"""
State Machine Diagram Handler
Handles generation of UML State Machine Diagrams
"""

import logging
from typing import Dict, Any
from ..core.base_handler import (
    BaseDiagramHandler,
    SINGLE_STATE_REQUIRED,
    SINGLE_STATE_OPTIONAL,
    SYSTEM_STATE_REQUIRED,
    SYSTEM_STATE_OPTIONAL,
)
from utilities.model_helpers import detailed_model_summary

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
1. State names should be descriptive (Idle, Processing, Complete)
2. entryAction, exitAction, doActivity are optional (can be empty strings)
3. Use camelCase for state names
4. Keep it SIMPLE and focused
5. Do NOT include any "position" field - positioning is handled automatically
6. Return ONLY the JSON, no explanations

Examples:
- "create idle state" -> {"stateName": "Idle", "stateType": "regular", "entryAction": "", "exitAction": "", "doActivity": ""}
- "create processing state" -> {"stateName": "Processing", "stateType": "regular", "entryAction": "start timer", "exitAction": "stop timer", "doActivity": "process data"}

Return ONLY the JSON, no explanations."""
    
    def generate_single_element(self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Generate a single state with deterministic positioning."""
        
        system_prompt = self.get_system_prompt()
        user_prompt = f"Create a state specification for: {user_request}"
        
        try:
            response = self.predict_with_retry(f"{system_prompt}\n\nUser Request: {user_prompt}")
            
            state_spec = self.parse_and_validate(
                response,
                required_keys=SINGLE_STATE_REQUIRED,
                optional_keys=SINGLE_STATE_OPTIONAL,
                label="StateMachine.single_element",
            )
            
            # Remove any hallucinated position and apply deterministic layout
            state_spec.pop("position", None)
            self.apply_single_layout(state_spec, existing_model)
            
            return {
                "action": "inject_element",
                "element": state_spec,
                "diagramType": "StateMachineDiagram",
                "message": f"Created state '{state_spec['stateName']}'."
            }
            
        except Exception as e:
            logger.error(f"[StateMachine] generate_single_element FAILED: {e}", exc_info=True)
            return self.generate_fallback_element(user_request)
    
    def generate_complete_system(self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Generate a complete state machine with deterministic positioning."""
        
        system_prompt = """You are a UML modeling expert. Create a COMPLETE state machine diagram.

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
1. Always start with ONE "initial" state
2. Include 3-6 regular states
3. End with ONE "final" state (optional)
4. Include meaningful transitions with triggers
5. Guards and effects are optional
6. Do NOT include any "position" field - positioning is handled automatically
7. Keep transitions logical and coherent

Return ONLY the JSON, no explanations."""
        
        try:
            response = self.predict_with_retry(f"{system_prompt}\n\nUser Request: {user_request}")
            
            system_spec = self.parse_and_validate(
                response,
                required_keys=SYSTEM_STATE_REQUIRED,
                optional_keys=SYSTEM_STATE_OPTIONAL,
                label="StateMachine.complete_system",
            )
            
            # Strip any hallucinated positions and apply deterministic layout
            for s in system_spec.get("states", []):
                s.pop("position", None)
            self.apply_system_layout(system_spec, existing_model)
            
            return {
                "action": "inject_complete_system",
                "systemSpec": system_spec,
                "diagramType": "StateMachineDiagram",
                "message": (
                    f"Created state machine '{system_spec.get('systemName', 'StateMachine')}' with "
                    f"{len(system_spec.get('states', []))} state(s) and "
                    f"{len(system_spec.get('transitions', []))} transition(s)."
                )
            }
            
        except Exception as e:
            logger.error(f"[StateMachine] generate_complete_system FAILED: {e}", exc_info=True)
            return self.generate_fallback_system()
    
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
            "message": f"Created a starter state '{state_name}'. Try describing your state machine in more detail."
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
            "message": "Created a starter state machine. Try describing your workflow in more detail for a richer result."
        }

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
        full_prompt = f"{system_prompt}\n\nUser Request: {user_prompt}"

        logger.info(f"[StateMachine] generate_modification called with: {user_request!r}")

        try:
            response = self.predict_with_retry(full_prompt)
            json_text = self.clean_json_response(response)
            modification_spec = self.parse_json_safely(json_text)

            if not modification_spec:
                raise ValueError(f"Failed to parse modification JSON: {json_text[:300]}")

            self.validate_modification_spec(modification_spec)

            modification_spec.setdefault('action', 'modify_model')
            modification_spec.setdefault('diagramType', self.get_diagram_type())

            if 'message' not in modification_spec:
                if 'modifications' in modification_spec and isinstance(modification_spec['modifications'], list):
                    mods = modification_spec['modifications']
                    actions_summary = ", ".join(m.get('action', '?') for m in mods)
                    target_names = set()
                    for m in mods:
                        t = m.get('target', {})
                        n = (
                            t.get('stateName')
                            or f"{t.get('sourceState', '?')} -> {t.get('targetState', '?')}"
                        )
                        target_names.add(n)
                    modification_spec['message'] = f"Applied {len(mods)} modifications ({actions_summary}) to {', '.join(target_names)}"
                else:
                    mod_action = modification_spec['modification'].get('action', 'modification')
                    target = modification_spec['modification'].get('target', {})
                    target_name = (
                        target.get('stateName')
                        or f"{target.get('sourceState', '?')} -> {target.get('targetState', '?')}"
                    )
                    modification_spec['message'] = f"Applied {mod_action} to {target_name}"

            return modification_spec

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
                "message": "Could not apply the modification automatically. Try rephrasing your request."
            }
