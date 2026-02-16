"""
State Machine Diagram Handler
Handles generation of UML State Machine Diagrams
"""

import logging
from typing import Dict, Any
from .base_handler import BaseDiagramHandler

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
    
    def generate_single_element(self, user_request: str, existing_model: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a single state with deterministic positioning."""
        
        system_prompt = self.get_system_prompt()
        user_prompt = f"Create a state specification for: {user_request}"
        
        try:
            response = self.predict_with_retry(f"{system_prompt}\n\nUser Request: {user_prompt}")
            
            json_text = self.clean_json_response(response)
            state_spec = self.parse_json_safely(json_text)
            
            if not state_spec:
                raise Exception("Failed to parse JSON response")
            
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
    
    def generate_complete_system(self, user_request: str, existing_model: Dict[str, Any] = None) -> Dict[str, Any]:
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
            
            json_text = self.clean_json_response(response)
            system_spec = self.parse_json_safely(json_text)
            
            if not system_spec:
                raise Exception("Failed to parse JSON response")
            
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
