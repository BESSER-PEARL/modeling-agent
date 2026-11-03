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
from typing import Dict, Any, Optional, Tuple
from ..core.base_handler import (
    BaseDiagramHandler,
)
from llm.model_config import MODEL_GENERATION_LARGE, MODEL_GENERATION_SMALL, MODEL_REASONING
from prompts import state_machine as state_machine_prompts
from schemas import SingleStateSpec as SingleStateSchema, SystemStateMachineSpec, StateMachineModificationResponse
from utilities.model_context import detailed_model_summary
from protocol.actions import Action, inject_complete_system, inject_element


logger = logging.getLogger(__name__)


class StateMachineHandler(BaseDiagramHandler):
    """Handler for State Machine Diagram generation"""

    _SYSTEM_FAILURE_MESSAGE = "I couldn't generate that state machine. Please try again or rephrase your request."
    _SINGLE_FAILURE_MESSAGE = "I couldn't generate that state. Please try again or rephrase your request."
    _LLM_FAILURE_CODE = "llm_failure"

    def get_diagram_type(self) -> str:
        return "StateMachineDiagram"

    def get_system_prompt(self) -> str:
        return state_machine_prompts.SYSTEM_PROMPT

    def _prepare_single_element(self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs) -> Tuple[str, str]:
        return self.get_system_prompt(), f"Create a state specification for: {user_request}"

    def _build_single_element(
        self, user_request: str, existing_model: Dict[str, Any] = None, *, prepared=None, **kwargs,
    ) -> Dict[str, Any]:
        """Generate a single state with structured outputs and deterministic positioning."""

        system_prompt, user_prompt = prepared

        # Single element → SMALL generation tier (latency-sensitive).
        parsed = self.predict_structured(
            user_prompt,
            SingleStateSchema,
            system_prompt=system_prompt,
            model=MODEL_GENERATION_SMALL,
        )
        state_spec = parsed.model_dump()

        # Remove any hallucinated position and apply deterministic layout
        state_spec.pop("position", None)
        self.apply_single_layout(state_spec, existing_model)

        return inject_element(
            state_spec,
            "StateMachineDiagram",
            self._build_single_state_message(state_spec),
        )

    def _get_system_generation_prompt(self) -> str:
        """Return the enhanced system prompt for complete state machine generation."""
        return state_machine_prompts.SYSTEM_GENERATION_PROMPT

    def _prepare_system(self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs) -> str:
        return self._get_system_generation_prompt()

    def _build_system(
        self,
        user_request: str,
        existing_model: Dict[str, Any] = None,
        raw_request: Optional[str] = None,
        *,
        prepared=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a complete state machine with two-pass structured outputs, pattern injection,
        validation-feedback loop, and deterministic layout.

        ``raw_request`` is the original user message before context enrichment
        (conversation history / workspace block); it drives the fast-path
        length check and keeps the reasoning prompt lean. Falls back to the
        full ``user_request`` when not provided.
        """

        system_prompt = prepared

        # --- Two-pass structured: reason about behavior first, then produce validated model ---
        # The reasoning prompt uses the raw request only; the enriched
        # context (history + workspace block) reaches the structured pass
        # via ``user_request`` exactly once.
        reasoning_prompt = state_machine_prompts.reasoning_prompt(raw_request or user_request)

        # Complete-system generation → LARGE tier; reasoning pass on
        # the REASONING tier (see model_config).
        parsed = self.predict_two_pass_structured(
            user_request=user_request,
            system_prompt=system_prompt,
            reasoning_prompt=reasoning_prompt,
            response_schema=SystemStateMachineSpec,
            raw_request=raw_request,
            model=MODEL_GENERATION_LARGE,
            reasoning_model=MODEL_REASONING,
        )
        system_spec = parsed.model_dump()

        # --- Validation-feedback loop for state machine quality ---
        system_spec = self._validate_and_refine_state_machine(system_spec, user_request)

        # Strip any hallucinated positions and apply deterministic layout
        for s in system_spec.get("states", []):
            s.pop("position", None)
        self.apply_system_layout(system_spec, existing_model)

        return inject_complete_system(
            system_spec,
            "StateMachineDiagram",
            self._build_system_message(system_spec),
        )

    # ------------------------------------------------------------------
    # State Machine Validation & Refinement
    # ------------------------------------------------------------------

    def _validate_and_refine_state_machine(
        self, spec: Dict[str, Any], user_request: str,
    ) -> Dict[str, Any]:
        """Validate a generated state machine and fix common issues.

        Fixes:
        - Drops transitions whose source/target names don't exist (they would
          otherwise materialize as phantom states during layout).
        - Adds a missing initial / final state and wires it into the flow.
        - Connects orphan regular states (no incoming/outgoing transition) into
          the happy path so every state is reachable.
        """
        states = spec.get("states", [])
        transitions = spec.get("transitions", [])

        if not states:
            return spec

        state_names = {s.get("stateName") for s in states}

        # Validate transition endpoints: a transition referencing a state that
        # doesn't exist is dropped; otherwise it could appear as a phantom node
        # when the layout engine builds the graph.
        valid_transitions = []
        for t in transitions:
            src = t.get("source")
            tgt = t.get("target")
            if src in state_names and tgt in state_names:
                valid_transitions.append(t)
            else:
                logger.info(
                    "[StateMachine] Validation: dropping transition with unknown "
                    "endpoint(s): source=%r target=%r", src, tgt,
                )
        transitions = valid_transitions

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

        # Connect orphan regular states into the flow. An orphan has no
        # incoming AND no outgoing transition, so it floats disconnected.
        # Wire them into the happy path (initial -> orphan -> final) so the
        # diagram stays connected.
        sources = {t.get("source") for t in transitions}
        targets = {t.get("target") for t in transitions}
        initial_state = next(
            (s.get("stateName") for s in states if s.get("stateType") == "initial"), None,
        )
        final_state = next(
            (s.get("stateName") for s in states if s.get("stateType") == "final"), None,
        )
        for s in states:
            if s.get("stateType", "regular") != "regular":
                continue
            name = s.get("stateName", "")
            if not name or name in sources or name in targets:
                continue
            if initial_state and initial_state != name:
                transitions.append({
                    "source": initial_state,
                    "target": name,
                    "trigger": "",
                    "guard": "",
                    "effect": "",
                })
                sources.add(initial_state)
                targets.add(name)
            if final_state and final_state != name:
                transitions.append({
                    "source": name,
                    "target": final_state,
                    "trigger": "",
                    "guard": "",
                    "effect": "",
                })
                sources.add(name)
                targets.add(final_state)
            logger.info(
                "[StateMachine] Validation: connected orphan state %r into the flow", name,
            )

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
        
        return inject_element(
            fallback_spec,
            "StateMachineDiagram",
            f"I created a starter **{state_name}** state. Describe the state machine flow in more detail and I'll add transitions and guards!",
        )
    
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

        return inject_complete_system(
            fallback_system,
            "StateMachineDiagram",
            "I created a starter state machine with a basic state. Describe your workflow (e.g. *'Create an order processing flow with states: pending, confirmed, shipped, delivered'*) and I'll build a richer model!",
        )

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

    def _prepare_modification(self, user_request: str, current_model: Dict[str, Any] = None, **kwargs) -> Tuple[str, str]:
        system_prompt = state_machine_prompts.MODIFY_SYSTEM_PROMPT_STATE_MACHINE

        # Build context from current model using centralized helper
        context_block = ''
        if current_model and isinstance(current_model, dict):
            summary = detailed_model_summary(current_model, 'StateMachineDiagram')
            if summary:
                context_block = f"\n\n{summary}"

        return system_prompt, f"Modify the state machine: {user_request}{context_block}"

    def _build_modification(
        self, user_request: str, current_model: Dict[str, Any] = None, *, prepared=None, **kwargs,
    ) -> Dict[str, Any]:
        """Generate modifications for existing state machine elements."""
        system_prompt, user_prompt = prepared
        return self._execute_modification(
            user_prompt, system_prompt, StateMachineModificationResponse,
        )

    def generate_fallback_modification(self, request: str) -> Dict[str, Any]:
        return {
            "action": Action.MODIFY_MODEL,
            "modification": {
                "action": "modify_state",
                "target": {"stateName": "Unknown"},
                "changes": {"name": "ModifiedState"}
            },
            "diagramType": self.get_diagram_type(),
            "message": "I couldn't apply that modification automatically. Could you rephrase it? For example: *'Add a transition from idle to active'* or *'Rename state X to Y'*."
        }
