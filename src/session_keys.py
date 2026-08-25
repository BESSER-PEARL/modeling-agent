"""Session key constants used across the modeling agent.

Centralizes magic string keys to prevent typos and enable IDE navigation.
"""

# Pending state keys
PENDING_COMPLETE_SYSTEM = "pending_complete_system"
PENDING_GUI_CHOICE = "pending_gui_choice"

# Web-app pause (bulletproof): set when a "create a web app" plan builds a GUI.
# The plan's auto-generation op is STRIPPED at the source so nothing can auto-run
# on any execution path; this flag then drives the "generate the web app?" prompt
# once the GUI is built. The user triggers generation explicitly afterwards
# (exactly like the class-diagram flow, which has no generation op in its plan).
PENDING_WEBAPP_GENERATE = "_pending_webapp_generate"

# Generation pending state
PENDING_GENERATOR_TYPE = "pending_generator_type"
PENDING_GENERATOR_CONFIG = "pending_generator_config"
CONFIG_PROMPT_ATTEMPTS = "_config_prompt_attempts"

# Diagram tracking
LAST_EXECUTED_DIAGRAM_TYPE = "_last_executed_diagram_type"

# Intent tracking
LAST_MATCHED_INTENT = "last_matched_intent"

# Greeting state
HAS_GREETED = "has_greeted"

# Workflow state
WORKFLOW_PENDING_GENERATOR = "_workflow_pending_generator"

# Smart-gen confirmation gate: every path that would run the
# Spec-Driven Agent (which spends the USER'S OWN API key) stashes
# the smart-gen payload here and asks for explicit confirmation first.
# Also used by the domain-mismatch handoff: on "Update model + generate"
# the workflow_body picks the stash up after rebuilding the model.
PENDING_SMART_GEN_INSTRUCTIONS = "_pending_smart_gen_instructions"
PENDING_SMART_GEN_PROVIDER = "_pending_smart_gen_provider"
# Unix timestamp set whenever the stash is (re)created. Stashes older
# than the TTL are rejected so an abandoned flow can never hijack a
# later, unrelated request (B-2 stale-stash fix).
PENDING_SMART_GEN_TIMESTAMP = "_pending_smart_gen_timestamp"
# When set to True, the next smart-route classification skips the
# domain-mismatch guard. Used by the "Generate anyway" path so the same
# request (or its resend) doesn't loop on the confirmation question.
SKIP_MISMATCH_CHECK_ONCE = "_skip_mismatch_check_once"
# One-shot flag set when the domain-mismatch "Update model + generate"
# quick action is offered. That action rebuilds the class diagram via a
# plain create; this flag tells the model-build choke point to resume the
# stashed smart-gen handoff right after the rebuild (so "+ generate" is
# honored instead of leaving the user to click "Generate application"
# again). Consumed once, at the next complete-system build.
MISMATCH_REGEN_PENDING = "_mismatch_regen_pending"

# Voice context
VOICE_CONTEXT = "_voice_context"

# Request caching
PARSED_ASSISTANT_REQUEST = "_parsed_assistant_request"
PARSED_REQUEST_EVENT_ID = "_parsed_request_event_id"

# Session history
SESSION_ACTION_HISTORY = "_session_action_history"

# Unified classifier per-message cache (see ``unified_classifier.py``).
# Stores a ``UnifiedClassification`` instance and the event id it was
# computed for, so multiple transition conditions / state bodies on
# the same message share one LLM call.
UNIFIED_CLASSIFICATION = "_unified_classification"
UNIFIED_CLASSIFICATION_EVENT_ID = "_unified_classification_event_id"
