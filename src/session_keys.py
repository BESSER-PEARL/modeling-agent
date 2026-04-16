"""Session key constants used across the modeling agent.

Centralizes magic string keys to prevent typos and enable IDE navigation.
"""

# Pending state keys
PENDING_COMPLETE_SYSTEM = "pending_complete_system"
PENDING_GUI_CHOICE = "pending_gui_choice"

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
