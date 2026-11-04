"""Session key constants used across the modeling agent.

Centralizes magic string keys to prevent typos and enable IDE navigation.
"""

# Every pending dialog (replace/keep, GUI mode, Spec-Driven confirmation,
# generator menu and settings) and its stash, as one dict of slots. Read and
# write it only through common.pending_dialogs.
PENDING_DIALOGS = "_pending_dialogs"

# Diagram tracking
LAST_EXECUTED_DIAGRAM_TYPE = "_last_executed_diagram_type"

# Intent tracking
LAST_MATCHED_INTENT = "last_matched_intent"

# Greeting state
HAS_GREETED = "has_greeted"

# Unix timestamp of the most recent COMPLETED smart/Spec-Driven generation.
# The unified classifier reads this (freshness-gated) to apply its SMART-GEN
# FOLLOW-UP rule from a structured signal instead of grepping reply copy out
# of the conversation history.
LAST_SMART_GEN_AT = "_last_smart_gen_at"
# Project id (``BesserProject.id``) the run recorded in LAST_SMART_GEN_AT
# belonged to. The BAF session is keyed on the stable per-browser user_id and
# SURVIVES a project switch (the frontend only rotates the payload sessionId,
# which scopes conversation memory — see AssistantClient.resetSession), so the
# timestamp alone says "some app was generated recently", NOT "an app exists
# for the project you are looking at" (a new project would otherwise get the
# fix/modify copy for a run from a previous one). Every consumer of
# LAST_SMART_GEN_AT must pair it with this id (see
# routing.generation_signals.recent_smart_gen_for_project).
LAST_SMART_GEN_PROJECT_ID = "_last_smart_gen_project_id"
# Project id observed when a Spec-Driven Agent run was ARMED (stash created).
# Promoted to LAST_SMART_GEN_PROJECT_ID when the run reports success. Arm time
# is the only place the project is knowable: the completion callback arrives as
# a ``frontend_event``, which carries no workspace context at all (see
# AssistantClient.sendFrontendEvent). Deliberately NOT a pending-dialog
# slot, so _clear_pending_smart_gen (which runs just before the run fires)
# doesn't wipe the attribution.
SMART_GEN_ARMED_PROJECT_ID = "_smart_gen_armed_project_id"
# Human-readable one-paragraph summary of the last completed smart run
# (outcome + file summary). Lets a follow-up QUESTION about the finished
# run ("what we generated?") be ANSWERED instead of re-arming a brand-new
# generation confirmation.
LAST_SMART_GEN_SUMMARY = "_last_smart_gen_summary"
# The conversation and linked repo's recap last written to memory, so it enters once.
REMEMBERED_REPO_RECAP = "_remembered_repo_recap"

# Voice context
VOICE_CONTEXT = "_voice_context"

# Request caching
PARSED_ASSISTANT_REQUEST = "_parsed_assistant_request"
PARSED_REQUEST_EVENT_ID = "_parsed_request_event_id"

# Telemetry: id of the incoming event a "prompt" telemetry event was
# already emitted for, so one user message never produces two events even
# when several replies flow out (see session_helpers._emit_prompt_telemetry).
TELEMETRY_EMITTED_EVENT_ID = "_telemetry_emitted_event_id"

# Session history
SESSION_ACTION_HISTORY = "_session_action_history"

# Unified classifier per-message cache (see ``unified_classifier.py``).
# Stores a ``UnifiedClassification`` instance and the event id it was
# computed for, so multiple transition conditions / state bodies on
# the same message share one LLM call.
UNIFIED_CLASSIFICATION = "_unified_classification"
UNIFIED_CLASSIFICATION_EVENT_ID = "_unified_classification_event_id"

# The user's verbatim app description, captured when the class diagram was
# built from it. The Spec-Driven Agent payload otherwise carries only
# ``refined_instructions`` (a 1-3 paragraph LLM summary), which drops the
# status vocabularies, named actions and business rules the gap analyser
# needs to diff against the model.
ORIGINAL_APP_REQUEST = "_original_app_request"
ORIGINAL_APP_REQUEST_PROJECT_ID = "_original_app_request_project_id"
# What the user wants built for this project (``target_kind`` +
# ``target_description`` from the classifier), remembered next to the
# original request with the same project scoping. Latest statement wins.
APP_TARGET = "_app_target"
APP_TARGET_PROJECT_ID = "_app_target_project_id"

# ISO 639-1 language the user writes in (classifier ``reply_language``);
# templated replies are translated into it at the reply choke point.
REPLY_LANGUAGE = "_reply_language"
# Prompts of the quick actions in the last reply. A click sends the prompt
# (English machine text) as the message, which says nothing about the user's
# language, so it must not reset REPLY_LANGUAGE.
LAST_SUGGESTED_PROMPTS = "_last_suggested_prompts"
