API and Module Reference
========================

This document maps all Python modules in the Modeling Agent codebase.

.. contents:: On this page
   :local:
   :depth: 2

Entry Point
-----------

``modeling_agent.py``
  Creates the BESSER agent, defines 10 intents and 10 states, wires state
  bodies, starts the session reaper, and starts the WebSocket runtime.

Core Runtime Modules
--------------------

``src/agent_context.py``
  Shared runtime context container. Stores module-level globals (``agent``,
  ``gpt``, ``gpt_text``, ``gpt_predict_json``, ``uml_rag``, ``diagram_factory``,
  ``openai_api_key``, ``stt``) populated at startup.

``src/agent_config.py``
  Tunable constants: tab and message limits, session grace period, streaming
  buffer, LLM temperatures and token budgets, conversation history depth.

``src/model_config.py``
  Per-call-site model routing:

  - ``MODEL_CLASSIFIER``, ``MODEL_GENERATION_LARGE``, ``MODEL_GENERATION_GUI``,
    ``MODEL_GENERATION_SMALL``, ``MODEL_REASONING``, ``MODEL_VISION``,
    ``MODEL_EMBEDDINGS`` — all overridable via ``BESSER_AGENT_MODEL_*``
  - ``supports_custom_temperature(model)`` — False for gpt-5 / o-series
  - ``reasoning_effort_for(model)`` — the ``reasoning_effort`` to pass, or
    ``None`` for models that take a temperature instead

``src/agent_setup.py``
  Initialization functions called during startup:

  - ``init_llm(agent)`` — Creates the two shared ``LLMOpenAI`` instances
    (structured/JSON + free text) and returns ``(gpt, gpt_text,
    gpt_predict_json)``; also installs the shared-client retry patch
  - ``init_stt(agent)`` — OpenAI ``whisper-1`` speech-to-text
  - ``init_rag(agent)`` — Builds ChromaDB-backed RAG (returns ``None`` on failure)
  - ``init_diagram_factory(gpt)`` — Creates ``DiagramHandlerFactory``
  - ``init_intent_classifier_config()`` — Returns
    ``SimpleIntentClassifierConfiguration(framework='tensorflow')``, the
    local free classifier used as BAF's default

``src/unified_classifier.py``
  The router. One structured-output call per message:

  - ``UnifiedClassification`` — the wide Pydantic verdict schema
  - ``classify_message(request, llm_provider, history, recent_smart_gen, pending_flow)``
  - ``get_or_classify(session, request, llm_provider)`` — per-message cache
  - ``_SYSTEM_PROMPT`` — the authoritative routing rulebook

``src/byok.py``
  Per-request bring-your-own-key routing:

  - ``set_current(provider, api_key, model, base_url)`` / ``reset_current(token)``
  - ``get_active_client()`` — the per-request client, or ``None``
  - ``resolve_model(provider, requested_model, user_model)``
  - ``SUPPORTED_PROVIDERS`` — ``openai``, ``anthropic``, ``mistral``

``src/state_bodies.py``
  All state body functions and transition wiring:

  - ``register_all(*, agent, states, intents)`` — Wires state bodies and
    transitions (keyword-only)
  - ``add_unified_transitions(state, intents_map, fallback_state, generation_state)``
  - ``greetings_body(session)`` — Welcome message handler
  - ``create_complete_system_body(session)``
  - ``modify_modeling_body(session)``
  - ``modeling_help_body(session)`` — Conceptual Q&A
  - ``describe_model_body(session)`` — Model summarization
  - ``uml_rag_body(session)`` — RAG query handler
  - ``generation_body(session)`` — Code generation routing
  - ``decline_body`` / ``out_of_scope_body`` / ``meta_question_body``
  - ``global_fallback_body(session)``

``src/execution/`` (package)
  Operation execution engine:

  - ``execute_planned_operations(session, request, default_mode, matched_intent)`` — ``planning.py``
  - ``execute_model_operation(session, request, operation, default_mode, ...)`` — ``model_operations.py``
  - ``handle_file_attachments(session, request)`` — ``file_handling.py``

``src/confirmation.py``
  Pending confirmation flow handlers. Each reads its own pending state off the
  session and returns ``True`` when it consumed the message:

  - ``handle_pending_system_confirmation(session) -> bool`` — replace / keep /
    new tab
  - ``handle_pending_gui_choice(session) -> bool`` — deterministic vs.
    AI-generated GUI

``src/session_keys.py``
  Every session-state key the agent writes, in one place — the pending-flow
  keys (``PENDING_COMPLETE_SYSTEM``, ``PENDING_GUI_CHOICE``,
  ``PENDING_GENERATOR_TYPE`` / ``_CONFIG``, ``PENDING_SMART_GEN_INSTRUCTIONS``,
  ``PENDING_WEBAPP_GENERATE``, ``PLAN_GENERATION_CONFIRM_FLAG``), the
  classification cache keys (``UNIFIED_CLASSIFICATION`` /
  ``UNIFIED_CLASSIFICATION_EVENT_ID``), the parsed-request cache keys, and
  ``VOICE_CONTEXT``. Import the constant, never the literal string.

``src/model_utils.py``
  - ``model_has_elements(model)`` — the shared "is this model non-empty?" check

``src/session_helpers.py``
  Reply utilities and transition conditions:

  - ``reply_payload(session, payload)`` / ``reply_message(session, text)``
  - ``reply_stream_start`` / ``reply_stream_chunk`` / ``reply_stream_done``
  - ``reply_progress(session, message, step, total)``
  - ``stream_llm_response(...)`` — streams a free-text reply, routed through
    BYOK when a user key is active (fetched whole, then sent)
  - ``json_intent_matches(session, params)`` — the priority-1 transition
    condition; reads the unified classifier's verdict
  - ``json_no_intent_matched(session)`` — the fallback condition
  - ``route_to_generation(session)`` — priority-2 condition; frontend events
    and pending generator flows only, no LLM call
  - ``replay_last_reply(session, request)`` — re-sends the buffered terminal
    reply after a reconnect
  - ``get_user_message`` / ``get_diagram_type`` / ``get_current_model``

``src/telemetry.py``
  Fire-and-forget pilot-experiment prompt telemetry. Posts one ``prompt``
  event per handled message to ``{BESSER_BACKEND_URL}/besser_api/telemetry/event``
  on a short-timeout daemon thread, only when the request carries a
  ``pilotParticipant`` label. Every exception is swallowed.

``src/errors.py``
  Unified error taxonomy:

  - ``ErrorCode`` — the enum of error codes
  - ``classify_error(error)`` → ``ErrorCode``
  - ``get_recovery_hint(error_code)``
  - ``build_error_response(error_code, message, ...)``

``src/reply_copy.py``
  Shared user-facing copy (decline acknowledgement, meta answer,
  out-of-scope redirect, confirmation prompts) kept in one place.

``src/suggestions.py``
  Context-aware suggestion engine — see `Suggestions`_ below for the API.

Protocol Layer
--------------

``src/protocol/types.py``
  Protocol data classes:

  - ``AssistantRequest`` — Canonical request object
  - ``WorkspaceContext`` — Editor state snapshot
  - ``FileAttachment`` — Uploaded file metadata
  - ``SUPPORTED_DIAGRAM_TYPES`` — Set of valid diagram type strings

``src/protocol/adapters.py``
  Payload extraction and normalization:

  - ``parse_assistant_request(session)`` — Main entry point
  - ``extract_event_payload(session)`` — Extract from BESSER event
  - ``parse_v2_payload(payload)`` — Parse v2 protocol format
  - ``normalize_diagram_type(diagram_type)`` — Normalize type strings
  - ``strip_diagram_prefix(message)`` — Remove diagram type prefixes

Orchestration Layer
-------------------

``src/orchestrator/__init__.py``
  Re-exports for convenience.

``src/orchestrator/request_planner.py``
  Multi-operation planning:

  - ``plan_assistant_operations(session, request, default_mode, matched_intent)``
  - ``_should_use_llm_planner(message, matched_intent, inferred_target_count)`` — Complexity check with fast-path (returns ``False`` for single-target intents)
  - ``_fallback_operations(request, default_mode)`` — Heuristic operations
  - ``_normalize_operations(operations)`` — Deduplicate and validate

``src/orchestrator/workspace_orchestrator.py``
  Diagram type targeting:

  - ``determine_target_diagram_type(request, last_intent)`` — Three-level resolution
  - ``KEYWORD_TARGETS`` — Explicit keyword-to-type mappings

Diagram Handlers
----------------

``src/diagram_handlers/core/base_handler.py``
  Abstract base handler with shared infrastructure:

  - ``BaseDiagramHandler`` (abstract class; 5 abstract methods)
  - ``predict_with_retry(prompt, max_retries=1, *, model=None, ...)``
  - ``predict_structured(prompt, response_schema, *, system_prompt, temperature, model)``
  - ``predict_two_pass_structured(user_request, system_prompt, reasoning_prompt, response_schema, ...)``
  - ``predict_two_pass(...)``
  - ``generate_modification(user_request, current_model, **kwargs)``
  - ``validate_and_refine(...)`` / ``self_correct(...)`` / ``parse_validate_or_correct(...)``
  - ``repair_json_response(malformed_json, schema_hint)`` /
    ``parse_and_validate_with_repair(...)``
  - ``apply_single_layout(...)`` / ``apply_system_layout(...)``
  - ``LLMPredictionError``, ``validate_spec``

``src/diagram_handlers/core/layout_engine.py``
  Deterministic canvas layout. Public entry point plus per-type layouts:

  - ``apply_layout(...)`` — dispatches by diagram type and mode
  - ``layout_class_single`` / ``layout_class_system`` (Sugiyama layered layout)
  - ``layout_object_single`` / ``layout_object_system``
  - ``layout_state_single`` / ``layout_state_system``
  - ``layout_agent_single`` / ``layout_agent_system``
  - ``layout_user_single`` / ``layout_user_system``
  - ``estimate_class_size`` / ``estimate_object_size`` / ``estimate_state_size`` /
    ``estimate_agent_element_size``
  - ``extract_occupied_rects(...)``
  - Constants: ``H_GAP``, ``V_GAP``, ``REL_EXTRA_GAP``, ``MARGIN``,
    ``GRID_SNAP``, ``CANVAS_MIN_X`` … ``CANVAS_MAX_Y``

``src/diagram_handlers/core/prompt_fragments.py``
  Shared prompt snippets reused across handlers — ``EXACT_NAMES_RULE``,
  ``REMOVE_ELEMENT_RULE``, ``MULTI_MOD_ARRAY_RULE``, ``POSITION_DISCLAIMER``.

``src/diagram_handlers/registry/factory.py``
  Handler factory:

  - ``HANDLER_CLASSES`` — the registered handler classes
  - ``DiagramHandlerFactory(llm)`` — keys the registry on each handler's own
    ``get_diagram_type()``
  - ``get_handler(diagram_type) -> Optional[BaseDiagramHandler]``
  - ``get_supported_types() -> list[str]``
  - ``is_supported(diagram_type) -> bool``

``src/diagram_handlers/registry/metadata.py``
  Per-type display metadata (labels, descriptions).

``src/diagram_handlers/types/class_diagram_handler.py``
  - ``ClassDiagramHandler(llm)``
  - Domain pattern detection and injection
  - Two-pass generation with validation loop

``src/diagram_handlers/types/state_machine_handler.py``
  - ``StateMachineHandler(llm)``
  - State pattern detection and injection
  - Initial/final/orphan state validation

``src/diagram_handlers/types/object_diagram_handler.py``
  - ``ObjectDiagramHandler(llm)``
  - Class reference catalog extraction
  - Heuristic value generation

``src/diagram_handlers/types/agent_diagram_handler.py``
  - ``AgentDiagramHandler(llm)``
  - 7-step normalization pipeline

``src/diagram_handlers/types/gui_nocode_diagram_handler.py``
  - ``GUINoCodeDiagramHandler(llm)``
  - Auto-generate and LLM generation modes

``src/diagram_handlers/types/quantum_circuit_diagram_handler.py``
  - ``QuantumCircuitDiagramHandler(llm)``
  - 60+ gate symbol mappings

``src/diagram_handlers/types/bpmn_diagram_handler.py``
  - ``BPMNDiagramHandler(llm)`` — diagram type ``"BPMN"``
  - ``_validate_and_refine`` / ``_connect_orphaned_nodes`` /
    ``_normalize_pool_refs`` / ``_infer_missing_lane_owners`` — deterministic
    post-generation repair
  - ``_validate_mod_refs`` — rejects modifications naming a nonexistent node

``src/diagram_handlers/types/user_profile_handler.py``
  - ``UserProfileDiagramHandler(llm)`` — diagram type ``"UserDiagram"``
  - Reference catalog from the bundled metamodel; inferred comparison operators

``src/diagram_handlers/types/gui_design_system.py``
  - ``DOMAINS`` — ``government``, ``finance``, ``health``, ``startup``, ``default``
  - ``THEMES`` — per-domain design tokens
  - ``stylesheet_rules(domain)`` — GrapesJS CSS rule objects
  - ``block_exemplars(domain)`` — themed ``.ds-*`` HTML composition patterns

``src/diagram_handlers/types/gui_html_converter.py``
  LLM-authored HTML → GrapesJS component-definition tree, sanitized and
  widget-spoof guarded. ``find_widget_slots`` / ``replace_widget_slot`` splice
  real data-bound widgets into ``<!--WIDGET:kind-->`` markers.

Auxiliary Handlers
------------------

``src/handlers/generation_handler.py``
  Code generation routing:

  - ``handle_generation_request(session, request)``
  - ``should_route_to_generation(session, request)`` — the cheap transition
    gate (frontend events and pending flows only)
  - ``handle_pending_smart_gen_confirmation(session)``
  - ``handle_pending_plan_generation_confirmation(session)``
  - ``detect_generator_type(message)`` — pure keyword/regex detection
  - ``parse_inline_generator_config(...)``
  - ``recent_smart_gen_for_project(session, window_seconds)``
  - ``GENERATOR_KEYWORDS`` — Keyword-to-generator mappings
  - ``GENERATOR_REQUIRED_FIELDS`` — Required configuration fields per generator
  - ``GENERATOR_PREREQUISITES`` — Diagrams each generator consumes
  - ``EXPORT_FORMATS`` / ``DIALECT_VALUES`` / ``MODE_VALUES`` / ``QISKIT_BACKENDS``

``src/handlers/smart_generation_handler.py``
  Smart (LLM-authored) generation dispatch:

  - ``GenerationClassification`` — the sub-routing schema
  - ``build_trigger_smart_generator_payload(classification, reason_prefix)``

``src/handlers/file_conversion_handler.py``
  File upload conversion:

  - ``convert_file_to_diagram_spec(...)`` — the entry point
  - ``detect_file_type(filename, content_text)`` /
    ``detect_plantuml_diagram_type(content)``
  - Per-format converters (private): ``_convert_plantuml``,
    ``_convert_knowledge_graph``, ``_convert_xmi``, ``_convert_pdf``,
    ``_convert_image``, ``_convert_generic_text``
  - Per-type validators: ``_validate_class_diagram_spec``,
    ``_validate_state_machine_spec``, ``_validate_object_diagram_spec``,
    ``_validate_agent_diagram_spec``, ``_validate_bpmn_spec``
  - ``PLANTUML_EXTENSIONS``, ``KG_EXTENSIONS``, ``IMAGE_EXTENSIONS``,
    ``PDF_EXTENSIONS``, ``XMI_EXTENSIONS``, ``CONVERTIBLE_DIAGRAM_TYPES``

  .. note::

     ``handle_file_attachments(session, request)`` lives in
     ``src/execution/file_handling.py``, not here.

``src/handlers/validation_handler.py``
  Bridge to the BESSER backend's diagram validator:

  - ``validate_diagram(diagram_json, diagram_type, api_url=None)`` — POSTs to
    ``{BESSER_BACKEND_URL}/besser_api/validate-diagram``

Utilities
---------

``src/utilities/model_resolution.py``
  Target model resolution:

  - ``resolve_target_model(request, target_type)``
  - ``resolve_class_diagram(request)``
  - ``resolve_object_reference_diagram(request, target_model)``
  - ``count_reference_classes(reference_diagram)``

``src/utilities/model_context.py``
  Model summarization, with a per-diagram-type summarizer for each supported
  type:

  - ``compact_model_summary(model_data, diagram_type)``
  - ``detailed_model_summary(model_data, diagram_type)``
  - ``is_diagram_nontrivial(model_data, diagram_type)``

``src/utilities/class_metadata.py``
  Class attribute/method extraction for GUI binding:

  - ``extract_class_metadata(model)``
  - ``format_class_metadata_for_prompt(class_metadata)``

``src/utilities/user_metamodel.py``
  The bundled User Profile metamodel:

  - ``load_user_metamodel()`` — cached; returns a ClassDiagram-shaped dict
  - ``load_user_metamodel_semantics()`` — curated element/attribute descriptions
  - ``format_user_metamodel_guide()`` / ``build_user_profile_help_prompt(message)``
  - ``is_user_profile_help(message)``

``src/utilities/llm_retry.py``
  Retry-with-backoff for the *shared* server LLM's raw SDK call:

  - ``patch_openai_client_for_retry(client, *, label='llm')`` —
    monkey-patches the SDK client's network-call layer once
  - ``with_retry(func, *, label)`` — the wrapper it installs
  - ``MAX_ATTEMPTS`` (4), ``BASE_DELAY_SECONDS`` (0.6),
    ``MAX_DELAY_SECONDS`` (6.0), ``JITTER_SECONDS`` (0.3)

``src/utilities/workspace_context.py``
  Cross-diagram reference and workspace helpers:

  - ``build_workspace_context_block(request, ...)``
  - ``record_session_action(session, action_summary)``

``src/utilities/request_builders.py``
  Derived request factories:

  - ``build_request_for_target(request, target_type, ...)``
  - ``build_generation_request(request, generator_type, config, message_override)``

Knowledge Libraries
-------------------

.. note::

   Both pattern libraries are defined but **not currently injected** into any
   LLM prompt. See :doc:`diagram_handlers` for why, and how to re-enable them.

``src/domain_patterns.py``
  10 expert domain patterns for ClassDiagram generation:

  - ``DOMAIN_PATTERNS`` — Dictionary of domain definitions
  - ``detect_domain_pattern(user_message)`` — Match message to domain
  - ``format_pattern_for_prompt(pattern)`` — Format for LLM injection
  - ``get_pattern_hint(user_message)`` — Convenience wrapper

``src/state_patterns.py``
  8 behavioral lifecycle patterns for StateMachine generation:

  - ``STATE_MACHINE_PATTERNS`` — Dictionary of pattern definitions
  - ``detect_state_pattern(user_request)`` — Match request to a pattern key
  - ``format_state_pattern_for_prompt(pattern_key)`` — Format for LLM injection
  - ``get_state_pattern_hint(user_request)`` — Convenience wrapper

Routing
-------

``src/routing/intents.py``
  Intent name constants:

  - ``GENERATION_INTENT_NAME`` — The string name for the generation intent

LLM Provider
------------

``src/llm/provider.py``
  Provider abstraction over the OpenAI SDK:

  - ``LLMProvider`` — ``parse(messages, schema, temperature, max_tokens, model)``
    for structured outputs, ``stream(...)`` for streaming text, and
    ``predict(prompt, **kwargs)`` for a plain completion. All omit
    ``temperature`` and pass ``reasoning_effort`` for gpt-5 / o-series models.
    ``.client`` and ``.model_name`` expose the underlying SDK client and the
    tier this provider was built for.
  - ``get_provider(llm, model_name)`` — process-wide accessor

Conversation Memory
-------------------

``src/memory/``
  Per-session conversation memory: a verbatim window plus a rolling LLM
  summary of everything older.

  - ``memory_session_key(session, request)`` — prefers the v2 ``sessionId``,
    which survives reconnects
  - ``get_memory(session_id, ...)`` / ``remove_memory(session_id)``
  - ``cleanup_stale_memories(max_age_seconds)`` — called by the reaper
  - ``ConversationMemory``

Schemas
-------

``src/schemas/``
  Pydantic schemas for structured LLM output, one module per diagram type:
  ``class_diagram``, ``compact_class_diagram`` (the token-trimmed generation
  variant), ``object_diagram``, ``state_machine``, ``agent_diagram``,
  ``gui_diagram``, ``quantum_circuit``, ``bpmn``, ``user_profile``. All are
  re-exported from ``src/schemas/__init__.py``.

Token Tracking
--------------

``src/tracking/token_tracker.py``
  Token usage and cost tracking per session and globally:

  - ``get_tracker()`` → ``TokenTracker``
  - ``TokenTracker.record_from_usage(usage, model=...)``

  .. note::

     ``_COST_PER_1K`` must carry an entry for every model the agent can run.
     An unknown model falls back to placeholder pricing and every reported
     cost silently becomes an estimate — add an entry whenever a
     ``MODEL_*`` default changes.

Suggestions
-----------

``src/suggestions.py``
  Context-aware next-step suggestions after operations:

  - ``get_suggested_actions(diagram_type, operation_mode, available_diagrams=None,
    model_summary=None, generator_type=None)`` — returns 2–4
    ``{label, prompt}`` dicts
  - ``format_suggestions_as_text(actions)``
  - ``get_post_spec_suggestions(detected_generator)`` /
    ``get_artifact_label(detected_generator)``
  - ``_DIAGRAM_SUGGESTION_HANDLERS`` — per-diagram-type suggestion builders,
    one entry per supported type

Test Infrastructure
-------------------

``tests/conftest.py``
  Shared test fixtures:

  - ``FakeSession`` — Lightweight session stand-in
  - ``FakeLLM`` — Stub LLM with round-robin responses
  - ``make_v2_payload(message, ...)`` — Build v2 protocol payloads
  - ``make_session(message, ...)`` — Pre-loaded session fixture
  - ``MINIMAL_CLASS_MODEL`` — Fixture model with one class
  - ``EMPTY_CLASS_MODEL`` — Empty model fixture
