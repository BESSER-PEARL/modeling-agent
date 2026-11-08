API and Module Reference
========================

This document maps all Python modules in the Modeling Agent codebase, one
section per package, in layer order (a package imports only those below it).

.. contents:: On this page
   :local:
   :depth: 2

Entry Point
-----------

``modeling_agent.py``
  Creates the BESSER agent, defines 10 intents and 10 states, wires state
  bodies, starts the session reaper, and starts the WebSocket runtime.

Application (``src/app/``)
--------------------------

``src/app/agent_setup.py``
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

``src/app/state_bodies.py``
  All state body functions and transition wiring:

  - ``register_all(*, agent, states, intents)`` — Wires state bodies and
    transitions (keyword-only)
  - ``add_unified_transitions(state, intents_map, fallback_state, generation_state)``
  - ``route_to_generation(session)`` — priority-2 condition; frontend events
    and pending generator flows only, no LLM call
  - ``greetings_body(session)`` — Welcome message handler
  - ``create_complete_system_body(session)``
  - ``modify_modeling_body(session)``
  - ``modeling_help_body(session)`` — Conceptual Q&A
  - ``describe_model_body(session)`` — Model summarization
  - ``uml_rag_body(session)`` — RAG query handler
  - ``generation_body(session)`` — Code generation routing
  - ``decline_body`` / ``out_of_scope_body`` / ``meta_question_body``
  - ``global_fallback_body(session)``

``src/app/confirmation.py``
  Resuming pending dialogs (``common.pending_dialogs``). Each handler returns
  ``True`` when it consumed the message:

  - ``resume_pending(session) -> bool`` — hands the message to the open
    dialogs in ``PRECEDENCE`` order, through ``RESUME_HANDLERS``
  - ``handle_pending_system_confirmation(session) -> bool`` — replace / keep /
    new tab
  - ``handle_pending_gui_choice(session) -> bool`` — deterministic vs.
    AI-generated GUI

``src/app/container_config.py``
  The Docker entrypoint (``python -m app.container_config``): writes
  ``/app/config.yaml`` from ``AgentSettings``, logs it with the ``api_key``
  line redacted, then execs ``python modeling_agent.py``.

  - ``render(settings)`` — the config text
  - ``redact(text)`` — the log copy

Ingestion (``src/ingestion/``)
------------------------------

``src/ingestion/file_handling.py``
  ``handle_file_attachments(session, request)`` — the attachment step of a turn:
  converts each file and executes the resulting model operation.

``src/ingestion/file_conversion_handler.py``
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

Execution (``src/execution/``)
------------------------------

``src/execution/`` (package)
  Operation execution engine:

  - ``execute_planned_operations(session, request, default_mode, matched_intent)`` — ``planning.py``
  - ``execute_model_operation(session, request, operation, default_mode, ...)`` — ``model_operations.py``

Code Generation (``src/codegen/``)
----------------------------------

``src/codegen/generation_handler.py``
  Code generation routing:

  - ``handle_generation_request(session, request)``
  - ``should_route_to_generation(session, request)`` — the cheap transition
    gate (frontend events and pending flows only)
  - ``handle_pending_smart_gen_confirmation(session)``
  - ``handle_pending_plan_generation_confirmation(session)``
  - ``parse_inline_generator_config(...)``
  - ``GENERATOR_REQUIRED_FIELDS`` — Required configuration fields per generator
  - ``EXPORT_FORMATS`` / ``DIALECT_VALUES`` / ``MODE_VALUES`` / ``QISKIT_BACKENDS``

``src/codegen/smart_generation_handler.py``
  Smart (LLM-authored) generation dispatch:

  - ``GenerationClassification`` — the sub-routing schema
  - ``build_trigger_smart_generator_payload(classification, reason_prefix)``

Orchestration (``src/orchestrator/``)
-------------------------------------

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

``src/orchestrator/request_builders.py``
  Derived request factories:

  - ``build_request_for_target(request, target_type, ...)``
  - ``build_generation_request(request, generator_type, config, message_override)``

Conversation (``src/conversation/``)
------------------------------------

``src/conversation/session_helpers.py``
  Reply utilities and transition conditions:

  - ``reply_payload(session, payload)`` / ``reply_message(session, text)``
  - ``reply_stream_start`` / ``reply_stream_chunk`` / ``reply_stream_done``
  - ``reply_progress(session, message, step, total)``
  - ``stream_llm_response(...)`` — streams a free-text reply, routed through
    BYOK when a user key is active (fetched whole, then sent)
  - ``json_intent_matches(session, params)`` — the priority-1 transition
    condition; reads the unified classifier's verdict
  - ``json_no_intent_matched(session)`` — the fallback condition
  - ``replay_last_reply(session, request)`` — re-sends the buffered terminal
    reply after a reconnect
  - ``get_user_message`` / ``get_diagram_type`` / ``get_current_model``

``src/conversation/suggestions.py``
  Context-aware next-step suggestions after operations:

  - ``get_suggested_actions(diagram_type, operation_mode, available_diagrams=None,
    model_summary=None, generator_type=None)`` — returns 2–4
    ``{label, prompt}`` dicts
  - ``format_suggestions_as_text(actions)``
  - ``get_post_spec_suggestions(detected_generator)`` /
    ``get_artifact_label(detected_generator)``
  - ``_DIAGRAM_SUGGESTION_HANDLERS`` — per-diagram-type suggestion builders,
    one entry per supported type

``src/conversation/reply_copy.py``
  Shared user-facing copy (decline acknowledgement, meta answer,
  out-of-scope redirect, confirmation prompts) kept in one place.

``src/conversation/telemetry.py``
  Opt-in, fire-and-forget study prompt telemetry. Posts one ``prompt``
  event per handled message to ``{BESSER_BACKEND_URL}/besser_api/telemetry/event``
  on a short-timeout daemon thread, only when the request carries a
  ``pilotParticipant`` label. Every exception is swallowed.

``src/conversation/agent_context.py``
  Shared runtime context container. Stores module-level globals (``agent``,
  ``gpt``, ``gpt_text``, ``gpt_predict_json``, ``uml_rag``, ``diagram_factory``,
  ``openai_api_key``, ``stt``) populated at startup.

Routing (``src/routing/``)
--------------------------

``src/routing/unified_classifier.py``
  The router. One structured-output call per message:

  - ``UnifiedClassification`` — the wide Pydantic verdict schema
  - ``classify_message(request, llm_provider, history, recent_smart_gen, pending_flow)``
  - ``get_or_classify(session, request, llm_provider)`` — per-message cache
  - its system prompt, the authoritative routing rulebook, is ``SYSTEM_PROMPT`` in
    ``src/prompts/classifier.py``

``src/routing/guards.py``
  The deterministic routing rules, pure functions over the request or message
  (see :doc:`intent_recognition`):

  - ``skips_classification``, ``frontend_event_verdict``, ``auto_fix_verdict``
    — before the classifier
  - ``reconcile_intent``, ``modeling_guard``, ``quick_response_key``,
    ``is_mixed_modeling_and_generation``, ``is_past_generation_question``,
    ``generation_route`` — in the state that received the message

``src/routing/generation_signals.py``
  Generation vocabulary and session signals shared by the classifier, the
  planner and the generation handler:

  - ``detect_generator_type(message)`` — pure keyword/regex detection
  - ``GENERATOR_KEYWORDS`` — Keyword-to-generator mappings
  - ``GENERATOR_PREREQUISITES`` — Diagrams each generator consumes
  - ``recent_smart_gen_for_project(session, window_seconds)``
  - ``_extract_github_reference(message)`` with ``_GITHUB_URL_RE`` /
    ``_GITHUB_CONTINUE_VERB_RE`` — the continue-from-GitHub guard

``src/routing/intents.py``
  Intent name constants:

  - ``GENERATION_INTENT_NAME`` — The string name for the generation intent

Conversation Memory
-------------------

``src/memory/``
  Per-session conversation memory: a verbatim window plus a rolling LLM
  summary of everything older.

  - ``memory_session_key(session, request)`` — prefers the v2 ``sessionId``,
    which survives reconnects
  - ``get_memory(session_id, ...)`` / ``remove_memory(session_id)``
  - ``cleanup_stale_memories(max_age_seconds)`` — the reaper runs it through
    ``common.session_store.sessions.expire``; it drops memories with no new
    message and no heartbeat for ``max_age_seconds``
  - ``touch_memory(session_id)`` — marks a conversation as in use; the
    platform calls it on each keep-alive heartbeat (whose ``user_id`` is the
    tab's ``sessionId``)
  - ``ConversationMemory.seed(turns)`` — rebuilds an empty memory from the
    client's ``recentHistory``
  - ``ConversationMemory``

Diagram Handlers
----------------

``src/diagram_handlers/core/base_handler.py``
  Abstract base handler with shared infrastructure:

  - ``BaseDiagramHandler`` (abstract class; 6 abstract methods)
  - ``predict_with_retry(prompt, max_retries=1, *, model=None, ...)``
  - ``predict_structured(prompt, response_schema, *, system_prompt, temperature, model)``
  - ``predict_two_pass_structured(user_request, system_prompt, reasoning_prompt, response_schema, ...)``
  - ``predict_two_pass(...)``
  - ``generate_complete_system`` / ``generate_single_element`` /
    ``generate_modification`` — template methods over the ``_prepare_*`` and
    ``_build_system`` / ``_build_single_element`` / ``_build_modification`` hooks
  - ``self_correct(...)`` / ``parse_validate_or_correct(...)``
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

``src/prompts/``
  The prompt text, one module per diagram type or state (``class_diagram``,
  ``bpmn``, ``gui``, ``classifier``, ``planner``, ``modeling_help``, …). Static
  prompts are constants, request-built ones are functions of the request
  values. ``fragments.py`` holds the snippets several handlers share —
  ``EXACT_NAMES_RULE``, ``REMOVE_ELEMENT_RULE``, ``MULTI_MOD_ARRAY_RULE``,
  ``POSITION_DISCLAIMER``.

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

LLM (``src/llm/``)
------------------

``src/llm/provider.py``
  The one door to the model SDKs (an import-linter contract keeps ``openai`` /
  ``anthropic`` imports inside ``llm``). ``LLMProvider(llm)`` wraps a BAF
  ``LLMOpenAI`` instance; every method runs on the request's BYOK key when one
  is active, else on the instance's shared client, records usage with the
  model actually called, applies the ``model_config`` sampling rules and
  raises ``ModelRefusal`` on a refusal:

  - ``complete(prompt, *, model, json_mode, temperature, max_tokens, images)``
    — free text or JSON mode; on the server key without a ``model`` (and
    images) it is BAF's ``predict`` with an estimated usage. Images follow
    the same rule as ``vision``
  - ``parse(messages, schema, *, temperature, max_tokens, model)`` — one
    structured-output attempt (OpenAI Structured Outputs, else JSON mode);
    raises ``OutputTruncated`` when the reply hits ``max_tokens``.
    ``supports_structured_outputs()`` tells which one it will use
  - ``stream(prompt, *, system_prompt, model, temperature, max_tokens)`` — the
    reply as text chunks (one chunk on a BYOK key)
  - ``vision(prompt, images, *, timeout)`` — the file-conversion vision call
    (JSON mode) on the user's key where it reads images, as the ``vision``
    entry of the BYOK tier table says (the chosen model for OpenAI and
    Anthropic, ``mistral-large-latest`` for Mistral), else on the server key
    and ``MODEL_VISION``; raises ``ImagesNotSupported``
    when the provider rejects the image request
  - ``get_provider(llm, model_name)`` — the process-wide provider over ``gpt``

``src/llm/byok.py``
  Per-request bring-your-own-key routing:

  - ``set_current(provider, api_key, model, base_url)`` / ``reset_current(token)``
  - ``get_active_client()`` — the per-request client, or ``None``
  - ``active_vision_model()`` — the model the active key reads images with, or ``None``
    when vision runs on the server key
  - ``resolve_model(provider, requested_model, user_model)``
  - ``SUPPORTED_PROVIDERS`` — ``openai``, ``anthropic``, ``mistral``, ``nebius``

``src/llm/byok_config.py``
  The per-request BYOK state, below both ``byok`` and ``tracking``:

  - ``BYOKConfig`` — the user's provider, key, model and base URL
  - ``current_byok`` — the context variable holding it
  - ``get_current()`` — the active ``BYOKConfig``, or ``None``

``src/llm/model_config.py``
  Per-call-site model routing:

  - ``MODEL_CLASSIFIER``, ``MODEL_GENERATION_LARGE``, ``MODEL_GENERATION_GUI``,
    ``MODEL_GENERATION_SMALL``, ``MODEL_REASONING``, ``MODEL_VISION``,
    ``MODEL_EMBEDDINGS`` — all overridable via ``BESSER_AGENT_MODEL_*``
  - ``supports_custom_temperature(model)`` — False for gpt-5+ / o-series and
    the Claude models that reject sampling params
  - ``reasoning_effort_for(model)`` — the ``reasoning_effort`` to pass, or
    ``None`` for models that take a temperature instead

``src/llm/tracking/token_tracker.py``
  Token usage and cost tracking per session and globally:

  - ``get_tracker()`` → ``TokenTracker``
  - ``TokenTracker.record_from_usage(usage, model=...)``

  .. note::

     Prices come from the model catalog (``BESSER_MODEL_CATALOG``, read by
     ``src/llm/model_catalog.py``), else from ``src/llm/tracking/data/model_prices.json``,
     a copy of BESSER's vendored price file (see :ref:`agent-prices`). A model
     neither prices falls back to placeholder pricing, so every reported cost
     for it becomes an estimate: price a new ``MODEL_*`` default in the
     catalog.

Utilities
---------

``src/utilities/model_utils.py``
  - ``model_has_elements(model)`` — the shared "is this model non-empty?" check

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
  - ``format_user_metamodel_guide()`` — the metamodel as prompt text; the help
    prompt around it is ``prompts.modeling_help.user_profile_help_prompt``
  - ``is_user_profile_help(message)``

``src/llm/retry.py``
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

Protocol Layer
--------------

``src/protocol/types.py``
  Protocol data classes:

  - ``AssistantRequest`` — Canonical request object
  - ``WorkspaceContext`` — Editor state snapshot
  - ``FileAttachment`` — Uploaded file metadata
  - ``SUPPORTED_DIAGRAM_TYPES`` — Set of valid diagram type strings

``src/protocol/actions.py``
  The wire action names and the repeated envelopes:

  - ``Action`` — ``StrEnum`` of every inbound and outbound ``action``
  - ``assistant_message(message, **fields)``, ``inject_element(...)``,
    ``inject_complete_system(...)``, ``model_reply(action, ...)``

``src/protocol/message_limits.py``
  ``validate_message_length(message)`` and ``UserMessageTooLong``: a request
  over ``MAX_USER_MESSAGE_CHARS`` is rejected, never truncated.

``src/protocol/adapters.py``
  Payload extraction and normalization:

  - ``parse_assistant_request(session)`` — Main entry point
  - ``extract_event_payload(session)`` — Extract from BESSER event
  - ``parse_v2_payload(payload)`` — Parse v2 protocol format
  - ``normalize_diagram_type(diagram_type)`` — Normalize type strings
  - ``strip_diagram_prefix(message)`` — Remove diagram type prefixes

Common (``src/common/``)
------------------------

``src/common/agent_config.py``
  Tunable constants: tab and message limits, session grace period, streaming
  buffer, LLM temperatures and token budgets, conversation history depth.

``src/common/settings.py``
  ``AgentSettings`` (pydantic-settings): every environment variable the agent
  reads, declared once with its default and parsing. Module constants build it
  at import, per-call reads at each use. See :doc:`configuration`.

``src/common/session_store.py``
  ``sessions``, the registry of the in-process per-session stores that expire
  their own entries (today conversation memory; the reply-replay buffers are
  bounded LRUs instead):

  - ``sessions.register(name, expire=...)`` — a store's expiry
  - ``sessions.expire(max_age_seconds)`` — runs every registered expiry (the
    session reaper); a failing one is logged at warning and skipped

  BAF's session variables (every key in ``session_keys``) belong to BAF and
  are freed by ``Agent.close_session``.

``src/common/session_keys.py``
  Every session-state key the agent writes, in one place — ``PENDING_DIALOGS``
  (all pending dialogs), the classification cache keys
  (``UNIFIED_CLASSIFICATION`` / ``UNIFIED_CLASSIFICATION_EVENT_ID``), the
  parsed-request cache keys, and ``VOICE_CONTEXT``. Import the constant, never
  the literal string.

``src/common/pending_dialogs.py``
  The questions the assistant is waiting on and their stashes, as slots of the
  one ``PENDING_DIALOGS`` session variable (``PENDING_COMPLETE_SYSTEM``,
  ``PENDING_GUI_CHOICE``, ``PENDING_GENERATOR_TYPE`` / ``_CONFIG``,
  ``PENDING_SMART_GEN_*``, ``PENDING_WEBAPP_GENERATE``,
  ``MISMATCH_REGEN_PENDING``, …):

  - ``get(session, slot)`` / ``put(session, slot, value)`` /
    ``clear(session, *slots)`` — the only way to read or write a slot
  - ``PRECEDENCE`` — ``GUI_CHOICE``, ``SYSTEM_CONFIRMATION``, ``SMART_GEN``,
    ``GENERATOR``: which open dialog a message is checked against first
  - ``is_open(session, kind)`` / ``current(session)`` → ``PendingDialog(kind,
    payload)`` or ``None``
  - ``one_per_plan()`` / ``may_open(slot)`` — a parallel plan opens one
    dialog; ``execute_model_operation`` returns ``Deferred(operation)`` for the others

``src/common/errors.py``
  Unified error taxonomy:

  - ``ErrorCode`` — the enum of error codes
  - ``classify_error(error)`` → ``ErrorCode``
  - ``get_recovery_hint(error_code)``
  - ``build_error_response(error_code, message, ...)``

Schemas
-------

``src/schemas/``
  Pydantic schemas for structured LLM output, one module per diagram type:
  ``class_diagram``, ``compact_class_diagram`` (the token-trimmed generation
  variant), ``object_diagram``, ``state_machine``, ``agent_diagram``,
  ``gui_diagram``, ``quantum_circuit``, ``bpmn``, ``user_profile``. All are
  re-exported from ``src/schemas/__init__.py``.

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
