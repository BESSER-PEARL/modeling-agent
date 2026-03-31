API and Module Reference
========================

This document maps all Python modules in the Modeling Agent codebase.

.. contents:: On this page
   :local:
   :depth: 2

Entry Point
-----------

``modeling_agent.py``
  Creates the BESSER agent, defines 8 intents and 8 states, wires state bodies,
  and starts the WebSocket runtime.

Core Runtime Modules
--------------------

``src/agent_context.py``
  Shared runtime context container. Stores module-level globals (``agent``,
  ``gpt``, ``gpt_text``, ``gpt_predict_json``, ``uml_rag``, ``diagram_factory``,
  ``openai_api_key``) populated at startup.

``src/agent_setup.py``
  Initialization functions called during startup:

  - ``init_llm(agent)`` — Creates two LLMOpenAI instances (JSON + text mode)
  - ``init_rag(agent)`` — Builds ChromaDB-backed RAG
  - ``init_diagram_factory(gpt)`` — Creates DiagramHandlerFactory
  - ``init_intent_classifier_config()`` — Configures LLM-based intent classifier

``src/state_bodies.py``
  All state body functions and transition wiring:

  - ``register_all(states, intents)`` — Wires state bodies and transitions
  - ``greetings_body(session)`` — Welcome message handler
  - ``create_complete_system_body(session)``
  - ``modify_modeling_body(session)``
  - ``modeling_help_body(session)`` — Conceptual Q&A
  - ``describe_model_body(session)`` — Model summarization
  - ``uml_rag_body(session)`` — RAG query handler
  - ``generation_body(session)`` — Code generation routing
  - ``workflow_body(session)`` — Multi-step workflow orchestration

``src/execution/`` (package)
  Operation execution engine:

  - ``execute_planned_operations(session, request, default_mode, matched_intent)`` — ``planning.py``
  - ``execute_model_operation(session, request, operation, default_mode, ...)`` — ``model_operations.py``
  - ``handle_file_attachments(session, request)`` — ``file_handling.py``

``src/confirmation.py``
  Pending confirmation flow handlers:

  - ``handle_pending_system_confirmation(session, request)``
  - ``handle_pending_gui_choice(session, request)``

``src/session_helpers.py``
  Protocol-agnostic reply utilities:

  - ``reply_payload(session, result)``
  - ``reply_message(session, text)``

``src/suggestions.py``
  Context-aware suggestion engine:

  - ``get_suggested_actions(session, request, diagram_type, result)``
  - ``format_suggestions_as_text(suggestions)``

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

  - ``BaseDiagramHandler`` (abstract class)
  - ``predict_with_retry(system_prompt, user_prompt)``
  - ``predict_two_pass(system_prompt, user_prompt)``
  - ``validate_and_refine(spec, system_prompt)``
  - ``repair_json_response(raw_text)``
  - ``apply_single_layout(spec, existing_model)``
  - ``apply_system_layout(spec, existing_model)``

``src/diagram_handlers/core/layout_engine.py``
  Deterministic canvas layout:

  - ``apply_layout(spec, diagram_type, mode, existing_model)``
  - ``compute_canvas_bounds(element_count)``
  - ``compute_grid_shape(element_count)``
  - ``find_free_position(placed_rects, element_size)``

``src/diagram_handlers/registry/factory.py``
  Handler factory:

  - ``DiagramHandlerFactory(llm)``
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

Auxiliary Handlers
------------------

``src/handlers/generation_handler.py``
  Code generation routing:

  - ``handle_generation_request(session, request)``
  - ``match_generator_type(message)``
  - ``parse_inline_config(message, generator_type)``
  - ``GENERATOR_KEYWORDS`` — Keyword-to-generator mappings
  - ``GENERATOR_REQUIRED_FIELDS`` — Required configuration fields per generator

``src/handlers/file_conversion_handler.py``
  File upload conversion:

  - ``handle_file_attachments(session, request)``
  - ``convert_plantuml(content)``
  - ``convert_knowledge_graph(content)``
  - ``convert_image(content_b64, mime_type)``
  - ``convert_generic_text(content)``

Utilities
---------

``src/utilities/model_resolution.py``
  Target model resolution:

  - ``resolve_target_model(request, target_type)``
  - ``resolve_class_diagram(request)``
  - ``resolve_object_reference_diagram(request, target_model)``
  - ``count_reference_classes(reference_diagram)``

``src/utilities/model_context.py``
  Model summarization:

  - ``compact_model_summary(model, diagram_type)``
  - ``detailed_model_summary(request)``

``src/utilities/class_metadata.py``
  Class attribute/method extraction for GUI binding:

  - ``extract_class_metadata(model)``

``src/utilities/workspace_context.py``
  Cross-diagram reference and workspace helpers:

  - ``build_workspace_context_block(request)``
  - ``record_session_action(session, action_summary)``
  - ``build_session_summary(session)``

``src/utilities/request_builders.py``
  Derived request factories:

  - ``build_request_for_target(request, target_type)``
  - ``build_generation_request(request, generator_type, config, message_override)``

Knowledge Libraries
-------------------

``src/domain_patterns.py``
  10 expert domain patterns for ClassDiagram generation:

  - ``DOMAIN_PATTERNS`` — Dictionary of domain definitions
  - ``detect_domain_pattern(message)`` — Match message to domain
  - ``format_pattern_for_prompt(pattern)`` — Format for LLM injection

``src/state_patterns.py``
  8 behavioral lifecycle patterns for StateMachine generation:

  - ``STATE_PATTERNS`` — Dictionary of pattern definitions
  - ``detect_state_pattern(message)`` — Match message to pattern
  - ``format_state_pattern_for_prompt(pattern)`` — Format for LLM injection

Routing
-------

``src/routing/intents.py``
  Intent name constants:

  - ``GENERATION_INTENT_NAME`` — The string name for the generation intent

LLM Provider
------------

``src/llm/``
  LLM provider abstraction (structured outputs, streaming).

Conversation Memory
-------------------

``src/memory/``
  Per-session conversation memory with sliding window and LLM summarization.

Schemas
-------

``src/schemas/``
  Pydantic schemas for structured LLM output (one per diagram type:
  ``agent_diagram``, ``class_diagram``, ``gui_diagram``, ``object_diagram``,
  ``quantum_circuit``, ``state_machine``).

Token Tracking
--------------

``src/tracking/``
  Token usage and cost tracking per session and globally.

Suggestions
-----------

``src/suggestions.py``
  Context-aware next-step suggestions after operations.

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
