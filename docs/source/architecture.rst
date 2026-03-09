Architecture
============

This document describes the internal architecture of the Modeling Agent. For the
full reference including all schemas, see :doc:`schema`.

System Overview
---------------

The modeling agent is a WebSocket-based conversational AI system built on the
BESSER Agentic Framework. It connects the BESSER Web Modeling Editor (a
React/TypeScript SPA) with an OpenAI GPT-4.1-mini LLM backend.

.. code-block:: text

   ┌───────────────────────────────────────────────────────┐
   │            BESSER Web Modeling Editor                  │
   │             (React/TypeScript SPA)                     │
   └──────────────────────┬────────────────────────────────┘
                          │ WebSocket (JSON v2 protocol)
                          ▼
   ┌───────────────────────────────────────────────────────┐
   │                    MODELING AGENT                      │
   │                                                       │
   │  Protocol     → State Machine  → Execution Engine     │
   │  Adapters       (8 states)       (plan + dispatch)    │
   │                                        │              │
   │  Orchestrator ←────────────────────────┘              │
   │  (planner +                                           │
   │   type resolver)  → Diagram Handlers → Layout Engine  │
   │                                                       │
   │  Quality Review   File Conversion   Generation Handler│
   └───────────────────────────────────────────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        OpenAI GPT   ChromaDB     UML Specs
        (JSON/Text/  (RAG store)  (PDF source)
         Vision)

Technology Stack
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Component
     - Technology
     - Notes
   * - Agent Framework
     - BESSER Agentic Framework v4.2.3
     - State machine, WebSocket platform, intent classification
   * - LLM
     - OpenAI GPT-4.1-mini
     - JSON mode (temp=0.2), text mode (temp=0.4), Vision
   * - RAG
     - LangChain + ChromaDB
     - Vector store over UML 2.5.1 specification
   * - Transport
     - WebSocket
     - Port 8765 (configurable)
   * - Runtime
     - Python 3.11+
     - Virtual environment with ~40 direct dependencies

Architectural Layers
--------------------

The system is organized into these layers, processed in order for each request:

1. **Protocol Layer** (``src/protocol/``): Parses raw WebSocket messages into
   canonical ``AssistantRequest`` objects.

2. **State Machine** (``modeling_agent.py`` + ``src/state_bodies.py``): Routes
   requests to the appropriate handler based on intent classification.

3. **Orchestration Layer** (``src/orchestrator/``): Plans multi-step operations
   and resolves target diagram types.

4. **Execution Engine** (``src/execution.py``): Dispatches operations to diagram
   handlers and manages confirmation flows.

5. **Diagram Handler System** (``src/diagram_handlers/``): Six specialized
   handlers that generate diagram JSON via LLM calls. See :doc:`diagram_handlers`.

6. **Utility Layer** (``src/utilities/``): Model resolution, context building,
   metadata extraction, and layout helpers.

7. **Knowledge Layer** (``src/domain_patterns.py``, ``src/state_patterns.py``):
   Expert domain patterns injected into LLM prompts.

Entry Point
-----------

``modeling_agent.py`` performs the following startup sequence:

1. Adds ``src/`` to ``sys.path``
2. Creates the BESSER ``Agent`` object
3. Calls four ``init_*`` functions from ``agent_setup``
4. Populates ``agent_context`` module-level globals
5. Defines all 8 states and 8 intents
6. Calls ``state_bodies.register_all()`` to wire state bodies and transitions
7. Starts WebSocket platform (``use_ui=False``)
8. Calls ``agent.run()``

Shared Runtime Context
----------------------

``src/agent_context.py`` stores module-level globals populated at startup:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Variable
     - Type
     - Description
   * - ``agent``
     - ``Agent``
     - BESSER Agent instance
   * - ``gpt``
     - ``LLMOpenAI``
     - JSON mode, GPT-4.1-mini, temp=0.2
   * - ``gpt_text``
     - ``LLMOpenAI``
     - Free-text mode, GPT-4.1-mini, temp=0.4
   * - ``gpt_predict_json``
     - ``Callable``
     - Closure enforcing JSON response format
   * - ``uml_rag``
     - ``RAG | None``
     - ChromaDB-backed RAG, None if unavailable
   * - ``diagram_factory``
     - ``DiagramHandlerFactory``
     - Factory for all 6 diagram handlers
   * - ``openai_api_key``
     - ``str``
     - API key from config.ini

All modules import these at call-time (not import-time) to ensure they are
populated when user messages arrive.

State Machine and Intent Classification
----------------------------------------

The agent uses 8 states with corresponding intents. Intent classification is
performed by the BESSER framework using LLM-based description matching.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Intent
     - Target State
     - Behavior
   * - ``hello_intent``
     - ``greetings_state``
     - Welcome message, quick patterns
   * - ``create_single_element_intent``
     - ``create_single_element_state``
     - Single class/state/object creation
   * - ``create_complete_system_intent``
     - ``create_complete_system_state``
     - Multi-element system design
   * - ``modify_model_intent``
     - ``modify_model_state``
     - Edit existing diagram elements
   * - ``modeling_help_intent``
     - ``modeling_help_state``
     - Conceptual Q&A with LLM
   * - ``describe_model_intent``
     - ``describe_model_state``
     - Analyze current project
   * - ``uml_spec_intent``
     - ``uml_rag_state``
     - UML spec lookups via RAG
   * - ``generation_intent``
     - ``generation_state``
     - Code generation routing

All three modeling states (``create_single_element``, ``create_complete_system``,
``modify_model``) share the same body function ``_modeling_state_body()`` with
different ``default_mode`` parameters.

Execution Engine
----------------

``src/execution.py`` is the core dispatch layer.

**execute_planned_operations():**

1. Calls ``plan_assistant_operations()`` to get operation list
2. Loops over operations:

   - ``type == "model"`` → ``execute_model_operation()``
   - ``type == "generation"`` → ``handle_generation_request()``

3. If an operation returns ``None`` (pending confirmation), saves remaining ops

**execute_model_operation()** — the most complex function:

1. Resolve diagram type (from operation or heuristic)
2. Resolve operation mode (single_element / complete_system / modify_model)
3. Existing-model guard (complete_system only)
4. GUI generation-mode choice (GUINoCodeDiagram only)
5. Handler lookup via ``diagram_factory``
6. Build modeling prompt (request + workspace context)
7. Dispatch to handler method
8. Inject metadata (diagramType, diagramId, replaceExisting)
9. Send reply payload to frontend
10. Record action in session history
11. Run quality review

Confirmation Flows
~~~~~~~~~~~~~~~~~~

Two confirmation flows pause execution and resume on the next user message:

- **Complete System Confirmation**: When model already exists, asks "replace or keep?"
- **GUI Generation-Mode Choice**: When ClassDiagram exists, asks "auto or LLM?"

Both store pending state in the session and resume via ``_common_preamble()``.

Concurrency and Caching
------------------------

**LLM Semaphore:** ``threading.Semaphore(4)`` in ``base_handler.py`` limits
concurrent OpenAI API calls across all sessions.

**Prompt/Response Cache:**

- Key: SHA-256 hash of (system_prompt + user_prompt)
- Max entries: 50
- TTL: 5 minutes
- Reduces duplicate LLM calls for identical requests

**Retry Strategy:** Exponential backoff with jitter (3 attempts), then fallback
generator.

Graceful Degradation
--------------------

Every modeling operation follows a 4-level degradation chain:

1. Primary LLM call (with cache + retry)
2. JSON repair via LLM
3. Type-specific fallback generator
4. Error response with ``retryable: true``

No exceptions propagate to the caller.

Design Patterns
---------------

**Module-Level Globals:** All LLM handles and the diagram factory are stored in
``agent_context.py`` to avoid circular imports.

**Protocol Decoupling:** ``AssistantRequest`` separates protocol parsing from
execution. Downstream code works only with typed Python objects.

**Handler Extensibility:** Adding a new diagram type requires implementing 4
abstract methods and registering in the factory. Layout, retry, caching, and
concurrency are inherited.

**Deterministic Layout:** LLMs never emit positions. The layout engine runs
after every generation, ensuring collision-free visual presentation.
