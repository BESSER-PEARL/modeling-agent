Architecture
============

This document describes the internal architecture of the Modeling Agent. For the
full reference including all schemas, see :doc:`schema`.

System Overview
---------------

The BESSER Modeling Agent is a WebSocket-based conversational AI system built on
the `BESSER Agentic Framework <https://besser-pearl.github.io/BESSER/>`_. It
connects the `BESSER Web Modeling Editor <https://editor.besser-pearl.org>`_ (a
React/TypeScript SPA) with OpenAI models, routed per call site through the
model tier table in ``src/model_config.py``. Code generation is powered by
`BESSER generators <https://besser-pearl.github.io/BESSER/generators.html>`_
(Django, Python, Java, SQL, SQLAlchemy, and more).

.. mermaid::

   graph TD
       FE["BESSER Web Modeling Editor<br/>(React/TypeScript SPA)"]
       FE -->|"WebSocket (JSON v2 protocol)"| PA

       subgraph AGENT["MODELING AGENT"]
           PA["Protocol Adapters"] --> UC["Unified Classifier<br/>(1 LLM call / message)"]
           UC --> SM["State Machine<br/>(10 states)"]
           SM --> EE["Execution Engine<br/>(plan + dispatch)"]
           EE --> ORCH["Orchestrator<br/>(planner + type resolver)"]
           ORCH --> DH["Diagram Handlers"]
           DH --> LE["Layout Engine"]
           EE --> FC["File Conversion"]
           EE --> GH["Generation Handler"]
       end

       DH --> LLM["OpenAI models<br/>(per-tier: classifier / generation / vision)"]
       SM --> RAG["ChromaDB<br/>(RAG store)"]
       RAG --> UML["UML Specs<br/>(PDF source)"]

Technology Stack
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Component
     - Technology
     - Notes
   * - Agent Framework
     - ``besser-agentic-framework[extras,llms,tensorflow] == 4.3.2``
     - State machine, WebSocket platform, local intent classification.
       One file is vendored over the pip install — see
       ``patches/websocket_platform.py``.
   * - Routing LLM
     - ``MODEL_CLASSIFIER`` (default ``gpt-4o-mini``)
     - One structured-output call per message (the unified classifier)
   * - Generation LLMs
     - ``MODEL_GENERATION_LARGE`` / ``_GUI`` / ``_SMALL``, ``MODEL_REASONING``
     - Structured diagram output; gpt-5 / o-series use ``reasoning_effort``
       instead of ``temperature``
   * - Vision LLM
     - ``MODEL_VISION`` (default ``gpt-5``)
     - Image / PDF → diagram conversion
   * - Local intent classifier
     - ``SimpleIntentClassifier`` (TensorFlow)
     - Free, trained at startup on each intent's training sentences;
       exception-path and voice/text-event fallback only
   * - RAG
     - LangChain + ChromaDB
     - Vector store over UML 2.5.1 specification
   * - Speech-to-text
     - OpenAI ``whisper-1``
     - Voice messages; language auto-detected unless pinned
   * - Transport
     - WebSocket
     - Port 8765 (configurable)
   * - Runtime
     - Python 3.11
     - ``python:3.11-slim`` base image

Architectural Layers
--------------------

The system is organized into these layers, processed in order for each request:

1. **Protocol Layer** (``src/protocol/``): Parses raw WebSocket messages into
   canonical ``AssistantRequest`` objects.

2. **Classification Layer** (``src/unified_classifier.py``): One
   structured-output LLM call per message, cached, returning the state-level
   intent plus every sub-routing field downstream code needs. See
   :doc:`intent_recognition`.

3. **State Machine** (``modeling_agent.py`` + ``src/state_bodies.py``): Routes
   requests to the appropriate handler based on that classification.

4. **Orchestration Layer** (``src/orchestrator/``): Plans multi-step operations
   and resolves target diagram types.

5. **Execution Engine** (``src/execution/``): Dispatches operations to diagram
   handlers and manages confirmation flows.

6. **Diagram Handler System** (``src/diagram_handlers/``): Eight specialized
   handlers that generate diagram JSON via LLM calls. See :doc:`diagram_handlers`.

7. **Utility Layer** (``src/utilities/``): Model resolution, context building,
   metadata extraction, and layout helpers.

8. **Knowledge Layer** (``src/domain_patterns.py``, ``src/state_patterns.py``):
   Expert domain patterns. Defined but **not currently injected** into LLM
   prompts — see :doc:`diagram_handlers`.

9. **LLM Abstraction** (``src/llm/``): Provider abstraction over
   OpenAI, encapsulating model selection and call conventions (structured
   outputs via ``parse()``, streaming via ``stream()``).

10. **BYOK Routing** (``src/byok.py``): Per-request routing of generation,
    structured-output, classification and conversational calls through a
    user-supplied API key, driven by a context var set at the WebSocket
    request boundary.

11. **Conversation Memory** (``src/memory/``): Per-session conversation
    history with a rolling LLM summary of everything older than the verbatim
    window.

12. **Schemas** (``src/schemas/``): Pydantic models for each diagram type,
    used by the structured-output pass of diagram handlers.

13. **Token Tracking** (``src/tracking/``): Per-session and global token
    usage and cost accounting.

14. **Suggestion Engine** (``src/suggestions.py``): Contextual next-step
    action suggestions returned to the frontend after each operation.

Entry Point
-----------

``modeling_agent.py`` performs the following startup sequence:

1. Adds ``src/`` to ``sys.path``
2. Creates the BESSER ``Agent`` object and the WebSocket platform (``use_ui=False``)
3. Calls the five ``init_*`` functions from ``agent_setup`` — ``init_llm``,
   ``init_stt``, ``init_rag``, ``init_diagram_factory``,
   ``init_intent_classifier_config``
4. Populates ``agent_context`` module-level globals
5. Defines all 10 states and 10 intents (each intent with training sentences,
   required by the local Simple classifier)
6. Calls ``state_bodies.register_all()`` to wire state bodies and transitions
7. Starts the session reaper thread
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
     - Structured / JSON call path. Defaults to the ``MODEL_CLASSIFIER``
       tier, temp=0.2; generation-quality call sites override the model
       per call via the ``model=`` plumbing.
   * - ``gpt_text``
     - ``LLMOpenAI``
     - Free-text mode, ``MODEL_CLASSIFIER`` tier, temp=0.4. Registered
       under the key ``<model>-text`` because BAF keys LLMs by name.
   * - ``gpt_predict_json``
     - ``Callable``
     - Closure enforcing ``response_format={"type": "json_object"}``,
       with an optional per-call model override
   * - ``uml_rag``
     - ``RAG | None``
     - ChromaDB-backed RAG, None if unavailable
   * - ``diagram_factory``
     - ``DiagramHandlerFactory``
     - Factory for all 8 diagram handlers
   * - ``openai_api_key``
     - ``str``
     - Server API key from config.yaml
   * - ``stt``
     - ``OpenAISpeech2Text``
     - Whisper speech-to-text for voice messages

All modules import these at call-time (not import-time) to ensure they are
populated when user messages arrive.

State Machine and Intent Classification
----------------------------------------

The agent uses 10 states with corresponding intents. The authoritative
classification comes from ``src/unified_classifier.py`` — one structured-output
LLM call per message, cached, read by the ``json_intent_matches`` transition
condition. BAF's own local ``SimpleIntentClassifier`` is only an
exception-path and voice/text-event fallback. See :doc:`intent_recognition`.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Intent
     - Target State
     - Behavior
   * - ``hello_intent``
     - ``greetings_state``
     - Welcome message, quick patterns
   * - ``create_complete_system_intent``
     - ``create_complete_system_state``
     - Multi-element system design
   * - ``modify_model_intent``
     - ``modify_model_state``
     - Single element creation and editing existing elements
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
     - Code generation routing (deterministic, smart, export, deploy,
       GitHub import)
   * - ``decline_intent``
     - ``decline_state``
     - Acknowledge an opt-out without building anything
   * - ``out_of_scope_intent``
     - ``out_of_scope_state``
     - Redirect a request for a non-software artifact
   * - ``meta_question_intent``
     - ``meta_question_state``
     - Answer "what can you do / why use you" questions

Both modeling states (``create_complete_system``, ``modify_model``) share the
same body function ``_modeling_state_body()`` with different ``default_mode``
parameters.

Execution Engine
----------------

``src/execution/`` is the core dispatch layer (package with ``planning.py``,
``model_operations.py``, ``file_handling.py``).

**execute_planned_operations():**

1. Calls ``plan_assistant_operations()`` to get operation list
2. Loops over operations:

   - ``type == "model"`` → ``execute_model_operation()``
   - ``type == "generation"`` → ``handle_generation_request()``

3. If an operation returns ``None`` (pending confirmation), saves remaining ops

**execute_model_operation()** — the most complex function:

1. Resolve diagram type (from operation or heuristic)
2. Resolve operation mode (``complete_system`` / ``modify_model`` — a
   ``modify_model`` op on a flow-style diagram that does not exist yet is
   promoted to ``complete_system``)
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

Several confirmation flows pause execution and resume on the next user
message. Each stores pending state in the session; the modeling ones resume
via ``_common_preamble()``, the generation ones via the generation state body.

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flow
     - Question asked
   * - Complete-system confirmation (``src/confirmation.py``)
     - A model already exists — replace it, keep it, or use a new tab?
   * - GUI generation-mode choice (``src/confirmation.py``)
     - "Fast & deterministic" (one screen per class, no LLM) or
       "AI-Generated (experimental)"?
   * - Spec-Driven Agent confirmation (``src/handlers/generation_handler.py``)
     - Confirm before handing off to the LLM-authored Spec-Driven Agent
   * - Plan-generation pause (``src/handlers/generation_handler.py``)
     - A mixed modeling + generation plan pauses after the modeling step and
       waits for an explicit "generate" before running the generator
   * - Destructive-modification confirmation (``src/execution/model_operations.py``)
     - A ``modify_model`` plan that would delete most or all of the existing
       diagram asks before applying
   * - Domain-mismatch confirmation
     - The smart request describes a different domain than the existing class
       diagram — confirm before rewriting

Whether the classifier treats the next message as an **answer** to a pending
question or as an unrelated **new request** is itself part of the
classification (``pending_flow_action``), so an off-topic instruction can
never be swallowed by a half-finished flow.

Progress Events
~~~~~~~~~~~~~~~

For multi-step plans (2+ operations), progress events are sent to the frontend
via ``reply_progress()`` so that the user sees real-time step indicators (e.g.,
"Step 1 of 3: Creating class diagram...").

Request Planner
----------------

``src/orchestrator/request_planner.py`` decomposes a user message into an
ordered list of operations using a 3-tier approach:

- **Tier 0 -- Fast heuristic regex patterns:** A bank of compiled regular
  expressions matches common request shapes (e.g., "create a web app", "generate
  Python code", "design a state machine"). This tier handles ~90% of simple
  requests with zero LLM calls.
- **Tier 1 -- Keyword-based fallback with intent-aware fast path:** When no
  regex matches, a keyword-based classifier determines the operation type and
  target diagram without calling the LLM.
- **Tier 2 -- LLM planner:** Only genuinely complex multi-step requests that
  escape both fast paths are sent to the LLM for decomposition.

Two-Pass Generation
-------------------

Diagram handlers use a two-pass strategy for complex requests
(``predict_two_pass_structured``): a reasoning pass (free-text LLM call on the
``MODEL_REASONING`` tier to plan the design) followed by a structured pass
(OpenAI Structured Outputs with Pydantic schema validation on a generation
tier).

For simple requests the reasoning pass is skipped entirely, saving one full
LLM round-trip. "Simple" is judged by the length of the **raw** user message
(``_TWO_PASS_MIN_LENGTH = 250`` characters) rather than the enriched prompt —
otherwise conversation history and the workspace-context block would push a
trivial request onto the expensive path.

Session Identity
-----------------

Conversation memory is keyed on ``memory_session_key()``
(``src/memory/__init__.py``), which prefers the v2 payload's ``sessionId``
(``AssistantRequest.session_id``) because it survives WebSocket reconnects.
It falls back to the BAF session id only when no parsed request is available
— the BAF id changes on every reconnect without a stable user query param,
which would silently drop all conversation context.

The frontend keeps that continuity by appending a persisted ``?user_id=``
query parameter to the WebSocket URL. Because the two sockets a browser tab
opens (the assistant widget and the workspace drawer) then share one session
key, stock BAF 4.3.2 would evict the live connection when either closed;
``patches/websocket_platform.py`` vendors an ownership-guarded delete plus
reply-route-to-sender to fix this.

Rate Limiting, Retries and Caching
-----------------------------------

**Rate Limiting:** Handled by the provider API directly. HTTP 429 and 5xx
responses are treated as transient by the retry layer; other 4xx are
permanent and fail fast.

**Shared-client retry** (``src/utilities/llm_retry.py``): the shared server
LLM's raw SDK call is patched once, at the client's network-call layer, so
every path through it — BAF ``predict``/``chat``, the provider's ``parse()``
and ``stream()``, and ``gpt_predict_json`` — inherits the same bounded
backoff. ``MAX_ATTEMPTS = 4`` (1 try + 3 retries), base delay 0.6 s, per-attempt
cap 6 s, ±0.3 s jitter — roughly 5 s of worst-case added latency, deliberately
bounded so a live chat never stalls. BYOK's per-request client is a separate
object this patch never touches.

**Handler-level retry** (``base_handler.predict_with_retry``): jittered
exponential backoff over ``1 + max_retries`` attempts (default 1 retry),
then the graceful-degradation chain below.

**Parsed-Request Cache:** ``parse_assistant_request()`` caches its result
per-event using ``id(session.event)`` as the key, avoiding redundant JSON
parsing within a single message cycle (it is called 3–5 times per message).

**Classification Cache:** ``get_or_classify()`` caches the unified
classification on the same event-id key, so one incoming message costs exactly
one classification call regardless of how many transition conditions ask.

Graceful Degradation
--------------------

Every modeling operation follows a 4-level degradation chain:

1. Primary LLM call (with retry)
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

**Handler Extensibility:** Adding a new diagram type requires implementing the
5 abstract methods of ``BaseDiagramHandler`` and registering the class in
``HANDLER_CLASSES``. Layout, retry, structured output, and two-pass generation
are inherited. See :doc:`diagram_handlers` for the full checklist.

**Per-Request BYOK Isolation:** A user's own API key never touches the shared
LLM objects. It lives in a ``contextvars.ContextVar`` set and reset at the
WebSocket request boundary, so concurrent sessions can never cross keys.

**Deterministic Layout:** LLMs never emit positions. The layout engine runs
after every generation, ensuring collision-free visual presentation.
