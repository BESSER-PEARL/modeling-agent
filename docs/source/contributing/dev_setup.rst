Development Setup
=================

This section covers how to set up a local development environment for the
Modeling Agent, including prerequisites, installation, and an overview of the
project structure.

.. contents:: On this page
   :local:
   :depth: 2

Prerequisites
-------------

- **Python 3.11+** — the codebase uses modern typing features and f-strings
- **An OpenAI API key** — for GPT-4.1-mini (the intent classifier and LLM planner)
- **Git** — for version control and branch management

Installation
------------

.. code-block:: bash

   git clone <repository-url>
   cd modeling-agent
   python -m venv venv
   source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows
   pip install -r requirements.txt

   # Copy and configure
   cp config_example.yaml config.yaml
   # Edit config.yaml with your OpenAI API key

Verify your setup:

.. code-block:: bash

   python -m pytest -v
   python modeling_agent.py  # Should start WebSocket on :8765

If the agent starts and logs ``WebSocket server listening on ws://localhost:8765``,
your setup is correct.


Project Structure
-----------------

.. code-block:: text

   modeling-agent/
   ├── modeling_agent.py          # Entry point: defines intents, states, wires transitions
   ├── config.yaml                # Runtime configuration (API keys, LLM settings)
   ├── config_example.yaml        # Template for config.yaml
   │
   ├── src/                       # All source code
   │   ├── protocol/              # WebSocket protocol layer
   │   │   ├── types.py           #   AssistantRequest, WorkspaceContext, SUPPORTED_DIAGRAM_TYPES
   │   │   └── adapters.py        #   parse_assistant_request(), v2 envelope unwrapping
   │   │
   │   ├── handlers/              # Request handlers
   │   │   └── generation_handler.py  # Code generation routing, config parsing, safety guards
   │   │
   │   ├── diagram_handlers/      # Diagram-specific handlers
   │   │   ├── core/              #   BaseDiagramHandler, layout engine, shared logic
   │   │   ├── types/             #   ClassDiagramHandler, StateMachineHandler, etc.
   │   │   └── registry/          #   DiagramHandlerFactory, metadata
   │   │
   │   ├── orchestrator/          # Multi-step planning and diagram type resolution
   │   │   ├── request_planner.py #   3-tier planner (heuristic → keyword → LLM)
   │   │   └── workspace_orchestrator.py  # Diagram type resolution (keywords → patterns → context)
   │   │
   │   ├── routing/               # Intent constants and routing helpers
   │   ├── state_bodies.py        # State body functions + transition wiring
   │   ├── session_helpers.py     # Reply helpers, streaming, intent matching conditions
   │   ├── execution/             # Operation execution engine (package)
   │   │   ├── planning.py        #   Multi-operation planning
   │   │   ├── model_operations.py #  Single operation dispatch
   │   │   └── file_handling.py   #   File attachment processing
   │   ├── agent_setup.py         # LLM initialization, classifier configuration
   │   ├── memory/                # Conversation memory (sliding window + summarization)
   │   ├── tracking/              # Token counting and cost tracking
   │   └── suggestions.py         # Context-aware "What's next?" suggestions
   │
   ├── tests/                     # Test suite
   │   ├── conftest.py            #   FakeSession, FakeLLM, make_v2_payload, fixtures
   │   ├── test_generation_handler.py
   │   ├── test_orchestrator.py
   │   ├── test_request_planner.py
   │   ├── test_diagram_handlers.py
   │   ├── test_protocol.py
   │   └── ...                    #   15 test files total
   │
   └── docs/                      # Sphinx documentation
       └── source/                #   RST files → ReadTheDocs


Key Module Responsibilities
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Responsibility
   * - ``modeling_agent.py``
     - Entry point. Defines all 8 intents, creates states, calls ``register_all()``
       to wire transitions.
   * - ``src/protocol/``
     - Parses the v2 WebSocket protocol. Unwraps the BESSER framework envelope,
       extracts ``AssistantRequest`` and ``WorkspaceContext``.
   * - ``src/handlers/generation_handler.py``
     - Code generation routing. Contains ``detect_generator_type()``,
       ``_is_modeling_request()``, ``_is_diagram_creation_request()``, and all
       pre-filter/safety-net logic for intent disambiguation.
   * - ``src/orchestrator/``
     - Multi-step planning (``request_planner.py``) and diagram type resolution
       (``workspace_orchestrator.py``). Converts a user message into an ordered
       list of operations.
   * - ``src/state_bodies.py``
     - State body functions (one per intent) and the transition wiring system
       (``add_unified_transitions()``).
   * - ``src/session_helpers.py``
     - Intent matching conditions (``json_intent_matches``,
       ``json_no_intent_matched``), reply helpers, streaming utilities, and
       cross-validation logic.
   * - ``src/diagram_handlers/``
     - One handler per diagram type. Each handler knows how to generate
       single elements, complete systems, and fallback responses for its type.
