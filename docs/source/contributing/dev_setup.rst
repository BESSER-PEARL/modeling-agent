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
- **An OpenAI API key** — the model used per call site comes from the tier
  table in ``src/llm/model_config.py`` (all tiers env-overridable)
- **Git** — for version control and branch management

Installation
------------

.. code-block:: bash

   git clone --branch develop https://github.com/BESSER-PEARL/modeling-agent.git
   cd modeling-agent
   python -m venv venv
   source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows
   pip install -r requirements-dev.txt  # runtime pins + test and lint tools
   pip install -e .                     # the agent itself (src/ is the import root)

   # Copy and configure
   cp config_example.yaml config.yaml
   # Edit config.yaml with your OpenAI API key

Verify your setup:

.. code-block:: bash

   python -m pytest tests --ignore=tests/live -v
   python modeling_agent.py  # Should start WebSocket on :8765

If the agent starts and logs ``WebSocket server listening on ws://localhost:8765``,
your setup is correct.


Project Structure
-----------------

.. code-block:: text

   modeling-agent/
   ├── modeling_agent.py          # Entry point: defines intents, states, wires transitions
   ├── pyproject.toml             # Package (src/ as import root), version, import-linter contracts
   ├── config.yaml                # Runtime configuration (API keys, LLM settings)
   ├── config_example.yaml        # Template for config.yaml
   │
   ├── src/                       # All source code, one package per layer (top to bottom)
   │   ├── app/                   # The top layer
   │   │   ├── agent_setup.py     #   LLM initialization, classifier configuration
   │   │   ├── state_bodies.py    #   State body functions + transition wiring
   │   │   └── confirmation.py    #   Pending replace/keep and GUI-mode flows
   │   │
   │   ├── ingestion/             # File attachments
   │   │   └── file_handling.py   #   File attachment processing
   │   │
   │   ├── execution/             # Operation execution engine
   │   │   ├── planning.py        #   Multi-operation planning
   │   │   └── model_operations.py #  Single operation dispatch
   │   │
   │   ├── codegen/               # Generator triggers
   │   │   └── generation_handler.py  # Code generation routing, config parsing, safety guards
   │   │
   │   ├── orchestrator/          # Multi-step planning and diagram type resolution
   │   │   ├── request_planner.py #   3-tier planner (heuristic → keyword → LLM)
   │   │   └── workspace_orchestrator.py  # Diagram type resolution (keywords → patterns → context)
   │   │
   │   ├── conversation/          # Reply helpers, reply language, suggestions, telemetry
   │   │   └── session_helpers.py #   Reply helpers, streaming, intent matching conditions
   │   │
   │   ├── routing/               # The unified classifier, generation signals, intent constants
   │   ├── memory/                # Conversation memory (sliding window + summarization)
   │   │
   │   ├── diagram_handlers/      # Diagram-specific handlers
   │   │   ├── core/              #   BaseDiagramHandler, layout engine, shared logic
   │   │   ├── types/             #   ClassDiagramHandler, StateMachineHandler, etc.
   │   │   └── registry/          #   DiagramHandlerFactory, metadata
   │   │
   │   ├── schemas/               # Structured-output schemas
   │   ├── llm/                   # Provider, BYOK, model tiers, token and cost tracking
   │   ├── utilities/             # Shared helpers
   │   ├── protocol/              # WebSocket protocol layer
   │   │   ├── types.py           #   AssistantRequest, WorkspaceContext, SUPPORTED_DIAGRAM_TYPES
   │   │   └── adapters.py        #   parse_assistant_request(), v2 envelope unwrapping
   │   │
   │   └── common/                # Constants, session-state keys, error taxonomy
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
     - Entry point. Defines all 10 states and 10 intents (each with the
       ``training_sentences`` the local classifier needs), calls
       ``register_all()`` to wire bodies and transitions, and starts the
       session reaper.
   * - ``src/protocol/``
     - Parses the v2 WebSocket protocol. Unwraps the BESSER framework envelope,
       extracts ``AssistantRequest`` and ``WorkspaceContext``.
   * - ``src/routing/unified_classifier.py``
     - The router. One structured-output LLM call per message returns the
       intent plus all sub-routing fields; its system prompt
       (``src/prompts/classifier.py``) is the authoritative rulebook.
   * - ``src/codegen/generation_handler.py``
     - Code generation routing. Contains ``GENERATOR_REQUIRED_FIELDS``, the
       config-prompt flows, and the deterministic GitHub-continue guard.
   * - ``src/routing/guards.py``
     - The deterministic routing rules (classification skips, ``frontend_event``
       and ``[auto-fix]`` verdicts, the modeling-state guards, quick replies,
       the generation route). State bodies only map their decisions to replies.
   * - ``src/routing/generation_signals.py``
     - Generation vocabulary shared below the handlers: ``detect_generator_type()``,
       ``GENERATOR_KEYWORDS`` / ``GENERATOR_PREREQUISITES``, the GitHub-continue
       regexes and the recent-smart-gen signal the classifier reads.
   * - ``src/orchestrator/``
     - Multi-step planning (``request_planner.py``) and diagram type resolution
       (``workspace_orchestrator.py``). Converts a user message into an ordered
       list of operations.
   * - ``src/app/state_bodies.py``
     - State body functions (one per intent) and the transition wiring system
       (``add_unified_transitions()``).
   * - ``src/conversation/session_helpers.py``
     - Intent matching conditions (``json_intent_matches``,
       ``json_no_intent_matched``), reply helpers, streaming utilities, and
       cross-validation logic.
   * - ``src/diagram_handlers/``
     - One handler per diagram type. Each handler knows how to generate
       single elements, complete systems, and fallback responses for its type.
