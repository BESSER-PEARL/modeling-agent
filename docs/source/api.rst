API and Module Map
==================

Entrypoint
----------

- ``modeling_agent.py``: creates the BESSER agent, intents/states, and starts runtime.

Core runtime modules
--------------------

- ``src/agent_setup.py``: initializes LLM, RAG, and diagram factory.
- ``src/agent_context.py``: shared runtime context container.
- ``src/state_bodies.py``: state handlers and transition wiring.
- ``src/execution.py``: operation execution engine.

Protocol and orchestration
--------------------------

- ``src/protocol/types.py``: protocol dataclasses.
- ``src/protocol/adapters.py``: payload extraction/normalization.
- ``src/orchestrator/request_planner.py``: multi-operation planning.
- ``src/orchestrator/workspace_orchestrator.py``: target diagram selection helpers.

Diagram handlers
----------------

- ``src/diagram_handlers/core``: abstract base handler + deterministic layout engine.
- ``src/diagram_handlers/types``: concrete handlers by diagram type.
- ``src/diagram_handlers/registry``: factory and metadata.
- ``src/diagram_handlers/*.py``: backward-compatible import shims.

Auxiliary handlers and utilities
--------------------------------

- ``src/handlers/generation_handler.py``
- ``src/handlers/file_conversion_handler.py``
- ``src/utilities/*``
