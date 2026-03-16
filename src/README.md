# Source Layout

Top-level modules under `src/` are organized by responsibility:

- `diagram_handlers/`: diagram generation logic.
- `handlers/`: non-diagram handlers (generation and file conversion).
- `orchestrator/`: planning and operation routing.
- `protocol/`: request/context protocol models and adapters.
- `routing/`: intent names and routing helpers.
- `utilities/`: shared helper utilities.
- `llm/`: LLM provider abstraction (structured outputs, streaming).
- `memory/`: Conversation memory with sliding-window and LLM summarization.
- `schemas/`: Pydantic schemas for OpenAI structured outputs per diagram type.
- `tracking/`: Token usage and cost tracking.
- `suggestions.py`: context-aware follow-up suggestion engine.
- `agent_setup.py`, `agent_context.py`, `execution.py`, `state_bodies.py`: runtime wiring.

## Diagram Handler Internals

`diagram_handlers/` uses a layered structure:

- `core/`: base abstractions and deterministic layout engine.
- `types/`: concrete handler implementations per diagram type.
- `registry/`: factory and supported-diagram metadata.

Backward-compatible shim modules remain at `diagram_handlers/*.py` for
existing imports.

