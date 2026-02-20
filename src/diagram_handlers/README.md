# Diagram Handlers Structure

This package is organized in three layers:

- `core/`: shared primitives used by all handlers.
  - `base_handler.py`: abstract base class, JSON parsing, validation helpers.
  - `layout_engine.py`: deterministic post-generation layout logic.
- `types/`: concrete handler implementations per diagram type.
  - `class_diagram_handler.py`
  - `object_diagram_handler.py`
  - `state_machine_handler.py`
  - `agent_diagram_handler.py`
  - `gui_nocode_diagram_handler.py`
  - `quantum_circuit_diagram_handler.py`
- `registry/`: assembly and metadata.
  - `factory.py`: handler instantiation and lookup.
  - `metadata.py`: supported diagram metadata and lookup helper.

Backward compatibility is kept through module shims at package root
(`base_handler.py`, `factory.py`, etc.), so existing imports continue to work.

