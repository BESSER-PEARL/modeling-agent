# Modeling Agent

The Modeling Agent is the conversational backend used by the BESSER Web Modeling Editor.
It interprets natural-language requests and returns structured modeling actions (create, modify,
generate, convert) over WebSocket-compatible payloads.

## What It Does

- Creates and modifies UML models from natural language.
- Supports multi-step orchestration (for example: model first, then generate code).
- Supports code generation triggers for multiple BESSER generators.
- Answers UML specification questions using RAG over UML documents.
- Converts uploaded files (PlantUML, KG, images, text) into diagram specifications.

## Supported Diagram Types

| Diagram type | Single element | Complete system | Modification |
| --- | --- | --- | --- |
| `ClassDiagram` | Yes | Yes | Yes |
| `ObjectDiagram` | Yes | Yes | Yes |
| `StateMachineDiagram` | Yes | Yes | Yes |
| `AgentDiagram` | Yes | Yes | Yes |
| `GUINoCodeDiagram` | Yes | Yes | Yes |
| `QuantumCircuitDiagram` | Yes | Yes | Yes |

## Supported Generators

`django`, `backend`, `web_app`, `sql`, `sqlalchemy`, `python`, `java`, `pydantic`, `jsonschema`, `smartdata`, `agent`, `qiskit`

## Repository Structure

```text
modeling-agent/
  modeling_agent.py                # Runtime entrypoint
  src/
    agent_setup.py                 # LLM/RAG/factory bootstrapping
    execution.py                   # Operation execution engine
    state_bodies.py                # Intent state logic
    suggestions.py                 # Context-aware suggestion engine
    llm/                           # LLM provider abstraction
    memory/                        # Conversation memory with sliding window
    schemas/                       # Pydantic schemas for structured LLM output (enforced name lengths, Literal actions)
    tracking/                      # Token usage and cost tracking
    protocol/                      # Request parsing and protocol types
    orchestrator/                  # Multi-operation planning and routing
    handlers/                      # Generation and file-conversion handlers
    utilities/                     # Shared context/model/request helpers
    diagram_handlers/
      core/                        # Base handler + deterministic layout
      types/                       # Concrete per-diagram handlers
      registry/                    # Factory + metadata registry
      *.py                         # Backward-compat import shims
  tests/
  docs/
```

## Request Protocol (v2)

The agent consumes assistant payloads with `protocolVersion: "2.0"`.
In BESSER WebSocket mode, this payload is often serialized inside the top-level `message` field.

Example payload:

```json
{
  "action": "user_message",
  "message": "{\"action\":\"user_message\",\"protocolVersion\":\"2.0\",\"clientMode\":\"workspace\",\"message\":\"create a User class\",\"context\":{\"activeDiagramType\":\"ClassDiagram\"}}"
}
```

The agent normalizes this into an internal `AssistantRequest` object (`src/protocol/types.py`).

## Setup

### Prerequisites

- Python 3.10+
- OpenAI API key

### Install

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Configure

1. Copy `config_example.yaml` to `config.yaml`.
2. Set `nlp.openai.api_key` in `config.yaml`.
3. Optional: copy `.env.example` to `.env` for local tooling.

```bash
cp config_example.yaml config.yaml
```

### Run

```bash
python modeling_agent.py
```

Default WebSocket host/port are configured in `config.yaml` (`websocket_platform`).

## Testing

```bash
# Full test suite
python -m pytest

# Focused suites
python -m pytest tests/test_diagram_handlers.py
python -m pytest tests/test_request_planner.py
python -m pytest tests/test_protocol.py
```

## Documentation

```bash
pip install -r docs/requirements.txt
cd docs
# Windows
make.bat html
# Linux/macOS
make html
```

## Notes for Contributors

- Keep behavior changes synchronized across `src/`, `tests/`, and `docs/source/`.
- Prefer deterministic handler outputs and shared helper functions under `src/utilities/`.
- Keep backward-compatibility shims when moving modules used by imports/tests.

## Security

- Never commit real API keys.
- Use `config_example.yaml` and `.env.example` as templates.
