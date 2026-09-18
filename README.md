# Modeling Agent

The Modeling Agent is the conversational backend used by the BESSER Web Modeling Editor.
It is a WebSocket service that turns natural language into **diagram operations**: it
interprets each request, decides what to do, and returns a structured action payload
(`inject_complete_system`, `modify_model`, `trigger_generator`, `trigger_smart_generator`, …)
that the editor applies to the canvas or hands to a generator.

## What It Does

- Creates and modifies models from natural language.
- Routes every message with **one** LLM classification call (`src/unified_classifier.py`),
  cached per message; a local, free TensorFlow `SimpleIntentClassifier` is the fallback.
- Supports multi-step orchestration (for example: model first, then generate code).
- Triggers BESSER's deterministic generators, or an LLM-authored **smart** generation
  path for stacks BESSER has no built-in generator for.
- Routes generation and conversation through a user-supplied API key (BYOK) when one is set.
- Answers UML specification questions using RAG over the OMG UML 2.5.1 specification.
- Converts uploaded files (PlantUML, knowledge graphs, XMI, PDFs, images, text) into
  diagram specifications.
- Accepts voice messages via OpenAI speech-to-text.

## Supported Diagram Types

The identifier is the value `get_diagram_type()` returns — the editor's storage-bucket
token, which the protocol, the factory and `SUPPORTED_DIAGRAM_TYPES` all key on. Note that
BPMN's token is `BPMN` (not `BPMNDiagram`) and the user-profile token is `UserDiagram`.

| Diagram type | Single element | Complete system | Modification |
| --- | --- | --- | --- |
| `ClassDiagram` | Yes | Yes | Yes |
| `ObjectDiagram` | Yes | Yes | Yes |
| `StateMachineDiagram` | Yes | Yes | Yes |
| `AgentDiagram` | Yes | Yes | Yes |
| `GUINoCodeDiagram` | Yes | Yes | Yes |
| `QuantumCircuitDiagram` | Yes | Yes | Yes |
| `BPMN` | Yes | Yes | Yes |
| `UserDiagram` | Yes | Yes | Yes |

## Supported Generators

Deterministic BESSER generators (`GENERATOR_KEYWORDS` in `src/handlers/generation_handler.py`):

`django`, `backend`, `web_app`, `sql`, `sqlalchemy`, `python`, `java`, `pydantic`,
`jsonschema`, `smartdata`, `agent`, `qiskit`, `rest_api`, `rdf`, plus the `export` and
`deploy` actions.

Anything outside that list — a non-BESSER language or framework, or a BESSER stack plus
extras the template cannot produce (auth, Docker, migrations, …) — is routed to the
**smart generator** via a `trigger_smart_generator` payload.

## Repository Structure

```text
modeling-agent/
  modeling_agent.py                # Runtime entrypoint: agent, states, intents, wiring
  config_example.yaml              # Template for the gitignored config.yaml
  Dockerfile                       # Image + entrypoint that writes config.yaml from env
  patches/websocket_platform.py    # Vendored BAF fix (see the note below)
  src/
    agent_setup.py                 # LLM/STT/RAG/factory bootstrapping
    agent_context.py               # Shared runtime globals populated at startup
    agent_config.py                # Tunable constants (limits, temperatures, budgets)
    model_config.py                # Per-call-site model tiers, env-overridable
    unified_classifier.py          # ONE classification call per message (routing brain)
    state_bodies.py                # State bodies + transition wiring
    session_helpers.py             # Reply/streaming helpers, transition conditions
    confirmation.py                # Pending replace/keep + GUI-mode flows
    byok.py                        # Per-request bring-your-own-key routing
    suggestions.py                 # Context-aware "what's next?" suggestions
    telemetry.py                   # Pilot-experiment prompt telemetry (fire-and-forget)
    llm/                           # LLM provider abstraction (structured output, streaming)
    memory/                        # Conversation memory + rolling summary
    schemas/                       # Pydantic schemas for structured LLM output
    tracking/                      # Token usage and cost tracking
    protocol/                      # Request parsing and protocol types
    routing/                       # Shared intent-name constants
    orchestrator/                  # Multi-operation planning + diagram-type resolution
    execution/                     # Operation execution engine (planning, model ops, files)
    handlers/                      # Generation, smart generation, file conversion, validation
    utilities/                     # Shared context/model/request helpers, LLM retry
    diagram_handlers/
      core/                        # Base handler + deterministic layout + prompt fragments
      types/                       # Concrete per-diagram handlers
      registry/                    # Factory + metadata registry
  tests/
  docs/
```

> `patches/websocket_platform.py` is copied over the pip-installed BAF 4.3.2 file at image
> build time. Stock BAF evicts the `_connections` slot unconditionally on close; with the
> stable `?user_id=` param the two sockets a browser tab opens share one session key, so a
> closing socket would drop the live one's replies. Re-vendor if the BAF version bumps.

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

Responses carry an `action` field. The terminal ones are `inject_complete_system` (with a
`systemSpec` of `classes` / `relationships` or the per-type equivalent), `inject_element`,
`modify_model` (single `modification` or a `modifications` array), `assistant_message`,
`trigger_generator`, `trigger_smart_generator`, `trigger_export`, `trigger_deploy`,
`trigger_github_import`, `auto_generate_gui`, `create_diagram_tab` and `agent_error`;
`progress` and `stream_start` / `stream_chunk` / `stream_done` are non-terminal.
See `docs/source/websocket_protocol.rst` for the full payload shapes.

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

Default WebSocket host/port are configured in `config.yaml` under
`platforms.websocket` (`localhost:8765`).

Optional environment overrides — the full list is in
`docs/source/configuration.rst`:

| Variable | Purpose |
| --- | --- |
| `BESSER_AGENT_MODEL_*` | Re-point one model tier (`CLASSIFIER`, `GENERATION_LARGE`, `GENERATION_GUI`, `GENERATION_SMALL`, `REASONING`, `VISION`, `EMBEDDINGS`), plus `REASONING_EFFORT` |
| `BESSER_BACKEND_URL` | BESSER backend base URL (diagram validation + telemetry) |
| `BESSER_AGENT_ALLOW_CUSTOM_BASE_URL` | Whether a BYOK request may supply its own API base URL |
| `BESSER_AGENT_STT_LANGUAGE` | Pin speech-to-text to a language instead of auto-detect |
| `LOG_PROMPTS` | Log full LLM prompts (local debugging only) |

## Deployment

The agent is one of **three services** deployed together — the BESSER backend
(`:9000`), this agent (`:8765`), and the editor frontend (`:8080`) — behind one nginx
reverse proxy that terminates TLS and exposes the agent at `wss://<host>/agent`.
See `docs/source/deployment.rst`.

```bash
docker build -t modeling-agent .
docker run -p 8765:8765 -e OPENAI_API_KEY=sk-... modeling-agent
```

The image's entrypoint writes `config.yaml` from the environment (redacting the API key
in its debug output). Boot is slow by design — BAF trains a NER model plus one intent
classifier per state before opening the socket (measured 3m38s), which is why the
Docker healthcheck uses a 300 s `start-period`.

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
