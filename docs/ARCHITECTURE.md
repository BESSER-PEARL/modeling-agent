# Modeling Agent — Architecture Documentation

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Structure](#2-repository-structure)
3. [System Architecture Diagram](#3-system-architecture-diagram)
4. [Configuration and Entry Point](#4-configuration-and-entry-point)
5. [Protocol Layer](#5-protocol-layer)
6. [Agent Core and Initialization](#6-agent-core-and-initialization)
7. [State Machine and Conversation Flow](#7-state-machine-and-conversation-flow)
8. [Execution Engine](#8-execution-engine)
9. [Orchestration Layer](#9-orchestration-layer)
10. [Diagram Handler System](#10-diagram-handler-system)
11. [Schema Reference](#11-schema-reference)
12. [Utility Layer](#12-utility-layer)
13. [Knowledge and Pattern Libraries](#13-knowledge-and-pattern-libraries)
14. [File Conversion Pipeline](#14-file-conversion-pipeline)
15. [Generation Handler](#15-generation-handler)
16. [Quality Review](#16-quality-review)
17. [Data Flow Diagrams](#17-data-flow-diagrams)
18. [Key Design Patterns](#18-key-design-patterns)
19. [Concurrency and Caching](#19-concurrency-and-caching)
20. [Testing Architecture](#20-testing-architecture)
21. [Deployment](#21-deployment)

---

## 1. Overview

The modeling agent is a **WebSocket-based conversational AI agent** that enables users to create, modify, and query software models through natural language. It bridges the BESSER Web Modeling Editor frontend with an LLM backend (OpenAI GPT-4.1-mini), turning free-text requests into structured diagram JSON payloads.

**What it does:**
- Accepts natural language over WebSocket (`ws://host:8765`)
- Classifies intent using an LLM-based classifier
- Plans one or more modeling operations for complex requests
- Dispatches to specialized diagram handlers per diagram type
- Returns structured JSON that the frontend editor renders directly
- Converts uploaded files (PlantUML, RDF, images) into diagram specifications
- Routes code generation, export, and deployment requests
- Answers UML specification questions via RAG (ChromaDB)

**Supported Diagram Types:**

| Type | Description | Output Format |
|------|-------------|---------------|
| `ClassDiagram` | UML class diagrams | Apollon-compatible JSON |
| `ObjectDiagram` | UML object/instance diagrams | Apollon-compatible JSON |
| `StateMachineDiagram` | UML state machine diagrams | Apollon-compatible JSON |
| `AgentDiagram` | BESSER conversational agent diagrams | Custom state/intent JSON |
| `GUINoCodeDiagram` | No-code GUI models | GrapesJS project JSON |
| `QuantumCircuitDiagram` | Quantum circuit diagrams | Quirk-format JSON |

**Technology Stack:**
- **Agent Framework:** BESSER Agentic Framework v4.2.3
- **LLM:** OpenAI GPT-4.1-mini (JSON mode temp=0.2, text mode temp=0.4)
- **RAG:** LangChain + ChromaDB vector store
- **Transport:** WebSocket (port 8765)
- **Runtime:** Python 3.11+

---

## 2. Repository Structure

```
modeling-agent/
├── modeling_agent.py                 # Main entry point
├── config.ini                        # Runtime configuration
├── requirements.txt                  # Python dependencies
├── .readthedocs.yaml                 # Read the Docs build config
├── .dockerignore                     # Docker ignore rules
├── .env                              # Environment variables (local)
├── CITATION.cff                      # Citation metadata
├── CODEOWNERS                        # GitHub code owners
├── CONTRIBUTING.md                   # Contribution guidelines
├── CODE_OF_CONDUCT.md                # Community code of conduct
├── GOVERNANCE.md                     # Governance rules
├── README.md                         # Project readme
├── uml_specs/                        # UML spec PDFs (for RAG ingestion)
│   └── formal-17-12-05.pdf           # OMG UML 2.5.1 specification
│
├── src/                              # All business logic
│   ├── agent_context.py              # Shared mutable globals
│   ├── agent_setup.py                # Initialization functions
│   ├── state_bodies.py               # All agent state body functions
│   ├── execution.py                  # Core execution engine
│   ├── confirmation.py               # Pending confirmation flow handlers
│   ├── session_helpers.py            # Protocol-agnostic reply utilities
│   ├── suggestions.py                # Contextual suggestion engine
│   ├── domain_patterns.py            # 10-domain expert pattern library
│   ├── state_patterns.py             # 8 behavioral lifecycle patterns
│   ├── quality_review.py             # Post-generation quality checks
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── provider.py               # LLM provider abstraction
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   └── conversation_memory.py    # Sliding window conversation memory
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── agent_diagram.py          # Agent diagram Pydantic schema
│   │   ├── class_diagram.py          # Class diagram Pydantic schema
│   │   ├── gui_diagram.py            # GUI diagram Pydantic schema
│   │   ├── object_diagram.py         # Object diagram Pydantic schema
│   │   ├── quantum_circuit.py        # Quantum circuit Pydantic schema
│   │   └── state_machine.py          # State machine Pydantic schema
│   │
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── token_tracker.py          # Token usage tracking
│   │
│   ├── protocol/
│   │   ├── __init__.py
│   │   ├── types.py                  # Data classes: AssistantRequest, WorkspaceContext
│   │   └── adapters.py               # Protocol parsing and normalization
│   │
│   ├── orchestrator/
│   │   ├── __init__.py               # Re-exports
│   │   ├── request_planner.py        # LLM-based multi-operation planner
│   │   └── workspace_orchestrator.py # Diagram type targeting
│   │
│   ├── handlers/
│   │   ├── __init__.py
│   │   ├── generation_handler.py     # Code generator routing
│   │   └── file_conversion_handler.py # Uploaded file -> diagram spec
│   │
│   ├── diagram_handlers/
│   │   ├── __init__.py
│   │   ├── factory.py                # Backward-compat re-export
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── base_handler.py       # Abstract base class for all handlers
│   │   │   └── layout_engine.py      # Deterministic canvas layout
│   │   ├── registry/
│   │   │   ├── __init__.py
│   │   │   ├── factory.py            # DiagramHandlerFactory
│   │   │   └── metadata.py           # Per-type display metadata
│   │   └── types/
│   │       ├── __init__.py
│   │       ├── class_diagram_handler.py
│   │       ├── object_diagram_handler.py
│   │       ├── state_machine_handler.py
│   │       ├── agent_diagram_handler.py
│   │       ├── gui_nocode_diagram_handler.py
│   │       └── quantum_circuit_diagram_handler.py
│   │
│   ├── utilities/
│   │   ├── __init__.py
│   │   ├── model_helpers.py          # Backward-compat re-exporter
│   │   ├── model_context.py          # Model summarization
│   │   ├── model_resolution.py       # Target model resolution
│   │   ├── workspace_context.py      # LLM context block builder
│   │   ├── request_builders.py       # Derived AssistantRequest factories
│   │   ├── class_metadata.py         # Class attribute/method extraction
│   │   └── layout_helpers.py         # Position extraction, anchor lines
│   │
│   └── routing/
│       ├── __init__.py
│       └── intents.py                # GENERATION_INTENT_NAME constant
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Shared fixtures (FakeSession, FakeLLM)
│   ├── test_diagram_handlers.py
│   ├── test_file_conversion.py
│   ├── test_generation_handler.py
│   ├── test_gui_chart_generation.py
│   ├── test_model_helpers.py
│   ├── test_orchestrator.py
│   ├── test_protocol.py
│   ├── test_request_planner.py
│   ├── test_suggestions.py           # Contextual suggestion engine tests
│   ├── test_schemas.py               # Pydantic schema validation tests
│   ├── test_conversation_memory.py   # Sliding window memory tests
│   ├── test_token_tracker.py         # Token usage tracking tests
│   ├── test_llm_provider.py          # LLM provider abstraction tests
│   └── test_base_handler.py          # Base handler utilities tests
│
└── docs/
    ├── Makefile
    ├── make.bat
    ├── requirements.txt              # Sphinx + RTD theme
    ├── ARCHITECTURE.md               # This file
    └── source/
        ├── conf.py                   # Sphinx configuration
        ├── index.rst
        ├── getting_started.rst
        ├── architecture.rst
        ├── schema.rst
        ├── diagram_handlers.rst
        ├── orchestration.rst
        ├── configuration.rst
        ├── usage.rst
        ├── api.rst
        ├── deployment.rst
        └── contributing.rst
```

---

## 3. System Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         BESSER Web Modeling Editor                            │
│                          (React/TypeScript SPA)                               │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ WebSocket (JSON v2 protocol)
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            MODELING AGENT                                     │
│                                                                              │
│  ┌───────────────┐    ┌───────────────────┐    ┌─────────────────────────┐  │
│  │   Protocol     │───▶│   State Machine    │───▶│   Execution Engine     │  │
│  │   Adapters     │    │   (8 states,       │    │                        │  │
│  │               │    │    8 intents)       │    │  execute_planned_      │  │
│  │  parse_       │    │                    │    │  operations()          │  │
│  │  assistant_   │    │  Intent Classifier  │    │                        │  │
│  │  request()    │    │  (LLM-based)       │    │  execute_model_       │  │
│  └───────────────┘    └───────────────────┘    │  operation()           │  │
│                                                 └───────────┬─────────────┘  │
│                                                             │                │
│  ┌──────────────────────────────────────────────────────────┤                │
│  │                                                          │                │
│  ▼                                                          ▼                │
│  ┌───────────────────┐    ┌──────────────────────────────────────────────┐  │
│  │   Orchestrator     │    │           Diagram Handler System             │  │
│  │                    │    │                                              │  │
│  │  request_planner   │    │  ┌──────────────────────────────────────┐   │  │
│  │  (LLM or           │    │  │       BaseDiagramHandler             │   │  │
│  │   heuristic)       │    │  │  predict_with_retry()                │   │  │
│  │                    │    │  │  predict_two_pass()                   │   │  │
│  │  workspace_        │    │  │  validate_and_refine()                │   │  │
│  │  orchestrator      │    │  │  apply_*_layout()                    │   │  │
│  │  (type targeting)  │    │  └─────────────────┬────────────────────┘   │  │
│  └───────────────────┘    │    ┌───────┬────────┴──┬──────────┬──────┐  │  │
│                            │    ▼       ▼           ▼          ▼      ▼  │  │
│                            │  Class  StateMachine Object   Agent   GUI  │  │
│                            │  Diagram Handler    Diagram  Diagram      │  │
│                            │  Handler            Handler  Handler      │  │
│                            │                                  Quantum  │  │
│                            │                                  Circuit  │  │
│                            └──────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Layout Engine   │  │ Domain &     │  │ Quality      │  │ File         │  │
│  │ (deterministic  │  │ State        │  │ Review       │  │ Conversion   │  │
│  │  positioning)   │  │ Patterns     │  │ (post-gen    │  │ Handler      │  │
│  └────────────────┘  └──────────────┘  │  suggestions)│  └──────────────┘  │
│                                         └──────────────┘                     │
└──────────────────────────────────────────────────────────────────────────────┘
                                │
                 ┌──────────────┼──────────────┐
                 │              │              │
                 ▼              ▼              ▼
    ┌──────────────────┐ ┌───────────┐ ┌──────────────┐
    │ OpenAI GPT-4.1   │ │ ChromaDB  │ │ UML Spec     │
    │                   │ │ Vector    │ │ PDFs         │
    │ - JSON  (t=0.2)  │ │ Store     │ │ (RAG source) │
    │ - Text  (t=0.4)  │ └───────────┘ └──────────────┘
    │ - Vision          │
    └──────────────────┘
```

---

## 4. Configuration and Entry Point

### `config.ini`

```ini
[websocket_platform]
websocket.host = 0.0.0.0
websocket.port = 8765
streamlit.host = localhost
streamlit.port = 5001

[api]
api.server.url = http://localhost:3001

[nlp]
nlp.language = en
nlp.openai.api_key = YOUR_KEY_HERE
nlp.intent.openai.model_name = gpt-4o-mini

[db]
db.monitoring = False
db.monitoring.dialect = postgresql
```

### `modeling_agent.py` — Main Entry Point

Startup sequence:
1. Adds `src/` to `sys.path`
2. Creates the BESSER `Agent` object
3. Calls four `init_*` functions from `agent_setup`
4. Populates `agent_context` module-level globals
5. Defines all 8 states and 8 intents
6. Calls `state_bodies.register_all()` to wire state bodies and transitions
7. Starts WebSocket platform: `agent.use_websocket_platform(use_ui=False)`
8. Calls `agent.run()`

---

## 5. Protocol Layer

### `src/protocol/types.py`

All agent code works with `AssistantRequest` objects after parsing.

**`AssistantRequest` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `action` | `str` | Request type (`"user_message"`, `"frontend_event"`) |
| `protocol_version` | `str` | `"2.0"` for v2 clients |
| `message` | `str` | Natural language request text |
| `diagram_type` | `str` | Target diagram type string |
| `diagram_id` | `str` | Target diagram instance ID |
| `current_model` | `dict` | Full model JSON currently active |
| `context` | `WorkspaceContext` | Structured workspace context |
| `attachments` | `list[FileAttachment]` | Uploaded file attachments |

**`WorkspaceContext`** captures the editor state:
- `active_diagram_type` — which diagram tab is open
- `active_diagram_id` — UUID of active diagram
- `active_model` — JSON model currently shown
- `project_snapshot` — all diagrams in the project keyed by type
- `diagram_summaries` — compact per-diagram summaries

**`FileAttachment`** for uploaded files:
- `filename` — original filename
- `content_b64` — base64-encoded content
- `mime_type` — MIME type

**Supported diagram types constant:**
```python
SUPPORTED_DIAGRAM_TYPES = {
    "ClassDiagram",
    "ObjectDiagram",
    "StateMachineDiagram",
    "AgentDiagram",
    "GUINoCodeDiagram",
    "QuantumCircuitDiagram",
}
```

### `src/protocol/adapters.py`

Parsing pipeline:
```
BESSER WebSocket Event
    ↓ extract_event_payload()
    ↓ _unwrap_v2_envelope()
    ↓ parse_v2_payload()
    ↓ strip_diagram_prefix()
    ↓ normalize_diagram_type()
    ▼
AssistantRequest (canonical)
```

Parsed requests are cached on the session keyed by event identity (`id(session.event)`), so multiple calls to `parse_assistant_request()` within the same message processing cycle avoid redundant JSON parsing. The cache is automatically invalidated when a new WebSocket message arrives.

---

## 6. Agent Core and Initialization

### `src/agent_context.py` — Shared Globals

Module-level mutable variables set at startup, read throughout:

```python
agent             # BESSER Agent instance
gpt               # LLMOpenAI, JSON mode, gpt-4.1-mini, temp=0.2
gpt_text          # LLMOpenAI, free-text, gpt-4.1-mini, temp=0.4
gpt_predict_json  # Closure: (prompt) -> dict
uml_rag           # RAG instance (ChromaDB), None if unavailable
diagram_factory   # DiagramHandlerFactory instance
openai_api_key    # str
```

### `src/agent_setup.py` — Initialization

| Function | Purpose |
|----------|---------|
| `init_llm(agent)` | Creates two LLMOpenAI instances (JSON + text mode) |
| `init_rag(agent)` | Builds ChromaDB-backed RAG for UML spec lookups |
| `init_diagram_factory(gpt)` | Creates DiagramHandlerFactory with all 6 handlers |
| `init_intent_classifier_config()` | LLM-based intent classifier (description mode) |

---

## 7. State Machine and Conversation Flow

### State Diagram

```
                        ┌─────────────────────┐
                        │   greetings_state    │◄──── Session Start
                        │   (welcome + route)  │
                        └──────────┬──────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
              ▼                    ▼                     ▼
┌─────────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
│ create_single_      │ │ create_complete_  │ │ modify_model_state   │
│ element_state       │ │ system_state      │ │                      │
│                     │ │                   │ │ default_mode=        │
│ default_mode=       │ │ default_mode=     │ │ "modify_model"       │
│ "single_element"    │ │ "complete_system" │ │                      │
└─────────────────────┘ └──────────────────┘ └──────────────────────┘
         │                       │                      │
         └───────────────────────┼──────────────────────┘
                                 │
                    All three call _modeling_state_body()
                    → execute_planned_operations()
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
              ▼                  ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
│ modeling_help_   │ │ describe_model_  │ │ uml_rag_state        │
│ state            │ │ state            │ │                      │
│                  │ │                  │ │ UML spec lookups     │
│ Conceptual Q&A   │ │ Summarizes model │ │ via ChromaDB RAG     │
└──────────────────┘ └──────────────────┘ └──────────────────────┘

              ┌──────────────────────────┐
              │    generation_state      │
              │                          │
              │ Code generators, export, │
              │ deployment               │
              └──────────────────────────┘
```

### 8 Intents

| Intent | Description | Target State |
|--------|-------------|--------------|
| `hello_intent` | Greeting / session start | `greetings_state` |
| `create_single_element_intent` | Add one element | `create_single_element_state` |
| `create_complete_system_intent` | Generate full diagram | `create_complete_system_state` |
| `modify_model_intent` | Modify existing elements | `modify_model_state` |
| `modeling_help_intent` | Conceptual questions | `modeling_help_state` |
| `describe_model_intent` | Describe current model | `describe_model_state` |
| `uml_spec_intent` | UML spec lookups | `uml_rag_state` |
| `generation_intent` | Generate code/deploy | `generation_state` |

### State Body Summary

| State | Body Function | Behavior |
|-------|---------------|----------|
| `greetings_state` | `greetings_body` | Welcome, quick patterns, then `_modeling_state_body` fallthrough |
| `create_single_element_state` | `create_single_element_body` | `_modeling_state_body(default_mode="single_element")` |
| `create_complete_system_state` | `create_complete_system_body` | `_modeling_state_body(default_mode="complete_system")` |
| `modify_model_state` | `modify_modeling_body` | `_modeling_state_body(default_mode="modify_model")` |
| `modeling_help_state` | `modeling_help_body` | `gpt_text.predict()` with UML system prompt |
| `describe_model_state` | `describe_model_body` | `detailed_model_summary()` + LLM narrative |
| `uml_rag_state` | `uml_rag_body` | RAG retrieval chain query |
| `generation_state` | `generation_body` | `handle_generation_request()` |

---

## 8. Execution Engine

### `src/execution.py`

The core dispatch layer between state bodies and diagram handlers.

**`execute_planned_operations(session, request, default_mode, matched_intent)`**

1. Calls `plan_assistant_operations()` to get operation list
2. Loops over operations:
   - `type == "model"` → `execute_model_operation()`
   - `type == "generation"` → `handle_generation_request()`
3. If a model operation returns `None` (pending confirmation), saves remaining ops and halts

**`execute_model_operation(session, request, operation, default_mode, ...)`**

The most complex function. Steps:

```
1. Resolve diagram type (from operation or heuristic)
2. Resolve operation mode (from operation or default)
3. Resolve request text (from operation or message)
4. Existing-model guard (complete_system only)
   → Ask "replace or keep?" if model exists
   → Store pending_complete_system, return None
5. GUI generation-mode choice (GUINoCodeDiagram only)
   → Ask "auto or LLM?" if class diagram exists
   → Store pending_gui_choice, return None
6. Handler lookup via diagram_factory
7. Build modeling prompt (request + workspace context)
8. Resolve GUI class metadata (if GUINoCodeDiagram)
9. Dispatch to handler method:
   - single_element → handler.generate_single_element()
   - modify_model   → handler.generate_modification()
   - complete_system → handler.generate_complete_system()
10. Inject metadata (diagramType, diagramId, replaceExisting)
11. Send reply_payload to frontend
12. Record action in session history
13. Run quality review, store suggestions
14. Return target_diagram_type
```

### Confirmation Flows

**Pending Complete-System Confirmation (`src/confirmation.py`):**

When user requests `complete_system` but model already exists:
1. Store `pending_complete_system` in session
2. Ask "Replace or keep existing model?"
3. Return `None` (halt remaining ops)
4. User responds ("replace", "keep", etc.)
5. `handle_pending_system_confirmation()` at top of next state body resumes execution

**Pending GUI Generation-Mode Choice:**

When user requests `GUINoCodeDiagram` and ClassDiagram exists:
1. Check for customization hints (chart, dashboard, custom, etc.)
2. If hints → skip to LLM path
3. If no hints → ask "auto or LLM?"
4. Store `pending_gui_choice` in session
5. User responds → auto or LLM path

---

## 9. Orchestration Layer

### `src/orchestrator/workspace_orchestrator.py`

**Three-level diagram type resolution:**

```
Level 1: Explicit keywords
  "class diagram" → ClassDiagram
  "object diagram" → ObjectDiagram
  "state machine" → StateMachineDiagram
  "agent diagram" → AgentDiagram
  "gui" → GUINoCodeDiagram
  "quantum circuit" → QuantumCircuitDiagram
         │
         ▼ (no match)
Level 2: Implicit semantic scoring
  Token/weight pairs per type, highest score wins
         │
         ▼ (no match)
Level 3: Context fallback
  Use active diagram type, or priority:
  ClassDiagram > ObjectDiagram > StateMachine > Agent > GUI > Quantum
```

### `src/orchestrator/request_planner.py`

Plans multi-step execution from a single user message.

**Decision flow:**
0. Fast heuristic pre-decomposition via regex patterns. Catches common shapes: "create a web app for X", "create/build X and generate Y", "generate X", "create a gui/state machine/agent/quantum circuit/object diagram/class diagram for X". If matched, skips the LLM planner entirely.
1. Build keyword fallback `_fallback_operations()` (always computed as safety net)
2. Check `_should_use_llm_planner()` — true for multi-clause messages with multiple diagram targets or generation requests; skipped when intent classifier already resolved a single target
3. If LLM: send planning prompt, get JSON `operations` array
4. `_normalize_operations()` — deduplicate, validate, enforce ClassDiagram-first ordering
5. Fall back to heuristic if LLM returns nothing valid

**Generator prerequisites** (auto-injected modeling ops):
```python
GENERATOR_PREREQUISITES = {
    "web_app":    ["ClassDiagram", "GUINoCodeDiagram"],
    "react":      ["ClassDiagram", "GUINoCodeDiagram"],
    "flutter":    ["ClassDiagram", "GUINoCodeDiagram"],
    "django":     ["ClassDiagram"],
    "backend":    ["ClassDiagram"],
    "sql":        ["ClassDiagram"],
    "sqlalchemy": ["ClassDiagram"],
    "python":     ["ClassDiagram"],
    "java":       ["ClassDiagram"],
    "pydantic":   ["ClassDiagram"],
    "jsonschema": ["ClassDiagram"],
    "rest_api":   ["ClassDiagram"],
    "agent":      ["AgentDiagram"],
    "qiskit":     ["QuantumCircuitDiagram"],
}
```

---

## 10. Diagram Handler System

### Handler Class Hierarchy

```
BaseDiagramHandler (abstract)
│
│  Required abstract methods:
│  ├── get_diagram_type() -> str
│  ├── get_system_prompt() -> str
│  ├── generate_single_element(request, model, **kw) -> dict
│  ├── generate_complete_system(request, model, **kw) -> dict
│  └── generate_fallback_element(request) -> dict
│
│  Shared concrete methods:
│  ├── generate_modification()     # Default modification with LLM
│  ├── predict_with_retry()        # LLM call + exponential backoff + jitter
│  ├── predict_structured()        # OpenAI structured outputs with Pydantic validation
│  ├── predict_two_pass_structured() # Reasoning pass + structured pass (skipped for simple requests <80 chars)
│  ├── predict_two_pass()          # Free-text reasoning → JSON output
│  ├── validate_and_refine()       # LLM self-critique loop
│  ├── repair_json_response()      # Last-resort JSON repair
│  ├── apply_single_layout()       # Layout for single elements
│  ├── apply_system_layout()       # Layout for full systems
│  └── _error_response()           # Standard error format
│
├── ClassDiagramHandler
│   ├── Domain pattern injection (10 domains)
│   ├── Two-pass generation + validation loop
│   ├── Impact analysis for modifications
│   └── Incremental fallback (class-by-class)
│
├── StateMachineHandler
│   ├── State pattern injection (8 patterns)
│   ├── Specialized validation (initial/final/orphan checks)
│   └── Fallback: 3-state machine (initial → active → final)
│
├── ObjectDiagramHandler
│   ├── Always requires ClassDiagram reference
│   ├── Reference catalog extraction
│   └── Heuristic value generator per attribute type
│
├── AgentDiagramHandler
│   ├── Elements: state (replies[]), intent (trainingPhrases[]), initial
│   ├── Rich normalization pipeline (7 normalizers)
│   └── Auto-inserts initial transition if missing
│
├── GUINoCodeDiagramHandler
│   ├── Output: GrapesJS project JSON
│   ├── Auto-generate mode (one page per class, no LLM)
│   ├── Chart color palettes for data binding
│   └── Class metadata injection for charts/tables
│
└── QuantumCircuitDiagramHandler
    ├── Output: Quirk-format JSON (columns of gates)
    ├── 60+ gate symbol mappings
    └── Algorithm detection (Grover, QFT)
```

### Layout Engine (`src/diagram_handlers/core/layout_engine.py`)

Single entry point: `apply_layout(spec, diagram_type, mode, existing_model)`

**Algorithm:**
1. Collect existing element positions
2. Compute dynamic canvas bounds (expand for large diagrams)
3. Calculate ideal grid shape (approximately square)
4. Assign grid positions, snapped to 20px grid
5. Check collision against all placed rectangles
6. Fall back to extending grid if no free position

**Parameters:**
- Canvas: -900 to 900 (x), -500 to 500 (y), expandable
- Gaps: h_gap=60px, v_gap=50px
- Margin: 40px minimum between elements
- Grid snap: 20px

### DiagramHandlerFactory

```python
class DiagramHandlerFactory:
    def __init__(self, llm):
        self._handlers = {
            "ClassDiagram":          ClassDiagramHandler(llm),
            "ObjectDiagram":         ObjectDiagramHandler(llm),
            "StateMachine":          StateMachineHandler(llm),
            "AgentDiagram":          AgentDiagramHandler(llm),
            "GUINoCodeDiagram":      GUINoCodeDiagramHandler(llm),
            "QuantumCircuitDiagram": QuantumCircuitDiagramHandler(llm),
        }

    def get_handler(self, diagram_type: str) -> Optional[BaseDiagramHandler]
    def get_supported_types(self) -> list[str]
    def is_supported(self, diagram_type: str) -> bool
```

---

## 11. Schema Reference

### AssistantRequest Schema

```json
{
  "action": "user_message",
  "protocolVersion": "2.0",
  "clientMode": "workspace",
  "message": "create a User class with id and email",
  "diagramType": "ClassDiagram",
  "diagramId": "uuid-string",
  "context": {
    "activeDiagramType": "ClassDiagram",
    "activeDiagramId": "uuid-string",
    "activeModel": { "...model JSON..." },
    "projectSnapshot": {
      "ClassDiagram": { "...model..." },
      "StateMachineDiagram": null
    },
    "diagramSummaries": {
      "ClassDiagram": "3 classes, 2 relationships"
    }
  },
  "attachments": [
    {
      "filename": "model.puml",
      "contentBase64": "QGN0YXJ0dW...",
      "mimeType": "text/plain"
    }
  ]
}
```

### Reply Payload Schema (Model Operations)

**inject_single_element:**
```json
{
  "action": "inject_single_element",
  "diagramType": "ClassDiagram",
  "diagramId": "uuid",
  "elementSpec": {
    "elements": {
      "elem-uuid": {
        "id": "elem-uuid",
        "name": "User",
        "type": "Class",
        "bounds": { "x": 100, "y": 100, "width": 200, "height": 150 },
        "attributes": {}
      }
    },
    "relationships": {}
  }
}
```

**inject_complete_system:**
```json
{
  "action": "inject_complete_system",
  "diagramType": "ClassDiagram",
  "diagramId": "uuid",
  "replaceExisting": true,
  "systemSpec": {
    "elements": { "...all elements..." },
    "relationships": { "...all relationships..." }
  }
}
```

**inject_modification:**
```json
{
  "action": "inject_modification",
  "diagramType": "ClassDiagram",
  "diagramId": "uuid",
  "modificationSpec": {
    "elementsToAdd": {},
    "elementsToUpdate": {},
    "elementsToRemove": [],
    "relationshipsToAdd": {},
    "relationshipsToRemove": []
  }
}
```

### ClassDiagram Element Schema

```json
{
  "id": "uuid",
  "name": "ClassName",
  "type": "Class",
  "bounds": { "x": 0, "y": 0, "width": 200, "height": 150 },
  "attributes": {
    "attr-uuid": {
      "id": "attr-uuid",
      "name": "attributeName",
      "type": "ClassAttribute",
      "bounds": { "x": 0, "y": 40, "width": 200, "height": 30 }
    }
  }
}
```

### ClassDiagram Relationship Schema

```json
{
  "id": "uuid",
  "type": "ClassBidirectional",
  "source": { "element": "class-uuid-1", "multiplicity": "1" },
  "target": { "element": "class-uuid-2", "multiplicity": "*" },
  "path": [
    { "x": 200, "y": 75 },
    { "x": 400, "y": 75 }
  ]
}
```

### StateMachine Element Schema

```json
{
  "id": "uuid",
  "name": "StateName",
  "type": "ObjectActivityNode",
  "bounds": { "x": 0, "y": 0, "width": 160, "height": 80 }
}
```

### AgentDiagram Element Schema

```json
{
  "type": "state",
  "name": "greeting_state",
  "replies": ["Hello! How can I help you?"],
  "x": 100,
  "y": 100
}
```

```json
{
  "type": "intent",
  "name": "hello_intent",
  "trainingPhrases": ["hi", "hello", "hey"],
  "x": 300,
  "y": 100
}
```

### GUINoCodeDiagram Schema (GrapesJS)

```json
{
  "pages": [
    {
      "name": "UserManagement",
      "component": "<div>...GrapesJS HTML...</div>",
      "styles": "...",
      "scripts": "..."
    }
  ]
}
```

### QuantumCircuitDiagram Schema (Quirk)

```json
{
  "cols": [
    [1, 1, "H"],
    ["*", 1, "X"],
    ["Measure", "Measure", "Measure"]
  ],
  "gates": []
}
```

### Generation Trigger Payload

```json
{
  "action": "trigger_generator",
  "generatorType": "django",
  "config": {
    "project_name": "myproject",
    "app_name": "myapp",
    "containerization": true
  },
  "diagramType": "ClassDiagram"
}
```

### Quality Suggestion Schema

```json
{
  "suggestions": [
    "Consider adding an 'id' attribute to the User class",
    "The Customer class has no relationships — consider connecting it"
  ],
  "whatsNext": [
    "Create object instances to test your class model",
    "Add a state machine to model User lifecycle"
  ]
}
```

---

## 12. Utility Layer

### Model Resolution (`src/utilities/model_resolution.py`)

**`resolve_target_model(request, target_type)`** — Priority:
1. `request.context.active_model` (if active type matches)
2. `request.context.project_snapshot[target_type]`
3. `request.current_model` (legacy fallback)

**`resolve_class_diagram(request)`** — Shortcut for ClassDiagram from snapshot/active.

### Model Context (`src/utilities/model_context.py`)

**`compact_model_summary()`** — One-line: `"3 classes, 2 relationships"`

**`detailed_model_summary()`** — Multi-line structural summary per diagram type, injected into LLM prompts.

### Workspace Context (`src/utilities/workspace_context.py`)

**`build_workspace_context_block()`** — The text block appended to every modeling prompt:
- Target and active diagram type
- Active model summary
- Layout anchors (existing element positions)
- Project info
- Cross-diagram references (ClassDiagram summary for StateMachine/GUI/Object generation)

### Class Metadata (`src/utilities/class_metadata.py`)

**`extract_class_metadata(model)`** — 4-pass extraction over ClassDiagram:
1. Find `type == "Class"` elements
2. Attach `ClassAttribute` elements to parent classes
3. Attach `ClassMethod` elements to parent classes
4. Extract association ends from relationships

Returns typed metadata with `isNumeric`/`isString` flags for chart binding.

### Layout Helpers (`src/utilities/layout_helpers.py`)

- `extract_element_position(element)` — Get (x, y) from element bounds
- `build_layout_anchor_lines(model, diagram_type)` — Format existing positions for LLM prompt
- `is_primary_layout_element(element, diagram_type)` — Check if element should get positioned

### Request Builders (`src/utilities/request_builders.py`)

- `build_request_for_target(request, target_type)` — Update request for chained operations
- `build_generation_request(request, generator_type, config, message_override)` — Prepare generation request

---

## 13. Knowledge and Pattern Libraries

### Domain Patterns (`src/domain_patterns.py`)

10 pre-defined expert domain patterns injected into ClassDiagram generation:

| Domain | Keywords (sample) | Core Classes |
|--------|-------------------|--------------|
| `ecommerce` | shop, store, order, cart | Product, Customer, Order, Cart, Payment |
| `library` | book, library, isbn, borrow | Book, Member, Loan, Author, Category |
| `hospital` | patient, doctor, medical | Patient, Doctor, Appointment, Prescription |
| `university` | student, course, enrollment | Student, Course, Professor, Enrollment |
| `banking` | account, bank, transaction | BankAccount, Customer, Transaction |
| `social_media` | post, user, follow, like | User, Post, Comment, Like, Follow |
| `hotel` | hotel, room, booking, guest | Hotel, Room, Guest, Booking |
| `restaurant` | restaurant, menu, order, table | Restaurant, MenuItem, Table, Order |
| `inventory` | inventory, warehouse, stock | InventoryItem, Warehouse, StockMovement |
| `project_management` | project, task, sprint, team | Project, Task, Team, Sprint, User |

Each pattern includes: `keywords`, `core_classes` (with attributes), `key_relationships` (with multiplicities), `notes`.

### State Patterns (`src/state_patterns.py`)

8 behavioral lifecycle patterns injected into StateMachine generation:

| Pattern | States |
|---------|--------|
| `order_processing` | new → payment_pending → confirmed → shipped → delivered |
| `authentication` | idle → authenticating → authenticated → expired |
| `document_workflow` | draft → review → approved / rejected |
| `task_management` | todo → in_progress → review → done |
| `booking_reservation` | requested → confirmed → checked_in → completed |
| `user_registration` | initiated → email_verification → profile_setup → active |
| `payment_processing` | initiated → processing → authorized → captured |
| `support_ticket` | open → assigned → in_progress → resolved → closed |

---

## 14. File Conversion Pipeline

### `src/handlers/file_conversion_handler.py`

Converts uploaded files into diagram spec JSON:

| File Type | Extension | Output |
|-----------|-----------|--------|
| PlantUML | `.puml`, `.plantuml`, `.wsd` | ClassDiagram or StateMachine |
| Knowledge Graph | `.ttl`, `.rdf`, `.owl`, `.n3`, `.jsonld` | ClassDiagram |
| Images | `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp` | ClassDiagram (via OpenAI Vision) |
| Generic text | any other | ClassDiagram (LLM interpretation) |

**Flow:**
```
File upload → detect_file_type() → per-type converter → validate → inject_complete_system payload
```

---

## 15. Generation Handler

### `src/handlers/generation_handler.py`

Routes to BESSER code generators or deployment services.

**Supported generators:**

| Generator | Keywords | Config Options |
|-----------|----------|----------------|
| `django` | django, django web app | `project_name`, `app_name`, `containerization` |
| `backend` | full backend, backend | — |
| `web_app` | web app, full stack | — |
| `sql` | sql ddl, sql schema | `dialect` (sqlite/postgresql/mysql/mssql) |
| `sqlalchemy` | sqlalchemy, sql alchemy | — |
| `python` | python classes, generate python | — |
| `java` | java classes, generate java | — |
| `pydantic` | pydantic, pydantic model | — |
| `jsonschema` | json schema, jsonschema | `mode` (regular/smart_data) |
| `smartdata` | smart data, smartdata | — |
| `agent` | besser agent, agent generator | — |
| `qiskit` | qiskit, quantum code | `backend`, `shots` |
| `export` | export project, export to json | `format` (json/buml) |
| `deploy` | deploy to render, deploy app | — |

**Action payloads returned:**
- `trigger_generator` — frontend triggers code generation
- `trigger_export` — frontend triggers JSON/BUML export
- `trigger_deploy` — frontend triggers deployment
- `assistant_message` — help text or error

---

## 16. Quality Review

### `src/quality_review.py`

Post-generation analysis producing suggestions (max 5):

**ClassDiagram checks:**
- Missing expected attributes (e.g., User without email/password)
- Isolated classes (no relationships)
- Missing ID attributes
- Missing common methods

**Cross-diagram suggestions:**
- ClassDiagram exists, no ObjectDiagram → suggest creating instances
- ClassDiagram exists, no StateMachine → suggest behavioral modeling
- ClassDiagram exists, no GUINoCodeDiagram → suggest building a GUI

Suggestions stored in session and consolidated with "What's next?" hints.

---

## 17. Data Flow Diagrams

### Incoming Request Flow

```
Frontend WebSocket Message
    │
    ▼
BESSER Agent Event (outer frame)
    │
    ▼
adapters.parse_assistant_request()
    ├── extract_event_payload()
    ├── _unwrap_v2_envelope()
    ├── parse_v2_payload()
    └── strip_diagram_prefix()
    │
    ▼
AssistantRequest (canonical)
    │
    ▼
state_bodies._common_preamble()
    ├── handle_pending_gui_choice()
    ├── handle_pending_system_confirmation()
    └── handle_file_attachments()
    │
    ▼
State Body Function (from intent classifier)
    │
    ▼
execute_planned_operations()
    │
    ▼
plan_assistant_operations() → [operation, operation, ...]
    │
    ▼
execute_model_operation() (per operation)
    ├── resolve_target_model()
    ├── build_workspace_context_block()
    ├── handler.generate_*()
    ├── apply_layout()
    └── review_generated_model()
    │
    ▼
reply_payload(session, result) → Frontend WebSocket
```

### ClassDiagram Complete System Generation Flow

```
user_request + workspace_context
    │
    ▼
detect_domain_pattern() → format_pattern_for_prompt()
    │                      "DOMAIN REFERENCE: ..."
    ▼
predict_two_pass()
    ├── Pass 1: gpt_text.predict() → free-text reasoning
    └── Pass 2: gpt.predict()      → structured JSON spec
    │
    ▼
validate_and_refine()
    └── gpt.predict(spec + "Find issues, return corrected JSON")
    │
    ▼
apply_system_layout()
    └── layout_engine.apply_layout()
    │
    ▼
{"action": "inject_complete_system", "systemSpec": {...}}
```

### Pending Confirmation Flow

```
execute_model_operation()
    │ complete_system + model_has_elements()
    ▼
Store pending_complete_system in session
Reply: "Replace or keep existing model?"
Return None (halt remaining ops)

    ... user replies "replace" or "keep" ...

_common_preamble()
    └── handle_pending_system_confirmation()
        ├── "replace" → _replace_existing = True
        └── "keep"    → _replace_existing = False
        │
        ▼
    execute_model_operation(_skip_existing_check=True)
        │
        ▼
    Resume remaining_operations (if any)
```

---

## 18. Key Design Patterns

### Module-Level Globals
All LLM handles and the diagram factory are stored in `agent_context.py` rather than passed as arguments. This avoids circular imports and keeps function signatures simple.

### Protocol Decoupling
`AssistantRequest` cleanly separates protocol parsing from execution. After `parse_assistant_request()`, all downstream code works with typed Python objects.

### Handler Extensibility
Adding a new diagram type requires:
1. Create handler extending `BaseDiagramHandler` (4 abstract methods)
2. Add to `DiagramHandlerFactory.__init__`
3. Add to `SUPPORTED_DIAGRAM_TYPES` in `protocol/types.py`
4. Add metadata to `registry/metadata.py`

Layout, retry, caching, quality review, and concurrency are inherited automatically.

### Deterministic Layout Post-Processing
LLMs never emit positions. The layout engine runs after every LLM call, ensuring consistent visual presentation and collision avoidance.

### Graceful Degradation
For every operation:
1. Primary LLM call (with retry)
2. JSON repair via LLM
3. Type-specific fallback generator
4. Error response with `retryable: true`

No exceptions propagate to the caller.

---

## 19. Concurrency and Caching

### LLM Concurrency Control
Rate limiting is handled by OpenAI's API (429 responses). The retry loop in `predict_with_retry()` catches rate-limit errors and surfaces them to the user. No local semaphore is used — concurrent sessions call the API directly.

### Prompt Caching
Prompt caching was removed because every prompt includes workspace context (model data) that changes on each interaction, yielding ~0% cache hit rate. The API-level caching provided by OpenAI (prompt caching for repeated prefixes) provides the actual speedup.

### Retry Strategy
Exponential backoff with jitter:
1. Attempt 1: immediate
2. Attempt 2: ~2s delay
3. Attempt 3: ~4s delay
4. All attempts exhausted → fallback generator

---

## 20. Testing Architecture

### Test Infrastructure (`tests/conftest.py`)

**FakeSession** — Lightweight stand-in for `besser.agent.core.session.Session`:
- Stores key-value pairs in `_store`
- Provides `get/set/delete/get_dictionary` interface
- Captures replies in `_replies` list
- `last_reply_json()` parses JSON from last reply

**FakeLLM** — Stub LLM:
- Round-robin returns from `responses` list
- Logs all prompts in `call_log`

**Helper Functions:**
- `make_v2_payload()` — Build wrapped v2 protocol payload
- `make_session()` — Pre-load session with v2 payload
- `MINIMAL_CLASS_MODEL`, `EMPTY_CLASS_MODEL` — Fixture models

### Test Suites

| Suite | Tests |
|-------|-------|
| `test_protocol.py` | Request parsing, v2 envelope unwrapping |
| `test_diagram_handlers.py` | Handler generation for all 6 diagram types |
| `test_file_conversion.py` | PlantUML, KG, image, text file conversions |
| `test_generation_handler.py` | Code generator routing and config parsing |
| `test_gui_chart_generation.py` | GUI chart binding and color palettes |
| `test_model_helpers.py` | Utility function correctness |
| `test_orchestrator.py` | Diagram type resolution and scoring |
| `test_request_planner.py` | Multi-step operation planning |
| `test_suggestions.py` | Contextual suggestion engine |
| `test_schemas.py` | Pydantic schema validation |
| `test_conversation_memory.py` | Sliding window memory |
| `test_token_tracker.py` | Token usage tracking |
| `test_llm_provider.py` | LLM provider abstraction |
| `test_base_handler.py` | Base handler utilities |

---

## 21. Deployment

### Requirements
- Python 3.11+
- OpenAI API key (GPT-4.1-mini access)
- BESSER Agentic Framework (`besser-agentic-framework[llms]`)

### Local Development
```bash
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
cp config.ini.example config.ini
# Edit config.ini with your OpenAI API key
python modeling_agent.py
```

### Production Architecture
```
nginx (:8080)
  ├── /           → static frontend files
  ├── /besser_api → Python backend (:9000)
  └── /agent      → WebSocket proxy → modeling agent (:8765)
```

### Docker
```bash
docker build -t modeling-agent .
docker run -p 8765:8765 --env-file .env modeling-agent
```
