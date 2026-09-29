# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Overview

`modeling-agent` is the conversational AI backend for the **BESSER Web Modeling Editor**
(`editor.besser-pearl.org`): a standalone Python service built on the **BESSER Agentic Framework (BAF)** that
talks to the frontend over a WebSocket, interprets natural-language modeling requests and returns structured
actions (`inject_complete_system`, `inject_element`, `modify_model`, `trigger_generator`,
`trigger_smart_generator`, …) that the frontend applies to the canvas or hands to a generator.

- **Caller**: the WME frontend's `packages/webapp/src/main/features/assistant/services/AssistantClient.ts`
  (see **Wire Protocol**).
- **Own repo, own release cadence**: not part of BESSER's version number and never included in a BESSER
  release PR. Ships as its own Docker image built from this repo's `Dockerfile`.
- **Branches**: `develop` is the integration branch, `main` is production (PR `develop` → `main`). Not the same
  names as BESSER's `development` / `master`.
- One long-lived process (`modeling_agent.py`) hosting BAF's `websocket_platform` on
  `config.yaml` → `platforms.websocket.port` (default 8765); reverse-proxied at `wss://<host>/agent`.
- **Boot is slow by design**: BAF trains a NER model plus one local intent classifier per state *before* the
  socket opens (several minutes). The Docker `HEALTHCHECK` uses `--start-period=300s`. A "hang" on first run is
  usually this.

Long-form docs: `docs/source/` — `intent_recognition.rst` (routing), `orchestration.rst` (planner and diagram-type
resolution), `diagram_handlers.rst`, `websocket_protocol.rst`, `configuration.rst`, `deployment.rst`,
`contributing/` (setup, testing, how-to guides).

## Setup and Testing

Python 3.11+.
```bash
python -m venv venv
source venv/bin/activate                 # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt          # pins besser-agentic-framework == 4.3.2
cp config_example.yaml config.yaml       # config.yaml is gitignored; never commit real keys (same for .env)
python modeling_agent.py                 # WebSocket on :8765

python -m pytest                                    # full suite
python -m pytest tests/test_unified_classifier.py   # routing verdicts
python -m pytest tests/test_diagram_handlers.py     # cross-handler contract tests
python -m pytest tests/test_bpmn.py                 # one diagram type
```
There is no CI workflow in this repo; run the suite locally.

- Handler tests instantiate handlers directly, e.g. `BPMNDiagramHandler(None)` (the LLM arg is only needed by
  methods that call the model), and test deterministic repair such as `_validate_and_refine` as a pure function
  on hand-built dicts. See `tests/test_bpmn.py`.
- `tests/conftest.py` provides `FakeSession`, `FakeLLM`, `make_v2_payload()`, `make_session()` and the
  `MINIMAL_CLASS_MODEL` / `EMPTY_CLASS_MODEL` fixtures.
- **Live probing**: unit tests catch structural regressions, not LLM generation-quality drift. The scripts in
  `tests/live/` (`probe_smoke.py`, `probe_full_agentic.py`, `wme_release_sweep.py`, …) run against a running
  agent (`AGENT_WS_URL` required) and speak the real double-encoded envelope. Run the same prompt N times and diff
  the specs — a single manual browser test often lands in the "looks fine" bucket even when the failure rate is
  high. `probe_smoke.py` is fast enough for a post-deploy gate. `tests/live/test_nl_generation_scenarios.py` is
  skipped unless `RUN_LIVE_AGENT_TESTS=1`.

## Request Flow

Routing is **one LLM call**, then a keyword/regex layer picks the diagram type. Confusing which layer owns a
decision is the most common source of bugs here. Full detail: `docs/source/intent_recognition.rst`.

```
WebSocket message
  → protocol/adapters.py + protocol/types.py        parse wire payload → AssistantRequest
  → src/unified_classifier.py                        LAYER 1: THE ROUTER
      classify_message() → UnifiedClassification, cached per BAF event by get_or_classify().
      Returns the intent (which of the 10 states) AND every sub-routing field downstream code
      needs (generation_route, generator_type, target_diagram_type, model_disposition,
      pending_flow_action, …) — no second prompt. Rules live in _SYSTEM_PROMPT.
  → src/state_bodies.py                              state body; modeling bodies funnel through
                                                     execution/planning.py::execute_planned_operations
  → orchestrator/workspace_orchestrator.py           LAYER 2: determine_target_diagram_type()
      0. classifier's target_diagram_type  1. KEYWORD_TARGETS  2. _IMPLICIT_PATTERNS
      3. active-diagram / project-snapshot fallback (FALLBACK_PRIORITY)
  → diagram_handlers/registry/factory.py             DiagramHandlerFactory.get_handler(type)
  → handler.generate_complete_system / generate_modification / …
  → response envelope sent back over the WebSocket
```

- **`_SYSTEM_PROMPT` in `unified_classifier.py` is the single authoritative rulebook.** Routing changes go there,
  or — for decisions the LLM is measurably unreliable at — in an explicit deterministic guard
  (`_names_unsupported_stack()` in `unified_classifier.py`; `_GITHUB_URL_RE` / `_GITHUB_CONTINUE_VERB_RE` in
  `handlers/generation_handler.py`). There is no keyword pre-filter layer to add special cases to.
- **Intent `description=` strings in `modeling_agent.py` stay one-liners.** They only feed BAF's local fallback
  classifier and do not drive routing.
- **BAF's own classifier is a local, non-LLM fallback** (`SimpleIntentClassifierConfiguration(framework='tensorflow')`
  in `agent_setup.py`, trained at startup on each intent's `training_sentences`). It routes only when the unified
  call failed (`fallback_intent`) and for voice / plain-text events. `training_sentences` are therefore
  **mandatory** on every intent — an intent without them breaks that classifier at startup.
- `config.yaml`'s `nlp.intent_threshold` (0.55) gates only that local classifier, not the unified one.
- **Transition priority** (`state_bodies.add_unified_transitions`): 0 `_ensure_unified_classification` (side
  effect, always False) → 1 `json_intent_matches` (cached verdict) → 2 `route_to_generation` (`frontend_event` and
  pending generator flows only; `should_route_to_generation()` makes no LLM call and runs no text heuristics) →
  3 `when_intent_matched` (voice / plain text) → 4 `json_no_intent_matched` (state fallback).
- **A pending flow suppresses intent matching by design**: while the assistant awaits an answer,
  `json_intent_matches()` returns False for every intent unless the classifier set
  `pending_flow_action="new_request"`. A message that looks "stuck" is usually this, not a routing bug.
- **Layer 2 is keyword/regex, not LLM.** Look there only if a message that reached a modeling state resolves to
  the *wrong diagram type*.

## Adding (or Auditing) a Diagram Type

Every type must be registered in **all** of these places — a missed one is easy to introduce and easy to miss
in manual testing. Walkthrough: `docs/source/contributing/howto_guides.rst`.

1. `src/schemas/<type>.py` — Pydantic schemas for structured output (single element, complete system,
   modification actions with `Literal` names); re-export from `src/schemas/__init__.py`
2. `src/diagram_handlers/types/<type>_diagram_handler.py` — the handler (see below)
3. `src/diagram_handlers/registry/factory.py` — add to `HANDLER_CLASSES` (keyed on the handler's own
   `get_diagram_type()`)
4. `src/unified_classifier.py` — token in `_TARGET_DIAGRAM_TYPES` **and** vocabulary in `_SYSTEM_PROMPT`. This is
   what makes the type reachable. Add one representative `training_sentence` in `modeling_agent.py`, but no
   vocabulary in its `description=`
5. `src/orchestrator/workspace_orchestrator.py` — `KEYWORD_TARGETS`, an `_IMPLICIT_PATTERNS` regex, and
   `FALLBACK_PRIORITY`. Patterns are evaluated in order, first match wins: place a new one where a broader pattern
   above it won't steal its vocabulary (BPMN sits above StateMachineDiagram because "process" is in both)
6. `src/state_bodies.py` — `_QUICK_RESPONSES["what_can_you_do"]` and `["help"]` (bump the hardcoded type count),
   the `_fallback_llm_reply` prompt (shared by `global_fallback_body` and `greetings_body`), optionally a
   `modeling_help_body` branch
7. `src/protocol/types.py` — `SUPPORTED_DIAGRAM_TYPES`; an `activeDiagramType` outside it is silently normalized
   to `ClassDiagram`
8. `src/diagram_handlers/registry/metadata.py` — `DIAGRAM_TYPE_METADATA`
9. `src/suggestions.py` — a suggestion list + `_DIAGRAM_SUGGESTION_HANDLERS` entry
10. Tests — `tests/test_diagram_handlers.py` plus a dedicated `tests/test_<type>.py`
11. Docs — `README.md` "Supported Diagram Types" table, `docs/source/diagram_handlers.rst`,
    `websocket_protocol.rst`, `getting_started.rst`
12. Frontend (separate repo) — the type must exist in the WME's diagram-type union and be a valid
    `activeDiagramType`

Capability lists are duplicated with no single source of truth (`_SYSTEM_PROMPT`, `_QUICK_RESPONSES`,
`_fallback_llm_reply`, `suggestions.py`, README): grep for an existing type's name (e.g. `"quantum"`) to find
every place. Two tokens break the `<Name>Diagram` convention: BPMN is `"BPMN"` (the editor's converter sets the
Apollon `model.type` to `"BPMNDiagram"` itself) and User Profile is `"UserDiagram"`.

**Test routing from outside the feature.** Opening the new type's tab and asking for changes never exercises
layer 1 or 2 — both already know the answer from `context.activeDiagramType`. Test from a *different* tab, with
phrasing that doesn't literally name the type.

## Diagram Handlers

Handlers extend `BaseDiagramHandler` (`src/diagram_handlers/core/base_handler.py`) and implement five abstract
methods: `get_diagram_type()` (the WME storage-bucket token), `get_system_prompt()` (design rules),
`generate_single_element`, `generate_complete_system` (primary path) and `generate_fallback_element` (error-path
stub). `generate_modification` is concrete on the base class; override only when needed. Layout, retry,
structured output and two-pass generation are inherited. Details: `docs/source/diagram_handlers.rst`.

- **Two-pass generation** (`predict_two_pass_structured`): a free-text reasoning pass, then a structured pass into
  the Pydantic schema. Skipped below `_TWO_PASS_MIN_LENGTH = 250` characters of the **raw** user message (not the
  enriched prompt, or history and workspace context would push trivial requests onto the expensive path). The
  reasoning prompt is the highest-leverage place to fix systematic completeness gaps (see
  `bpmn_diagram_handler.py`'s `reasoning_prompt`, which stops the model merging several decision points into one
  gateway).
- **Deterministic post-generation repair** (`_validate_and_refine` / `_connect_orphaned_nodes` in
  `bpmn_diagram_handler.py`): prefer fixing the root cause in the prompt; add a repair pass only for invariants the
  LLM statistically violates and where a cheap, evidence-grounded heuristic exists — never a blind guess.
- **LLMs never emit positions.** `core/layout_engine.py` runs after every generation (Sugiyama layout for class
  diagrams, per-type grids otherwise).

## Wire Protocol

Full reference: `docs/source/websocket_protocol.rst`. The traps:

- **Inbound is double-JSON-encoded.** BAF's `Payload.decode()` reads only `action`, `message`, `history`, so the
  frontend's v2 payload (`protocolVersion`, `clientMode`, `sessionId`, `context.activeDiagramType`, …) is
  JSON-stringified into the `message` field of a `user_message` envelope. `_unwrap_v2_envelope()` in
  `protocol/adapters.py` recovers it and merges it over the outer one.
- **Outbound streams are wrapped again**: each chunk is `{"action": "agent_reply_str", "message": "<JSON string>"}`
  whose inner JSON is `stream_start` / `stream_chunk` / `stream_done`. When probing the agent directly, replicate
  both layers and unwrap them in the order `AssistantClient.ts` does (`extractActionPayload`) — a single-level
  unwrap treats every response as unknown and hangs.
- **`user_id` is a URL query param** (`wss://host/agent?user_id=…`), read by `_extract_user_id_from_request` in
  `patches/websocket_platform.py`. It keys the BAF session; conversation memory keys on the inner `sessionId`
  (`memory.memory_session_key`).
- Other inbound actions: `user_voice` (base64 audio → whisper-1), `user_set_variable` (arms BYOK via
  `user_api_key` / `user_api_provider` / `user_api_model` / `user_api_base`, passes `_voice_context`, carries the
  keep-alive heartbeat), `frontend_event` (generator result echo, routed deterministically, never classified),
  `replay_last_response` (re-send the buffered terminal reply after a reconnect).
- Emitted actions — terminal: `inject_element`, `inject_complete_system`, `modify_model`, `assistant_message`,
  `agent_error`, `create_diagram_tab`, `trigger_generator`, `trigger_smart_generator`, `trigger_github_import`,
  `trigger_export`, `trigger_deploy`, `auto_generate_gui`; non-terminal: `progress`, `stream_start`,
  `stream_chunk`, `stream_done`. There is **no `switch_diagram` action**.

## Where Things Live

```text
modeling_agent.py                 Entrypoint: BAF agent, 10 states/intents, wiring
config_example.yaml               Template for the gitignored config.yaml
Dockerfile                        Entrypoint writes config.yaml from env
patches/websocket_platform.py     Vendored over pip-installed BAF 4.3.2 (see below)
src/
  agent_setup.py / agent_context.py   LLM/RAG/STT/factory bootstrapping; shared module-level context
  agent_config.py                 Tunable constants (MAX_TABS, temperatures, token budgets, …)
  model_config.py                 Per-call-site model tiers, all BESSER_AGENT_MODEL_* overridable
  unified_classifier.py           Layer 1 router
  state_bodies.py                 State bodies, transitions, global fallback, quick responses
  session_helpers.py / session_keys.py   Reply/stream helpers; every session-state key (import, don't use literals)
  confirmation.py                 Pending replace/keep and GUI-mode choice flows
  byok.py                         Per-request bring-your-own-key routing (contextvar)
  orchestrator/                   workspace_orchestrator.py (layer 2), request_planner.py (multi-op planning)
  diagram_handlers/               core/ (base_handler, layout_engine, prompt_fragments), types/ (one handler per
                                  type + gui_design_system.py, gui_html_converter.py), registry/ (factory, metadata)
  schemas/<type>.py               Structured-output schemas
  execution/                      planning.py (multi-op dispatch), model_operations.py, file_handling.py, progress.py
  handlers/                       generation_handler.py, smart_generation_handler.py, file_conversion_handler.py,
                                  validation_handler.py (bridge to the BESSER backend validator)
  protocol/                       types.py, adapters.py
  memory/                         Verbatim window + rolling LLM summary per session
  llm/provider.py                 Provider abstraction (structured outputs, streaming)
  tracking/token_tracker.py       Token/cost tracking (_COST_PER_1K)
  suggestions.py                  "What's next?" suggestions
  domain_patterns.py, state_patterns.py   Defined but NOT injected into prompts
```

**`patches/websocket_platform.py`** is copied over BAF's file at image build. Stock BAF evicts the `_connections`
slot unconditionally on close; with the stable `?user_id=`, a tab's two sockets (widget + drawer) share one
session key, so a closing socket would drop the live one's replies. The patch adds an ownership-guarded delete and
reply-route-to-sender, and sets/resets the per-request BYOK contextvar. **Re-vendor it if the BAF pin changes.**

## LLM Pitfalls

- **Two shared LLM instances**: `gpt` is the structured/JSON path, `gpt_text` is free text (help, greetings, RAG,
  streaming). Both default to the `MODEL_CLASSIFIER` tier; generation call sites pass `model=` from
  `model_config.py`. Passing the wrong one breaks silently or raises a JSON-parse error deep in `predict_structured`.
- **Reasoning models reject `temperature`** (gpt-5+/gpt-6, o-series, and recent Claude families). Use
  `model_config.supports_custom_temperature(model)` and `reasoning_effort_for(model)` at every call site; never
  hardcode either parameter.
- **When a `MODEL_*` default changes, add it to `_COST_PER_1K`** in `tracking/token_tracker.py` — an unknown model
  falls back to placeholder pricing and every reported cost silently becomes an estimate.
- The Docker entrypoint generates `config.yaml` from the environment, redacts the `api_key` line from its own log
  output, and emits `platforms.websocket.origins` from `BESSER_AGENT_WS_ORIGIN` / `BESSER_AGENT_WS_ORIGIN_ALT`
  plus localhost dev origins.
