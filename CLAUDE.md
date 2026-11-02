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
pip install -r requirements-dev.txt      # runtime pins (besser-agentic-framework == 4.5.3) + test/lint tools
pip install -e .                         # the agent itself; src/ is the import root
cp config_example.yaml config.yaml       # config.yaml is gitignored; never commit real keys (same for .env)
python modeling_agent.py                 # WebSocket on :8765

python -m pytest                                    # full suite
python -m pytest tests/test_unified_classifier.py   # routing verdicts
python -m pytest tests/test_diagram_handlers.py     # cross-handler contract tests
python -m pytest tests/test_bpmn.py                 # one diagram type
lint-imports                                        # import layers (contracts in pyproject.toml)
```
CI (`.github/workflows/tests.yml`) runs this suite, ruff and `lint-imports` on every PR and push.

**Packaging**: `pyproject.toml` installs `src/` as the import root, so the agent's packages are top-level
names (`from protocol.types import ...`, `from conversation import session_helpers`) in the app, the tests and
the image alike. Never import through `src.` and never add `sys.path` inserts. Every module lives in a package
(only the entrypoint `modeling_agent.py` sits at the repo root); a new package goes into the import-linter
layers (`tests/test_packaging.py` checks). Its `version` is the agent's only version.

**Import layering** (each top-level package is one layer and imports only those below it): `app` → `ingestion` → `execution` → `codegen` / `orchestrator` → `conversation` → `routing` → `memory` / `diagram_handlers` → `prompts` → `llm` / `schemas` → `utilities` → `protocol` → `common`. Packages on one level (`/`) do not import each other. Inside a package: `app` is `state_bodies` / `agent_setup` → `confirmation`; `conversation` is `session_helpers` → `reply_language` → the rest; `routing` is `unified_classifier` → `guards` → `generation_signals` / `intents`; `llm` is `provider` → `byok` → `tracking` → `byok_config` / `model_catalog` / `model_config` / `retry`, and only `llm` imports `openai` / `anthropic`. `lint-imports` enforces all of it (`[tool.importlinter]` in `pyproject.toml`), with the one known cycle (`codegen.generation_handler` → `execution`: a confirmed generation re-runs the planned operations) ignored.

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
  → routing/unified_classifier.py                    LAYER 1: THE ROUTER
      classify_message() → UnifiedClassification, cached per BAF event by get_or_classify().
      Returns the intent (which of the 10 states) AND every sub-routing field downstream code
      needs (generation_route, generator_type, target_diagram_type, model_disposition,
      pending_flow_action, …) — no second prompt. Rules live in prompts/classifier.py.
  → app/state_bodies.py                              state body; modeling bodies funnel through
                                                     execution/planning.py::execute_planned_operations
  → orchestrator/workspace_orchestrator.py           LAYER 2: determine_target_diagram_type()
      0. classifier's target_diagram_type  1. KEYWORD_TARGETS  2. _IMPLICIT_PATTERNS
      3. active-diagram / project-snapshot fallback (FALLBACK_PRIORITY)
  → diagram_handlers/registry/factory.py             DiagramHandlerFactory.get_handler(type)
  → handler.generate_complete_system / generate_modification / …
  → response envelope sent back over the WebSocket
```

- **`SYSTEM_PROMPT` in `prompts/classifier.py` is the single authoritative rulebook.** Routing changes go there,
  or — for decisions the LLM is measurably unreliable at — in an explicit deterministic guard
  (`_names_unsupported_stack()` in `unified_classifier.py`; `_GITHUB_URL_RE` / `_GITHUB_CONTINUE_VERB_RE` in
  `routing/generation_signals.py`). There is no keyword pre-filter layer to add special cases to.
- **Every other deterministic routing rule lives in `routing/guards.py`**, as a pure function over the request or
  message (and the verdict where it depends on it): before the classifier, `skips_classification`,
  `frontend_event_verdict` and `auto_fix_verdict`; after routing, in the receiving state, `reconcile_intent`,
  `modeling_guard` (injection input, the classifier-outage net, self-contradictory and contentless creates, first
  match wins), `quick_response_key`, `is_mixed_modeling_and_generation`, `is_past_generation_question` and
  `generation_route`. State bodies and the generation handler only map a guard's decision to a reply. A new
  text rule goes there, with a row in `tests/test_routing_guards.py`, never inline in a state body. Answers to a
  pending question (replace/keep, GUI choice, generator confirmations) are parsed by the pending-flow handlers
  (`app/confirmation.py`, `codegen/generation_handler.py`), not by these guards.
- **Intent `description=` strings in `modeling_agent.py` stay one-liners.** They only feed BAF's local fallback
  classifier and do not drive routing.
- **BAF's own classifier is a local, non-LLM fallback** (`SimpleIntentClassifierConfiguration(framework='tensorflow')`
  in `agent_setup.py`, trained at startup on each intent's `training_sentences`). It routes only when the unified
  call failed (`fallback_intent`) and for voice / plain-text events (a voice transcript is a text event, which the
  unified hook never sees: `when_intent_matched` is its only route). `training_sentences` are therefore
  **mandatory** on every intent — an intent without them breaks that classifier at startup.
- `config.yaml`'s `nlp.intent_threshold` (0.55) gates only that local classifier, not the unified one.
- **Transition priority** (`state_bodies.add_unified_transitions`): 0 `_ensure_unified_classification` (side
  effect, always False) → 1 `json_intent_matches` (cached verdict) → 2 `route_to_generation` (`frontend_event` and
  pending generator flows only; `should_route_to_generation()` makes no LLM call and runs no text heuristics) →
  3 `when_intent_matched` (voice / plain text) → 4 `json_no_intent_matched` (state fallback).
- **A pending flow suppresses intent matching by design**: while the assistant awaits an answer,
  `json_intent_matches()` returns False for every intent unless the classifier set
  `pending_flow_action="new_request"`. A message that looks "stuck" is usually this, not a routing bug.
- **Pending dialogs are one session variable** (`PENDING_DIALOGS`), read and written only through
  `common/pending_dialogs.py` (`get` / `put` / `clear`; a slot name is not a session key). Several can be
  open at once (the mismatch "Update model + generate" chain, a paused "design X and generate Y" plan), so
  `PRECEDENCE` fixes the order: `resume_pending()` in `_common_preamble` tries the open dialogs' handlers
  (`app/confirmation.RESUME_HANDLERS`) in it, and the classifier is shown the first open one. Generator
  menu and settings answers reach `handle_generation_request` through routing. A new dialog adds a slot, a
  kind in `PRECEDENCE`, a resume handler and a question in `_pending_flow_context`
  (`docs/source/orchestration.rst`, "Pending Dialogs").
- **Layer 2 is keyword/regex, not LLM.** Look there only if a message that reached a modeling state resolves to
  the *wrong diagram type*.

## Adding (or Auditing) a Diagram Type

Every type must be registered in **all** of these places — a missed one is easy to introduce and easy to miss
in manual testing. Walkthrough: `docs/source/contributing/howto_guides.rst`.

1. `src/schemas/<type>.py` — Pydantic schemas for structured output (single element, complete system,
   modification actions with `Literal` names); re-export from `src/schemas/__init__.py`
2. `src/diagram_handlers/types/<type>_diagram_handler.py` — the handler (see below), with its prompt text in
   `src/prompts/<type>.py`; replies built with `protocol.actions`. Add its cases to
   `tests/test_handler_envelopes.py` and regenerate the fixture
3. `src/diagram_handlers/registry/factory.py` — add to `HANDLER_CLASSES` (keyed on the handler's own
   `get_diagram_type()`)
4. `src/routing/unified_classifier.py` — token in `_TARGET_DIAGRAM_TYPES` **and** vocabulary in `SYSTEM_PROMPT` (`src/prompts/classifier.py`). This is
   what makes the type reachable. Add one representative `training_sentence` in `modeling_agent.py`, but no
   vocabulary in its `description=`
5. `src/orchestrator/workspace_orchestrator.py` — `KEYWORD_TARGETS`, an `_IMPLICIT_PATTERNS` regex, and
   `FALLBACK_PRIORITY`. Patterns are evaluated in order, first match wins: place a new one where a broader pattern
   above it won't steal its vocabulary (BPMN sits above StateMachineDiagram because "process" is in both)
6. `src/app/state_bodies.py` — `_QUICK_RESPONSES["what_can_you_do"]` and `["help"]` (bump the hardcoded type count),
   the fallback prompt in `src/prompts/fallback.py` (shared by `global_fallback_body` and `greetings_body`), optionally a
   `modeling_help_body` branch
7. `src/protocol/types.py` — `SUPPORTED_DIAGRAM_TYPES`; an `activeDiagramType` outside it is silently normalized
   to `ClassDiagram`
8. `src/diagram_handlers/registry/metadata.py` — `DIAGRAM_TYPE_METADATA`
9. `src/conversation/suggestions.py` — a suggestion list + `_DIAGRAM_SUGGESTION_HANDLERS` entry
10. Tests — `tests/test_diagram_handlers.py` plus a dedicated `tests/test_<type>.py`
11. Docs — `README.md` "Supported Diagram Types" table, `docs/source/diagram_handlers.rst`,
    `websocket_protocol.rst`, `getting_started.rst`
12. Frontend (separate repo) — the type must exist in the WME's diagram-type union and be a valid
    `activeDiagramType`

Capability lists are duplicated with no single source of truth (`prompts/classifier.py`, `_QUICK_RESPONSES`,
`prompts/fallback.py`, `suggestions.py`, README): grep for an existing type's name (e.g. `"quantum"`) to find
every place. Two tokens break the `<Name>Diagram` convention: BPMN is `"BPMN"` (the editor's converter sets the
Apollon `model.type` to `"BPMNDiagram"` itself) and User Profile is `"UserDiagram"`.

**Test routing from outside the feature.** Opening the new type's tab and asking for changes never exercises
layer 1 or 2 — both already know the answer from `context.activeDiagramType`. Test from a *different* tab, with
phrasing that doesn't literally name the type.

## Diagram Handlers

Handlers extend `BaseDiagramHandler` (`src/diagram_handlers/core/base_handler.py`) and implement five abstract
methods: `get_diagram_type()` (the WME storage-bucket token), `get_system_prompt()` (design rules),
`_build_single_element`, `_build_system` (primary path) and `generate_fallback_element` (error-path stub), plus
`generate_fallback_system` and, for edits, `_build_modification` / `generate_fallback_modification`. The public
`generate_complete_system` / `generate_single_element` / `generate_modification` are **template methods** on the
base class: they call the `_build_*` hook and turn an `LLMPredictionError` into the handler's
`_SYSTEM_FAILURE_MESSAGE` / `_SINGLE_FAILURE_MESSAGE` / `_MODIFY_FAILURE_MESSAGE` reply (with `_LLM_FAILURE_CODE`
/ `_MODIFY_FAILURE_CODE`) and any other exception into the fallback; override the `_*_failure_reply` /
`_*_crash_reply` hooks only when a handler answers differently. Layout, retry, structured output and two-pass
generation are inherited. Details: `docs/source/diagram_handlers.rst`.

- **Two-pass generation** (`predict_two_pass_structured`): a free-text reasoning pass, then a structured pass into
  the Pydantic schema. Skipped below `_TWO_PASS_MIN_LENGTH = 250` characters of the **raw** user message (not the
  enriched prompt, or history and workspace context would push trivial requests onto the expensive path). The
  reasoning prompt is the highest-leverage place to fix systematic completeness gaps (see
  `prompts/bpmn.py`'s `reasoning_prompt`, which stops the model merging several decision points into one
  gateway).
- **Deterministic post-generation repair** (`_validate_and_refine` / `_connect_orphaned_nodes` in
  `bpmn_diagram_handler.py`): prefer fixing the root cause in the prompt; add a repair pass only for invariants the
  LLM statistically violates and where a cheap, evidence-grounded heuristic exists — never a blind guess.
- **LLMs never emit positions.** `core/layout_engine.py` runs after every generation (Sugiyama layout for class
  diagrams, per-type grids otherwise).

## Prompts

All prompt text lives in `src/prompts/`, one module per diagram type or state (`classifier`, `class_diagram`,
`bpmn`, `gui`, …, `fragments` for rules several handlers share, `planner`, `modeling_help`, `describe_model`,
`fallback`, `uml_rag`, `file_conversion`, `reply_language`, `spec_driven`, `model_operations`,
`conversation_memory`). A static prompt is a constant (`MODIFY_SYSTEM_PROMPT_CLASS`); a prompt built from the
request is a function of exactly those values (`reasoning_prompt(user_request)`). Call sites import the module
(`from prompts import bpmn as bpmn_prompts`) and look the prompt up by name.

- Edit wording in the prompt module, never at a call site.
- System prompts are composed at import time and must stay byte-identical across calls, or the prompt cache
  misses (`docs/source/configuration.rst`, "Static prefix first"): nothing per-request in a constant.
- `tests/test_prompt_snapshots.py` pins a hash of every prompt; a deliberate change updates its hash (the failure
  prints the new one) in the same commit.

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
  `replay_last_response` (after a reconnect: with `turnId` + `appliedSeqs`, re-send only that turn's unacknowledged terminal replies; without, the session's last terminal reply).
- Emitted actions — terminal: `inject_element`, `inject_complete_system`, `modify_model`, `assistant_message`,
  `agent_error`, `create_diagram_tab`, `trigger_generator`, `trigger_smart_generator`, `trigger_github_import`,
  `trigger_export`, `trigger_deploy`, `auto_generate_gui`; non-terminal: `progress`, `stream_start`,
  `stream_chunk`, `stream_done`. There is **no `switch_diagram` action**.
- **Every action name is `protocol.actions.Action`** (a `StrEnum`: equal to its string, serialized as it): never
  write an action literal (`tests/test_protocol_actions.py` fails on one outside `protocol/`). Build the repeated
  envelopes with its builders (`assistant_message`, `inject_element`, `inject_complete_system`, `model_reply`);
  they keep the key order, and `tests/test_handler_envelopes.py` pins every handler's JSON byte for byte.

## Where Things Live

```text
modeling_agent.py                 Entrypoint: BAF agent, 10 states/intents, wiring
pyproject.toml                    Package (src/ as import root, editable install), version, import-linter contracts
config_example.yaml               Template for the gitignored config.yaml
Dockerfile                        Image; its entrypoint is app.container_config (config.yaml from env)
patches/websocket_platform.py     Vendored over pip-installed BAF 4.5.3 (see below)
src/
  app/                            The top layer: agent_setup.py (LLM/RAG/STT/factory bootstrapping), state_bodies.py
                                  (state bodies, transitions, global fallback, quick responses), confirmation.py
                                  (resume_pending, replace/keep and GUI-mode choice flows), container_config.py (Docker
                                  entrypoint: config.yaml from AgentSettings)
  ingestion/                      file_handling.py (the attachment step of a turn), file_conversion_handler.py
                                  (file → diagram spec)
  execution/                      planning.py (multi-op dispatch), model_operations.py, progress.py
  codegen/                        generation_handler.py (generator requests and pending generator flows),
                                  smart_generation_handler.py (Spec-Driven Agent trigger)
  orchestrator/                   workspace_orchestrator.py (layer 2), request_planner.py (multi-op planning),
                                  request_builders.py (re-targeted requests)
  conversation/                   session_helpers.py (reply/stream helpers), reply_language.py, reply_copy.py,
                                  suggestions.py ("What's next?"), telemetry.py, agent_context.py (runtime handles
                                  the entrypoint fills in)
  routing/                        unified_classifier.py (layer 1 router), guards.py (deterministic routing
                                  rules), generation_signals.py (generator vocabulary + signals), intents.py
  memory/                         Verbatim window + rolling LLM summary per session
  diagram_handlers/               core/ (base_handler, layout_engine), types/ (one handler per
                                  type + gui_design_system.py, gui_html_converter.py), registry/ (factory, metadata)
  prompts/                        Prompt text, one module per diagram type or state (see Prompts)
  schemas/<type>.py               Structured-output schemas
  llm/                            provider.py (the only SDK caller: text, JSON, structured, stream, vision), retry.py
                                  (shared-client retry), byok.py (per-request bring-your-own key), byok_config.py
                                  (its contextvar), model_config.py (per-call-site model tiers, all
                                  BESSER_AGENT_MODEL_* overridable), model_catalog.py,
                                  tracking/ (token/cost tracking; prices: resources/model_prices.json)
  utilities/                      Model summaries/resolution, JSON repair, model_utils.py
  protocol/                       actions.py (wire action names + envelope builders), types.py, adapters.py,
                                  message_limits.py
  common/                         agent_config.py (tunable constants), settings.py (AgentSettings: every env var,
                                  read nowhere else), session_store.py (registry of per-session stores, one
                                  close(key)), session_keys.py (every session-state key: import, don't use
                                  literals), pending_dialogs.py (pending dialogs: slots, precedence), errors.py
  resources/                      Price file and the bundled User Profile metamodel
```

**`patches/websocket_platform.py`** is copied over BAF's file at image build. Stock BAF evicts the `_connections`
slot unconditionally on close; with the stable `?user_id=`, a tab's two sockets (widget + drawer) share one
session key, so a closing socket would drop the live one's replies. The patch adds an ownership-guarded delete and
reply-route-to-sender, and sets/resets the per-request BYOK contextvar. **Re-vendor it if the BAF pin changes.**

## LLM Pitfalls

- **One door to the SDKs**: every LLM call goes through `llm.provider.LLMProvider` (`complete`, `parse`, `stream`,
  `vision`, `predict`), which routes the request's BYOK key, records usage with the model actually called, applies
  the sampling rules and raises `ModelRefusal`. Only `llm/` may import `openai` / `anthropic` (import-linter);
  never post to a model API with `requests`/`httpx`.
- **Two shared LLM instances**: `gpt` is the structured/JSON path, `gpt_text` is free text (help, greetings, RAG,
  streaming); code gets one injected (handlers, `stream_llm_response`) and calls it through `LLMProvider(llm)`.
  Both default to the `MODEL_CLASSIFIER` tier; generation call sites pass `model=` from `model_config.py`. Passing
  the wrong one breaks silently or raises a JSON-parse error deep in `predict_structured`.
- **Reasoning models reject `temperature`** (gpt-5+/gpt-6, o-series, and recent Claude families).
  `model_config.sampling_params(model, temperature)` picks `temperature` or `reasoning_effort`; `LLMProvider`
  applies it, so call sites never pass either parameter themselves.
- **A new `MODEL_*` default needs a price**: the model catalog or `resources/model_prices.json` (a copy of BESSER's
  vendored price file; refresh both together) — an unknown model
  falls back to placeholder pricing and every reported cost silently becomes an estimate.
- The Docker entrypoint (`python -m app.container_config`) generates `config.yaml` from `AgentSettings`, redacts
  the `api_key` line from its own log output, and emits `platforms.websocket.origins` from `BESSER_AGENT_WS_ORIGIN` / `BESSER_AGENT_WS_ORIGIN_ALT`
  plus localhost dev origins.
