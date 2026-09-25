# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Overview

`modeling-agent` is the conversational AI backend for the **BESSER Web Modeling Editor**
(`editor.besser-pearl.org`). It's a standalone Python service built on the
**BESSER Agentic Framework (BAF)** that talks to the frontend over a WebSocket, interprets
natural-language modeling requests, and returns structured actions (`inject_complete_system`,
`inject_element`, `modify_model`, `trigger_generator`, `trigger_smart_generator`, …) that the
frontend applies to the diagram canvas or hands to a generator.

- **Frontend caller**: `besser/utilities/web_modeling_editor/frontend`'s
  `packages/webapp/src/main/features/assistant/services/AssistantClient.ts` — see
  **Wire Protocol** below for the exact message shape it sends/expects.
- **Deployed alongside** BESSER's main releases but has its **own repo, its own
  `develop`→`main` branch convention, and its own release cadence** — it is
  *not* part of BESSER's version number and is never included in a BESSER release PR.
  It ships as its own Docker image, built from this repo's `Dockerfile`.
- Runs as a single long-lived process (`modeling_agent.py`) hosting BAF's
  `websocket_platform` (`config.yaml` → `platforms.websocket.port`, default 8765).
  Reverse-proxied at `wss://<host>/agent` (`editor.besser-pearl.org` in production).
- **Boot is slow by design.** BAF trains a NER model plus one local intent classifier per
  state *before* the socket opens — measured 3m38s container-start to listening. That's why
  the Docker `HEALTHCHECK` uses `--start-period=300s`. A "hang" on first run is usually this.

## Request Flow (read this before touching routing or adding a diagram type)

Routing happens in **one LLM call**, then a keyword/regex layer picks the diagram type.
Confusing which layer owns a decision is the single most common source of bugs here.

```
WebSocket message
  → protocol/adapters.py + protocol/types.py   (parse wire payload → AssistantRequest)
  → src/unified_classifier.py                  (THE ROUTER — one structured-output call)
      classify_message() → UnifiedClassification, cached per BAF event id by
      get_or_classify(). Returns the state-level intent AND every sub-routing
      field downstream code needs (generation_route, generator_type,
      target_diagram_type, model_disposition, pending_flow_action, …), so no
      second prompt is ever needed. The rules live in _SYSTEM_PROMPT.
      decides WHICH STATE handles this message (10 states):
      greetings_state / create_complete_system_state / modify_model_state /
      modeling_help_state / describe_model_state / uml_rag_state /
      generation_state / decline_state / out_of_scope_state / meta_question_state
      — or falls through to global_fallback_body (src/state_bodies.py).
  → state_bodies.py's state body for the matched intent
      (create_complete_system_body / modify_modeling_body / … all funnel through
      _modeling_state_body → execution/planning.py's execute_planned_operations)
  → orchestrator/workspace_orchestrator.py's determine_target_diagram_type()
      decides WHICH DIAGRAM TYPE the message targets, using (in priority order):
      0. the classifier's target_diagram_type, when it named one
      1. explicit keyword match (KEYWORD_TARGETS)
      2. discriminating regex patterns (_IMPLICIT_PATTERNS)
      3. active-diagram-type / project-snapshot fallback (FALLBACK_PRIORITY)
  → diagram_handlers/registry/factory.py's DiagramHandlerFactory.get_handler(diagram_type)
  → the concrete handler's generate_complete_system / generate_modification / … method
  → response envelope ({"action": "inject_complete_system", "diagramType": ..., ...})
      sent back over the WebSocket.
```

**Layer 1 (routing) is `src/unified_classifier.py`, not BAF.** `_SYSTEM_PROMPT` in that
file is the single authoritative rulebook — positive examples, what an intent is *not*, and
disambiguation against confusable intents all live there. The `description=` strings on
`agent.new_intent()` in `modeling_agent.py` are deliberately **one-liners**; they no longer
drive routing and must stay short.

**BAF's own classifier is a fallback, and it is not an LLM.**
`agent_setup.init_intent_classifier_config()` returns
`SimpleIntentClassifierConfiguration(framework='tensorflow')` — a local, free classifier
trained at startup on each intent's `training_sentences`. It only decides routing in two
cases: (1) the unified call failed and returned `fallback_intent`, and (2) voice /
plain-text events, which arrive without a JSON payload and route via `when_intent_matched`.
`training_sentences` are therefore **required** on every intent — an intent with none breaks
that classifier at startup.

**Transition priority** (wired in `state_bodies.add_unified_transitions`):

```
0  _ensure_unified_classification(session)   side-effect hook; ALWAYS returns False
1  json_intent_matches(session, {...})       reads the cached classifier verdict
2  route_to_generation(session)              frontend_event + pending generator flows ONLY
3  when_intent_matched(intent)               voice / plain-text events
4  json_no_intent_matched(session)           → the state's fallback
```

Priority 2 is deliberately cheap: `should_route_to_generation()` makes **no** LLM call and
runs **no** text heuristics. The old keyword pre-filters (`_is_modeling_request()`,
`_is_diagram_creation_request()`, the `json_intent_matches` cross-validation) were **deleted**
— there is no pre-filter layer to add a special case to. Routing changes belong in
`_SYSTEM_PROMPT`, or, for decisions the LLM is measurably unreliable at, in an explicit
deterministic guard (`_names_unsupported_stack()` in `unified_classifier.py`, the
`_GITHUB_URL_RE` / `_GITHUB_CONTINUE_VERB_RE` pair in `generation_handler.py`).

**Layer 2 (diagram-type resolution, `orchestrator/workspace_orchestrator.py`) is
keyword/regex-based, not LLM-based** — this is *not* where the routing bug lives if a
request already reached `create_complete_system_state`; check here only if a message that
did reach a modeling state resolves to the *wrong* diagram type.

## Adding (or auditing) a diagram type — the checklist

Every diagram type must be registered in **all** of these places. Missing any one of them
is exactly the kind of stale-list bug that's easy to introduce (add a handler, forget the
surrounding scaffolding) and easy to miss in manual testing (testing "while already on the
new diagram's tab" never exercises the routing/discoverability layer):

1. **`src/schemas/<type>.py`** — Pydantic schemas for the LLM's structured output
   (single-element spec, complete-system spec, modification actions with `Literal` action
   names). Re-export from `src/schemas/__init__.py`.
2. **`src/diagram_handlers/types/<type>_diagram_handler.py`** — concrete
   `BaseDiagramHandler` subclass (see **Diagram Handler Pattern** below).
3. **`src/diagram_handlers/registry/factory.py`** — add the handler class to
   `HANDLER_CLASSES`. The registry keys itself on each handler's own `get_diagram_type()`,
   so the token can't drift from the handler that owns it.
4. **`src/unified_classifier.py`** — add the token to `_TARGET_DIAGRAM_TYPES` **and** teach
   `_SYSTEM_PROMPT` the type's vocabulary. This is the step that actually makes the type
   reachable; without it a request naming the type can fall through to the fallback body.
   Do **not** put the vocabulary in `modeling_agent.py`'s intent `description=` strings —
   those only feed the local fallback classifier. Do add one representative
   `training_sentence` there.
5. **`src/orchestrator/workspace_orchestrator.py`** — add entries to `KEYWORD_TARGETS`,
   an `_IMPLICIT_PATTERNS` regex, and append the type to `FALLBACK_PRIORITY`.
   `_IMPLICIT_PATTERNS` is evaluated in list order and the first match wins — place a new
   pattern where a broader one above it won't steal its vocabulary (BPMN sits above
   StateMachineDiagram precisely because "process" is in both).
6. **`src/state_bodies.py`** — two places:
   - `_QUICK_RESPONSES["what_can_you_do"]` and `["help"]` (the static capability-list
     text — bump the hardcoded type count in `["help"]`).
   - `_fallback_llm_reply`'s prompt (the "You are a modeling assistant that helps
     with X, Y, Z" string) — shared by `global_fallback_body` and `greetings_body`.
   - Optionally `modeling_help_body`'s per-diagram-type conceptual-help prompt branch
     (`if diagram_type == "..."`) if the type warrants a specialized help persona.
7. **`src/protocol/types.py`** — add to `SUPPORTED_DIAGRAM_TYPES`. An `activeDiagramType`
   outside this set is silently normalized to `ClassDiagram`.
8. **`src/diagram_handlers/registry/metadata.py`** — display metadata
   (`DIAGRAM_TYPE_METADATA`).
9. **`src/suggestions.py`** — a suggestion list plus an entry in
   `_DIAGRAM_SUGGESTION_HANDLERS`.
10. **Docs** — `README.md`'s "Supported Diagram Types" table,
    `docs/source/diagram_handlers.rst`, `docs/source/websocket_protocol.rst`,
    `docs/source/getting_started.rst`.
11. **Frontend** (separate repo) — the type must exist in the WME's own diagram-type union
    and be a valid `activeDiagramType` context value for `AssistantClient.ts` to send.

Two existing tokens do **not** follow the `<Name>Diagram` convention: BPMN's is `"BPMN"`
(the editor's converter sets the Apollon `model.type` to `"BPMNDiagram"` itself) and the
User Profile handler's is `"UserDiagram"`.

## Diagram Handler Pattern

Every concrete handler in `src/diagram_handlers/types/` extends
`BaseDiagramHandler` (`src/diagram_handlers/core/base_handler.py`) and implements its
**five abstract methods**:

```python
get_diagram_type() -> str          # the WME storage-bucket token, e.g. "BPMN"
get_system_prompt() -> str          # DESIGN RULES for the LLM's structured generation
generate_single_element(...)        # append one node to an existing diagram
generate_complete_system(...)       # build a full diagram from scratch (primary path)
generate_fallback_element(...)      # error-path stub when the LLM fails
```

`generate_modification(...)` is **concrete** on the base class (a default LLM path) —
override it only when the default isn't enough. Layout, retry, structured output and
two-pass generation are all inherited.

**Two-pass structured generation** (`base_handler.predict_two_pass_structured`,
used by `generate_complete_system`): a free-text *reasoning pass* (chain-of-thought
planning against the request, on the `MODEL_REASONING` tier) followed by a *structured pass*
that converts the reasoning into a schema-validated Pydantic object. The reasoning pass is
skipped for short requests — the threshold is `_TWO_PASS_MIN_LENGTH = 250` characters,
measured on the **raw** user message, not the enriched prompt (otherwise history and the
workspace-context block would push a trivial request onto the expensive path). The reasoning
prompt is the highest-leverage place to fix systematic completeness gaps — see
`bpmn_diagram_handler.py`'s `reasoning_prompt` for an example that explicitly tells the model
not to silently merge multiple described decision points into one gateway (a real bug found
via statistical probing: ~87% of runs on an otherwise-correct prompt silently dropped a
described decision point before this fix).

**Post-generation validation without an LLM round-trip** (`_validate_and_refine` /
`_connect_orphaned_nodes` in `bpmn_diagram_handler.py`): deterministic Python repair for
structural invariants the LLM sometimes violates despite the system prompt stating them
(e.g. "every node has an incoming flow"). Prefer fixing the *root cause* via the prompt when
possible; use a deterministic repair pass for invariants where prompt-following is
statistically unreliable and a cheap, evidence-grounded heuristic exists (don't guess blindly
— e.g. reconnecting an orphaned node from a gateway that has fewer outgoing flows than its
design rules require, not from an arbitrary node).

**LLMs never emit positions.** `core/layout_engine.py` runs after every generation.
Class diagrams get a Sugiyama layered layout (`layout_class_system`); everything else gets
per-type grid layouts. Constants: `H_GAP = 100`, `V_GAP = 80`, `REL_EXTRA_GAP = 60`,
`MARGIN = 40`, `GRID_SNAP = 20`.

## Wire Protocol

BAF's `Payload.decode()` reads only three top-level keys off a message:
`action`, `message`, `history`. The frontend's actual v2 payload
(`protocolVersion`, `clientMode`, `sessionId`, `context.activeDiagramType`, …) is therefore
**JSON-stringified into the `message` field of a `user_message` envelope** — i.e. the wire
payload is double-JSON-encoded:

```json
{
  "action": "user_message",
  "message": "{\"action\":\"user_message\",\"protocolVersion\":\"2.0\",\"clientMode\":\"workspace\",\"sessionId\":\"...\",\"message\":\"<actual user text>\",\"context\":{\"activeDiagramType\":\"BPMN\",...}}",
  "history": false
}
```

`_unwrap_v2_envelope()` in `protocol/adapters.py` recovers the inner payload and merges it
over the outer one.

**`user_id` is not an envelope key.** It's a WebSocket **URL query parameter**
(`wss://host/agent?user_id=…`), read off the opening HTTP request by
`_extract_user_id_from_request` in `patches/websocket_platform.py`. It keys the BAF session;
conversation memory keys on the *inner* `sessionId` (`memory.memory_session_key`).

Not every inbound message is a `user_message`: `user_voice` (base64 audio → whisper-1),
`user_set_variable` (arms BYOK via `user_api_key` / `user_api_provider` / `user_api_model` /
`user_api_base`, passes `_voice_context`, and carries the keep-alive heartbeat),
`frontend_event` (a generator result echo — routed deterministically, never classified), and
`replay_last_response` (re-send the buffered terminal reply after a reconnect).

Responses streamed back from the LLM are wrapped **again**: each chunk arrives as
`{"action": "agent_reply_str", "message": "<JSON string>", "history": false}`, where the
inner JSON string is `{"action": "stream_start"|"stream_chunk"|"stream_done", "streamId": ...}`.
When probing the agent directly (bypassing the browser/frontend), replicate both encoding
layers and unwrap them in the same order the frontend's `AssistantClient.ts` does
(`extractActionPayload`) — a naive single-level unwrap will silently treat every response as
an unrecognized action and hang waiting for a message that already arrived.

**Emitted actions.** Terminal: `inject_element`, `inject_complete_system`, `modify_model`,
`assistant_message`, `agent_error`, `create_diagram_tab`, `trigger_generator`,
`trigger_smart_generator`, `trigger_github_import`, `trigger_export`, `trigger_deploy`,
`auto_generate_gui`. Non-terminal: `progress`, `stream_start`, `stream_chunk`,
`stream_done`. There is **no `switch_diagram` action** — nothing in `src/` emits one.

`config.yaml`'s `nlp.intent_threshold` (default 0.55) is BAF's own confidence floor for the
local `SimpleIntentClassifier`. It does **not** gate the unified classifier and therefore
does not decide which state handles a normal message — changing it only affects the
fallback path.

## Repository Structure

```text
modeling-agent/
  modeling_agent.py                # Entrypoint: BAF agent + 10 states/intents + reaper + wiring
  config.yaml                      # WebSocket host/port/origins, intent_threshold, OpenAI key (gitignored)
  config_example.yaml              # Template for config.yaml
  Dockerfile                       # python:3.11-slim; entrypoint writes config.yaml from env
  patches/websocket_platform.py    # Vendored over pip-installed BAF 4.3.2 (see below)
  src/
    agent_setup.py                 # LLM/RAG/STT/diagram-factory/classifier-config bootstrapping
    agent_context.py               # Shared module-level context (gpt, gpt_text, factory, rag, stt, ...)
    agent_config.py                # Tunable constants (MAX_TABS, temperatures, token budgets, ...)
    model_config.py                # Per-call-site model tiers, all BESSER_AGENT_MODEL_* overridable
    unified_classifier.py          # THE ROUTER — one classification call per message
    state_bodies.py                # State bodies + transition wiring + global fallback + quick responses
    session_helpers.py             # Reply/stream helpers, transition conditions, reply replay
    session_keys.py                # Every session-state key constant (import these, not literals)
    confirmation.py                # Pending replace/keep + GUI-mode choice flows
    byok.py                        # Per-request bring-your-own-key routing (contextvar)
    telemetry.py                   # Opt-in, fire-and-forget study prompt telemetry
    reply_copy.py                  # Shared user-facing copy strings
    routing/intents.py             # Intent-name constants shared across modules
    orchestrator/
      workspace_orchestrator.py    # determine_target_diagram_type — keyword/regex layer 2
      request_planner.py           # Multi-step / multi-operation planning
    diagram_handlers/
      core/base_handler.py         # Abstract base + two-pass structured generation + layout
      core/layout_engine.py        # Deterministic post-LLM position computation
      core/prompt_fragments.py     # Shared prompt snippets (EXACT_NAMES_RULE, etc.)
      types/<type>_diagram_handler.py  # One concrete handler per diagram type (8)
      types/gui_design_system.py   # Per-domain GUI themes + .ds-* stylesheet rules
      types/gui_html_converter.py  # LLM HTML → GrapesJS tree, sanitized, widget-slot splicing
      registry/factory.py          # DiagramHandlerFactory — HANDLER_CLASSES tuple
      registry/metadata.py         # Per-type metadata (labels, icons, keywords)
    schemas/<type>.py              # Pydantic schemas for structured LLM output per type
    execution/
      planning.py                  # execute_planned_operations — the multi-op dispatch point
      model_operations.py          # Apply generated specs; destructive-change confirmation
      file_handling.py             # handle_file_attachments
      progress.py                  # Streaming progress events
    handlers/
      generation_handler.py        # Deterministic generators, export/deploy, GitHub import
      smart_generation_handler.py  # trigger_smart_generator payload assembly
      file_conversion_handler.py   # PlantUML/KG/XMI/PDF/image/text → diagram spec
      validation_handler.py        # Bridge to the BESSER backend's diagram validator
    protocol/
      types.py                     # AssistantRequest, WorkspaceContext, SUPPORTED_DIAGRAM_TYPES
      adapters.py                  # Wire payload → AssistantRequest parsing (+ per-event cache)
    memory/conversation_memory.py  # Verbatim window + rolling LLM summary per session
    llm/provider.py                # LLM provider abstraction (structured outputs, streaming)
    tracking/token_tracker.py      # Token usage/cost tracking (_COST_PER_1K per model)
    utilities/                     # model_context, model_resolution, workspace_context,
                                   # class_metadata, request_builders, user_metamodel, llm_retry
    suggestions.py                 # "What's next?" QuickAction suggestion engine
    domain_patterns.py             # 10 domain patterns — defined but NOT injected
    state_patterns.py              # 8 lifecycle patterns — defined but NOT injected
  tests/
  docs/
```

`patches/websocket_platform.py` is copied over the pip-installed BAF 4.3.2 file at image
build time. Stock BAF evicts the `_connections` slot unconditionally on close; with the
stable `?user_id=` param the two sockets a browser tab opens (assistant widget + workspace
drawer) share one session key, so a closing socket would drop the live one's replies. The
vendored file adds an ownership-guarded delete plus reply-route-to-sender, and is also where
the per-request BYOK context var is set and reset. **Re-vendor if the BAF pin bumps.**

## Testing

```bash
python -m pytest                                    # full suite
python -m pytest tests/test_unified_classifier.py    # routing verdicts
python -m pytest tests/test_bpmn.py                  # one diagram type
python -m pytest tests/test_diagram_handlers.py       # cross-handler contract tests
python -m pytest tests/test_request_planner.py
python -m pytest tests/test_protocol.py
```

Handler-level tests instantiate handlers directly with `BPMNDiagramHandler(None)` (the LLM
arg is only needed by methods that actually call the model) — see `tests/test_bpmn.py` for
the established pattern, including testing `_validate_and_refine` as a pure function against
hand-built node/flow dicts (no LLM call needed to test deterministic repair logic).
`tests/conftest.py` supplies `FakeSession`, `FakeLLM`, `make_v2_payload()`, `make_session()`
and the `MINIMAL_CLASS_MODEL` / `EMPTY_CLASS_MODEL` fixtures.

**Statistical / live probing**: unit tests catch structural regressions but not LLM
generation-quality drift (completeness, consistency across runs). For that, use the scripts
in `tests/live/` (`probe_smoke.py`, `probe_full_agentic.py`, `wme_release_sweep.py`) against
a running agent (`AGENT_WS_URL` is required) — they speak the real double-encoded envelope (see **Wire Protocol**).
Run the same prompt N times and diff the resulting specs: this is how the ~87%
stock-gateway-drop and orphaned-end-event bugs were actually found; a single manual test in
the browser has good odds of landing in the "looks fine" bucket even when the underlying
rate is bad. `probe_smoke.py` is fast enough to use as a post-deploy gate.

## Common Pitfalls

1. **Manual testing "from inside" the feature doesn't exercise routing.** Testing a new
   diagram type by opening its tab and asking to create/modify things never exercises the
   classifier (layer 1) or diagram-type resolution (layer 2) — both already know the
   answer from `context.activeDiagramType`. Test from a *different* tab, and with phrasing
   that doesn't literally name the diagram type, to catch discoverability gaps.
2. **Capability descriptions are duplicated across files with no single source of truth.**
   `unified_classifier._SYSTEM_PROMPT`, `state_bodies.py`'s `_QUICK_RESPONSES` and
   `_fallback_llm_reply` prompt, `suggestions.py`, and `README.md`'s table all independently
   enumerate supported diagram types. Grep for an existing type's name (e.g. `"quantum"`)
   across the repo when adding a new one — that's the fastest way to find every place that
   needs updating.
3. **Routing rules go in `_SYSTEM_PROMPT`, never in an intent `description=`.** The
   description strings are one-line summaries for the local fallback classifier only. Long
   keyword essays there are dead weight and drift silently out of sync with real behavior.
   `training_sentences`, by contrast, are mandatory — the Simple classifier trains on them.
4. **A pending flow can suppress intent matching — by design.** When the assistant is
   awaiting an answer, `json_intent_matches()` returns `False` for all intents *unless* the
   classifier labelled the message `pending_flow_action="new_request"`. A message that looks
   "stuck" in a state is usually this working correctly, not a routing bug.
5. **`develop` is this repo's integration branch; `main` is production.** Work lands on
   `develop`, gets PR'd to `main`. Don't confuse this with BESSER's own
   `development`/`master` convention — they're separate repos with separate branch names
   for the same roles.
6. **Local `develop`/`main` checkouts drift silently.** If you've been working in this repo
   across a long session, `git status --short` and `git branch --show-current` before editing
   — merges via `gh pr merge` update the *remote* branch immediately but not your local ref
   until you `git pull`, and switching branches with uncommitted changes based on a newer
   ref than your local target can fail the checkout outright (stash, switch+pull, pop).
7. **Two shared LLM instances, don't mix them up.** `gpt` (`agent_setup.init_llm`) is the
   structured/JSON call path; `gpt_text` is free-text (help, greetings, RAG, streaming).
   Both default to the `MODEL_CLASSIFIER` tier — generation-quality call sites pass
   `model=` explicitly from `src/model_config.py`. Passing the wrong instance to a call that
   expects the other's response format will break silently or raise a JSON-parse error deep
   in `predict_structured`.
8. **gpt-5 / o-series models reject `temperature`.** Use
   `model_config.supports_custom_temperature(model)` and `reasoning_effort_for(model)` at
   every call site instead of hardcoding either parameter. And whenever a `MODEL_*` default
   changes, add the matching entry to `_COST_PER_1K` in `tracking/token_tracker.py` —
   an unknown model falls back to placeholder pricing and every reported cost silently
   becomes an estimate.
9. **`config.yaml` is gitignored — real API keys never get committed.** Copy from
   `config_example.yaml`; the same applies to `.env` / `.env.example`. The Docker entrypoint
   generates `config.yaml` from the environment and redacts the `api_key` line from its own
   debug output. It emits `platforms.websocket.origins` from `BESSER_AGENT_WS_ORIGIN` /
   `BESSER_AGENT_WS_ORIGIN_ALT` plus the localhost dev origins.
