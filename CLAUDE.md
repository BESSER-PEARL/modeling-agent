# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Overview

`modeling-agent` is the conversational AI backend for the **BESSER Web Modeling Editor**
(`editor.besser-pearl.org`). It's a standalone Python service built on the
**BESSER Agent Framework (BAF)** that talks to the frontend over a WebSocket, interprets
natural-language modeling requests, and returns structured actions (`inject_complete_system`,
`modify_model`, `switch_diagram`, `trigger_generator`, …) that the frontend applies to the
diagram canvas.

- **Frontend caller**: `besser/utilities/web_modeling_editor/frontend`'s
  `packages/webapp/src/main/features/assistant/services/AssistantClient.ts` — see
  **Wire Protocol** below for the exact message shape it sends/expects.
- **Deployed alongside** BESSER's main releases but has its **own repo, its own
  `develop`→`main` branch convention, and its own release cadence** — it is
  *not* part of BESSER's version number and is never included in a BESSER release PR.
  Deploy it with `./scripts/deploy.sh agent` from the BESSER repo.
- Runs as a single long-lived process (`modeling_agent.py`) hosting BAF's
  `websocket_platform` (`config.yaml` → `platforms.websocket.port`, default 8765).
  Production is reverse-proxied at `wss://editor.besser-pearl.org/agent`.

## Request Flow (read this before touching routing or adding a diagram type)

A message goes through **two independent routing layers** before it reaches a diagram
handler. Confusing which layer owns a decision is the single most common source of bugs
here (including the "agent says it can't do BPMN" bug this file exists to help prevent).

```
WebSocket message
  → protocol/adapters.py + protocol/types.py   (parse wire payload → AssistantRequest)
  → BAF's LLM intent classifier                (modeling_agent.py intent descriptions)
      decides WHICH STATE handles this message:
      create_complete_system_state / modify_model_state / modeling_help_state /
      describe_model_state / generation_state / workflow_state / greetings_state
      — or falls through to global_fallback_body (src/state_bodies.py) if no
        intent matches confidently.
  → state_bodies.py's state body for the matched intent
      (create_complete_system_body / modify_modeling_body / … all funnel through
      _modeling_state_body → execution/planning.py's execute_planned_operations)
  → orchestrator/workspace_orchestrator.py's determine_target_diagram_type()
      decides WHICH DIAGRAM TYPE the message targets, using (in priority order):
      1. explicit keyword match (KEYWORD_TARGETS)
      2. discriminating regex patterns (_IMPLICIT_PATTERNS)
      3. active-diagram-type / project-snapshot fallback (FALLBACK_PRIORITY)
  → diagram_handlers/registry/factory.py's DiagramHandlerFactory.get_handler(diagram_type)
  → the concrete handler's generate_complete_system / generate_modification / … method
  → response envelope ({"action": "inject_complete_system", "diagramType": ..., ...})
      sent back over the WebSocket.
```

**Layer 1 (intent classification, `modeling_agent.py`) is LLM-based and description-driven** —
BAF's `LLMIntentClassifierConfiguration` feeds each intent's `description` string (keywords,
examples, disambiguation rules) to the LLM as classification context. If a diagram type's
vocabulary is missing from these descriptions, requests using that vocabulary can miss the
right intent entirely and fall through to `global_fallback_body`, which has **its own
separate, hardcoded capability-list prompt** that must *also* mention every diagram type —
this is a distinct bug from layer-1 misrouting and is easy to miss even after fixing layer 1.

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
   (single-element spec, complete-system spec, modification actions).
2. **`src/diagram_handlers/types/<type>_diagram_handler.py`** — concrete
   `BaseDiagramHandler` subclass (see **Diagram Handler Pattern** below).
3. **`src/diagram_handlers/registry/factory.py`** — add the handler class to
   `HANDLER_CLASSES`.
4. **`modeling_agent.py`** — add the diagram type's vocabulary to:
   - `create_complete_system_intent`'s description (both the "generate X" disambiguation
     list and a dedicated keyword-example paragraph, matching the existing quantum/BPMN
     paragraphs).
   - `generation_intent`'s "these are NOT generation" exclusion list (so "generate a
     BPMN diagram" isn't misclassified as code generation).
   - `modify_model_intent`'s description, if the type has element-level modification verbs
     worth calling out explicitly (see the BPMN modification paragraph for the pattern).
5. **`src/orchestrator/workspace_orchestrator.py`** — add entries to `KEYWORD_TARGETS`,
   an `_IMPLICIT_PATTERNS` regex, and append the type to `FALLBACK_PRIORITY`.
6. **`src/state_bodies.py`** — two places:
   - `_QUICK_RESPONSES["what_can_you_do"]` and `["help"]` (the static capability-list
     text — bump the hardcoded type count).
   - `global_fallback_body`'s LLM prompt (the "You are a modeling assistant that helps
     with X, Y, Z" string) — the fallback that fires when intent classification misses.
   - Optionally `modeling_help_body`'s per-diagram-type conceptual-help prompt branch
     (`if diagram_type == "..."`) if the type warrants a specialized help persona.
7. **`src/protocol/types.py`** — add to `SUPPORTED_DIAGRAM_TYPES`.
8. **`README.md`** — "Supported Diagram Types" table (this repo's README has drifted from
   BPMN already; don't let the next type repeat that).
9. **Frontend** (separate repo) — the type must exist in the WME's own diagram-type enum
   and be a valid `activeDiagramType` context value for `AssistantClient.ts` to send.

## Diagram Handler Pattern

Every concrete handler in `src/diagram_handlers/types/` extends
`BaseDiagramHandler` (`src/diagram_handlers/core/base_handler.py`) and implements:

```python
get_diagram_type() -> str          # the WME storage-bucket token, e.g. "BPMN"
get_system_prompt() -> str          # DESIGN RULES for the LLM's structured generation
generate_single_element(...)        # append one node to an existing diagram
generate_complete_system(...)       # build a full diagram from scratch (primary path)
generate_fallback_element(...)      # error-path stub when the LLM fails
generate_modification(...)          # apply add/remove/rename ops to an existing diagram
```

**Two-pass structured generation** (`base_handler.predict_two_pass_structured`,
used by `generate_complete_system`): a free-text *reasoning pass* (chain-of-thought
planning against the request) followed by a *structured pass* that converts the reasoning
into a schema-validated Pydantic object. The reasoning prompt is the highest-leverage place
to fix systematic completeness gaps — see `bpmn_diagram_handler.py`'s `reasoning_prompt` for
an example that explicitly tells the model not to silently merge multiple described decision
points into one gateway (a real bug found via statistical probing: ~87% of runs on an
otherwise-correct prompt silently dropped a described decision point before this fix).

**Post-generation validation without an LLM round-trip** (`_validate_and_refine` /
`_connect_orphaned_nodes` in `bpmn_diagram_handler.py`): deterministic Python repair for
structural invariants the LLM sometimes violates despite the system prompt stating them
(e.g. "every node has an incoming flow"). Prefer fixing the *root cause* via the prompt when
possible; use a deterministic repair pass for invariants where prompt-following is
statistically unreliable and a cheap, evidence-grounded heuristic exists (don't guess blindly
— e.g. reconnecting an orphaned node from a gateway that has fewer outgoing flows than its
design rules require, not from an arbitrary node).

## Wire Protocol

BAF's `WebSocketPlatform` only preserves four top-level keys on a message:
`action`, `message`, `user_id`, `history`. The frontend's actual v2 payload
(`protocolVersion`, `clientMode`, `sessionId`, `context.activeDiagramType`, …) is therefore
**JSON-stringified into the `message` field of a `user_message` envelope** — i.e. the wire
payload is double-JSON-encoded:

```json
{
  "action": "user_message",
  "user_id": "<sessionId>",
  "message": "{\"action\":\"user_message\",\"protocolVersion\":\"2.0\",\"clientMode\":\"widget\",\"sessionId\":\"...\",\"message\":\"<actual user text>\",\"context\":{\"activeDiagramType\":\"BPMN\",...}}"
}
```

Responses streamed back from the LLM are wrapped **again**: each chunk arrives as
`{"action": "agent_reply_str", "message": "<JSON string>", "history": false}`, where the
inner JSON string is `{"action": "stream_start"|"stream_chunk"|"stream_done", "streamId": ...}`.
When probing the agent directly (bypassing the browser/frontend), replicate both encoding
layers and unwrap them in the same order the frontend's `AssistantClient.ts` does
(`extractActionPayload`) — a naive single-level unwrap will silently treat every response as
an unrecognized action and hang waiting for a message that already arrived.

`config.yaml`'s `nlp.intent_threshold` (default 0.55) gates how confident the LLM
classifier must be before a specialized intent state — rather than the global fallback —
handles a message.

## Repository Structure

```text
modeling-agent/
  modeling_agent.py                # Entrypoint: BAF agent + states/intents + wiring
  config.yaml                      # WebSocket host/port, intent_threshold, OpenAI key (gitignored)
  config_example.yaml              # Template for config.yaml
  src/
    agent_setup.py                 # LLM/RAG/STT/diagram-factory bootstrapping
    agent_context.py               # Shared module-level context (gpt, factory, rag, ...)
    state_bodies.py                # Per-intent state logic + global fallback + quick responses
    routing/intents.py              # Intent-name constants shared across modules
    orchestrator/
      workspace_orchestrator.py    # determine_target_diagram_type — keyword/regex layer 2
      request_planner.py           # Multi-step / multi-operation planning
    diagram_handlers/
      core/base_handler.py         # Abstract base + two-pass structured generation + layout
      core/layout_engine.py        # Deterministic post-LLM position computation
      core/prompt_fragments.py     # Shared prompt snippets (EXACT_NAMES_RULE, etc.)
      types/<type>_diagram_handler.py  # One concrete handler per diagram type
      registry/factory.py          # DiagramHandlerFactory — HANDLER_CLASSES tuple
      registry/metadata.py         # Per-type metadata (labels, etc.)
    schemas/<type>.py              # Pydantic schemas for structured LLM output per type
    execution/
      planning.py                  # execute_planned_operations — the layer-2 dispatch point
      model_operations.py          # Apply generated specs to the working model
      file_handling.py             # PlantUML/KG/image conversion
      progress.py                  # Streaming progress events
    protocol/
      types.py                     # AssistantRequest, WorkspaceContext, SUPPORTED_DIAGRAM_TYPES
      adapters.py                  # Wire payload → AssistantRequest parsing
    memory/conversation_memory.py  # Sliding-window conversation memory per session
    llm/provider.py                # LLM provider abstraction (structured outputs, streaming)
    tracking/token_tracker.py      # Token usage/cost tracking
    utilities/                     # model_context, model_resolution, workspace_context, ...
    handlers/                      # generation_handler, validation_handler, file_conversion_handler
    suggestions.py                 # "What's next?" QuickAction suggestion engine
    confirmation.py                # Confirmation-flow helpers
  tests/
  docs/
```

## Testing

```bash
python -m pytest                                    # full suite
python -m pytest tests/test_bpmn.py                  # one diagram type
python -m pytest tests/test_diagram_handlers.py       # cross-handler contract tests
python -m pytest tests/test_request_planner.py
python -m pytest tests/test_protocol.py
```

Handler-level tests instantiate handlers directly with `BPMNDiagramHandler(None)` (the LLM
arg is only needed by methods that actually call the model) — see `tests/test_bpmn.py` for
the established pattern, including testing `_validate_and_refine` as a pure function against
hand-built node/flow dicts (no LLM call needed to test deterministic repair logic).

**Statistical / live probing**: unit tests catch structural regressions but not LLM
generation-quality drift (completeness, consistency across runs). For that, script direct
WebSocket calls against the live deployment (see **Wire Protocol** above for the exact
envelope) running the same prompt N times and diffing the resulting specs — this is how the
~87% stock-gateway-drop and orphaned-end-event bugs were actually found; a single manual
test in the browser has good odds of landing in the "looks fine" bucket even when the
underlying rate is bad.

## Common Pitfalls

1. **Manual testing "from inside" the feature doesn't exercise routing.** Testing a new
   diagram type by opening its tab and asking to create/modify things never exercises intent
   classification (layer 1) or diagram-type resolution (layer 2) — both already know the
   answer from `context.activeDiagramType`. Test from a *different* tab, and with phrasing
   that doesn't literally name the diagram type, to catch discoverability gaps.
2. **Capability descriptions are duplicated across files with no single source of truth.**
   `modeling_agent.py` intent descriptions, `state_bodies.py`'s `_QUICK_RESPONSES` and
   `global_fallback_body` prompt, and `README.md`'s table all independently enumerate
   supported diagram types. Grep for an existing type's name (e.g. `"quantum"`) across the
   repo when adding a new one — that's the fastest way to find every place that needs updating.
3. **`develop` is this repo's integration branch; `main` is production.** Work lands on
   `develop`, gets PR'd to `main`, then `./scripts/deploy.sh agent` (run from the BESSER repo)
   builds and deploys `main`. Don't confuse this with BESSER's own `development`/`master`
   convention — they're separate repos with separate branch names for the same roles.
4. **Local `develop`/`main` checkouts drift silently.** If you've been working in this repo
   across a long session, `git status --short` and `git branch --show-current` before editing
   — merges via `gh pr merge` update the *remote* branch immediately but not your local ref
   until you `git pull`, and switching branches with uncommitted changes based on a newer
   ref than your local target can fail the checkout outright (stash, switch+pull, pop).
5. **Two LLM instances, don't mix them up.** `gpt` (`agent_setup.init_llm`) is JSON-mode /
   structured-output only; `gpt_text` is free-text (help, greetings, RAG, streaming). Passing
   the wrong one to a call that expects the other's response format will break silently or
   raise a JSON-parse error deep in `predict_structured`.
6. **`config.yaml` is gitignored — real API keys never get committed.** Copy from
   `config_example.yaml`; the same applies to `.env` / `.env.example`.
