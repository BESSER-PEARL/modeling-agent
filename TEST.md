# Testing the Modeling Agent

This document describes how the `modeling-agent` test suite is organized, how to run it,
and what each test file covers.

## How the tests are organized

Tests live in two places with very different characters:

| Location | Character | Runs in CI? | Cost |
| --- | --- | --- | --- |
| `tests/*.py` | Deterministic unit / integration tests. No live LLM — everything is stubbed (`FakeSession`, `FakeLLM`, recording fakes, monkeypatched classifiers). | Run manually (there is no CI). | Free, fast. |
| `tests/live/*.py` | **LIVE** harnesses that open a real WebSocket to the *deployed* agent at `wss://experimental.besser-pearl.org/agent` and drive real generations. | **Never** — they hit a paid model. | Cost tokens; run manually before a release. |

`tests/conftest.py` provides the shared scaffolding used across the flat suite:
`FakeSession` (a stand-in for BAF's `Session`), `FakeLLM` (canned-response stub),
`make_v2_payload` / `make_session` (build the double-JSON-encoded v2 wire payload the
frontend sends), and the `MINIMAL_CLASS_MODEL` / `EMPTY_CLASS_MODEL` fixtures. It also
inserts `src/` onto `sys.path` so tests can use bare-style imports (`from protocol.types import ...`).

## Running the tests

```bash
# Full deterministic suite (1177 tests collected)
python -m pytest

# Focused suites
python -m pytest tests/test_bpmn.py                  # one diagram type
python -m pytest tests/test_diagram_handlers.py       # cross-handler contract tests
python -m pytest tests/test_request_planner.py
python -m pytest tests/test_protocol.py
```

There is **no `pytest.ini` / `pyproject.toml` / `setup.cfg`** — tests are discovered with
pytest defaults (`test_*.py`), and `conftest.py` handles the `src/` path wiring.

### CI status

**There is no CI in this repo.** No `.github/workflows/` directory exists — the suite is
run manually. Keep that in mind: a green local run is the only gate on a change.

### Running the live sweeps

The `tests/live/` harnesses are **run by hand** (usually before a release) and print a
categorized `ok / FLAW / FAIL` report to stdout. They are `__main__` scripts, not pytest
tests (the one exception is the pytest wrapper in `test_nl_generation_scenarios.py`, which
is **skipped unless `RUN_LIVE_AGENT_TESTS=1`**).

```bash
# Comprehensive ~97-scenario release sweep
AGENT_WS_URL=wss://experimental.besser-pearl.org/agent CONC=4 \
    python tests/live/wme_100_sweep.py

# Focused subset (comma-separated categories)
ONLY=system,modify,edge python tests/live/wme_100_sweep.py

# NL→generator routing matrix (as a pytest deploy gate)
RUN_LIVE_AGENT_TESTS=1 python -m pytest tests/live/test_nl_generation_scenarios.py
```

Environment knobs (defaults in parentheses):

| Knob | Used by | Meaning |
| --- | --- | --- |
| `AGENT_WS_URL` | all live harnesses | Target agent (`wss://experimental.besser-pearl.org/agent`). |
| `CONC` | `wme_100_sweep` (4), `wme_release_sweep` (3) | Parallel WebSocket connections — kept low to protect the single live agent. |
| `GEN_TIMEOUT` | `wme_100_sweep`, `wme_release_sweep` (150) | Per-reply timeout, seconds. |
| `ONLY` | `wme_100_sweep`, `wme_release_sweep` | CSV filter of scenario categories (`system,webapp,other,modify,generate,vague,edge,meta`). |
| `REPEAT` | `wme_release_sweep` (1) | Repeat the whole scenario set N times. |
| `REPEATS` | `test_nl_generation_scenarios` (2) | Probes per scenario (classifier is non-deterministic; a pass threshold applies). |
| `RUN_LIVE_AGENT_TESTS` | `test_nl_generation_scenarios` (unset) | Gate for the pytest wrapper — unset ⇒ skipped. |

## Inventory

Counts below are `def test_` function counts. Many files use `@pytest.mark.parametrize`, so
the collected total (**1173**) is larger than the raw function count. Unless noted, a file
has **no** skip/xfail markers.

### Routing, classification & planning

| File | ~Funcs | What it verifies |
| --- | --- | --- |
| `test_non_modeling_guard.py` | 2 (param) | `state_bodies._request_is_non_modeling` — injection / persona-hijack / shell-command text is declined; real (even non-English) modeling requests pass. |
| `test_create_vagueness_guard.py` | 2 (param) | `state_bodies._create_request_is_too_vague` — pure-filler creates ("create", "make an app") flagged vague; a real domain noun proceeds. |
| `test_unified_classifier.py` | 19 | `unified_classifier` — `classify_message`, `get_or_classify` event-id caching, schema literals, `_SYSTEM_PROMPT` content, fallback/demotion, and smart-gen follow-up routing. |
| `test_orchestrator.py` | 17 | `orchestrator/workspace_orchestrator.py` target resolution — explicit/implicit targets, `determine_target_diagram_type(s)`, `resolve_diagram_id`. |
| `test_request_planner.py` | 23 | `orchestrator/request_planner.py` — segment splitting, target matching, operation normalization/fallback, `plan_assistant_operations` (no-op fake LLM). |
| `test_two_pass_fast_path.py` | 4 | `base_handler.predict_two_pass_structured` fast-path — trivial (<250-char) requests skip the reasoning pass (one LLM call), keyed on `raw_request`. |

### Diagram handlers — class diagram, guards & constraints

| File | ~Funcs | What it verifies |
| --- | --- | --- |
| `test_class_typed_attribute_guard.py` | 6 | `ClassDiagramHandler._rewrite_class_typed_attributes` — class-typed attributes become associations; enums/primitives handled; unknown PascalCase → String. |
| `test_enum_relationship_guard.py` | 12 | `_rewrite_enum_relationships` (system) + `_rewrite_enum_relationship_mods` (modify) — enums never appear as relationship endpoints. |
| `test_relationship_name_dedup.py` | 7 | `_dedupe_relationship_names` — duplicate association names made unique (camelCase then numeric fallback); unnamed left alone. |
| `test_class_modification_missing_target.py` | 10 | `_build_model_index` / `_drop_phantom_target_ops` — deterministic missing-target detection for class removals/modifies (adds never validated). |
| `test_class_modification_names.py` | 19 (param) | `schemas.class_diagram._is_placeholder` / `_clean_name` + `add_class` name resolution — junk placeholder names nulled, real names promoted. |
| `test_ocl_constraint_capture.py` | 10 | OCL constraint capture — `OCLConstraintSpec`, `_validate_constraints` (drops unknown-context/empty), end-to-end into `inject_complete_system`. |
| `test_prompt_caching.py` | 5 (param) | Modification prompts are static module-level constants (no UUID/timestamp), clear the ~1024-token cache threshold, and are passed separately (not concatenated). |

### Diagram handlers — agent, BPMN, base & factory

| File | ~Funcs | What it verifies |
| --- | --- | --- |
| `test_agent_diagram.py` | 10 | `AgentDiagramHandler` code-reply wrapping — `replyType="code"` replies always wrapped into `def name(session):`, with name-hint sanitization. |
| `test_agent_modification_safety.py` | 18 (param) | Agent modification data-loss guards — validate mods against the *current* model; no hallucinated targets; clarify on vague input. |
| `test_agent_orphan_transition.py` | 17 | Agent `_validate_modifications` same-batch pending-rescue + orphan backstop + inherited friendly-message helpers. |
| `test_bpmn.py` | 24 | BPMN end-to-end — routing, schemas, `_validate_and_refine` structural repair, fallback, add/remove modifications, suggestions. |
| `test_base_handler.py` | 42 | `BaseDiagramHandler` utilities in depth — `validate_spec`, `clean_json_response`, `parse_json_safely`, `classify_error`, `validate_modification_spec`, `extract_name_from_request`. |
| `test_diagram_handlers.py` | 22 | `DiagramHandlerFactory` (+ `get_diagram_type_info` / metadata) plus a slice of the base-handler utilities. |
| `test_model_helpers.py` | 86 | `utilities.model_context` (compact/detailed summaries across all types, `is_diagram_nontrivial`) + `utilities.model_resolution` resolvers. |

### GUI (no-code web app)

| File | ~Funcs | Markers | What it verifies |
| --- | --- | --- | --- |
| `test_auto_gui_message.py` | 4 | — | `confirmation._build_auto_gui_message` — the "Auto-generate GUI" completion message (names pages, counts, truncates, degrades gracefully). Runs in an isolated subprocess to sandbox a hard `baf` import. |
| `test_gui_chart_generation.py` | 33 (param) | — | Chart/table/dashboard section generation + `utilities.class_metadata` binding. |
| `test_gui_design_system.py` | 15 (param×DOMAINS) | — | `gui_design_system` — every domain resolves a full theme; `stylesheet_rules`/`block_exemplars` emit `.ds-*` output; CSP-safety (no webfonts/`@import`/`url(`). |
| `test_gui_html_converter.py` | 26 (param) | — | `gui_html_converter` — LLM HTML → GrapesJS tree, tag/class/style retention, sanitisation (script/on*/external stripped), widget-slot helpers. |
| `test_gui_modification.py` | 13 (param) | — | GUI modification ops — deterministic fast-paths (rename/recolor/reorder), spec application, and the "LLM failure never empties the model" safety net. |
| `test_gui_phase0.py` | 15 | 1× `importorskip("openai.lib._pydantic")` | Phase 0 quick-wins — truncation salvage, `two_column`, `stats_grid` value/binding preservation. |
| `test_gui_phase3.py` | 22 | 1× `importorskip("openai.lib._pydantic")` | Phase 3 — LLM-authored `.ds-*` HTML sections + structured widget binding spliced at `<!--WIDGET:-->`, domain styles, graceful fallback. |

### Generation & smart-generation flow

| File | ~Funcs | What it verifies |
| --- | --- | --- |
| `test_generation_handler.py` | 59 (param) | `generation_handler` — `detect_generator_type`, config parse/prompt, `should_route_to_generation` gatekeeper, dispatch, Django→SQL pivot-mid-config fix. |
| `test_smart_generation_handler.py` | 23 (param) | Smart / Spec-Driven classifier — `classify_generation_request` (safe fallbacks, verbatim passthrough), `build_trigger_smart_generator_payload`, dispatch. |
| `test_smart_generation_handler_gate.py` | 10 (param) | The B-2 "confirm-before-smart" gate — never fires on first contact; stashes with a 30-min TTL; explicit/natural confirm fires, cancel/mixed never do. |
| `test_smart_generator_result_event.py` | 6 | `generator_result` frontend event with `metadata.smart=True` — hides cost/internal name, records outcome to memory, maps error codes to tone. |
| `test_webapp_generation_gate.py` | 4 | The web-app auto-generation **pause** — a GUI-create + `web_app` plan has the generation op stripped at source (`execution/planning.py`) and shows a generate nudge. Distinct from the smart-gen gate. |
| `test_empty_workspace_bridge.py` | 11 | Same-turn create→generate bridging — an in-turn create is written back into the snapshot so a later generate step isn't wrongly refused as "empty workspace". |
| `test_confirmation.py` | 52 | `confirmation.py` — `keyword_matches` (whole-word `\b`-bounded), `model_has_elements`, and the REPLACE/KEEP/CANCEL/NEW_TAB keyword lists. |
| `test_suggestions.py` | 25 (param) | `suggestions.get_suggested_actions` per diagram type/mode + `format_suggestions_as_text`, with `{label,prompt}`-only and ≤4-suggestions invariants. |

### Protocol, memory, LLM & tracking

| File | ~Funcs | Markers | What it verifies |
| --- | --- | --- | --- |
| `test_protocol.py` | 23 (param) | — | v2 wire-payload parsing (`protocol/adapters.py`). **Contains the known red** (`test_with_active_model`). |
| `test_conversation_memory.py` | 14 | — | `ConversationMemory` sliding window, summarizer trigger, `build_context`, the global registry, and thread safety. |
| `test_memory_session_key.py` | 5 | — | `memory_session_key` derivation chain — `request.session_id` → payload `sessionId` → BAF `session.id` → `id(session)`. |
| `test_llm_provider.py` | 18 | — | `LLMProvider` — construction, `predict`/`parse`/`stream` delegation, token recording, error propagation, no-client fallbacks. |
| `test_llm_retry.py` | 8 | — | `llm_retry` — permanent quota/billing/invalid-key errors fail fast (1 call); genuine 429/5xx retry up to `MAX_ATTEMPTS`. |
| `test_byok_base_url.py` | 4 | — | BYOK custom `base_url` SSRF gate (`BESSER_AGENT_ALLOW_CUSTOM_BASE_URL`) + `redacted()`. |
| `test_model_config.py` | 6 | — | Env-driven model routing table + defaults/overrides, cost-table consistency, `supports_custom_temperature` / `reasoning_effort_for`. |
| `test_token_tracker.py` | 12 | — | `TokenTracker` — record/accumulate, per-session isolation, cost estimation, `summary()`, singleton, thread safety. |
| `test_ws_reply_outbox.py` | 6 | `importorskip("patches.websocket_platform")` | The "generate did nothing" reconnect fix — `_send`/`_buffer_reply`/`_flush_outbox` buffer on a dead conn and redeliver in order on reclaim. |
| `test_file_conversion.py` | 74 | — | `file_conversion_handler` — `detect_file_type` + PlantUML / RDF-KG / image / text → diagram spec across Class/StateMachine/Object/Agent/BPMN. |
| `test_schemas.py` | 191 (param) | — | Every Pydantic schema under `src/schemas/` — field defaults, `Literal`/enum constraints, min-length/`ge` validators, across class/state-machine/object/agent/GUI/quantum. |

### Live harnesses (`tests/live/`, manual)

| File | What it does | Notes |
| --- | --- | --- |
| `test_nl_generation_scenarios.py` | NL→generator **routing matrix** — drives real phrasings ("generate a database", …) and asserts each lands on an acceptable generator and never a forbidden one (e.g. database requests must never hit `django`). | Hosts the shared `_send` / `_wait_meaningful` WebSocket helpers imported by both sweeps. Only pytest-collected live test (1 test, skipped unless `RUN_LIVE_AGENT_TESTS=1`). |
| `wme_100_sweep.py` | Comprehensive ~97-scenario sweep: `system / webapp / other / modify / generate / vague / edge / meta`, classified ok/FLAW/FAIL. | Consolidates the earlier sweeps (its `edge` category replaces the old standalone edge-case script). **Imports `_flaws_for_system` from `wme_release_sweep.py`.** |
| `wme_release_sweep.py` | 28-scenario fidelity sweep (`system / webapp / other`) checking the "useless model" flaws — isolated classes, dup names, dangling/enum endpoints, web-app auto-run. | **Hosts `_flaws_for_system`**, the fidelity checker reused by `wme_100_sweep.py`. |
