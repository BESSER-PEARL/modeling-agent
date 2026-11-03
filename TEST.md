# Testing the Modeling Agent

This document describes how the `modeling-agent` test suite is organized, how to run it,
and what each test file covers.

## How the tests are organized

Tests live in two places with very different characters:

| Location | Character | Runs in CI? | Cost |
| --- | --- | --- | --- |
| `tests/*.py` | Deterministic unit / integration tests. No live LLM — everything is stubbed (`FakeSession`, `FakeLLM`, recording fakes, monkeypatched classifiers). | Yes: `.github/workflows/tests.yml` on every PR and push. | Free, fast. |
| `tests/live/*.py` | **LIVE** harnesses that open a real WebSocket to a running agent (`AGENT_WS_URL`, required) and drive real generations. | **Never** — they hit a paid model. | Cost tokens; run manually before a release. |

`tests/conftest.py` provides the shared scaffolding used across the flat suite:
`FakeSession` (a stand-in for BAF's `Session`), `FakeLLM` (canned-response stub),
`make_v2_payload` / `make_session` (build the double-JSON-encoded v2 wire payload the
frontend sends), and the `MINIMAL_CLASS_MODEL` / `EMPTY_CLASS_MODEL` fixtures. Tests import the agent the way it
imports itself, by top-level name (`from protocol.types import ...`), never `src.`: the
agent is installed with `pip install -e .`, which puts `src/` on the path.

## Running the tests

```bash
# Once: test tools and the agent (editable install)
pip install -r requirements-dev.txt
pip install -e .

# Full deterministic suite
python -m pytest

# Focused suites
python -m pytest tests/test_bpmn.py                  # one diagram type
python -m pytest tests/test_diagram_handlers.py       # cross-handler contract tests
python -m pytest tests/test_request_planner.py
python -m pytest tests/test_protocol.py
```

`pyproject.toml` declares the package but has no pytest section: tests are discovered with
pytest defaults (`test_*.py`). Without the editable install they fail with
`ModuleNotFoundError`.

### CI status

`.github/workflows/tests.yml` runs `python -m pytest tests -q --tb=short --ignore=tests/live`
on Python 3.11, `ruff check src patches modeling_agent.py` (rules in `ruff.toml`) and
`lint-imports` (layer contracts in `pyproject.toml`) on every
PR and on pushes to `develop` and `main`. `docs.yml` builds the documentation. The live
harnesses in `tests/live/` are never run in CI.

### Running the live sweeps

The `tests/live/` harnesses are **run by hand** (usually before a release) and print a
categorized `ok / FLAW / FAIL` report to stdout. They are `__main__` scripts, not pytest
tests (the one exception is the pytest wrapper in `test_nl_generation_scenarios.py`, which
is **skipped unless `RUN_LIVE_AGENT_TESTS=1`**). Every harness requires `AGENT_WS_URL`;
the WebSocket `Origin` header is derived from it (`wss://host` → `https://host`, otherwise
`http://localhost:8080`) and can be overridden with `AGENT_WS_ORIGIN`.

```bash
# Comprehensive ~97-scenario release sweep
AGENT_WS_URL=wss://<your-host>/agent CONC=4 \
    python tests/live/wme_100_sweep.py

# Focused subset (comma-separated categories)
AGENT_WS_URL=ws://localhost:8765 ONLY=system,modify,edge python tests/live/wme_100_sweep.py

# NL→generator routing matrix (as a pytest deploy gate)
AGENT_WS_URL=ws://localhost:8765 RUN_LIVE_AGENT_TESTS=1 python -m pytest tests/live/test_nl_generation_scenarios.py
```

Environment knobs (defaults in parentheses):

| Knob | Used by | Meaning |
| --- | --- | --- |
| `AGENT_WS_URL` | all live harnesses | Target agent, e.g. `ws://localhost:8765` or `wss://<your-host>/agent` (required, no default). |
| `CONC` | `wme_100_sweep` (4), `wme_release_sweep` (3) | Parallel WebSocket connections — kept low to protect the single live agent. |
| `GEN_TIMEOUT` | `wme_100_sweep`, `wme_release_sweep` (150) | Per-reply timeout, seconds. |
| `ONLY` | `wme_100_sweep`, `wme_release_sweep` | CSV filter of scenario categories (`system,webapp,other,modify,generate,vague,edge,meta`). |
| `REPEAT` | `wme_release_sweep` (1) | Repeat the whole scenario set N times. |
| `REPEATS` | `test_nl_generation_scenarios` (2) | Probes per scenario (classifier is non-deterministic; a pass threshold applies). |
| `RUN_LIVE_AGENT_TESTS` | `test_nl_generation_scenarios` (unset) | Gate for the pytest wrapper — unset ⇒ skipped. |

## Inventory

Counts below are the tests `python -m pytest tests --collect-only -q --ignore=tests/live`
collects per file, each parametrized case counted: **3,451** in all. Without BAF 4.5 the
three files that import `patches.websocket_platform` are skipped at import, leaving 3,436.

### Routing, classification & planning

| File | Tests | What it verifies |
| --- | --- | --- |
| `test_non_modeling_guard.py` | 2 | `routing.guards.is_non_modeling` — injection / persona-hijack / shell-command text is declined; real (even non-English) modeling requests pass. |
| `test_create_vagueness_guard.py` | 2 | `routing.guards.is_too_vague_create` — pure-filler creates ("create", "make an app") flagged vague; a real domain noun proceeds. |
| `test_routing_guards.py` | 73 | `routing.guards` tables: classification skips and the `frontend_event` / `[auto-fix]` verdicts, `modeling_guard` precedence (and the reply each decision gets in the modeling state), quick-response keys, mixed and past-generation checks, `generation_route`, and the transition order that keeps BAF's local classifier as the voice route and outage net. |
| `test_unified_classifier.py` | 26 | `unified_classifier` — `classify_message`, `get_or_classify` event-id caching, schema literals, `prompts.classifier.SYSTEM_PROMPT` content, fallback/demotion, and smart-gen follow-up routing. |
| `test_orchestrator.py` | 20 | `orchestrator/workspace_orchestrator.py` target resolution — explicit/implicit targets, `determine_target_diagram_type(s)`, `resolve_diagram_id`. |
| `test_request_planner.py` | 30 | `orchestrator/request_planner.py` — segment splitting, target matching, operation normalization/fallback, `plan_assistant_operations` (no-op fake LLM). |
| `test_two_pass_fast_path.py` | 4 | `base_handler.predict_two_pass_structured` fast-path — trivial (<250-char) requests skip the reasoning pass (one LLM call), keyed on `raw_request`. |

### Diagram handlers — class diagram, guards & constraints

| File | Tests | What it verifies |
| --- | --- | --- |
| `test_class_typed_attribute_guard.py` | 6 | `ClassDiagramHandler._rewrite_class_typed_attributes` — class-typed attributes become associations; enums/primitives handled; unknown PascalCase → String. |
| `test_enum_relationship_guard.py` | 12 | `_rewrite_enum_relationships` (system) + `_rewrite_enum_relationship_mods` (modify) — enums never appear as relationship endpoints. |
| `test_relationship_name_dedup.py` | 7 | `_dedupe_relationship_names` — duplicate association names made unique (camelCase then numeric fallback); unnamed left alone. |
| `test_class_modification_missing_target.py` | 10 | `_build_model_index` / `_drop_phantom_target_ops` — deterministic missing-target detection for class removals/modifies (adds never validated). |
| `test_class_modification_names.py` | 36 | `schemas.class_diagram._is_placeholder` / `_clean_name` + `add_class` name resolution — junk placeholder names nulled, real names promoted. |
| `test_ocl_constraint_capture.py` | 10 | OCL constraint capture — `OCLConstraintSpec`, `_validate_constraints` (drops unknown-context/empty), end-to-end into `inject_complete_system`. |
| `test_prompt_snapshots.py` | 1 | Pins a hash of every prompt in `src/prompts/` (constants as they are, functions rendered with fixed samples); an unmeant byte change fails. |
| `test_prompt_caching.py` | 8 | Modification prompts are static module-level constants (no UUID/timestamp), clear the ~1024-token cache threshold, and are passed separately (not concatenated). |

### Diagram handlers — agent, BPMN, base & factory

| File | Tests | What it verifies |
| --- | --- | --- |
| `test_agent_diagram.py` | 23 | `AgentDiagramHandler` code-reply wrapping — `replyType="code"` replies always wrapped into `def name(session):`, with name-hint sanitization. |
| `test_agent_modification_safety.py` | 36 | Agent modification data-loss guards — validate mods against the *current* model; no hallucinated targets; clarify on vague input. |
| `test_agent_orphan_transition.py` | 17 | Agent `_validate_modifications` same-batch pending-rescue + orphan backstop + inherited friendly-message helpers. |
| `test_bpmn.py` | 33 | BPMN end-to-end — routing, schemas, `_validate_and_refine` structural repair, fallback, add/remove modifications, suggestions. |
| `test_base_handler.py` | 42 | `BaseDiagramHandler` utilities in depth — `validate_spec`, `clean_json_response`, `parse_json_safely`, `classify_error`, `validate_modification_spec`, `extract_name_from_request`. |
| `test_diagram_handlers.py` | 22 | `DiagramHandlerFactory` (+ `get_diagram_type_info` / metadata) plus a slice of the base-handler utilities. |
| `test_model_helpers.py` | 97 | `utilities.model_context` (compact/detailed summaries across all types, `is_diagram_nontrivial`) + `utilities.model_resolution` resolvers. |

### GUI (no-code web app)

| File | Tests | Markers | What it verifies |
| --- | --- | --- | --- |
| `test_auto_gui_message.py` | 4 | — | `confirmation._build_auto_gui_message` — the "Auto-generate GUI" completion message (names pages, counts, truncates, degrades gracefully). Runs in an isolated subprocess to sandbox a hard `baf` import. |
| `test_gui_chart_generation.py` | 42 | — | Chart/table/dashboard section generation + `utilities.class_metadata` binding. |
| `test_gui_design_system.py` | 47 | — | `gui_design_system` — every domain resolves a full theme; `stylesheet_rules`/`block_exemplars` emit `.ds-*` output; CSP-safety (no webfonts/`@import`/`url(`). |
| `test_gui_html_converter.py` | 32 | — | `gui_html_converter` — LLM HTML → GrapesJS tree, tag/class/style retention, sanitisation (script/on*/external stripped), widget-slot helpers. |
| `test_gui_modification.py` | 15 | — | GUI modification ops — deterministic fast-paths (rename/recolor/reorder), spec application, and the "LLM failure never empties the model" safety net. |
| `test_gui_phase0.py` | 15 | 1× `importorskip("openai.lib._pydantic")` | Phase 0 quick-wins — truncation salvage, `two_column`, `stats_grid` value/binding preservation. |
| `test_gui_phase3.py` | 24 | 1× `importorskip("openai.lib._pydantic")` | Phase 3 — LLM-authored `.ds-*` HTML sections + structured widget binding spliced at `<!--WIDGET:-->`, domain styles, graceful fallback. |

### Generation & Spec-Driven Agent hand-off flow

| File | Tests | What it verifies |
| --- | --- | --- |
| `test_generation_handler.py` | 69 | `generation_handler` — `detect_generator_type`, config parse/prompt, `should_route_to_generation` gatekeeper, dispatch, Django→SQL pivot-mid-config fix. |
| `test_smart_generation_handler.py` | 17 | Generation sub-routing via the unified classifier (fallback safety, dispatch) and `build_trigger_smart_generator_payload`. |
| `test_pending_dialogs.py` | 11 | `common.pending_dialogs`: every slot under one session key, `PRECEDENCE` shared by `resume_pending` and the classifier's pending question, resume fall-through, thread-safe puts, and no slot read as a session key in `src/`. |
| `test_routing_cleanup.py` | 43 | Behavioral pins for the routing cleanup: confirmation pivots, config-flow suppression + cancel, structured smart-gen recency, generator-registry sync. |
| `test_smart_generation_handler_gate.py` | 16 | The "confirm-before-smart" gate — never fires on first contact; stashes with a 30-min TTL; explicit/natural confirm fires, cancel/mixed never do. |
| `test_smart_generator_result_event.py` | 10 | `generator_result` frontend event with `metadata.smart=True` — hides cost/internal name, records outcome to memory, maps error codes to tone. |
| `test_webapp_generation_gate.py` | 5 | The web-app auto-generation **pause** — a GUI-create + `web_app` plan has the generation op stripped at source (`execution/planning.py`) and shows a generate nudge. Distinct from the smart-gen gate. |
| `test_empty_workspace_bridge.py` | 11 | Same-turn create→generate bridging — an in-turn create is written back into the snapshot so a later generate step isn't wrongly refused as "empty workspace". |
| `test_confirmation.py` | 52 | `confirmation.py` — `keyword_matches` (whole-word `\b`-bounded), `utilities.model_utils.model_has_elements`, and the REPLACE/KEEP/CANCEL/NEW_TAB keyword lists. |
| `test_suggestions.py` | 40 | `suggestions.get_suggested_actions` per diagram type/mode + `format_suggestions_as_text`, with `{label,prompt}`-only and ≤4-suggestions invariants. |

### Protocol, memory, LLM & tracking

| File | Tests | Markers | What it verifies |
| --- | --- | --- | --- |
| `test_protocol.py` | 32 | — | v2 wire-payload parsing (`protocol/adapters.py`). |
| `test_conversation_memory.py` | 14 | — | `ConversationMemory` sliding window, summarizer trigger, `build_context`, the global registry, and thread safety. |
| `test_memory_session_key.py` | 5 | — | `memory_session_key` derivation chain — `request.session_id` → payload `sessionId` → BAF `session.id` → `id(session)`. |
| `test_llm_provider.py` | 17 | — | `LLMProvider` — construction, `complete`/`parse`/`stream` delegation, token recording, error propagation, no-client fallbacks. |
| `test_llm_provider_calls.py` | 31 | — | Every migrated call site sends the request it sent before (vision, `gpt_predict_json`, handler free text, stream, structured); vision on OpenAI / Anthropic / Mistral keys on their image model, on the server key for Nebius / gateway keys, a 400 says the model can't read the image; one usage record per call; no LLM host outside `llm/`. |
| `test_byok_reach.py` | 7 | — | A user's key reaches the memory summarizer, the no-RAG UML answer and parallel plan groups (pool threads get the request's context); vision is offered only when some key can read images. |
| `test_llm_retry.py` | 8 | — | `llm.retry` — permanent quota/billing/invalid-key errors fail fast (1 call); genuine 429/5xx retry up to `MAX_ATTEMPTS`. |
| `test_byok_base_url.py` | 4 | — | BYOK custom `base_url` SSRF gate (`BESSER_AGENT_ALLOW_CUSTOM_BASE_URL`) + `redacted()`. |
| `test_model_config.py` | 6 | — | Env-driven model routing table + defaults/overrides, cost-table consistency, `supports_custom_temperature` / `reasoning_effort_for`. |
| `test_token_tracker.py` | 52 | — | `TokenTracker` — record/accumulate, per-session isolation, cost estimation, `summary()`, singleton, thread safety. |
| `test_ws_reply_outbox.py` | 6 | `importorskip("patches.websocket_platform")` | The "generate did nothing" reconnect fix — `_send`/`_buffer_reply`/`_flush_outbox` buffer on a dead conn and redeliver in order on reclaim. |
| `test_file_conversion.py` | 77 | — | `file_conversion_handler` — `detect_file_type` + PlantUML / RDF-KG / image / text → diagram spec across Class/StateMachine/Object/Agent/BPMN. |
| `test_schemas.py` | 256 | — | Every Pydantic schema under `src/schemas/` — field defaults, `Literal`/enum constraints, min-length/`ge` validators, across class/state-machine/object/agent/GUI/quantum. |

### Other files

| File | Tests | What it verifies |
| --- | --- | --- |
| `test_abstract_on_leaf_guard.py` | 9 | ``isAbstract`` on a leaf class instead of on the base. |
| `test_agent_settings.py` | 652 | AgentSettings reads every variable exactly as the code it replaced did. |
| `test_agentic_bpmn.py` | 28 | Agentic BPMN: mode selection, prompt composition, the validated generation path, and project-scoped agentDiagramRef / reflectionReviewerLaneId. |
| `test_agentic_schemas.py` | 33 | Schema guarantees for agentic BPMN, Component and Deployment structured output. |
| `test_association_class_link_guard.py` | 19 | One bad token in the LLM's output (an association class naming a missing class, ...) drops only its member, not the whole model. |
| `test_association_end_guard.py` | 11 | Association-end uniqueness at CREATION time. |
| `test_auto_fix_route.py` | 3 | [auto-fix] protocol marker: the editor's validate-and-repair loop sends machine-generated repair requests. |
| `test_bpmn_batch_refs.py` | 2 | The BPMN reference guardrail keeps flows to a task added earlier in the same modification batch. |
| `test_byok_anthropic_caching.py` | 2 | Anthropic BYOK structured calls put the static prefix in a cached system block. |
| `test_byok_structured.py` | 17 | Structured generation must run on the user's own key when they saved one. |
| `test_byok_tier_routing.py` | 19 | BYOK routes each call by the model_config tier it requests. |
| `test_class_diagram_fidelity_rules.py` | 14 | Class-diagram fidelity rules from a hotel-booking generation (Qwen3-30B). |
| `test_class_identifier_sanitize.py` | 7 | Tests for the identifier-name sanitizer guard. |
| `test_class_refactor_operations.py` | 9 | extract_class / merge_classes / promote_attribute on the modify path (#35). |
| `test_claude_sampling_params.py` | 23 | Claude models that reject sampling params never receive them. |
| `test_compact_spec.py` | 27 | Compact structured-output spec: string decoding + expansion + handler wiring. |
| `test_component_deployment_registration.py` | 17 | ComponentDiagram / DeploymentDiagram must be registered in every routing and capability point, not only in the handler factory (see CLAUDE.md checklist). |
| `test_component_deployment_routing.py` | 24 | Layer-2 routing for Component / Deployment diagrams. |
| `test_component_diagram.py` | 12 | Tests for ComponentDiagramHandler. |
| `test_container_config.py` | 5 | The container's config.yaml is byte-identical to what the old Dockerfile heredoc wrote, and its log copy redacts the key the way the old ``sed`` did. |
| `test_contradiction_guard.py` | 2 | Self-contradiction guard: a CREATE request that negates the very content it asks to create ("a class diagram with no classes", "an empty diagram", "don't model anything") can't yield a real model — the agent should clarify instead of building a token 1-class model. |
| `test_cost_accounting.py` | 14 | Token/cost accounting matches what the providers bill. |
| `test_delete_reconnects_orphans.py` | 5 | Deleting a class must not silently orphan the ones it connected. |
| `test_deployment_diagram.py` | 12 | Tests for DeploymentDiagramHandler. |
| `test_duplicate_end_autofix.py` | 15 | Deterministic repair for "two association ends with the same name". |
| `test_editor_contract.py` | 17 | Shared contract between the agent's structured output and the editor. |
| `test_error_fallback_budget.py` | 6 | A provider error must not fan out into the per-class incremental fallback. |
| `test_generate_button_guard.py` | 7 | The post-build "Generate the code" button must reach the Spec-Driven flow. |
| `test_github_import_route.py` | 27 | Continue-from-GitHub chat path: "continue from github.com/x/y" must emit a ``trigger_github_import`` action (the frontend calls the backend import endpoint, loads the project, and arms the modify machinery — the agent only detects the intent and extracts owner/repo/branch). |
| `test_gui_actions.py` | 14 | 'Experimental AI design' GUIs must have working links and buttons. |
| `test_gui_agent_overhaul.py` | 17 | GUI agent overhaul: structured generation, custom themes, non-destructive failure, and the Studio-grade modification path (outline + batch + edit_section). |
| `test_gui_aggregation.py` | 6 | AI-designed charts and metric cards use BESSER's aggregation. |
| `test_gui_crud_forms.py` | 15 | AI-designed record buttons and forms act on the class's data. |
| `test_gui_from_mockup.py` | 16 | A screenshot or mockup attached in the chat becomes a GUI (#643). |
| `test_gui_generation_routing.py` | 5 | 'Generate the GUI' must route to GUI-diagram creation, never the web-app code generator. |
| `test_gui_modify_actions.py` | 18 | Modifying an AI-designed GUI must wire new content like generation does. |
| `test_gui_reasoning_context.py` | 2 | The design-reasoning pass must plan against the real class diagram. |
| `test_gui_widgets.py` | 7 | AI-designed data widgets must be specific and bound to the right class. |
| `test_handler_envelopes.py` | 153 | Every handler's reply envelope, byte for byte, under a fake LLM provider. |
| `test_hotel_model_contract.py` | 15 | The hotel prose end to end: agent spec -> editor JSON -> BUML -> model contract. |
| `test_inheritance_direction_guard.py` | 11 | An abstract base declared as the CHILD of its own subclasses. |
| `test_inherited_end_collision.py` | 6 | Association ends that collide through an inheritance hierarchy. |
| `test_intent_reconcile.py` | 4 | reconcile_intent: the unified classifier is the authoritative create-vs-modify router. |
| `test_json_params_temperature.py` | 17 | JSON-mode calls must not send a temperature the model rejects. |
| `test_json_repair.py` | 42 | Tolerant parsing of LLM JSON: a stray regex backslash must not discard a model. |
| `test_mandatory_cycle_guard.py` | 21 | Mandatory creation cycles (neither instance can be created first) are broken before BESSER's model-contract check rejects them. |
| `test_member_name_lookup.py` | 18 | A method must be findable by its bare name. |
| `test_mixed_plan_generation_pause.py` | 21 | Mixed "design X and generate Y" plans pause before generating — for EVERY generator type, not just web_app. |
| `test_model_catalog.py` | 12 | The deployment's model catalog: gateway endpoints and rates, shared with BESSER. |
| `test_model_refusal.py` | 21 | A model that declines a request must be reported as a refusal, not a failure. |
| `test_modification_type_normalisation.py` | 8 | Types arriving through a modification get the same guards as a full generation (``LocalDate`` becomes ``date`` on both paths). |
| `test_multiplicity_guidance.py` | 10 | The prompt must tell the model to read cardinality out of the user's prose. |
| `test_new_llm_models.py` | 39 | GPT-6 and the current Claude models get valid request params and real prices. |
| `test_object_reference_ids.py` | 5 | Object-diagram classId / attributeId come from the reference class diagram. |
| `test_original_request_passthrough.py` | 19 | The run must receive the user's spec, not only a summary of it. |
| `test_packaging.py` | 10 | Packaging: src/ is the one import root, and pyproject.toml declares all of it. |
| `test_parallel_association_merge.py` | 71 | Duplicate associations for one fact (two FKs, or an FK plus a join table) are merged; genuinely distinct ones are kept. |
| `test_parallel_plan_dialogs.py` | 3 | A parallel plan opens one dialog at a time. |
| `test_per_instance_multiplicity.py` | 36 | Per-instance multiplicity encoding in the compact class-diagram spec. |
| `test_pilot_telemetry.py` | 14 | Opt-in study telemetry: context propagation + fire-and-forget posting. |
| `test_progress_wording.py` | 10 | Progress lines for a complete build name the diagram being built. |
| `test_protocol_actions.py` | 11 | protocol.actions: the wire action names and the envelope builders. |
| `test_reference_guardrail.py` | 13 | The shared reference guardrail (BaseDiagramHandler._validate_mod_refs). |
| `test_reply_replay.py` | 7 | Reconnect recovery: buffer the last terminal reply and replay it on demand. |
| `test_reply_turn_replay.py` | 21 | Turn-scoped reply replay. |
| `test_run8_spec_normalization.py` | 12 | End-to-end normalization of a hotel spec as the LLM actually produced it. |
| `test_session_store.py` | 3 | The session store: the per-session stores that expire entries register their expiry, and the reaper's one ``expire`` runs them all. |
| `test_shadowed_attributes_guard.py` | 10 | Shadowed-attribute guard: a subclass must not redefine an attribute an ancestor already declares (BUML validates attribute shadowing). |
| `test_shared_client_retry_budget.py` | 1 | The shared client's transport retries must not nest under llm_retry. |
| `test_spec_fidelity_guidance.py` | 21 | The prompt must carry over what the user actually stated. |
| `test_structured_truncation.py` | 2 | Truncated structured output: retry once with the concise instruction. |
| `test_target_kind.py` | 60 | Any target, not only web apps: the classifier names it, it is remembered per project, and the planner, confirmation and replies honour it. |
| `test_type_alias_normalisation.py` | 23 | A Java-flavoured type name must not silently become a string. |
| `test_uml_rag_indexing.py` | 4 | The UML-spec RAG must load its PDF: nothing used to, so every container's vector store stayed empty and "according to the UML spec" answers had no context. |
| `test_undeclared_enumeration_guard.py` | 17 | A closed value set must survive the model forgetting to declare its enum. |
| `test_user_profile.py` | 20 | Tests for the User Profile (UserDiagram) handler, schemas, and routing. |
| `test_ux_pilot_fixes.py` | 28 | Focused tests for UX fixes: new-diagram-tab wording, the AI-Generated screen-creation suggestion, and the clickable API-key link. |
| `test_ws_reply_slot.py` | 6 | Reply-slot ownership across browser tabs. |
| `test_ws_session_key.py` | 3 | Session keying on connect. |

### Live harnesses (`tests/live/`, manual)

| File | What it does | Notes |
| --- | --- | --- |
| `test_nl_generation_scenarios.py` | NL→generator **routing matrix** — drives real phrasings ("generate a database", …) and asserts each lands on an acceptable generator and never a forbidden one (e.g. database requests must never hit `django`). | Hosts the shared `_send` / `_wait_meaningful` WebSocket helpers imported by both sweeps. Only pytest-collected live test (1 test, skipped unless `RUN_LIVE_AGENT_TESTS=1`). |
| `wme_100_sweep.py` | Comprehensive ~97-scenario sweep: `system / webapp / other / modify / generate / vague / edge / meta`, classified ok/FLAW/FAIL. | Consolidates the earlier sweeps (its `edge` category replaces the old standalone edge-case script). **Imports `_flaws_for_system` from `wme_release_sweep.py`.** |
| `wme_release_sweep.py` | 28-scenario fidelity sweep (`system / webapp / other`) checking the "useless model" flaws — isolated classes, dup names, dangling/enum endpoints, web-app auto-run. | **Hosts `_flaws_for_system`**, the fidelity checker reused by `wme_100_sweep.py`. |
