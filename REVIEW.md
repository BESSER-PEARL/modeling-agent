# Review instructions

These rules apply to automated pull request review. Project conventions live in `CLAUDE.md`; read it as the
reference for every finding below.

## What this review enforces

A change must fit the codebase as it already is. Grade it against this repository's existing patterns, not a
generic best-practice rubric.

- **Follow the existing convention.** When the codebase already has a way to do something, a new but equivalent
  way is a finding: session-state keys from `session_keys.py` (never literals), tunables in `agent_config.py`,
  model tiers in `model_config.py`, the reply and stream helpers in `session_helpers.py`, handlers built on
  `BaseDiagramHandler`, test doubles from `tests/conftest.py` (`FakeSession`, `FakeLLM`, `make_v2_payload()`).
- **Fix routing in the layer that owns it.** Layer 1 (`unified_classifier.py`) decides the intent and every
  sub-routing field; layer 2 (`workspace_orchestrator.py`) picks the diagram type. A keyword special case in
  a state body that overrides the classifier, or a second classification prompt, is a finding.
- **No papering over.** Flag an empty `except` that swallows an error, a deterministic repair pass added where
  the prompt should be fixed (repair is only for invariants the LLM statistically violates, with an
  evidence-grounded heuristic), a duplicated helper instead of an import, and a test made to pass by
  loosening its assertion.
- **Cite the canonical example.** A convention finding names the file that shows the correct pattern. If you
  cannot point to one, do not post the finding.

## What Important means here

Important is for findings that break behavior, the wire protocol, or a security rule:

- incorrect logic, crashes, a reply that never reaches the client, a session or memory leak
- a wire-protocol change (action names, envelope shape, double encoding) without the matching frontend change in
  BESSER-WEB-MODELING-EDITOR (`features/assistant/services/`); say which file there must change
- a side-effect action (`trigger_smart_generator`, `trigger_deploy`, `trigger_export`, ...) that can be triggered
  from model prose instead of a structured, classified intent
- an API key or BYOK value logged, persisted, or leaked across sessions (`byok.py` scopes it per request)
- `temperature` or `reasoning_effort` hardcoded at a call site instead of `supports_custom_temperature()` /
  `reasoning_effort_for()`
- a new `MODEL_*` default missing from `_COST_PER_1K` in `tracking/token_tracker.py`
- the structured `gpt` and free-text `gpt_text` instances swapped
- an LLM asked to emit positions (layout belongs to `core/layout_engine.py`)
- `patches/websocket_platform.py` changed, or the BAF pin in `requirements.txt` changed without re-vendoring it

Missing tests for a routing or handler change are Important. Cosmetic convention breaks, naming, and style
are Nit at most.

## Always check

- **New or changed diagram type**: every registration point listed under "Adding (or Auditing) a Diagram
  Type" in `CLAUDE.md` is covered; list the missed ones by number.
- **`_IMPLICIT_PATTERNS` order**: a new pattern sits where a broader pattern above it will not steal its
  vocabulary.
- **Two-pass threshold**: `_TWO_PASS_MIN_LENGTH` is measured on the raw user message, not the enriched prompt.
- **Tests**: handler logic tested as pure functions on hand-built dicts (see `tests/test_bpmn.py`); routing
  changes covered in `tests/test_unified_classifier.py`; shared handler behavior in
  `tests/test_diagram_handlers.py`.
- **Docs**: routing, protocol, and handler changes update the matching page in `docs/source/`
  (`intent_recognition.rst`, `websocket_protocol.rst`, `diagram_handlers.rst`, `configuration.rst`).
- **Config**: a new required setting is added to `config_example.yaml` and the Docker entrypoint; real keys never
  appear in `config.yaml` or `.env` in the diff.

## Do not report

- Prompt wording and LLM output quality, unless the change contradicts a rule in `CLAUDE.md`. Unit review cannot
  measure generation quality; the live probes in `tests/live/` do.
- `uml_specs/` and other data files, and `ca-certs-extra/`.
- Generic advice such as "consider more tests" without a concrete path.

## Cap the nits

Report at most five Nits per review.
