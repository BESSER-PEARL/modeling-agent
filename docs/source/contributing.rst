Contributing
============

Thank you for your interest in contributing to the Modeling Agent. This guide
covers everything you need to know to get started, from development setup to
submitting pull requests.

.. contents:: On this page
   :local:
   :depth: 2

Development Setup
-----------------

Prerequisites
~~~~~~~~~~~~~

- Python 3.11+
- An OpenAI API key (for GPT-4.1-mini)
- Git

Installation
~~~~~~~~~~~~

.. code-block:: bash

   git clone <repository-url>
   cd modeling-agent
   python -m venv venv
   source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows
   pip install -r requirements.txt

   # Copy and configure
   cp config_example.yaml config.yaml
   # Edit config.yaml with your OpenAI API key

Verify your setup:

.. code-block:: bash

   python -m pytest -v
   python modeling_agent.py  # Should start WebSocket on :8765


Project Structure
-----------------

.. code-block:: text

   modeling-agent/
   ├── modeling_agent.py          # Entry point: defines intents, states, wires transitions
   ├── config.yaml                # Runtime configuration (API keys, LLM settings)
   ├── config_example.yaml        # Template for config.yaml
   │
   ├── src/                       # All source code
   │   ├── protocol/              # WebSocket protocol layer
   │   │   ├── types.py           #   AssistantRequest, WorkspaceContext, SUPPORTED_DIAGRAM_TYPES
   │   │   └── adapters.py        #   parse_assistant_request(), v2 envelope unwrapping
   │   │
   │   ├── handlers/              # Request handlers
   │   │   └── generation_handler.py  # Code generation routing, config parsing, safety guards
   │   │
   │   ├── diagram_handlers/      # Diagram-specific handlers
   │   │   ├── core/              #   BaseDiagramHandler, layout engine, shared logic
   │   │   ├── types/             #   ClassDiagramHandler, StateMachineHandler, etc.
   │   │   └── registry/          #   DiagramHandlerFactory, metadata
   │   │
   │   ├── orchestrator/          # Multi-step planning and diagram type resolution
   │   │   ├── request_planner.py #   3-tier planner (heuristic → keyword → LLM)
   │   │   └── workspace_orchestrator.py  # Diagram type resolution (keywords → patterns → context)
   │   │
   │   ├── routing/               # Intent constants and routing helpers
   │   ├── state_bodies.py        # State body functions + transition wiring
   │   ├── session_helpers.py     # Reply helpers, streaming, intent matching conditions
   │   ├── execution.py           # Operation execution engine
   │   ├── agent_setup.py         # LLM initialization, classifier configuration
   │   ├── memory.py              # Conversation memory (sliding window + summarization)
   │   ├── tracking.py            # Token counting and cost tracking
   │   └── suggestions.py         # Context-aware "What's next?" suggestions
   │
   ├── tests/                     # Test suite
   │   ├── conftest.py            #   FakeSession, FakeLLM, make_v2_payload, fixtures
   │   ├── test_generation_handler.py
   │   ├── test_orchestrator.py
   │   ├── test_request_planner.py
   │   ├── test_diagram_handlers.py
   │   ├── test_protocol.py
   │   └── ...                    #   14 test files total
   │
   └── docs/                      # Sphinx documentation
       └── source/                #   RST files → ReadTheDocs


Running Tests
-------------

.. code-block:: bash

   # Full suite
   python -m pytest

   # Verbose with test names
   python -m pytest -v

   # Specific file
   python -m pytest tests/test_generation_handler.py

   # Specific test class
   python -m pytest tests/test_generation_handler.py::TestDetectGeneratorType

   # Specific test
   python -m pytest tests/test_generation_handler.py::TestDetectGeneratorType::test_detects_django -v

Test Suites
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - What It Tests
   * - ``test_protocol.py``
     - Request parsing, v2 envelope unwrapping, diagram type extraction
   * - ``test_generation_handler.py``
     - Generator detection, keyword matching, config parsing, safety nets,
       modeling request guards, diagram creation detection
   * - ``test_orchestrator.py``
     - Diagram type resolution (explicit keywords, discriminating patterns,
       context fallback)
   * - ``test_request_planner.py``
     - Multi-step operation planning, heuristic decomposition, LLM planner
       decision logic
   * - ``test_diagram_handlers.py``
     - Handler generation for all 6 diagram types
   * - ``test_schemas.py``
     - Pydantic schema validation for all diagram types
   * - ``test_base_handler.py``
     - Base handler utilities (cache stubs, JSON parsing, error classification)
   * - ``test_conversation_memory.py``
     - Conversation memory (sliding window, summarization, thread safety)
   * - ``test_file_conversion.py``
     - PlantUML, KG, image, text file conversions
   * - ``test_gui_chart_generation.py``
     - GUI chart binding and color palettes
   * - ``test_model_helpers.py``
     - Utility function correctness
   * - ``test_llm_provider.py``
     - LLM provider abstraction
   * - ``test_suggestions.py``
     - Suggestion engine (context-aware, per-diagram-type)
   * - ``test_token_tracker.py``
     - Token counting and cost tracking
   * - ``test_confirmation.py``
     - Confirmation flow logic (replace/keep/merge)

Test Infrastructure
~~~~~~~~~~~~~~~~~~~

All tests use lightweight fakes from ``tests/conftest.py``:

**FakeSession** — Stand-in for ``besser.agent.core.session.Session``:

.. code-block:: python

   session = FakeSession()
   session.set("my_key", "my_value")
   assert session.get("my_key") == "my_value"

   # Capture replies
   session.reply("hello")
   assert session.replies == ["hello"]
   assert session.last_reply_json() == None  # not valid JSON

**FakeLLM** — Stub LLM returning canned responses:

.. code-block:: python

   llm = FakeLLM('{"classes": []}')
   result = llm.predict("generate a class diagram")
   assert result == '{"classes": []}'
   assert llm.call_log == ["generate a class diagram"]

**make_session()** — Create a session pre-loaded with a v2 payload:

.. code-block:: python

   session = make_session("create a User class", diagram_type="ClassDiagram")

Writing New Tests
~~~~~~~~~~~~~~~~~

1. Import from the module under test and ``tests.conftest``
2. Use ``_make_request()`` helpers for creating ``AssistantRequest`` objects
3. Use ``FakeSession`` for session state
4. Test both positive and negative cases
5. Test edge cases (empty strings, None values, plurals)


How to Add a New Diagram Type
------------------------------

1. **Create handler** in ``src/diagram_handlers/types/``:

   .. code-block:: python

      class MyDiagramHandler(BaseDiagramHandler):
          def get_diagram_type(self) -> str:
              return "MyDiagram"

          def get_system_prompt(self) -> str:
              return "You are a MyDiagram expert..."

          def generate_single_element(self, request, model) -> dict: ...
          def generate_complete_system(self, request, model) -> dict: ...
          def generate_fallback_element(self, request) -> dict: ...

2. **Register** in ``src/diagram_handlers/registry/factory.py`` →
   ``DiagramHandlerFactory.__init__``

3. **Add type** to ``SUPPORTED_DIAGRAM_TYPES`` in ``src/protocol/types.py``

4. **Add display metadata** in ``src/diagram_handlers/registry/metadata.py``

5. **Add explicit keywords** to ``KEYWORD_TARGETS`` in
   ``src/orchestrator/workspace_orchestrator.py``

6. **Add discriminating pattern** to ``_IMPLICIT_PATTERNS`` in the same file

7. **Add tests** in ``tests/test_diagram_handlers.py``

8. **Update docs** in ``docs/source/diagram_handlers.rst``


How to Add a New Generator
--------------------------

1. **Add keywords** to ``GENERATOR_KEYWORDS`` in
   ``src/handlers/generation_handler.py``:

   .. code-block:: python

      GENERATOR_KEYWORDS: Dict[str, List[str]] = {
          ...
          "my_gen": ["my generator", "generate my_gen"],
      }

2. **Add required fields** (if any) to ``GENERATOR_REQUIRED_FIELDS``:

   .. code-block:: python

      GENERATOR_REQUIRED_FIELDS["my_gen"] = ["setting1", "setting2"]

3. **Add inline config parsing** in ``parse_inline_generator_config()``

4. **Add prerequisites** to ``GENERATOR_PREREQUISITES`` in
   ``src/orchestrator/request_planner.py``

5. **Add config prompt** in ``_build_config_prompt()``

6. **Add tests** in ``tests/test_generation_handler.py``

7. **Update docs** in ``docs/source/usage.rst``


How to Add a New Intent
-----------------------

1. **Define the intent** in ``modeling_agent.py``:

   .. code-block:: python

      my_intent = agent.new_intent(
          name="my_intent",
          description="When the user wants to do X. Keywords: ..."
      )

   .. warning::

      Intent descriptions are the **primary signal** for the LLM classifier.
      Include explicit positive examples, negative examples (what it's NOT),
      and disambiguation rules for confusable intents.

2. **Create a state** in ``modeling_agent.py``:

   .. code-block:: python

      my_state = agent.new_state("my_state")

3. **Write the state body** in ``src/state_bodies.py``:

   .. code-block:: python

      def my_body(session: Session):
          request = parse_assistant_request(session)
          # ... handle the intent ...
          reply_message(session, "Done!")

4. **Register** in ``register_all()`` (same file):

   - Add to ``states`` dict
   - Add to ``intents`` dict
   - Add to ``intent_map``

5. **Add tests** for the state body logic


How to Modify Intent Recognition
---------------------------------

The intent recognition system has multiple layers. Choose the right one:

**To fix a misclassification for a specific phrase:**

- First, update the **intent description** in ``modeling_agent.py`` with an
  explicit example. This is the highest-impact, lowest-risk change.

**To add a pre-filter guard (zero latency):**

- Add logic to ``_is_modeling_request()`` or ``_is_diagram_creation_request()``
  in ``src/handlers/generation_handler.py``.
- These run before the LLM and catch obvious patterns.

**To add a generator keyword:**

- Add to ``GENERATOR_KEYWORDS`` in ``src/handlers/generation_handler.py``.
- If the keyword is short/ambiguous (≤6 chars), add it to ``_BOUNDARY_KEYWORDS``
  for word-boundary matching.

**To add a diagram type keyword:**

- For exact phrases: add to ``KEYWORD_TARGETS`` in
  ``src/orchestrator/workspace_orchestrator.py``.
- For discriminating patterns: add to ``_IMPLICIT_PATTERNS`` (same file).

**To change transition routing:**

- Modify ``add_unified_transitions()`` in ``src/state_bodies.py``.

.. warning::

   Always test with both the intended phrase AND similar phrases that should
   NOT match. For example, when adding "backend" as a generator keyword,
   verify that "go back to the backend concept" does NOT trigger generation.


Code Style & Conventions
------------------------

- **Imports**: Standard library → third-party → local, separated by blank lines
- **Type hints**: Use for function signatures; skip for obvious local variables
- **Error handling**: Only at system boundaries (user input, LLM responses,
  external APIs). Trust internal code.
- **Naming**: ``snake_case`` for functions/variables, ``PascalCase`` for classes,
  ``UPPER_CASE`` for constants
- **Comments**: Only where the logic isn't self-evident. No docstrings for
  obvious methods.
- **No premature abstraction**: Three similar lines is better than a premature
  helper function


Debugging Tips
--------------

**Intent misclassification:**

1. Check what intent GPT-4.1-mini predicted (enable debug logging)
2. Test ``_is_modeling_request(msg)`` and ``_is_diagram_creation_request(msg)``
3. Check ``detect_generator_type(msg)`` for false positives
4. Verify the intent description covers the failing case

**Request not reaching the right handler:**

1. Check transition priority (Priority 1: intent match, Priority 2: generation
   route, Priority 4: fallback)
2. Check if ``pending_generator_type`` or ``pending_complete_system`` is set on
   the session (these suppress intent matching)

**Wrong diagram type selected:**

1. Check ``_collect_explicit_targets(msg)`` for keyword matches
2. Check ``_rank_implicit_targets(msg)`` for pattern matches
3. Check ``determine_target_diagram_type(request)`` for the final result


Common Pitfalls
---------------

1. **"generate" is ambiguous**: ``"generate django"`` = code generation, but
   ``"generate a class diagram"`` = diagram creation. Always test both.

2. **Substring matching**: ``"sql"`` matches inside ``"sqlalchemy"``. Use
   ``_BOUNDARY_KEYWORDS`` for short keywords.

3. **Pending state suppresses intent matching**: When
   ``pending_generator_type`` is set, ``json_intent_matches()`` returns
   ``False`` for ALL intents, so the message stays in the generation state.

4. **Frontend context can be stale**: After injecting a diagram, the next
   message from the frontend may carry the pre-injection model snapshot.

5. **Dict ordering in GENERATOR_KEYWORDS matters**: Keywords are checked in
   insertion order. ``"sqlalchemy"`` must come before ``"sql"`` to avoid
   the shorter keyword matching first.


Pull Request Process
--------------------

1. Create a feature branch from ``main``
2. Make your changes with tests
3. Ensure ``python -m pytest`` passes
4. Update documentation if behavior changed
5. Submit a PR using the template in ``.github/pull_request_template.md``

**Commit message format**: Use conventional commits — ``fix:``, ``feat:``,
``refactor:``, ``docs:``, ``test:``

**What reviewers look for**:

- Tests cover both positive and negative cases
- No regressions in existing tests
- Intent descriptions updated if classification behavior changed
- Documentation updated for user-visible changes

Code of Conduct
---------------

This project follows the code of conduct defined in ``CODE_OF_CONDUCT.md``.
Please read it before contributing.
