Testing
=======

This section covers how to run the test suite, what each test file covers,
the test infrastructure (fakes and fixtures), and how to write new tests.

.. contents:: On this page
   :local:
   :depth: 2

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
-----------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - What It Tests
   * - ``test_protocol.py``
     - Request parsing, v2 envelope unwrapping, diagram type extraction
   * - ``test_generation_handler.py``
     - Generator detection, keyword matching, config parsing, safety nets,
       modeling request guards, diagram creation detection, pattern-based
       domain detection, cross-validation logic
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
-------------------

All tests use lightweight fakes from ``tests/conftest.py``. These avoid the
need for a running LLM, WebSocket server, or the BESSER framework at test time.

FakeSession
~~~~~~~~~~~

Stand-in for ``besser.agent.core.session.Session``. Stores key-value pairs and
captures replies for assertion:

.. code-block:: python

   session = FakeSession()
   session.set("my_key", "my_value")
   assert session.get("my_key") == "my_value"

   # Capture replies
   session.reply("hello")
   assert session.replies == ["hello"]
   assert session.last_reply_json() == None  # not valid JSON

FakeLLM
~~~~~~~

Stub LLM that returns canned responses. Logs all prompts it receives:

.. code-block:: python

   llm = FakeLLM('{"classes": []}')
   result = llm.predict("generate a class diagram")
   assert result == '{"classes": []}'
   assert llm.call_log == ["generate a class diagram"]

make_session()
~~~~~~~~~~~~~~

Create a ``FakeSession`` pre-loaded with a v2 payload:

.. code-block:: python

   session = make_session("create a User class", diagram_type="ClassDiagram")

_make_request()
~~~~~~~~~~~~~~~

Create an ``AssistantRequest`` for direct function testing:

.. code-block:: python

   from protocol.types import AssistantRequest, WorkspaceContext

   def _make_request(message, action="user_message"):
       return AssistantRequest(
           action=action,
           message=message,
           context=WorkspaceContext(
               project_snapshot={"name": "TestProject", "diagrams": {}},
           ),
       )


Writing New Tests
-----------------

Guidelines
~~~~~~~~~~

1. Import from the module under test and ``tests.conftest``
2. Use ``_make_request()`` helpers for creating ``AssistantRequest`` objects
3. Use ``FakeSession`` for session state
4. **Test both positive and negative cases** — every ``is True`` needs a
   corresponding ``is False``
5. Test edge cases: empty strings, ``None`` values, plurals, unusual phrasing

Test Organization
~~~~~~~~~~~~~~~~~

- Group related tests in a ``class Test<Feature>``
- Name tests descriptively: ``test_generate_class_diagram_not_routed``
- Use docstrings for non-obvious test rationale

Example: Testing a New Pre-Filter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class TestMyNewFilter:
       def test_positive_case(self):
           """Describe what should match."""
           assert my_filter("create a class diagram") is True

       def test_negative_case(self):
           """Describe what should NOT match."""
           assert my_filter("generate django code") is False

       def test_edge_case_empty(self):
           assert my_filter("") is False

       def test_edge_case_none(self):
           assert my_filter(None) is False

       def test_similar_phrase_that_should_not_match(self):
           """Regression guard for known confusable phrase."""
           assert my_filter("go back to the backend") is False
