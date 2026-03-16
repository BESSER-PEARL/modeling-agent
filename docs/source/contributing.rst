Contributing
============

Thank you for your interest in contributing to the Modeling Agent.

.. contents:: On this page
   :local:
   :depth: 2

Development Setup
-----------------

.. code-block:: bash

   git clone <repository-url>
   cd modeling-agent
   python -m venv venv
   source venv/bin/activate  # or .\\venv\\Scripts\\Activate.ps1 on Windows
   pip install -r requirements.txt

Running Tests
-------------

.. code-block:: bash

   # Full test suite
   python -m pytest

   # With verbose output
   python -m pytest -v

   # Specific suite
   python -m pytest tests/test_diagram_handlers.py
   python -m pytest tests/test_protocol.py
   python -m pytest tests/test_request_planner.py

Test Suites
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Coverage
   * - ``test_protocol.py``
     - Request parsing, v2 envelope unwrapping
   * - ``test_diagram_handlers.py``
     - Handler generation for all 6 diagram types
   * - ``test_file_conversion.py``
     - PlantUML, KG, image, text file conversions
   * - ``test_generation_handler.py``
     - Code generator routing and config parsing
   * - ``test_gui_chart_generation.py``
     - GUI chart binding and color palettes
   * - ``test_model_helpers.py``
     - Utility function correctness
   * - ``test_orchestrator.py``
     - Diagram type resolution and scoring
   * - ``test_request_planner.py``
     - Multi-step operation planning
   * - ``test_base_handler.py``
     - Base handler utilities (cache stubs, JSON parsing, error classification)
   * - ``test_conversation_memory.py``
     - Conversation memory (sliding window, summarization, thread safety)
   * - ``test_llm_provider.py``
     - LLM provider abstraction
   * - ``test_schemas.py``
     - Pydantic schema validation for all diagram types
   * - ``test_suggestions.py``
     - Suggestion engine (context-aware, per-diagram-type)
   * - ``test_token_tracker.py``
     - Token counting and cost tracking

Test Infrastructure
~~~~~~~~~~~~~~~~~~~

Tests use lightweight fakes defined in ``tests/conftest.py``:

- **FakeSession** — Stand-in for ``besser.agent.core.session.Session`` with
  key-value store and reply capture.
- **FakeLLM** — Stub LLM returning round-robin responses with prompt logging.
- **make_v2_payload()** — Build wrapped v2 protocol payloads for testing.
- **make_session()** — Pre-load a session with a v2 payload.

Building Documentation
----------------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   cd docs

   # Windows
   make.bat html

   # Linux/macOS
   make html

Output goes to ``docs/build/html/``.

Guidelines
----------

Code Changes
~~~~~~~~~~~~

- Keep behavior changes synchronized across ``src/``, ``tests/``, and
  ``docs/source/``.
- Prefer deterministic handler outputs and shared helper functions under
  ``src/utilities/``.
- Keep backward-compatibility shims when moving modules used by imports/tests.

Adding a New Diagram Type
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create a handler class extending ``BaseDiagramHandler`` in
   ``src/diagram_handlers/types/``.
2. Implement the 5 required abstract methods:

   - ``get_diagram_type()``
   - ``get_system_prompt()``
   - ``generate_single_element()``
   - ``generate_complete_system()``
   - ``generate_fallback_element()``

3. Register in ``DiagramHandlerFactory.__init__``
   (``src/diagram_handlers/registry/factory.py``).
4. Add to ``SUPPORTED_DIAGRAM_TYPES`` in ``src/protocol/types.py``.
5. Add display metadata in ``src/diagram_handlers/registry/metadata.py``.
6. Add tests in ``tests/test_diagram_handlers.py``.
7. Update documentation in ``docs/source/diagram_handlers.rst``.

Adding a New Generator
~~~~~~~~~~~~~~~~~~~~~~

1. Add keywords to ``GENERATOR_KEYWORDS`` in
   ``src/handlers/generation_handler.py``.
2. Add prerequisites to ``GENERATOR_PREREQUISITES`` in
   ``src/orchestrator/request_planner.py``.
3. Add inline config parsing if needed.
4. Add tests in ``tests/test_generation_handler.py``.
5. Update ``docs/source/usage.rst``.

Pull Request Process
~~~~~~~~~~~~~~~~~~~~

1. Create a feature branch from ``main``.
2. Make your changes with tests.
3. Ensure ``python -m pytest`` passes.
4. Update documentation if behavior changed.
5. Submit a pull request using the template in ``.github/pull_request_template.md``.

Code of Conduct
---------------

This project follows the code of conduct defined in ``CODE_OF_CONDUCT.md``.
Please read it before contributing.
