Getting Started
===============

Overview
--------

The BESSER Modeling Agent is the conversational AI backend for the
`BESSER Web Modeling Editor <https://editor.besser-pearl.org>`_. It receives
user requests over WebSocket, normalizes them into a unified protocol, plans one
or more operations, and returns structured responses for model updates or
code-generation triggers via
`BESSER generators <https://besser-pearl.github.io/BESSER/generators.html>`_.

Key capabilities:

- Diagram creation and modification via natural language.
- Multi-operation orchestration (modeling + generation in a single request).
- Code generation: BESSER's deterministic generators, plus a hand-off to the
  LLM-authored **Spec-Driven Agent** for stacks BESSER has no built-in generator for.
- Bring-your-own-key routing, so generation can run on the user's own OpenAI,
  Anthropic or Mistral key. See :doc:`configuration`.
- UML specification Q&A with RAG (Retrieval-Augmented Generation) over the OMG
  UML 2.5.1 specification. See :doc:`configuration` for RAG setup.
- File conversion from PlantUML, knowledge-graph files, XMI, PDFs, images, and
  plain text.
- Voice input via OpenAI speech-to-text.

For a detailed walkthrough of the request lifecycle, see :doc:`end_to_end_flow`.

Supported Diagram Types
-----------------------

The identifier is the token the protocol uses — see :doc:`diagram_handlers` for
why two of them do not end in ``Diagram``.

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Diagram Type
     - Description
     - Output Format
   * - ``ClassDiagram``
     - UML class diagrams
     - `Apollon <https://apollon-library.readthedocs.io/>`_-compatible JSON
   * - ``ObjectDiagram``
     - UML object/instance diagrams
     - Apollon-compatible JSON
   * - ``StateMachineDiagram``
     - UML state machine diagrams
     - Apollon-compatible JSON
   * - ``AgentDiagram``
     - BESSER conversational agent diagrams
     - Custom state/intent JSON
   * - ``GUINoCodeDiagram``
     - No-code GUI models
     - GrapesJS project JSON
   * - ``QuantumCircuitDiagram``
     - Quantum circuit diagrams
     - Quirk-format JSON
   * - ``BPMN``
     - BPMN process diagrams, with optional pools and lanes
     - BPMN node/flow spec (the editor lays it out)
   * - ``UserDiagram``
     - BESSER user-profile models — a target user as attribute-matching
       criteria drawn from a bundled metamodel
     - Object-diagram-shaped spec with comparison operators

Prerequisites
-------------

- Python 3.11 (3.10 minimum).
- An OpenAI API key. The default model tiers are ``gpt-4o-mini`` for routing
  and the gpt-5.6 family for generation — all env-overridable, see
  :doc:`configuration`.

Install
-------

.. code-block:: bash

   python -m venv .venv

   # Windows PowerShell
   .\\.venv\\Scripts\\Activate.ps1

   # Linux/macOS
   source .venv/bin/activate

   python -m pip install --upgrade pip
   pip install -r requirements.txt

Configuration
-------------

1. Copy ``config_example.yaml`` to ``config.yaml``.
2. Set ``nlp.openai.api_key`` with your OpenAI key.

.. code-block:: bash

   copy config_example.yaml config.yaml   # Windows
   cp config_example.yaml config.yaml     # Linux/macOS

See :doc:`configuration` for all available settings.

Run
---

.. code-block:: bash

   python modeling_agent.py

Default host/port are configured in ``config.yaml`` under ``platforms.websocket``.
The agent listens on ``ws://localhost:8765`` by default.

.. note::

   **Startup is slow.** Before the WebSocket opens, BAF trains a NER model plus
   one local intent classifier per state (10 states), which can take several
   minutes (about 3.5 on the Docker image) before the socket listens. A
   first run that seems to hang is usually just this.

If you see an ``OPENAI_API_KEY`` error, check your ``config.yaml`` or ``.env``
file. See :doc:`configuration` for details.

Validation
----------

.. code-block:: bash

   # Full test suite
   python -m pytest

   # Focused suites
   python -m pytest tests/test_diagram_handlers.py
   python -m pytest tests/test_protocol.py
   python -m pytest tests/test_request_planner.py

Documentation Build
-------------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   cd docs

   # Windows
   make.bat html

   # Linux/macOS
   make html

The built documentation will be in ``docs/build/html/``.
