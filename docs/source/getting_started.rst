Getting Started
===============

Overview
--------

The Modeling Agent receives user requests over WebSocket, normalizes them into a
unified protocol, plans one or more operations, and returns structured responses
for model updates or code-generation triggers.

Key capabilities:

- UML diagram creation and modification via natural language.
- Multi-operation orchestration (modeling + generation in a single request).
- UML specification Q&A with RAG over the OMG UML 2.5.1 specification.
- File conversion from PlantUML, knowledge-graph files, images, and plain text.

Supported Diagram Types
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Diagram Type
     - Description
     - Output Format
   * - ``ClassDiagram``
     - UML class diagrams
     - Apollon-compatible JSON
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

Prerequisites
-------------

- Python 3.10 or newer (3.11+ recommended).
- OpenAI API key with GPT-4.1-mini access.

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

1. Copy ``config.yaml.example`` to ``config.yaml``.
2. Set ``nlp.openai.api_key`` with your OpenAI key.

.. code-block:: bash

   copy config.yaml.example config.yaml   # Windows
   cp config.yaml.example config.yaml     # Linux/macOS

See :doc:`configuration` for all available settings.

Run
---

.. code-block:: bash

   python modeling_agent.py

Default host/port are configured in ``config.yaml`` under ``[websocket_platform]``.
The agent listens on ``ws://0.0.0.0:8765`` by default.

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
