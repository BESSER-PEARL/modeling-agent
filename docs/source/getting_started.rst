Getting Started
===============

Overview
--------

Modeling Agent receives user requests, normalizes them into a unified protocol,
plans one or more operations, and returns structured responses for model updates
or code-generation triggers.

Key capabilities:

- UML diagram creation and modification.
- Multi-operation orchestration (modeling + generation).
- UML specification Q&A with RAG.
- File conversion from PlantUML, knowledge-graph files, images, and plain text.

Prerequisites
-------------

- Python 3.10 or newer.
- OpenAI API key.

Install
-------

.. code-block:: bash

   python -m venv .venv
   # Windows PowerShell
   .\\.venv\\Scripts\\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt

Configuration
-------------

1. Copy ``config.ini.example`` to ``config.ini``.
2. Set ``nlp.openai.api_key``.

.. code-block:: bash

   copy config.ini.example config.ini

Run
---

.. code-block:: bash

   python modeling_agent.py

Default host/port are configured in ``config.ini`` under ``[websocket_platform]``.

Validation
----------

.. code-block:: bash

   python -m pytest
   python -m pytest tests/test_diagram_handlers.py
   python -m pytest tests/test_protocol.py

Documentation Build
-------------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   cd docs
   make.bat html
