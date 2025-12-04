Getting Started
===============

Welcome to the Modeling Agent project. This guide walks you through what
the agent does, how to install and configure it locally, and how to run
basic validation before contributing changes. A release notes section at
the end captures the highlights for ``v0.1.0``.

Project overview
----------------

The Modeling Agent is a conversational backend built on the BESSER
Agentic framework. It interprets natural language requests and generates
UML diagram elements that can be rendered inside
``editor.besser-pearl.org`` or other compatible tooling.

Core capabilities
~~~~~~~~~~~~~~~~~

- Generate class, object, state machine, and agent diagrams on demand.
- Modify existing diagrams by adding or updating nodes, attributes, and
  relationships.
- Use GPT-based large language models to understand domain-specific
  terminology and context.
- Stream diagram updates to clients over WebSocket connections for
  real-time feedback.

Repository layout
-----------------

The top-level directories you will likely interact with are:

``diagram_handlers/``
    Specialized logic for each supported diagram type.

``docs/``
    Sphinx documentation source files (including this guide).

``tests/``
    Pytest suites covering handlers, routing, and end-to-end workflows.

``modeling_agent.py``
    The main application entry point, including the intent router and LLM
    integration.

Prerequisites
-------------

- Python 3.10 or newer.
- A virtual environment (``venv`` or similar) for isolating dependencies.
- Access to an OpenAI API key or compatible LLM endpoint credentials.
- Git for managing branches and syncing with the upstream repository.

Quick installation
------------------

.. code-block:: bash

   git clone https://github.com/<your-org>/modeling-agent.git
   cd modeling-agent
   python -m venv .venv
   source .venv/bin/activate  # On Windows use .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt

Configuration
-------------

Copy the provided template and populate it with your environment
settings:

.. code-block:: bash

   cp config.ini.example config.ini

Update ``config.ini`` (or ``.env`` if you prefer environment variables)
with the following minimum values:

``API_KEY``
    Your OpenAI or compatible LLM provider key.

``MODEL``
    The model identifier, e.g. ``gpt-4``.

``TEMPERATURE``
    Sampling temperature. The default ``0.7`` balances creativity and
    determinism.

``PORT``
    WebSocket port for the agent; defaults to ``8765``.

Running the agent
-----------------

With your virtual environment active and configuration saved, start the
service:

.. code-block:: bash

   python modeling_agent.py

The agent listens on ``ws://localhost:8765`` by default. Connect a
client (such as the BESSER web editor widget) to this endpoint to
exchange diagram instructions in real time.

Testing
-------

Run the automated suites before opening pull requests:

.. code-block:: bash

   pytest

Add the ``-v`` flag for verbose output or filter to a specific file, for
example ``pytest tests/test_class_diagram_handler.py``. The GitHub
workflow mirrors this command, so keeping tests green locally reduces CI
iterations.

Documentation workflow
----------------------

The documentation uses Sphinx. After editing any ``.rst`` files, rebuild
the HTML output to catch warnings:

.. code-block:: bash

   sphinx-build -b html docs/source docs/_build/html

Only commit source files inside ``docs/source``—the ``_build`` directory
is a local artifact.

Troubleshooting tips
--------------------

- If LLM calls fail, verify your API key and ensure outbound network
  access is available from your environment.
- Use the logging output in ``application.log`` to inspect the request
  payloads sent to diagram handlers.
- When diagram generation appears inconsistent, clear any cached diagram
  state on the client side and resend the request.

Release notes
-------------

``v0.1.0`` (2025-10-23)
~~~~~~~~~~~~~~~~~~~~~~~

- Initial public release of the Modeling Agent project.
- Added support for class, object, state machine, and agent diagram
  generation backed by GPT-based reasoning.
- Implemented WebSocket server in ``modeling_agent.py`` for real-time
  collaboration with the BESSER editor.
- Bundled pytest suites covering diagram handlers and routing logic.
- Shipped Sphinx documentation scaffolding for deployment to
  Read the Docs.
