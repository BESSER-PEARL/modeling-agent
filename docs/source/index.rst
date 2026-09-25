BESSER Modeling Agent
=====================

The **BESSER Modeling Agent** is the conversational AI backend for the
`BESSER Web Modeling Editor <https://editor.besser-pearl.org>`_.
It is part of the `BESSER <https://besser-pearl.github.io/BESSER/>`_ platform
(*better software faster*) — an open-source low-code platform for smart software
modeling developed by the `BESSER-PEARL <https://github.com/BESSER-PEARL>`_ team.

The agent is a WebSocket service that turns natural language into **diagram
operations**. It interprets each request, decides what to do, and returns a
structured action payload — ``inject_complete_system``, ``modify_model``,
``trigger_generator``, ``trigger_smart_generator`` and friends — that the
editor applies to the canvas or hands to a generator.

**Key capabilities:**

- Create and modify diagrams from natural language
- Multi-step orchestration (model first, then generate code)
- 8 diagram types: Class, Object, State Machine, Agent, GUI (no-code),
  Quantum Circuit, BPMN, and User Profile
- Code generation via `BESSER generators <https://besser-pearl.github.io/BESSER/generators.html>`_
  (Django, Python, Java, SQL, and more), plus the LLM-authored Spec-Driven Agent for
  stacks BESSER has no built-in generator for
- Bring-your-own-key (BYOK) routing, so generation can run on the user's own
  OpenAI, Anthropic or Mistral key
- UML specification Q&A via :term:`RAG` (ChromaDB)
- File conversion from PlantUML, RDF/knowledge graphs, XMI, PDFs, images, and
  plain text
- Voice input via OpenAI speech-to-text

**Part of the BESSER ecosystem:**

.. list-table::
   :widths: 30 70

   * - `BESSER platform <https://besser-pearl.github.io/BESSER/>`_
     - The core modeling and code generation platform
   * - `BESSER Agentic Framework <https://besser-pearl.github.io/BESSER/>`_
     - State machine, WebSocket, and intent classification infrastructure
   * - `BESSER Web Modeling Editor <https://editor.besser-pearl.org>`_
     - The React/TypeScript frontend this agent powers
   * - `BESSER Modeling Agent <https://github.com/BESSER-PEARL/modeling-agent>`_
     - This repository — the conversational AI backend

**New here?** Start with :doc:`getting_started`, then read :doc:`end_to_end_flow`
for the full request lifecycle.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started
   configuration
   glossary

.. toctree::
   :maxdepth: 2
   :caption: How It Works

   end_to_end_flow
   architecture
   intent_recognition
   orchestration

.. toctree::
   :maxdepth: 2
   :caption: Reference

   schema
   websocket_protocol
   diagram_handlers
   usage
   api

.. toctree::
   :maxdepth: 2
   :caption: Operations & Contributing

   deployment
   contributing

.. toctree::
   :maxdepth: 1
   :caption: Release Notes

   releases

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
