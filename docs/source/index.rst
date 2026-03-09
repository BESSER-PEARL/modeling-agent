Modeling Agent Documentation
============================

The Modeling Agent is the conversational AI backend for the BESSER Web Modeling Editor.
It interprets natural-language requests over WebSocket and returns structured diagram JSON
payloads that the frontend renders directly.

**Key capabilities:**

- Create and modify UML diagrams from natural language
- Multi-step orchestration (model first, then generate code)
- 6 diagram types: Class, Object, StateMachine, Agent, GUI, Quantum Circuit
- Code generation triggers for Django, Python, Java, SQL, and more
- UML specification Q&A via RAG (ChromaDB)
- File conversion from PlantUML, RDF, images, and text

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started
   configuration

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   architecture
   schema
   diagram_handlers
   orchestration

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   usage
   api

.. toctree::
   :maxdepth: 2
   :caption: Operations

   deployment
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
