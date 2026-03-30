Modeling Agent Documentation
============================

The Modeling Agent is the conversational AI backend for the
`BESSER Web Modeling Editor <https://github.com/BESSER-PEARL>`_.
It interprets natural-language requests over WebSocket and returns structured
diagram JSON payloads that the frontend renders directly.

**Key capabilities:**

- Create and modify UML diagrams from natural language
- Multi-step orchestration (model first, then generate code)
- 6 diagram types: Class, Object, StateMachine, Agent, GUI, Quantum Circuit
- Code generation triggers for Django, Python, Java, SQL, and more
- UML specification Q&A via :term:`RAG` (ChromaDB)
- File conversion from PlantUML, RDF, images, and text

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
