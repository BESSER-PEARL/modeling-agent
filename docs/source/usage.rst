Usage
=====

Common Modeling Requests
------------------------

Examples of natural-language requests:

- ``create a User class with id and email``
- ``create an object instance of User called admin``
- ``create a login state machine``
- ``create a multi agent support workflow``
- ``create a GUI diagram for the current class model``
- ``create a quantum circuit for bell state``

Multi-step Requests
-------------------

The planner can split combined requests into ordered operations, for example:

- ``create a class diagram for a bookstore and then generate django backend``

Generation Requests
-------------------

Supported generator types:

- ``django``
- ``backend``
- ``web_app``
- ``sql``
- ``sqlalchemy``
- ``python``
- ``java``
- ``pydantic``
- ``jsonschema``
- ``smartdata``
- ``agent``
- ``qiskit``

File Conversion
---------------

Attachments are converted into system specs when supported:

- PlantUML: ``.puml``, ``.plantuml``, ``.pu``
- Knowledge graph files: ``.ttl``, ``.rdf``, ``.owl``, ``.jsonld`` and related RDF variants
- Images: ``.png``, ``.jpg``, ``.jpeg``, ``.gif``, ``.webp``, ``.bmp``, ``.svg``
- Generic text files (LLM interpretation)
