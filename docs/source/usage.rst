Usage
=====

This guide covers common interaction patterns with the Modeling Agent.
For the technical details of how requests are processed, see
:doc:`end_to_end_flow`. For the JSON schemas behind each response, see
:doc:`schema`.

.. contents:: On this page
   :local:
   :depth: 2

How Operation Mode is Selected
------------------------------

The agent infers the operation mode from your phrasing:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Mode
     - Triggered by
     - Example
   * - **Single element**
     - Creating one specific item
     - "create a User class", "add a Cancelled state"
   * - **Complete system**
     - Describing a whole domain or system
     - "create a class diagram for an e-commerce system"
   * - **Modification**
     - Referring to something that already exists
     - "rename Order to PurchaseOrder", "add email to User"

.. note::

   **ObjectDiagram** requires a **ClassDiagram** to exist first — the agent
   uses class definitions to generate object instances with realistic values.

Common Modeling Requests
------------------------

The agent interprets natural language and determines both the diagram type and
the operation mode automatically.

Class Diagram Examples
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   # Single element
   create a User class with id and email

   # Complete system
   create a class diagram for an e-commerce system with products, orders, and customers

   # Modification
   add a password attribute to the User class
   rename the Order class to PurchaseOrder

Object Diagram Examples
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   # Single instance
   create an object instance of User called admin

   # Complete system
   create object instances for all classes in the model

State Machine Examples
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   # Complete system
   create a login state machine
   create an order processing state diagram

   # Single element
   add a "Cancelled" state to the state machine

Agent Diagram Examples
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   create a multi-agent support workflow
   create a chatbot for restaurant ordering

GUI Diagram Examples
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   create a GUI diagram for the current class model
   create a dashboard with charts for user statistics

Quantum Circuit Examples
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   create a quantum circuit for Bell state
   create a 3-qubit Grover search circuit

Multi-step Requests
-------------------

The planner can split combined requests into ordered operations:

.. code-block:: text

   create a class diagram for a bookstore and then generate django backend

   create a hospital management system, add a state machine for patient
   admission, and build a GUI

   create a banking system and generate SQL schema

The planner ensures correct ordering (e.g., ClassDiagram is always created
before ObjectDiagram or GUI generation).

Generation Requests
-------------------

Supported generator types and their keywords:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Generator
     - Example Request
     - Output
   * - ``django``
     - ``generate django backend``
     - Django project trigger
   * - ``backend``
     - ``generate full backend``
     - Backend code trigger
   * - ``web_app``
     - ``generate web application``
     - Full-stack app trigger
   * - ``sql``
     - ``generate SQL schema``
     - SQL DDL trigger
   * - ``sqlalchemy``
     - ``generate SQLAlchemy models``
     - ORM model trigger
   * - ``python``
     - ``generate Python classes``
     - Python code trigger
   * - ``java``
     - ``generate Java classes``
     - Java code trigger
   * - ``pydantic``
     - ``generate Pydantic models``
     - Pydantic model trigger
   * - ``jsonschema``
     - ``generate JSON schema``
     - JSON Schema trigger
   * - ``smartdata``
     - ``generate smart data``
     - Smart data trigger
   * - ``agent``
     - ``generate BESSER agent``
     - Agent code trigger
   * - ``qiskit``
     - ``generate Qiskit code``
     - Quantum code trigger

Inline Configuration
~~~~~~~~~~~~~~~~~~~~

Some generators accept inline configuration:

.. code-block:: text

   # Django with config
   generate django backend with project name "myproject" and app name "store"

   # SQL with dialect
   generate SQL schema for PostgreSQL

   # Qiskit with backend
   generate Qiskit code using Aer simulator with 1024 shots

Export and Deploy
~~~~~~~~~~~~~~~~~

.. code-block:: text

   export project to JSON
   export to BUML
   deploy to Render

File Conversion
---------------

Attachments are converted into diagram specifications:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - File Type
     - Extensions
     - Output
   * - PlantUML
     - ``.puml``, ``.plantuml``, ``.wsd``
     - ClassDiagram or StateMachineDiagram
   * - Knowledge Graph
     - ``.ttl``, ``.rdf``, ``.owl``, ``.n3``, ``.jsonld``
     - ClassDiagram
   * - Images
     - ``.png``, ``.jpg``, ``.jpeg``, ``.gif``, ``.webp``
     - ClassDiagram (via OpenAI Vision)
   * - Generic text
     - any other
     - ClassDiagram (via LLM interpretation)

Upload a file alongside your message, and the agent will automatically convert
it to the appropriate diagram type.

UML Specification Queries
-------------------------

The agent can answer questions about the UML specification using RAG:

.. code-block:: text

   what is a composite state in UML?
   explain the difference between aggregation and composition
   what are the metaclasses in UML?

Model Description
-----------------

Ask the agent to describe your current model:

.. code-block:: text

   describe my current model
   what does my class diagram contain?
   summarize the project

Quick Start Commands
--------------------

.. code-block:: text

   # Greeting / capabilities
   hello
   what can you do?

   # Help
   help
   ?

Quality Suggestions
-------------------

After generating a diagram, the agent may offer quality suggestions:

- Missing expected attributes (e.g., User without email/password)
- Isolated classes with no relationships
- Missing ID attributes
- Cross-diagram suggestions (e.g., "create a state machine for User lifecycle")

These appear as "What's next?" hints after each generation.
