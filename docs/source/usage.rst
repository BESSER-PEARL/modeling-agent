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

The agent infers the operation mode from your phrasing. There are two plan
modes (``ALLOWED_MODEL_MODES``):

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Mode
     - Triggered by
     - Example
   * - ``complete_system``
     - Describing a whole domain or system to build from scratch
     - "create a class diagram for an e-commerce system"
   * - ``modify_model``
     - Referring to something that already exists, **or** creating one
       specific element
     - "rename Order to PurchaseOrder", "add email to User",
       "create a User class", "add a Cancelled state"

.. note::

   A ``modify_model`` operation targeting a flow-style diagram that does not
   exist yet is automatically promoted to ``complete_system``, so "add a task
   to the order process" creates the process when there is none.

.. note::

   **ObjectDiagram** requires a **ClassDiagram** to exist first — the agent
   uses class definitions to generate object instances with realistic values.
   **UserDiagram** does not: its reference catalog is a bundled metamodel.

The agent also judges how the request relates to what is already on the
canvas (``model_disposition``): extend the current model, replace it, build in
a new tab, or reuse it for generation without changing it. When a request
would discard or overwrite an existing model, it **asks first** rather than
guessing destructively.

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

BPMN Examples
~~~~~~~~~~~~~

.. code-block:: text

   # Complete system
   model an order fulfillment process as a BPMN diagram
   create a business process for document review with two reviewers

   # Modification
   add a task to the order fulfillment process
   add an exclusive gateway after the review task

User Profile Examples
~~~~~~~~~~~~~~~~~~~~~

Profiles describe a *target user* as matching criteria (``age >= 18``,
``level == B2``) rather than concrete instance values.

.. code-block:: text

   create a target user profile for elderly users with sight issues
   add a language competence of at least B2 to the profile

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
   * - ``rest_api``
     - ``generate a REST API``
     - REST API trigger
   * - ``rdf``
     - ``generate an RDF vocabulary``
     - RDF trigger

Smart Generation
~~~~~~~~~~~~~~~~

Anything outside that list goes to the **smart generator** — an LLM-authored
codebase rather than a template. Two kinds of request take this route:

- A language or framework BESSER has no deterministic generator for
  (Rails, Rust, Kotlin, Next.js, Spring Boot, Go, Laravel, .NET, …).
- A BESSER stack **plus** extras the template cannot produce — auth, JWT,
  OAuth, Docker, migrations, tests, rate limiting, custom middleware.

.. code-block:: text

   build a Rails 7 app with Devise auth from my model
   generate a FastAPI backend with JWT and Docker

Smart runs spend the user's own API key, so the agent always **asks for
explicit confirmation** before starting one. If the request describes a
different domain than the class diagram already on the canvas (classes say
"Team/Player", the request says "a shoe store"), it offers three choices
instead of silently rewriting: update the model and generate, generate
anyway, or cancel.

Continuing from GitHub
~~~~~~~~~~~~~~~~~~~~~~

A project that was generated and pushed to GitHub can be resumed:

.. code-block:: text

   continue from github.com/owner/repo
   resume work on github.com/owner/repo on branch develop

The agent hands the frontend a ``trigger_github_import`` action; the frontend
calls the backend's import endpoint, loads the project, and arms incremental
modification. The agent itself never contacts GitHub.

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
     - ``.puml``, ``.plantuml``, ``.pu``
     - Diagram type detected from the PlantUML source
   * - Knowledge Graph
     - ``.ttl``, ``.rdf``, ``.owl``, ``.n3``, ``.nt``, ``.nq``, ``.trig``,
       ``.jsonld``
     - ClassDiagram
   * - XMI / Ecore
     - ``.xmi``, ``.uml``, ``.ecore``
     - ClassDiagram
   * - PDF
     - ``.pdf``
     - Converted via the vision model
   * - Images
     - ``.png``, ``.jpg``, ``.jpeg``, ``.gif``, ``.webp``, ``.bmp``, ``.svg``
     - Converted via the vision model
   * - Generic text
     - any other
     - ClassDiagram (via LLM interpretation)

Upload a file alongside your message and the agent converts it automatically.
Conversion can target any of ``ClassDiagram``, ``StateMachineDiagram``,
``ObjectDiagram``, ``AgentDiagram`` or ``BPMN``
(``CONVERTIBLE_DIAGRAM_TYPES``); the target is auto-detected when you do not
name one. If the resulting diagram would overwrite something already on the
canvas, the agent asks whether to replace, keep, or use a new tab first.

Voice Input
-----------

Voice messages are transcribed with OpenAI speech-to-text and then handled
exactly like typed messages. The language is auto-detected by default; a
deployment can pin one with ``BESSER_AGENT_STT_LANGUAGE``. Because a
transcript arrives as plain text with no JSON context, the frontend sends the
workspace context separately just before the audio.

Using Your Own API Key
----------------------

If you supply your own OpenAI, Anthropic or Mistral key in the editor, the
agent routes its generation and conversational calls through a per-request
client built from that key instead of the shared server key. Smart generation
always uses your key. See :doc:`configuration` for what is and is not routed.

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
