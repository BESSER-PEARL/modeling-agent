WebSocket Protocol Reference
============================

This document is the complete reference for the WebSocket protocol between the
Modeling Agent (backend) and the BESSER Web Modeling Editor (frontend). For JSON
schema details, see :doc:`schema`. For the end-to-end flow including frontend
processing, see :doc:`end_to_end_flow`.

.. contents:: On this page
   :local:
   :depth: 2

Connection
----------

The agent listens on a WebSocket endpoint (default ``ws://localhost:8765``).
The frontend connects and communicates using JSON messages in the **v2 protocol
format**.

Protocol Version
~~~~~~~~~~~~~~~~

The current protocol version is **2.0**. All messages include a
``protocolVersion`` field. The backend detects the version and adapts response
formatting accordingly.


Inbound Messages (Frontend → Backend)
--------------------------------------

All frontend messages use the ``user_message`` action, wrapped in a
BESSER framework envelope.

Envelope Structure
~~~~~~~~~~~~~~~~~~

The BESSER framework wraps the v2 payload inside an outer envelope:

.. code-block:: json

   {
     "action": "user_message",
     "user_id": "session-uuid",
     "message": "<JSON string of v2 payload>"
   }

The ``message`` field contains a **JSON-encoded string** of the inner v2 payload.

V2 Payload Structure
~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "action": "user_message",
     "protocolVersion": "2.0",
     "clientMode": "workspace",
     "message": "create a User class with id and email",
     "context": {
       "activeDiagramType": "ClassDiagram",
       "activeDiagramId": "550e8400-e29b-41d4-a716-446655440000",
       "activeModel": { "...model JSON..." },
       "projectSnapshot": {
         "name": "MyProject",
         "diagrams": {
           "ClassDiagram": [{ "id": "diag-1", "title": "Main", "model": {} }],
           "StateMachineDiagram": null
         }
       },
       "diagramSummaries": [
         { "diagramType": "ClassDiagram", "summary": "3 classes, 2 relationships" }
       ],
       "sessionId": "abc-123",
       "currentDiagramIndices": { "ClassDiagram": 0 }
     },
     "attachments": [
       {
         "filename": "model.puml",
         "content": "QGN0YXJ0dW1s...",
         "mimeType": "text/plain"
       }
     ]
   }

Field Reference
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Required
     - Description
   * - ``action``
     - Yes
     - Always ``"user_message"``
   * - ``protocolVersion``
     - Yes
     - ``"2.0"``
   * - ``clientMode``
     - No
     - ``"workspace"`` (default) or ``"chat"``
   * - ``message``
     - Yes
     - The user's natural-language message (max 12,000 characters)
   * - ``context.activeDiagramType``
     - No
     - Currently active diagram tab type (e.g., ``"ClassDiagram"``)
   * - ``context.activeDiagramId``
     - No
     - UUID of the active diagram
   * - ``context.activeModel``
     - No
     - Current model JSON from the canvas (may be stale after injection)
   * - ``context.projectSnapshot``
     - No
     - Full project state including all diagram tabs
   * - ``context.diagramSummaries``
     - No
     - Short text summaries of each diagram
   * - ``context.currentDiagramIndices``
     - No
     - Active tab index per diagram type (for multi-tab support)
   * - ``attachments``
     - No
     - Array of uploaded files (PlantUML, images, RDF, etc.)

Supported Diagram Types
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ClassDiagram
   ObjectDiagram
   StateMachineDiagram
   AgentDiagram
   GUINoCodeDiagram
   QuantumCircuitDiagram


Outbound Messages (Backend → Frontend)
----------------------------------------

All responses are JSON objects with an ``action`` field that determines the
message type.

inject_element
~~~~~~~~~~~~~~

Adds a single element to the diagram canvas.

.. code-block:: json

   {
     "action": "inject_element",
     "diagramType": "ClassDiagram",
     "diagramId": "diagram-001",
     "element": {
       "className": "User",
       "attributes": [
         {"name": "id", "type": "String", "visibility": "public"}
       ],
       "methods": [],
       "position": {"x": 100, "y": 200}
     },
     "message": "Created the **User** class.",
     "suggestedActions": [
       {"label": "Add Order class", "prompt": "Add an Order class"}
     ]
   }

inject_complete_system
~~~~~~~~~~~~~~~~~~~~~~

Injects a full diagram (all elements + relationships) at once.

.. code-block:: json

   {
     "action": "inject_complete_system",
     "diagramType": "ClassDiagram",
     "diagramId": "diagram-001",
     "systemSpec": {
       "systemName": "E-commerce System",
       "classes": [
         {
           "className": "User",
           "attributes": [
             {"name": "id", "type": "String", "visibility": "public"},
             {"name": "email", "type": "String", "visibility": "public"}
           ],
           "methods": [],
           "position": {"x": 100, "y": 200}
         }
       ],
       "relationships": [
         {
           "type": "Association",
           "source": "User",
           "target": "Order",
           "sourceMultiplicity": "1",
           "targetMultiplicity": "0..*",
           "name": "creates"
         }
       ]
     },
     "replaceExisting": false,
     "createNewTab": false,
     "message": "Built the **E-commerce System** with 2 classes.",
     "suggestedActions": [
       {"label": "Add Product class", "prompt": "Add a Product class"}
     ]
   }

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``replaceExisting``
     - ``true`` to replace the current diagram; ``false`` to merge
   * - ``createNewTab``
     - ``true`` to create a new tab for the diagram
   * - ``suggestedActions``
     - Optional list of follow-up action buttons for the user

modify_model (single)
~~~~~~~~~~~~~~~~~~~~~

Apply a single modification to an existing element.

.. code-block:: json

   {
     "action": "modify_model",
     "diagramType": "ClassDiagram",
     "diagramId": "diagram-001",
     "modification": {
       "action": "modify_class",
       "target": {"className": "User"},
       "changes": {"name": "Customer"}
     },
     "message": "Renamed **User** to **Customer**."
   }

modify_model (batch)
~~~~~~~~~~~~~~~~~~~~

Apply multiple modifications in a single message.

.. code-block:: json

   {
     "action": "modify_model",
     "diagramType": "ClassDiagram",
     "diagramId": "diagram-001",
     "modifications": [
       {
         "action": "add_attribute",
         "target": {"className": "User"},
         "changes": {"name": "phone", "type": "String", "visibility": "public"}
       },
       {
         "action": "remove_method",
         "target": {"className": "Order"},
         "changes": {"name": "deprecatedMethod"}
       }
     ],
     "message": "Added **phone** to User and removed **deprecatedMethod** from Order."
   }

Nested Modification Actions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Action
     - Diagram Type
     - Purpose
   * - ``modify_class``
     - ClassDiagram
     - Rename, change visibility, etc.
   * - ``add_class``
     - ClassDiagram
     - Add a new class
   * - ``remove_element``
     - ClassDiagram
     - Remove a class or relationship
   * - ``add_relationship``
     - ClassDiagram
     - Add a new relationship
   * - ``add_attribute``
     - ClassDiagram
     - Add an attribute to a class
   * - ``remove_method``
     - ClassDiagram
     - Remove a method from a class
   * - ``modify_attribute``
     - ClassDiagram
     - Modify an existing attribute
   * - ``modify_object``
     - ObjectDiagram
     - Modify an object instance
   * - ``modify_state``
     - StateMachine, Agent
     - Modify a state
   * - ``modify_element``
     - Any (generic)
     - Generic element modification (fallback)

assistant_message
~~~~~~~~~~~~~~~~~

Text-only message displayed in the chat panel.

.. code-block:: json

   {
     "action": "assistant_message",
     "message": "The User class represents authenticated users in the system."
   }

Streaming Protocol
~~~~~~~~~~~~~~~~~~

For long LLM responses, the backend streams text chunk-by-chunk:

.. code-block:: json

   {"action": "stream_start", "streamId": "abc12345"}
   {"action": "stream_chunk", "streamId": "abc12345", "chunk": "The class diagram ", "done": false}
   {"action": "stream_chunk", "streamId": "abc12345", "chunk": "represents the core ", "done": false}
   {"action": "stream_chunk", "streamId": "abc12345", "chunk": "domain model.", "done": false}
   {"action": "stream_done", "streamId": "abc12345", "fullText": "The class diagram represents the core domain model.", "done": true}

**Buffer threshold:** ~200 characters. Chunks are buffered until they reach this
threshold or hit a natural break (``.``, ``!``, ``?``, ``:``, newline), balancing
between WebSocket flood and perceived latency.

progress
~~~~~~~~

Loading/progress indicator update.

.. code-block:: json

   {
     "action": "progress",
     "message": "Generating class diagram...",
     "step": 1,
     "total": 3
   }

agent_error
~~~~~~~~~~~

Error payload sent when something goes wrong.

.. code-block:: json

   {
     "action": "agent_error",
     "message": "Failed to generate the diagram.",
     "errorCode": "LLM_PARSE_ERROR",
     "retryable": true
   }

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Error Code
     - Description
   * - ``LLM_PARSE_ERROR``
     - LLM returned unparseable JSON (retryable)
   * - ``HANDLER_ERROR``
     - Diagram handler raised an exception
   * - ``VALIDATION_ERROR``
     - Generated diagram failed schema validation
   * - ``CONVERSION_ERROR``
     - File conversion failed

switch_diagram
~~~~~~~~~~~~~~

Switch the active diagram tab in the editor.

.. code-block:: json

   {
     "action": "switch_diagram",
     "diagramType": "StateMachineDiagram",
     "reason": "Switching to StateMachineDiagram based on your request."
   }

trigger_generator
~~~~~~~~~~~~~~~~~

Trigger code generation from the current model. The frontend should invoke the
appropriate generator with the provided config.

.. code-block:: json

   {
     "action": "trigger_generator",
     "generatorType": "django",
     "config": {
       "project_name": "hotel_app",
       "app_name": "core_app",
       "containerization": false
     },
     "message": "Starting **django** code generation — this may take a moment."
   }

trigger_export
~~~~~~~~~~~~~~

Trigger model export.

.. code-block:: json

   {
     "action": "trigger_export",
     "format": "json",
     "message": "Exporting your project as **JSON** — the download should start shortly."
   }

trigger_deploy
~~~~~~~~~~~~~~

Open the deploy dialog on the frontend.

.. code-block:: json

   {
     "action": "trigger_deploy",
     "platform": "render",
     "config": {},
     "message": "Opening the **Deploy to Render** dialog..."
   }

auto_generate_gui
~~~~~~~~~~~~~~~~~

Auto-generate a GUI diagram from the class diagram.

.. code-block:: json

   {
     "action": "auto_generate_gui",
     "sourceDiagramType": "ClassDiagram",
     "message": "Auto-generating GUI from the class diagram..."
   }


Message Flow Examples
---------------------

Simple Class Creation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Frontend → Backend:
     { "action": "user_message", "message": "create a User class with id and email",
       "context": { "activeDiagramType": "ClassDiagram" } }

   Backend → Frontend:
     { "action": "progress", "message": "Generating class diagram..." }
     { "action": "inject_element", "diagramType": "ClassDiagram",
       "element": { "className": "User", "attributes": [...] },
       "message": "Created the User class." }

Multi-Step: Create Model → Generate Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Frontend → Backend:
     { "message": "create a library system and generate django" }

   Backend → Frontend:
     { "action": "progress", "message": "Creating class diagram..." }
     { "action": "inject_complete_system", "diagramType": "ClassDiagram",
       "systemSpec": { ... } }
     { "action": "assistant_message",
       "message": "To generate your Django project, I need a few details..." }

   Frontend → Backend:
     { "message": "project_name=library app_name=books containerization=true" }

   Backend → Frontend:
     { "action": "trigger_generator", "generatorType": "django",
       "config": { "project_name": "library", "app_name": "books",
                   "containerization": true } }

Streaming Response (Help/Explanation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Frontend → Backend:
     { "message": "explain what is an association in UML" }

   Backend → Frontend:
     { "action": "stream_start", "streamId": "a1b2c3d4" }
     { "action": "stream_chunk", "streamId": "a1b2c3d4",
       "chunk": "An association in UML represents..." }
     { "action": "stream_chunk", "streamId": "a1b2c3d4",
       "chunk": "a structural relationship between..." }
     { "action": "stream_done", "streamId": "a1b2c3d4",
       "fullText": "An association in UML represents a structural..." }
