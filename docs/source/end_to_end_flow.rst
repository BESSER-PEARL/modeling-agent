End-to-End Flow
===============

This page explains the full request lifecycle in the
`BESSER Web Modeling Editor <https://editor.besser-pearl.org>`_ — from user
message through the BESSER Modeling Agent backend, LLM, frontend converter,
Redux store, and finally the Apollon editor rendering.

.. contents:: On this page
   :local:
   :depth: 2

The Big Picture
---------------

The system has three parts:

.. mermaid::

   graph LR
       subgraph FE["FRONTEND (browser)"]
           ChatUI["Chat UI"]
           Editor["Editor (Apollon)"]
           Redux["Redux Store"]
       end

       subgraph BE["BACKEND (Python)"]
           LLMCalls["LLM calls"]
           Prompts["Prompts"]
           Schemas["Schemas"]
       end

       FE <-->|"WebSocket<br/>JSON messages"| BE
       BE -->|"API calls"| OpenAI["OpenAI (GPT-4.1)"]

Frontend (browser)
~~~~~~~~~~~~~~~~~~

- **Chat UI**: The chat panel where the user types requests and sees responses.
- **Editor (Apollon)**: The visual diagram editor rendering boxes, arrows, and
  classes. This is a third-party library — we do not control its internals.
- **Redux Store**: A JavaScript object in memory holding all project data (every
  diagram, every tab, every model). When anything changes, the UI updates
  automatically.

Backend (Docker / Python)
~~~~~~~~~~~~~~~~~~~~~~~~~

- **WebSocket server**: Listens for messages from the frontend chat.
- **Intent recognition**: Determines what the user wants (create? modify? generate?).
- **Orchestrator**: Plans which operations to run (e.g., "create ClassDiagram then
  generate code").
- **Diagram handlers**: One per diagram type. Each handler builds prompts and parses
  LLM responses for its diagram type.
- **LLM calls**: Sends prompts to OpenAI and gets structured JSON back, validated by
  Pydantic schemas.

OpenAI API
~~~~~~~~~~

The backend sends a prompt like *"You are a UML expert. Create a class diagram for
a shoe store."* OpenAI returns structured JSON matching a Pydantic schema
(e.g., ``{ classes: [...], relationships: [...] }``). The backend validates the
response and sends it to the frontend.

Creation Flow
-------------

Full walkthrough of what happens when the user types **"create a shoe store"**.

Step 1–2: User message to WebSocket
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The user types in the chat panel. The frontend builds a JSON payload and sends it
over WebSocket:

.. code-block:: json

   {
     "action": "user_message",
     "protocolVersion": "2.0",
     "message": "create a shoe store",
     "context": {
       "activeDiagramType": "ClassDiagram",
       "activeDiagramId": "654bfb6f-...",
       "projectSnapshot": {
         "diagrams": {
           "ClassDiagram": [{ "model": { "elements": {}, "relationships": {} } }]
         }
       }
     }
   }

The ``context.projectSnapshot`` contains the entire project state so the backend
knows what already exists.

Step 3–4: Backend receives and classifies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The protocol adapter (``protocol/adapters.py``) parses the raw message into an
``AssistantRequest`` object. The intent classifier asks the LLM what the user
wants and returns an intent like ``create_complete_system_intent``.

Step 5–6: State machine and orchestrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The BESSER Agent Framework routes to the matching state
(``create_complete_system_state``). The orchestrator
(``orchestrator/request_planner.py``) decides:

- Target diagram type: ``ClassDiagram``
- Mode: ``complete_system``
- Request text: ``"create a shoe store"``

Step 7–9: Handler calls the LLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``execution/model_operations.py`` calls the ``ClassDiagramHandler`` which builds a
system prompt and calls ``client.beta.chat.completions.parse()`` with a Pydantic
schema. OpenAI returns:

.. code-block:: json

   {
     "systemName": "Shoe Store Management System",
     "classes": [
       {
         "className": "Shoe",
         "attributes": [
           { "name": "shoeId", "type": "str" },
           { "name": "brand", "type": "str" },
           { "name": "price", "type": "float" }
         ],
         "methods": [
           { "name": "updatePrice", "parameters": [{ "name": "newPrice", "type": "float" }], "returnType": "any" }
         ]
       },
       {
         "className": "Store",
         "attributes": [
           { "name": "storeId", "type": "int" },
           { "name": "name", "type": "str" }
         ]
       }
     ],
     "relationships": [
       {
         "type": "Association",
         "source": "Store",
         "target": "Shoe",
         "sourceMultiplicity": "1",
         "targetMultiplicity": "0..*"
       }
     ]
   }

This is validated by Pydantic (``SystemClassSpec``). Class names are capped at 30
characters, actions use ``Literal`` types, and all fields are strictly typed.

Step 10: Backend sends response
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The handler wraps the LLM response into a WebSocket payload:

.. code-block:: json

   {
     "action": "inject_complete_system",
     "diagramType": "ClassDiagram",
     "diagramId": "654bfb6f-...",
     "systemSpec": { "...the LLM response..." },
     "message": "Built the **Shoe Store** class diagram with 2 class(es).",
     "suggestedActions": [
       { "label": "Add attributes", "prompt": "add more attributes" }
     ]
   }

Step 11–12: Frontend converter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The frontend receives the WebSocket message and detects it is an injection
command. The **ConverterFactory** transforms the backend's simple format into the
Apollon editor's detailed format:

.. code-block:: text

   Backend format (simple):              Apollon format (detailed):
   {                                     {
     classes: [                            elements: {
       { className: "Shoe",       -->        "uuid-1": {
         attributes: [...] }                   type: "Class",
     ]                                         name: "Shoe",
   }                                           bounds: { x: 0, y: 0, w: 220, h: 150 },
                                               attributes: ["uuid-2"],
                                               methods: ["uuid-3"]
                                             },
                                             "uuid-2": {
                                               type: "ClassAttribute",
                                               name: "shoeId",
                                               visibility: "private",
                                               attributeType: "str"
                                             }
                                           },
                                           relationships: { ... }
                                         }

The converter generates UUIDs, calculates positions using a grid layout, maps
relationship types, and sets bounds/visibility. It is a pure function — no
editor, no DOM, no React.

Step 13–14: Redux and editor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The converted model is written to the Redux store:

.. code-block:: typescript

   dispatch(updateDiagramModelThunk({ model: newModel }))

Then the editor revision counter is bumped:

.. code-block:: typescript

   dispatch(bumpEditorRevision())

The Apollon editor watches this counter. When it changes, the editor destroys
itself and re-creates with the new model from Redux. The user sees the diagram.

Modification Flow
-----------------

When the user says **"change storeId to int"**, the flow is shorter:

1. Frontend sends ``{ message: "change storeId to int", context: { ... } }``
2. Backend classifies intent: ``modify_model_intent``
3. Handler builds prompt with the **current model** so the LLM knows what exists
4. LLM returns structured modifications:

.. code-block:: json

   {
     "modifications": [{
       "action": "modify_attribute",
       "target": { "className": "Store", "attributeName": "storeId" },
       "changes": { "type": "int" }
     }]
   }

5. Backend sends ``{ "action": "modify_model", "modification": { ... } }``
6. Frontend **ModifierFactory** applies changes to the current model (pure function)
7. Updated model written to Redux + editor revision bumped
8. Editor re-renders with the updated attribute

Injection Payload Reference
----------------------------

The backend sends three types of injection payloads over WebSocket.

inject_complete_system
~~~~~~~~~~~~~~~~~~~~~~

Creates a full diagram from scratch (replaces existing content):

.. code-block:: json

   {
     "action": "inject_complete_system",
     "diagramType": "ClassDiagram",
     "diagramId": "654bfb6f-...",
     "systemSpec": {
       "systemName": "Shoe Store",
       "classes": [
         {
           "className": "Shoe",
           "attributes": [{ "name": "shoeId", "type": "str" }],
           "methods": [{ "name": "updatePrice", "parameters": [{ "name": "newPrice" }] }]
         }
       ],
       "relationships": [
         { "type": "Association", "source": "Store", "target": "Shoe",
           "sourceMultiplicity": "1", "targetMultiplicity": "0..*" }
       ]
     },
     "message": "Built the Shoe Store class diagram.",
     "suggestedActions": [{ "label": "Add attributes", "prompt": "add more attributes" }]
   }

modify_model
~~~~~~~~~~~~

Applies modifications to existing diagram elements. Supports single and batch:

**Single modification:**

.. code-block:: json

   {
     "action": "modify_model",
     "diagramType": "ClassDiagram",
     "modification": {
       "action": "modify_attribute",
       "target": { "className": "Store", "attributeName": "storeId" },
       "changes": { "type": "int" }
     },
     "message": "Updated attribute storeId in Store."
   }

**Batch modifications:**

.. code-block:: json

   {
     "action": "modify_model",
     "diagramType": "ClassDiagram",
     "modifications": [
       { "action": "remove_element", "target": { "className": "Shoe" } },
       { "action": "remove_element", "target": { "sourceClass": "Store", "targetClass": "Shoe" } }
     ],
     "message": "Applied 2 changes."
   }

inject_element
~~~~~~~~~~~~~~

Adds a single element to an existing diagram:

.. code-block:: json

   {
     "action": "inject_element",
     "diagramType": "ClassDiagram",
     "element": {
       "className": "Review",
       "attributes": [
         { "name": "rating", "type": "int" },
         { "name": "comment", "type": "str" }
       ]
     },
     "message": "Added Review class."
   }

Frontend Processing
-------------------

When the frontend receives an injection payload, it processes it as follows:

.. mermaid::

   flowchart TD
       WS["WebSocket message arrives"] --> AC["AssistantClient.handleMessage()"]
       AC -->|"Parse JSON, detect action type"| MI["useModelInjection.handleInjection()"]
       MI --> S1["1. Determine target diagram type"]
       S1 --> S2["2. Switch diagram tab if needed"]
       S2 --> S3["3. Push undo snapshot"]
       S3 --> S4{"4. Action type?"}
       S4 -->|inject_complete_system| CONV["ConverterFactory.convert(systemSpec)"]
       S4 -->|modify_model| MOD["ModifierFactory.applyModification(model, mod)"]
       S4 -->|inject_element| CONV2["ConverterFactory (single element)"]
       CONV --> S5["5. dispatch(updateDiagramModelThunk)"]
       MOD --> S5
       CONV2 --> S5
       S5 --> S6["6. dispatch(bumpEditorRevision)"]
       S6 --> S7["7. Show success message in chat"]
       S7 --> RENDER["Apollon editor re-renders from Redux"]

Key Architecture Insight
~~~~~~~~~~~~~~~~~~~~~~~~

Redux is the model owner. The agent writes, the editor reads:

.. mermaid::

   graph TD
       Redux["Redux Store<br/>(source of truth)"]
       Redux --> AgentW["Agent writes<br/>(converter)"]
       Redux --> EditorR["Editor reads<br/>(renderer)"]
       Redux --> OtherR["Other components read<br/>(export, preview)"]

This means:

- The agent can work without the editor being mounted
- Tab switching does not break injections
- Multiple diagram types can be updated in sequence
- The editor can be replaced without touching the agent

Component Responsibilities
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Component
     - Job
     - Analogy
   * - **Chat UI**
     - Show messages, take user input
     - The cashier taking your order
   * - **AssistantClient**
     - Send/receive WebSocket messages
     - The phone line to the kitchen
   * - **Backend**
     - Understand intent, call LLM, build response
     - The chef deciding what to cook
   * - **OpenAI (LLM)**
     - Generate the actual diagram content
     - The recipe book
   * - **Converter**
     - Transform LLM output to editor format
     - The waiter plating the food
   * - **Redux Store**
     - Hold the project data (single source of truth)
     - The serving counter
   * - **Apollon Editor**
     - Render the diagram visually
     - The plate the customer eats from
