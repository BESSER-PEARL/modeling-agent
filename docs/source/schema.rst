Schema Reference
================

This document describes all JSON schemas used by the Modeling Agent for request
parsing, response generation, and inter-component communication.

.. contents:: On this page
   :local:
   :depth: 2

Protocol Schemas
----------------

AssistantRequest (Inbound)
~~~~~~~~~~~~~~~~~~~~~~~~~~

The canonical request format after protocol parsing. Raw WebSocket messages are
normalized into this structure by ``src/protocol/adapters.py``.

.. code-block:: json

   {
     "action": "user_message",
     "protocolVersion": "2.0",
     "clientMode": "workspace",
     "message": "create a User class with id and email",
     "diagramType": "ClassDiagram",
     "diagramId": "550e8400-e29b-41d4-a716-446655440000",
     "context": {
       "activeDiagramType": "ClassDiagram",
       "activeDiagramId": "550e8400-e29b-41d4-a716-446655440000",
       "activeModel": {},
       "projectSnapshot": {
         "ClassDiagram": {},
         "StateMachineDiagram": null,
         "ObjectDiagram": null
       },
       "diagramSummaries": {
         "ClassDiagram": "3 classes, 2 relationships"
       }
     },
     "attachments": [
       {
         "filename": "model.puml",
         "contentBase64": "QGN0YXJ0dW1s...",
         "mimeType": "text/plain"
       }
     ]
   }

.. list-table:: AssistantRequest Fields
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Type
     - Description
   * - ``action``
     - ``str``
     - Request type: ``"user_message"``, ``"frontend_event"``
   * - ``protocolVersion``
     - ``str``
     - Always ``"2.0"`` for v2 clients
   * - ``clientMode``
     - ``str``
     - ``"workspace"`` or ``"simple"``
   * - ``message``
     - ``str``
     - Natural language request text
   * - ``diagramType``
     - ``str``
     - Target diagram type (may be empty)
   * - ``diagramId``
     - ``str``
     - UUID of target diagram instance
   * - ``context``
     - ``object``
     - WorkspaceContext object (see below)
   * - ``attachments``
     - ``array``
     - List of FileAttachment objects

WorkspaceContext
~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "activeDiagramType": "ClassDiagram",
     "activeDiagramId": "uuid-string",
     "activeModel": {
       "elements": {},
       "relationships": {}
     },
     "projectSnapshot": {
       "ClassDiagram": { "elements": {}, "relationships": {} },
       "StateMachineDiagram": null,
       "ObjectDiagram": null,
       "AgentDiagram": null,
       "GUINoCodeDiagram": null,
       "QuantumCircuitDiagram": null
     },
     "diagramSummaries": {
       "ClassDiagram": "3 classes, 2 relationships"
     }
   }

.. list-table:: WorkspaceContext Fields
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Type
     - Description
   * - ``activeDiagramType``
     - ``str``
     - Currently active diagram tab
   * - ``activeDiagramId``
     - ``str``
     - UUID of active diagram
   * - ``activeModel``
     - ``object``
     - Full model JSON currently displayed
   * - ``projectSnapshot``
     - ``object``
     - All diagrams keyed by type (null if empty)
   * - ``diagramSummaries``
     - ``object``
     - Compact per-diagram summaries

FileAttachment
~~~~~~~~~~~~~~

.. code-block:: json

   {
     "filename": "model.puml",
     "contentBase64": "QGN0YXJ0dW1s...",
     "mimeType": "text/plain"
   }

Response Schemas
----------------

inject_single_element
~~~~~~~~~~~~~~~~~~~~~

Returned when a single element is created (e.g., one class, one state).

.. code-block:: json

   {
     "action": "inject_single_element",
     "diagramType": "ClassDiagram",
     "diagramId": "uuid",
     "elementSpec": {
       "elements": {
         "elem-uuid": {
           "id": "elem-uuid",
           "name": "User",
           "type": "Class",
           "bounds": { "x": 100, "y": 100, "width": 200, "height": 150 },
           "attributes": {
             "attr-uuid": {
               "id": "attr-uuid",
               "name": "email",
               "type": "ClassAttribute",
               "bounds": { "x": 0, "y": 40, "width": 200, "height": 30 }
             }
           }
         }
       },
       "relationships": {}
     }
   }

inject_complete_system
~~~~~~~~~~~~~~~~~~~~~~

Returned when a full diagram is generated (e.g., complete class model).

.. code-block:: json

   {
     "action": "inject_complete_system",
     "diagramType": "ClassDiagram",
     "diagramId": "uuid",
     "replaceExisting": true,
     "systemSpec": {
       "elements": {
         "class-1": { "id": "class-1", "name": "User", "type": "Class", "..." : "..." },
         "class-2": { "id": "class-2", "name": "Order", "type": "Class", "..." : "..." }
       },
       "relationships": {
         "rel-1": {
           "id": "rel-1",
           "type": "ClassBidirectional",
           "source": { "element": "class-1", "multiplicity": "1" },
           "target": { "element": "class-2", "multiplicity": "*" }
         }
       }
     }
   }

inject_modification
~~~~~~~~~~~~~~~~~~~

Returned when modifying an existing diagram.

.. code-block:: json

   {
     "action": "inject_modification",
     "diagramType": "ClassDiagram",
     "diagramId": "uuid",
     "modificationSpec": {
       "elementsToAdd": {
         "new-elem": { "id": "new-elem", "name": "NewClass", "type": "Class" }
       },
       "elementsToUpdate": {
         "existing-elem": { "name": "RenamedClass" }
       },
       "elementsToRemove": ["old-elem-id"],
       "relationshipsToAdd": {},
       "relationshipsToRemove": ["old-rel-id"]
     }
   }

trigger_generator
~~~~~~~~~~~~~~~~~

Returned when code generation is requested.

.. code-block:: json

   {
     "action": "trigger_generator",
     "generatorType": "django",
     "config": {
       "project_name": "myproject",
       "app_name": "myapp",
       "containerization": true
     },
     "diagramType": "ClassDiagram"
   }

trigger_export
~~~~~~~~~~~~~~

.. code-block:: json

   {
     "action": "trigger_export",
     "format": "json"
   }

trigger_deploy
~~~~~~~~~~~~~~

.. code-block:: json

   {
     "action": "trigger_deploy",
     "target": "render"
   }

assistant_message
~~~~~~~~~~~~~~~~~

Generic text response (help, errors, confirmations).

.. code-block:: json

   {
     "action": "assistant_message",
     "message": "I created a User class with id and email attributes."
   }

auto_generate_gui
~~~~~~~~~~~~~~~~~

Triggers automatic GUI generation from ClassDiagram (no LLM).

.. code-block:: json

   {
     "action": "auto_generate_gui",
     "diagramType": "GUINoCodeDiagram",
     "sourceType": "ClassDiagram"
   }

Diagram Element Schemas
-----------------------

ClassDiagram Elements
~~~~~~~~~~~~~~~~~~~~~

**Class:**

.. code-block:: json

   {
     "id": "uuid",
     "name": "ClassName",
     "type": "Class",
     "bounds": { "x": 0, "y": 0, "width": 200, "height": 150 },
     "attributes": {
       "attr-uuid": {
         "id": "attr-uuid",
         "name": "attributeName",
         "type": "ClassAttribute",
         "bounds": { "x": 0, "y": 40, "width": 200, "height": 30 }
       }
     },
     "methods": {
       "method-uuid": {
         "id": "method-uuid",
         "name": "methodName()",
         "type": "ClassMethod",
         "bounds": { "x": 0, "y": 70, "width": 200, "height": 30 }
       }
     }
   }

**Relationship types:**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Type
     - UML Meaning
   * - ``ClassBidirectional``
     - Association (bidirectional)
   * - ``ClassUnidirectional``
     - Association (unidirectional)
   * - ``ClassInheritance``
     - Generalization
   * - ``ClassAggregation``
     - Aggregation (hollow diamond)
   * - ``ClassComposition``
     - Composition (filled diamond)
   * - ``ClassDependency``
     - Dependency (dashed arrow)
   * - ``ClassRealization``
     - Interface realization

**Relationship:**

.. code-block:: json

   {
     "id": "uuid",
     "type": "ClassBidirectional",
     "source": {
       "element": "source-class-uuid",
       "multiplicity": "1",
       "role": "owner"
     },
     "target": {
       "element": "target-class-uuid",
       "multiplicity": "*",
       "role": "items"
     },
     "path": [
       { "x": 200, "y": 75 },
       { "x": 400, "y": 75 }
     ]
   }

StateMachine Elements
~~~~~~~~~~~~~~~~~~~~~

**State:**

.. code-block:: json

   {
     "id": "uuid",
     "name": "StateName",
     "type": "ObjectActivityNode",
     "bounds": { "x": 0, "y": 0, "width": 160, "height": 80 }
   }

**Special states:**

- ``ObjectActivityInitialNode`` — Initial pseudo-state (filled circle)
- ``ObjectActivityFinalNode`` — Final state (circle with border)

**Transition:**

.. code-block:: json

   {
     "id": "uuid",
     "type": "ObjectActivityControlFlow",
     "name": "event [guard] / action",
     "source": { "element": "source-state-uuid" },
     "target": { "element": "target-state-uuid" }
   }

ObjectDiagram Elements
~~~~~~~~~~~~~~~~~~~~~~

**Object (instance):**

.. code-block:: json

   {
     "id": "uuid",
     "name": "objectName : ClassName",
     "type": "ObjectName",
     "bounds": { "x": 0, "y": 0, "width": 200, "height": 120 },
     "attributes": {
       "attr-uuid": {
         "id": "attr-uuid",
         "name": "email = \"admin@example.com\"",
         "type": "ObjectAttribute"
       }
     }
   }

AgentDiagram Elements
~~~~~~~~~~~~~~~~~~~~~

**State:**

.. code-block:: json

   {
     "type": "state",
     "name": "greeting_state",
     "replies": ["Hello! How can I help you?", "Welcome!"],
     "x": 100,
     "y": 100
   }

**Intent:**

.. code-block:: json

   {
     "type": "intent",
     "name": "hello_intent",
     "trainingPhrases": ["hi", "hello", "hey", "good morning"],
     "x": 300,
     "y": 100
   }

**Initial element:**

.. code-block:: json

   {
     "type": "initial",
     "x": 50,
     "y": 50
   }

**Transition:**

.. code-block:: json

   {
     "source": "initial",
     "target": "greeting_state",
     "intent": null
   }

GUINoCodeDiagram Schema (GrapesJS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "pages": [
       {
         "name": "UserManagement",
         "component": "<div class='container'>...</div>",
         "styles": ".container { padding: 20px; }",
         "scripts": ""
       }
     ]
   }

QuantumCircuitDiagram Schema (Quirk)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "cols": [
       [1, 1, "H"],
       ["*", 1, "X"],
       ["Measure", "Measure", "Measure"]
     ],
     "gates": []
   }

**Gate notation:** Each column is an array of operations per qubit. ``1`` means
identity (no operation), ``"H"`` is Hadamard, ``"X"`` is Pauli-X, ``"*"`` is a
control qubit, ``"Measure"`` is measurement.

Internal Schemas
----------------

Operation (Request Planner Output)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "type": "model",
     "diagram_type": "ClassDiagram",
     "mode": "complete_system",
     "request_text": "create a bookstore class diagram"
   }

.. code-block:: json

   {
     "type": "generation",
     "generator": "django",
     "config": { "project_name": "myapp" }
   }

Quality Suggestion
~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "suggestions": [
       "Consider adding an 'id' attribute to the User class",
       "The Customer class has no relationships"
     ],
     "whatsNext": [
       "Create object instances to test your class model",
       "Add a state machine to model User lifecycle"
     ]
   }

Class Metadata (extracted from ClassDiagram)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "classes": [
       {
         "name": "User",
         "attributes": [
           { "name": "id", "isNumeric": true, "isString": false },
           { "name": "email", "isNumeric": false, "isString": true },
           { "name": "age", "isNumeric": true, "isString": false }
         ],
         "methods": ["login()", "logout()"],
         "associations": ["Order", "Profile"]
       }
     ]
   }

Domain Pattern
~~~~~~~~~~~~~~

.. code-block:: json

   {
     "domain": "ecommerce",
     "keywords": ["shop", "store", "order", "cart", "product"],
     "core_classes": [
       {
         "name": "Product",
         "attributes": ["name: String", "price: Float", "stock: Integer"]
       },
       {
         "name": "Customer",
         "attributes": ["name: String", "email: String"]
       }
     ],
     "key_relationships": [
       {
         "source": "Customer",
         "target": "Order",
         "type": "association",
         "sourceMultiplicity": "1",
         "targetMultiplicity": "*"
       }
     ],
     "notes": "Include cart management and payment processing"
   }

State Pattern
~~~~~~~~~~~~~

.. code-block:: json

   {
     "pattern": "order_processing",
     "keywords": ["order", "purchase", "checkout"],
     "states": [
       "Initial",
       "PendingPayment",
       "PaymentProcessing",
       "Confirmed",
       "Preparing",
       "Shipped",
       "Delivered",
       "Final"
     ],
     "transitions": [
       { "source": "Initial", "target": "PendingPayment", "trigger": "place_order" },
       { "source": "PendingPayment", "target": "PaymentProcessing", "trigger": "submit_payment" }
     ]
   }

Supported Type Constants
------------------------

.. code-block:: python

   SUPPORTED_DIAGRAM_TYPES = {
       "ClassDiagram",
       "ObjectDiagram",
       "StateMachineDiagram",
       "AgentDiagram",
       "GUINoCodeDiagram",
       "QuantumCircuitDiagram",
   }

   GENERATOR_KEYWORDS = {
       "django": ["django"],
       "backend": ["full backend", "backend"],
       "web_app": ["web app", "web application"],
       "sqlalchemy": ["sqlalchemy", "sql alchemy"],
       "sql": ["sql ddl", "sql schema"],
       "python": ["python classes", "generate python"],
       "java": ["java classes", "generate java"],
       "pydantic": ["pydantic"],
       "jsonschema": ["json schema", "jsonschema"],
       "smartdata": ["smart data", "smartdata"],
       "agent": ["besser agent", "agent generator"],
       "qiskit": ["qiskit", "quantum code"],
       "export": ["export project", "export to json"],
       "deploy": ["deploy to render", "deploy app"],
   }
