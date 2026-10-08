Schema Reference
================

This document describes all JSON schemas used by the Modeling Agent for request
parsing, response generation, and inter-component communication. For the
WebSocket transport layer, see :doc:`websocket_protocol`. For user-facing
examples, see :doc:`usage`.

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
       "projectSnapshot": {
         "name": "MyProject",
         "diagrams": {
           "ClassDiagram": [
             { "id": "diag-1", "title": "Main", "model": {} }
           ]
         }
       },
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

.. list-table:: AssistantRequest Fields
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Type
     - Description
   * - ``action``
     - ``str``
     - Request type: ``"user_message"`` (default), ``"frontend_event"``,
       ``"replay_last_response"``. Voice and session-variable messages arrive
       as ``"user_voice"`` / ``"user_set_variable"`` and are handled before
       this object is built.
   * - ``protocolVersion``
     - ``str``
     - Always ``"2.0"`` for v2 clients
   * - ``clientMode``
     - ``str``
     - Which surface sent the message — ``"workspace"`` (the drawer, the
       adapter's default) or ``"widget"``
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
     "projectSnapshot": {
       "name": "MyProject",
       "diagrams": {
         "ClassDiagram": [
           { "id": "diag-1", "title": "Main", "model": { "elements": {}, "relationships": {} } }
         ],
         "StateMachineDiagram": []
       }
     },
     "diagramSummaries": [
       { "diagramType": "ClassDiagram", "diagramId": "diag-1", "title": "Main" }
     ],
     "currentDiagramIndices": { "ClassDiagram": 0 },
     "sessionId": "abc-123",
     "pilotParticipant": null
   }

.. list-table:: WorkspaceContext Fields
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Type
     - Description
   * - ``activeDiagramType``
     - ``str``
     - Currently active diagram tab. Normalized to ``ClassDiagram`` if it is
       not in ``SUPPORTED_DIAGRAM_TYPES``.
   * - ``activeDiagramId``
     - ``str``
     - UUID of active diagram
   * - ``activeModel``
     - ``object``
     - **Deprecated and ignored.** The active model is resolved from
       ``projectSnapshot`` using ``activeDiagramType`` and
       ``currentDiagramIndices``; a first-tab-with-a-model fallback applies.
   * - ``projectSnapshot.diagrams``
     - ``object``
     - Maps each diagram type to an **array** of tabs
       (``{id, title, model}``). A bare dict is accepted as the legacy
       single-diagram format.
   * - ``diagramSummaries``
     - ``array``
     - ``{diagramType, diagramId, title}`` entries. Derived from
       ``projectSnapshot`` when absent.
   * - ``currentDiagramIndices``
     - ``object``
     - Active tab index per diagram type (default 0)
   * - ``pilotParticipant``
     - ``str``
     - Optional opt-in study participant label, validated against
       ``^[A-Za-z0-9_-]{1,16}$`` and dropped otherwise

FileAttachment
~~~~~~~~~~~~~~

.. code-block:: json

   {
     "filename": "model.puml",
     "content": "QGN0YXJ0dW1s...",
     "mimeType": "text/plain"
   }

Response Schemas
----------------

.. important::

   The agent emits its own **simple** spec format, not the editor's Apollon
   element/relationship maps. The frontend's ``ConverterFactory`` translates
   between them — it generates UUIDs, bounds and Apollon type names. Nothing
   below carries editor UUIDs or bounds.

inject_element
~~~~~~~~~~~~~~

Returned when a single element is created (e.g., one class, one state).

.. code-block:: json

   {
     "action": "inject_element",
     "diagramType": "ClassDiagram",
     "diagramId": "uuid",
     "element": {
       "className": "User",
       "attributes": [
         { "name": "id", "type": "String", "visibility": "public" },
         { "name": "email", "type": "String", "visibility": "private" }
       ],
       "methods": []
     },
     "message": "Added **User** …"
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
       "systemName": "E-commerce System",
       "classes": [
         {
           "className": "User",
           "attributes": [ { "name": "id", "type": "String", "visibility": "public" } ],
           "methods": []
         },
         {
           "className": "Order",
           "attributes": [ { "name": "total", "type": "Float", "visibility": "public" } ],
           "methods": []
         }
       ],
       "relationships": [
         {
           "type": "Association",
           "source": "User",
           "target": "Order",
           "sourceMultiplicity": "1",
           "targetMultiplicity": "0..*",
           "name": "places"
         }
       ]
     },
     "message": "Built the **E-commerce System** with 2 classes."
   }

Each diagram type has its own ``systemSpec`` shape — ``SystemClassSpec``,
``SystemStateMachineSpec``, ``SystemObjectSpec``, ``SystemAgentSpec``,
``SystemGUISpec``, ``SystemQuantumCircuitSpec``, ``SystemBPMNSpec``,
``SystemUserProfileSpec``. See `Structured Output Schemas (Pydantic)`_.

modify_model
~~~~~~~~~~~~

Returned when modifying an existing diagram. Single modifications use
``modification``; batches use ``modifications``.

.. code-block:: json

   {
     "action": "modify_model",
     "diagramType": "ClassDiagram",
     "diagramId": "uuid",
     "modifications": [
       {
         "action": "add_attribute",
         "target": { "className": "User" },
         "changes": { "name": "phone", "type": "String", "visibility": "public" }
       },
       {
         "action": "remove_element",
         "target": { "className": "LegacyOrder" }
       }
     ],
     "message": "Applied 2 changes."
   }

The nested ``action`` is a ``Literal`` on each diagram's modification schema —
see :doc:`websocket_protocol` for the per-type list.

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
     "platform": "render",
     "config": {},
     "message": "Opening the **Deploy to Render** dialog…"
   }

trigger_smart_generator
~~~~~~~~~~~~~~~~~~~~~~~

Returned when the request needs the LLM-authored generator rather than a
BESSER built-in.

.. code-block:: json

   {
     "action": "trigger_smart_generator",
     "instructions": "Rails 7, PostgreSQL via Active Record, Devise auth",
     "provider": "anthropic",
     "llmModel": "claude-sonnet-4-6",
     "message": "Generating your application from your specs…"
   }

trigger_github_import
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "action": "trigger_github_import",
     "owner": "besser-pearl",
     "repo": "my-generated-app",
     "branch": null,
     "message": "Importing **besser-pearl/my-generated-app** from GitHub…"
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

Triggers automatic GUI generation from the ClassDiagram (no LLM). The
frontend builds one page per class.

.. code-block:: json

   {
     "action": "auto_generate_gui",
     "diagramType": "GUINoCodeDiagram",
     "message": "I created screens for Book, Author and Member.",
     "suggestedActions": [
       { "label": "Generate the web app", "prompt": "generate the web app" }
     ]
   }

create_diagram_tab
~~~~~~~~~~~~~~~~~~

Creates a new tab for a diagram type. Emitted when the user answers a
replace/keep confirmation with "new tab".

.. code-block:: json

   {
     "action": "create_diagram_tab",
     "diagramType": "ClassDiagram"
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

The agent handler emits specs (not editor JSON). States and the initial node
get canvas positions from the layout engine; intents and the other components
(LLMs, RAG databases, tools, skills, workspaces, GUIs) go to the editor's
``components`` section without bounds.

**State:**

.. code-block:: json

   {
     "type": "state",
     "stateName": "greetingState",
     "replies": [
       { "text": "Hello! How can I help you?", "replyType": "text" },
       { "text": "Answer briefly.", "replyType": "llm", "llm_name": "gpt4" }
     ],
     "fallbackBodies": [
       { "text": "Sorry, I did not get that.", "replyType": "text" }
     ]
   }

Each reply carries a ``replyType`` (one of the 17 ``ReplyType`` values, see
:doc:`diagram_handlers`) plus the fields that type needs.

**Intent:**

.. code-block:: json

   {
     "type": "intent",
     "intentName": "HelloIntent",
     "intentDescription": "The user greets the agent",
     "trainingPhrases": ["hi", "hello", "hey", "good morning"]
   }

**Initial element:**

.. code-block:: json

   {
     "type": "initial"
   }

**Transition:**

.. code-block:: json

   {
     "source": "initial",
     "target": "greetingState",
     "condition": "auto",
     "conditionValue": "",
     "label": ""
   }

``condition`` is ``when_intent_matched`` (``conditionValue`` holds the intent
name), ``when_no_intent_matched``, or ``auto``.

**Components** (in a complete system, one list per type):

.. code-block:: json

   {
     "llms": [{ "name": "gpt4", "provider": "openai", "num_previous_messages": 3 }],
     "ragElements": [{ "name": "faqKB", "llm_name": "gpt4", "k": 4 }],
     "tools": [{ "name": "getWeather", "description": "...", "code": "def get_weather(city): ..." }],
     "skills": [{ "name": "politeness", "content": "Always greet the user." }],
     "workspaces": [{ "name": "docs", "path": "./docs", "writable": false }],
     "guis": [{ "gui_id": "orderForm", "is_form": true }]
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

Field names are camelCase and the mode is one of ``complete_system`` /
``modify_model`` (``ALLOWED_MODEL_MODES`` in
``src/orchestrator/request_planner.py``).

.. code-block:: json

   {
     "type": "model",
     "diagramType": "ClassDiagram",
     "mode": "complete_system",
     "request": "create a bookstore class diagram"
   }

.. code-block:: json

   {
     "type": "generation",
     "generatorType": "django",
     "config": { "project_name": "myapp" }
   }

UnifiedClassification (Router Output)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Pydantic object returned by the single per-message classification call
(``src/unified_classifier.py``). Every downstream state body reads from it, so
no second LLM call is ever needed to refine routing. See
:doc:`intent_recognition` for the full field reference.

.. code-block:: json

   {
     "intent": "generation_intent",
     "generation_route": "smart",
     "generator_type": null,
     "refined_instructions": "Rails 7 with PostgreSQL and Devise auth",
     "provider": "anthropic",
     "domain_mismatch": false,
     "suggested_new_domain": null,
     "target_diagram_type": null,
     "model_disposition": "reuse_for_generation",
     "needs_clarification": false,
     "clarifying_question": null,
     "pending_flow_action": null,
     "pending_flow_answer": null,
     "reason": "User named a non-BESSER stack, so the smart route applies."
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

Domain Pattern (currently disabled)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

State Pattern (currently disabled)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Structured Output Schemas (Pydantic)
-------------------------------------

The Modeling Agent uses `OpenAI Structured Outputs <https://platform.openai.com/docs/guides/structured-outputs>`_
backed by Pydantic models to guarantee valid JSON from the LLM. All schemas
live in ``src/schemas/`` and enforce naming constraints, valid action types,
and enum-safe field values at the schema level.

ClassDiagram Schemas
~~~~~~~~~~~~~~~~~~~~

**Location:** ``src/schemas/class_diagram.py``

**Generation schemas:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Schema
     - Purpose
   * - ``SingleClassSpec``
     - One UML class (``className`` max 30 chars, PascalCase)
   * - ``SystemClassSpec``
     - Complete class diagram (list of ``SingleClassSpec`` + ``RelationshipSpec``)
   * - ``AttributeSpec``
     - Attribute with name (max 50), type, visibility, isDerived, defaultValue, isOptional
   * - ``MethodSpec``
     - Method with name (max 50), returnType, parameters, visibility, implementationType, code
   * - ``RelationshipSpec``
     - Relationship with Literal type (Association, Inheritance, Composition, Aggregation, Realization, Dependency)

**Modification schemas:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Schema
     - Purpose
   * - ``ClassModificationResponse``
     - Wraps a list of ``ClassModification`` items
   * - ``ClassModification``
     - Single modification with a Literal ``action`` and ``ClassModificationChanges``

**Valid modification actions (enforced by Literal type):**

``add_class``, ``modify_class``, ``add_attribute``, ``modify_attribute``,
``add_method``, ``modify_method``, ``add_relationship``, ``modify_relationship``,
``remove_element``, ``extract_class``, ``split_class``, ``merge_classes``,
``promote_attribute``, ``add_enum``, ``add_ocl_constraint``

**OCL constraints:** ``SystemClassSpec`` may carry a ``constraints`` list of
``OCLConstraintSpec`` invariants. The agent generates and emits them, but the
frontend converter and the editor's class-diagram JSON have no slot for them
yet, so they are not persisted. See :doc:`websocket_protocol`.

Compact Class Diagram Schema
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``src/schemas/compact_class_diagram.py``

A trimmed generation schema — ``CompactClassSpec``,
``CompactRelationshipSpec``, ``CompactSystemClassSpec`` — used instead of the
full ``SystemClassSpec`` when ``BESSER_AGENT_COMPACT_SPEC`` is enabled (the
default). Fewer tokens on both the prompt and the completion side for the same
diagram.

StateMachine Schemas
~~~~~~~~~~~~~~~~~~~~

**Location:** ``src/schemas/state_machine.py``

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Schema
     - Purpose
   * - ``SingleStateSpec``
     - One state (``stateName`` max 30 chars)
   * - ``SystemStateMachineSpec``
     - Complete state machine (states + transitions + codeBlocks)
   * - ``StateMachineModification``
     - Literal actions: ``add_state``, ``modify_state``, ``add_transition``, ``modify_transition``, ``add_code_block``, ``remove_element``

ObjectDiagram Schemas
~~~~~~~~~~~~~~~~~~~~~

**Location:** ``src/schemas/object_diagram.py``

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Schema
     - Purpose
   * - ``SingleObjectSpec``
     - One object instance (``objectName`` max 30, ``className`` max 30)
   * - ``SystemObjectSpec``
     - Complete object diagram (objects + links)
   * - ``ObjectModification``
     - Literal actions: ``add_object``, ``modify_object``, ``modify_attribute_value``, ``add_link``, ``remove_element``

AgentDiagram Schemas
~~~~~~~~~~~~~~~~~~~~

**Location:** ``src/schemas/agent_diagram.py``

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Schema
     - Purpose
   * - ``ReplyType``
     - ``Literal`` of the 17 state action types: ``text``, ``llm``,
       ``llm_chat``, ``rag``, ``db_reply``, ``code``, ``web_crawl_llm``,
       ``ws_markdown``, ``ws_html``, ``ws_speech``, ``ws_options``,
       ``ws_location``, ``ws_file``, ``ws_image``, ``ws_dataframe``,
       ``ws_plotly``, ``gui_reply``. ``REPLY_TYPE_HINTS`` (checked against it at
       import) and ``reply_type_help()`` feed the same list into the prompts.
   * - ``AgentReplyFields``
     - Shared base with the type-specific action fields: ``ragDatabaseName``,
       ``system_message``, ``llm_name``, ``inputPromptMode``
       (``Literal["last_user_message", "custom"]``), ``customInputPrompt``,
       ``storeInSession``, ``sendReply``, ``dbSelectionType``
       (``Literal["default", "custom"]``), ``dbCustomName``, ``dbQueryMode``
       (``Literal["llm_query", "sql"]``), ``dbOperation``
       (``Literal["any", "select", "insert", "update", "delete"]``),
       ``dbSqlQuery``, ``initial_url``, ``ws_message``, ``ws_options``,
       ``ws_latitude``, ``ws_longitude``, ``guiId``
   * - ``AgentReplySpec``
     - One state action: ``AgentReplyFields`` + ``text`` + ``replyType``
       (default ``text``)
   * - ``AgentStateSpec``
     - Agent state (``stateName`` max 30) with ``replies`` and ``fallbackBodies``
   * - ``AgentIntentSpec``
     - Intent (``intentName`` max 30) with ``intentDescription`` and
       ``trainingPhrases``
   * - ``AgentLLMSpec``, ``AgentRagSpec``, ``AgentToolSpec``,
       ``AgentSkillSpec``, ``AgentWorkspaceSpec``, ``AgentGUISpec``
     - Component specs (fields listed in :doc:`diagram_handlers`)
   * - ``AgentSingleElementSpec``
     - One element: ``type`` is ``state``, ``intent``, or ``initial``, with the
       matching state / intent fields
   * - ``AgentTransitionSpec``
     - ``source``, ``target``, ``condition``
       (``Literal["when_intent_matched", "when_no_intent_matched", "auto"]``),
       ``conditionValue``, ``label``, ``sourceDirection``, ``targetDirection``
   * - ``SystemAgentSpec``
     - Complete agent diagram: ``systemName``, ``hasInitialNode``,
       ``initialNode``, ``states`` (at least one), ``transitions``, and the
       component lists ``intents``, ``ragElements``, ``llms``, ``tools``,
       ``skills``, ``workspaces``, ``guis``
   * - ``AgentModificationTarget``
     - ``stateName``, ``intentName``, ``sourceStateName``,
       ``targetStateName``, ``transitionId``, and ``name`` (component name for
       the ``add_*`` component actions)
   * - ``AgentModificationChanges``
     - ``AgentReplyFields`` + ``name``, ``replies``, ``trainingPhrases``,
       ``intentDescription``, ``intentName``, ``condition``, ``text``,
       ``replyType``, ``trainingPhrase``, and the component fields
       (``provider``, ``num_previous_messages``, ``global_context``,
       ``description``, ``code``, ``content``, ``path``, ``writable``,
       ``llm_prompt``, ``k``, ``embedding_provider``, ``gui_id``, ``persist``,
       ``is_form``, ``width``)
   * - ``AgentModification``
     - 15 ``Literal`` actions: ``add_state``, ``modify_state``, ``add_intent``,
       ``modify_intent``, ``add_transition``, ``remove_transition``,
       ``add_state_body``, ``add_intent_training_phrase``, ``add_rag_element``,
       ``add_llm``, ``add_tool``, ``add_skill``, ``add_workspace``, ``add_gui``,
       ``remove_element``
   * - ``AgentModificationResponse``
     - ``modifications`` (at least one ``AgentModification``)

GUINoCode & QuantumCircuit Schemas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``src/schemas/gui_diagram.py``, ``src/schemas/quantum_circuit.py``

Quantum circuits define ``SystemQuantumCircuitSpec`` over
``QuantumOperationSpec`` entries, with ``SingleQuantumGateSpec`` for a single
gate and ``QuantumModificationSpec`` for edits.

The GUI schemas come in two families:

- **Typed builders** — ``SystemGUISpec`` / ``GUIPageSpec`` / ``GUISectionSpec``
  with ``GUIBindSpec`` describing a data-bound widget (kind, source class,
  columns, rows, sample data), plus ``GUIStatItem``, ``GUITableRow`` and
  ``GUISampleDataPoint``.
- **Authored HTML** — ``AuthoredSystemGUISpec`` / ``AuthoredGUIPageSpec`` /
  ``AuthoredGUISectionSpec`` plus ``GUIThemeSpec``. Here the LLM writes themed
  ``.ds-*`` markup containing ``<!--WIDGET:kind-->`` markers, and the server
  splices real data-bound widgets into those slots. See
  :doc:`diagram_handlers`.

Modifications use ``GUIModificationSpec`` / ``GUIModificationBatchSpec``.

BPMN Schemas
~~~~~~~~~~~~

**Location:** ``src/schemas/bpmn.py``

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Schema
     - Purpose
   * - ``BPMNNodeSpec``
     - A process node — event, task (with ``taskType``) or gateway (with
       ``gatewayType``)
   * - ``BPMNFlowSpec``
     - A flow between two nodes. The agent never sets the flow *type*; the
       editor derives message vs. sequence from pool membership.
   * - ``BPMNPoolSpec`` / ``BPMNLaneSpec``
     - Participants and the roles within them. Generation-only — the
       modification path has no ``add_pool`` / ``add_lane`` action.
   * - ``SystemBPMNSpec``
     - Complete process: nodes + flows + optional pools/lanes
   * - ``BPMNModification``
     - Literal actions: ``add_task``, ``add_gateway``, ``add_event``,
       ``add_flow``, ``modify_node``, ``remove_flow``, ``remove_element``

UserProfile Schemas
~~~~~~~~~~~~~~~~~~~

**Location:** ``src/schemas/user_profile.py``

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Schema
     - Purpose
   * - ``UserProfileAttributeSpec``
     - One matching **criterion**: a name, a comparison operator
       (``<``, ``<=``, ``==``, ``>=``, ``>``) and a value — not a plain
       instance value
   * - ``SingleUserProfileSpec``
     - One class-instance box drawn from the bundled metamodel, carrying the
       metamodel's ``classId`` verbatim
   * - ``UserProfileLinkSpec``
     - A link between two boxes
   * - ``SystemUserProfileSpec``
     - Complete profile: boxes + links
   * - ``UserProfileModification``
     - Literal actions: ``add_object``, ``modify_object``,
       ``modify_attribute_value``, ``add_link``, ``remove_element``

Schema Validation Guarantees
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All schemas enforce:

- **Name length limits** — ``max_length=30`` on element names (classes, states, objects,
  intents), ``max_length=50`` on member names (attributes, methods, parameters, page names).
  Prevents the LLM from generating absurdly long names.

- **Literal action types** — Modification ``action`` fields use ``Literal`` instead of
  bare ``str``, so the LLM cannot hallucinate invalid actions. If it tries, OpenAI
  Structured Outputs rejects the response and triggers a retry.

- **Literal enum fields** — Fields like ``relationshipType``, ``implementationType``,
  ``stateType``, ``condition``, and ``replyType`` in modification schemas use ``Literal``
  types matching their generation-schema counterparts.

Supported Type Constants
------------------------

.. code-block:: python

   # src/protocol/types.py
   SUPPORTED_DIAGRAM_TYPES = {
       "ClassDiagram",
       "ObjectDiagram",
       "StateMachineDiagram",
       "AgentDiagram",
       "GUINoCodeDiagram",
       "QuantumCircuitDiagram",
       "BPMN",
       "UserDiagram",
   }

   # src/handlers/generation_handler.py — keys only; each maps to a
   # keyword list. "export" and "deploy" are actions, not generators.
   GENERATOR_KEYWORDS.keys() == {
       "django", "web_app", "backend", "sqlalchemy", "sql", "python",
       "java", "pydantic", "jsonschema", "smartdata", "agent", "qiskit",
       "rest_api", "rdf", "export", "deploy",
   }

   # Required config fields, asked for before trigger_generator is emitted.
   GENERATOR_REQUIRED_FIELDS = {
       "django": [], "backend": [], "sql": ["dialect"],
       "sqlalchemy": ["dbms"], "jsonschema": ["mode"], "smartdata": [],
       "qiskit": ["backend", "shots"], "rest_api": [], "rdf": [],
       "export": ["format"], "deploy": [],
   }

   EXPORT_FORMATS = ["json", "buml"]
   DIALECT_VALUES = ["sqlite", "postgresql", "mysql", "mssql", "mariadb", "oracle"]
   MODE_VALUES = ["regular", "smart_data"]
   QISKIT_BACKENDS = ["aer_simulator", "fake_backend", "ibm_quantum"]

   # src/diagram_handlers/registry/factory.py — registered handler classes.
   # The registry key is each handler's own get_diagram_type() value.
   HANDLER_CLASSES = (
       ClassDiagramHandler, ObjectDiagramHandler, StateMachineHandler,
       AgentDiagramHandler, GUINoCodeDiagramHandler,
       QuantumCircuitDiagramHandler, BPMNDiagramHandler,
       UserProfileDiagramHandler,
   )
