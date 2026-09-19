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

Most frontend messages use the ``user_message`` action, wrapped in a
BESSER framework envelope. The other inbound actions are listed below.

Envelope Structure
~~~~~~~~~~~~~~~~~~

BAF's ``Payload.decode()`` reads only three top-level keys off a message —
``action``, ``message`` and ``history`` — so the v2 payload is
**JSON-stringified into the ``message`` field** of a ``user_message``
envelope. The wire payload is therefore double-JSON-encoded:

.. code-block:: json

   {
     "action": "user_message",
     "message": "<JSON string of v2 payload>",
     "history": false
   }

``_unwrap_v2_envelope()`` in ``src/protocol/adapters.py`` recovers the inner
payload and merges it over the outer one.

.. note::

   Session identity does **not** travel in the envelope. The frontend appends a
   persisted ``?user_id=`` query parameter to the WebSocket **URL**, and the
   platform reads it off the HTTP request that opened the socket
   (``_extract_user_id_from_request`` in ``patches/websocket_platform.py``).
   Inside the payload, continuity comes from the v2 ``sessionId`` field.

.. note::

   The same double-encoding applies in reverse to streamed replies: each
   chunk arrives wrapped as ``{"action": "agent_reply_str", "message":
   "<JSON string>", "history": false}``, where the inner string is the
   ``stream_start`` / ``stream_chunk`` / ``stream_done`` payload. When probing
   the agent directly (bypassing the browser), replicate and unwrap **both**
   layers — a single-level unwrap silently treats every response as an
   unrecognized action and hangs waiting for a message that already arrived.

Other inbound actions
~~~~~~~~~~~~~~~~~~~~~

Not every message from the frontend is a ``user_message``:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Action
     - Purpose
   * - ``user_message``
     - A text turn, optionally with base64 ``attachments``
   * - ``user_voice``
     - Base64 audio, transcribed by OpenAI speech-to-text. The workspace
       context is sent just before it as a ``_voice_context`` session
       variable, because the transcript arrives as plain text with no JSON
       context of its own.
   * - ``user_set_variable``
     - Sets a BAF session variable. Used to arm BYOK
       (``user_api_key`` / ``user_api_provider`` / ``user_api_model`` /
       ``user_api_base``), to pass ``_voice_context``, and as a keep-alive
       heartbeat.
   * - ``frontend_event``
     - Reports the outcome of an action the frontend executed (e.g. a
       generator finishing). Routed deterministically to the generation
       state — never classified.
   * - ``replay_last_response``
     - Asks the agent to re-send its last completed terminal reply. Used
       after a reconnect that dropped a long-running reply mid-flight.

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
     - **Deprecated and ignored.** The agent resolves the active model from
       ``projectSnapshot`` using ``activeDiagramType`` and
       ``currentDiagramIndices``. The field is tolerated if an older frontend
       still sends it.
   * - ``context.projectSnapshot``
     - No
     - Full project state. ``diagrams`` maps each diagram type to an **array**
       of tabs (``{id, title, model}``); a bare dict is accepted as the
       legacy single-diagram format.
   * - ``context.diagramSummaries``
     - No
     - Array of ``{diagramType, diagramId, title}``. Derived from
       ``projectSnapshot`` when absent.
   * - ``context.currentDiagramIndices``
     - No
     - Active tab index per diagram type (for multi-tab support)
   * - ``context.pilotParticipant``
     - No
     - Pilot-experiment participant label (e.g. ``"P3"``), present only when
       the tab was opened through a facilitator link. Validated against
       ``^[A-Za-z0-9_-]{1,16}$`` and dropped otherwise. Never a name or email.
   * - ``attachments``
     - No
     - Array of uploaded files (PlantUML, images, RDF, XMI, PDF, text)

Supported Diagram Types
~~~~~~~~~~~~~~~~~~~~~~~~

``SUPPORTED_DIAGRAM_TYPES`` in ``src/protocol/types.py``. An
``activeDiagramType`` outside this set is normalized to ``ClassDiagram``.

.. code-block:: text

   ClassDiagram
   ObjectDiagram
   StateMachineDiagram
   AgentDiagram
   GUINoCodeDiagram
   QuantumCircuitDiagram
   BPMN            # NOT "BPMNDiagram" — the editor's converter sets that itself
   UserDiagram     # User Profile models


Outbound Messages (Backend → Frontend)
----------------------------------------

All responses are JSON objects with an ``action`` field that determines the
message type.

Action index
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Action
     - Terminal?
     - Meaning
   * - ``inject_element``
     - Yes
     - Add one element to the canvas
   * - ``inject_complete_system``
     - Yes
     - Inject a full diagram
   * - ``modify_model``
     - Yes
     - Apply one or many modifications
   * - ``assistant_message``
     - Yes
     - Text-only reply (also the shape of a structured error)
   * - ``agent_error``
     - Yes
     - Error the frontend surfaces with a recovery affordance
   * - ``create_diagram_tab``
     - Yes
     - Create a new tab for a diagram type
   * - ``trigger_generator``
     - Yes
     - Run a deterministic BESSER generator
   * - ``trigger_smart_generator``
     - Yes
     - Run the LLM-authored smart generator
   * - ``trigger_github_import``
     - Yes
     - Import a BESSER project from a GitHub repo and resume work on it
   * - ``trigger_export``
     - Yes
     - Export the project
   * - ``trigger_deploy``
     - Yes
     - Open the deploy dialog
   * - ``auto_generate_gui``
     - Yes
     - Deterministically build the GUI diagram from the class diagram
   * - ``progress``
     - No
     - Progress / keep-alive tick
   * - ``stream_start`` / ``stream_chunk`` / ``stream_done``
     - ``stream_done`` only
     - Streamed free-text reply

A subset of the terminal replies is buffered per stable session key so a reply
completed while the socket was reconnecting can be replayed on request:
``inject_complete_system``, ``modify_model``, ``auto_generate_gui``,
``trigger_generator``, ``trigger_github_import`` and ``assistant_message``
(``_TERMINAL_REPLY_ACTIONS`` / ``replay_last_reply`` in
``src/session_helpers.py``).

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
     - Optional list of ``{label, prompt}`` follow-up buttons. Clicking one
       sends its ``prompt`` as the next user message.

.. note::

   ``systemSpec`` is the handler's **simple** format (``classes`` /
   ``relationships`` arrays for a class diagram, and the per-type equivalent
   for the others), *not* the editor's Apollon element/relationship maps. The
   frontend's ``ConverterFactory`` performs that translation — see
   :doc:`end_to_end_flow`.

   Class-diagram ``systemSpec`` objects may also carry a ``constraints`` list
   of OCL invariants. The frontend persists these as ``ClassOCLConstraint``
   elements with ``ClassOCLLink`` attachments to their context classes. A
   persisted constraint is not necessarily enforced by a target generator;
   unsupported rules remain work for the spec-driven agent.

   An ``Association`` may carry ``associationClass: "Enrollment"``. Declare
   ``Enrollment`` in ``classes`` with the attributes belonging to the pairing,
   and attach it to exactly one direct relationship (for example
   ``Student``--``Course``). The frontend emits a ``ClassLinkRel`` whose source
   is the attribute class and whose target is that relationship's ID. BESSER
   converts this to a native ``AssociationClass``; no two extra ordinary
   endpoint associations are needed. Compact LLM output uses ``ac`` for the
   same attachment, expanded before this message is sent. Missing or null
   ``associationClass`` preserves the ordinary relationship behavior.

   This attachment is supported by complete-system generation; the existing
   incremental modification protocol does not yet expose it. Deploy the
   matching frontend converter with the agent schema change: older converters
   ignore this field and lose the native attachment.

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

The valid nested actions are ``Literal`` types on each diagram's Pydantic
modification schema (``src/schemas/``), so the LLM cannot hallucinate one —
Structured Outputs rejects the response and the call retries.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Diagram Type
     - Valid nested actions
   * - ``ClassDiagram``
     - ``add_class``, ``modify_class``, ``add_attribute``,
       ``modify_attribute``, ``add_method``, ``modify_method``,
       ``add_relationship``, ``modify_relationship``, ``remove_element``,
       ``extract_class``, ``split_class``, ``merge_classes``,
       ``promote_attribute``, ``add_enum``, ``add_ocl_constraint``
   * - ``StateMachineDiagram``
     - ``add_state``, ``modify_state``, ``add_transition``,
       ``modify_transition``, ``add_code_block``, ``remove_element``
   * - ``ObjectDiagram``, ``UserDiagram``
     - ``add_object``, ``modify_object``, ``modify_attribute_value``,
       ``add_link``, ``remove_element``
   * - ``AgentDiagram``
     - ``add_state``, ``modify_state``, ``add_intent``, ``modify_intent``,
       ``add_transition``, ``remove_transition``, ``add_state_body``,
       ``add_intent_training_phrase``, ``add_rag_element``,
       ``remove_element``
   * - ``BPMN``
     - ``add_task``, ``add_gateway``, ``add_event``, ``add_flow``,
       ``modify_node``, ``remove_flow``, ``remove_element``
   * - Any (generic)
     - ``modify_element`` — the base handler's fallback shape

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

Error payload sent when something goes wrong and the frontend should surface a
recovery affordance (for example an inline "Add your API key" button, which it
keys on ``rate_limit`` / ``auth_error``).

.. code-block:: json

   {
     "action": "agent_error",
     "errorCode": "rate_limit",
     "message": "We've hit the shared free usage limit for the AI service...",
     "suggestedRecovery": "Add your own API key",
     "retryable": true
   }

Most structured errors are instead sent as an ``assistant_message`` carrying
the same fields plus ``"error": true`` — see ``build_error_response()`` in
``src/errors.py``:

.. code-block:: json

   {
     "action": "assistant_message",
     "error": true,
     "errorCode": "parse_error",
     "message": "I had trouble structuring that response.",
     "suggestedRecovery": "try rephrasing your request more specifically",
     "retryable": false,
     "diagramType": "ClassDiagram"
   }

Error codes
^^^^^^^^^^^

``ErrorCode`` in ``src/errors.py``. Values are lowercase snake_case. Each code
carries a default user-facing message, a recovery hint, and a ``retryable``
flag that the caller may override.

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Error Code
     - Retryable
     - Description
   * - ``llm_failure``
     - yes
     - The AI service is temporarily unavailable
   * - ``parse_error``
     - yes
     - The model returned something that could not be structured
   * - ``validation_error``
     - yes
     - The generated model had structural issues
   * - ``schema_error``
     - —
     - Structured output failed schema validation
   * - ``generation_error``
     - —
     - Generation failed for another reason
   * - ``generation_handler_error``
     - —
     - The generation handler raised
   * - ``timeout``
     - yes
     - The request was too complex to process in time
   * - ``rate_limit``
     - yes
     - Provider rate limit or shared quota reached
   * - ``auth_error``
     - —
     - The API key was rejected
   * - ``context_error``
     - —
     - Required workspace context was missing or unusable
   * - ``unsupported``
     - —
     - The request asks for something the agent does not support
   * - ``prerequisite_missing``
     - —
     - A generator's required diagram does not exist yet
   * - ``handler_missing``
     - —
     - No handler registered for the target diagram type
   * - ``unknown``
     - —
     - Unclassified failure

create_diagram_tab
~~~~~~~~~~~~~~~~~~

Create a new tab for a diagram type. Emitted when the user answers a
replace/keep confirmation with "new tab"; the injection payload that follows
carries ``replaceExisting: true`` so it fills the freshly created tab.

.. code-block:: json

   {
     "action": "create_diagram_tab",
     "diagramType": "ClassDiagram"
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

trigger_smart_generator
~~~~~~~~~~~~~~~~~~~~~~~

Run the LLM-authored **smart** generator. Emitted when the request names a
stack BESSER has no deterministic generator for, or a BESSER stack plus
extras the template cannot produce (auth, JWT, Docker, migrations, tests, …).
Built by ``build_trigger_smart_generator_payload()`` in
``src/handlers/smart_generation_handler.py``.

.. code-block:: json

   {
     "action": "trigger_smart_generator",
     "instructions": "Rails 7 with PostgreSQL via Active Record and Devise auth",
     "provider": "anthropic",
     "llmModel": "claude-sonnet-4-6",
     "message": "Generating your application from your specs…"
   }

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``instructions``
     - The polished prompt for the generator. Required — the frontend aborts
       on an empty value. Names the stack and any non-functional requirements
       the user mentioned; it does **not** restate the class diagram, which
       the generator already has.
   * - ``provider``
     - Suggested provider. The frontend's BYOK selection can override it.
   * - ``llmModel``
     - Default model for that provider

The smart generator itself does **not** run over this WebSocket — the
frontend calls the BESSER backend's spec-driven HTTP/SSE endpoints with the
user's key. This action only hands it the instructions.

trigger_github_import
~~~~~~~~~~~~~~~~~~~~~

Resume work on a project that was previously generated and pushed to GitHub.
The agent never touches GitHub itself: the frontend calls the backend's import
endpoint, loads the returned project, and arms the incremental-modify
machinery.

.. code-block:: json

   {
     "action": "trigger_github_import",
     "owner": "besser-pearl",
     "repo": "my-generated-app",
     "branch": null,
     "message": "Importing **besser-pearl/my-generated-app** from GitHub..."
   }

``branch`` is ``null`` when the user named none; the backend then uses the
repository's default branch.

.. note::

   This route is matched **deterministically** by regex, not inferred by the
   classifier — an import must never be invented, missed, or swallowed. The
   bare ``owner/repo`` form only fires alongside an explicit continuation verb,
   so "create a diagram like github.com/x/y" is never hijacked.

auto_generate_gui
~~~~~~~~~~~~~~~~~

Deterministically build the GUI diagram from the class diagram — one page per
class, no LLM call. The frontend does the building via
``autoGenerateGUIFromClassDiagram``; ``message`` confirms completion and names
the pages that were created.

.. code-block:: json

   {
     "action": "auto_generate_gui",
     "diagramType": "GUINoCodeDiagram",
     "message": "I created screens for Book, Author and Member.",
     "suggestedActions": [
       {"label": "Generate the web app", "prompt": "generate the web app"}
     ]
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
     { "action": "progress", "message": "Thinking about your Class Diagram design..." }
     { "action": "inject_complete_system", "diagramType": "ClassDiagram",
       "systemSpec": { ... } }
     { "action": "assistant_message",
       "message": "Your model is ready. Shall I generate the Django project?",
       "suggestedActions": [ { "label": "Generate", "prompt": "generate" } ] }

   Frontend → Backend:
     { "message": "generate" }

   Backend → Frontend:
     { "action": "trigger_generator", "generatorType": "django",
       "config": { "project_name": "library", "app_name": "library_app",
                   "containerization": true } }

.. important::

   A mixed "design X **and** generate Y" plan deliberately **pauses** after the
   modeling step instead of running the generator straight through. The
   generator is stashed behind ``PLAN_GENERATION_CONFIRM_FLAG`` and only runs
   once the user explicitly confirms, so a user who only wanted the model does
   not get a code-generation run they never asked for.

   Generators with required config (``sql`` needs ``dialect``, ``sqlalchemy``
   needs ``dbms``, ``qiskit`` needs ``backend`` and ``shots``, ``export`` needs
   ``format``) ask for those values in the same way before the
   ``trigger_generator`` payload is emitted; ``GENERATOR_REQUIRED_FIELDS`` in
   ``src/handlers/generation_handler.py`` is the source of truth.

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
