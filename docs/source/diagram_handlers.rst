Diagram Handlers
================

The diagram handler system is the core of the Modeling Agent. Each supported
diagram type has a specialized handler that inherits from ``BaseDiagramHandler``
and implements type-specific generation logic. For the Pydantic schemas each
handler uses, see :doc:`schema`. To add a new handler, see
:doc:`contributing/howto_guides`.

.. contents:: On this page
   :local:
   :depth: 2

Handler Class Hierarchy
-----------------------

.. code-block:: text

   BaseDiagramHandler (abstract)
   │
   ├── ClassDiagramHandler          # UML class diagrams            → "ClassDiagram"
   ├── ObjectDiagramHandler         # UML object diagrams           → "ObjectDiagram"
   ├── StateMachineHandler          # UML state machines            → "StateMachineDiagram"
   ├── AgentDiagramHandler          # BESSER agent diagrams         → "AgentDiagram"
   ├── GUINoCodeDiagramHandler      # GrapesJS GUI models           → "GUINoCodeDiagram"
   ├── QuantumCircuitDiagramHandler # Quirk quantum circuits        → "QuantumCircuitDiagram"
   ├── BPMNDiagramHandler           # BPMN process diagrams         → "BPMN"
   └── UserProfileDiagramHandler    # BESSER user-profile models    → "UserDiagram"

.. note::

   The string on the right is the value returned by ``get_diagram_type()`` —
   the **WME storage-bucket token**, which is what the protocol, the factory
   and ``SUPPORTED_DIAGRAM_TYPES`` all key on. Two of them do not follow the
   ``<Name>Diagram`` convention: BPMN's token is ``"BPMN"`` (the editor's
   converter sets the Apollon ``model.type`` to ``"BPMNDiagram"`` itself),
   and the User Profile handler's token is ``"UserDiagram"``.

BaseDiagramHandler
------------------

**Location:** ``src/diagram_handlers/core/base_handler.py``

Abstract base class providing shared infrastructure for all handlers.

Abstract Methods (must override)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Description
   * - ``get_diagram_type() -> str``
     - Return the diagram type string (e.g., ``"ClassDiagram"``)
   * - ``get_system_prompt() -> str``
     - Return the LLM system prompt for this handler
   * - ``generate_single_element(request, model, **kw) -> dict``
     - Generate a single diagram element
   * - ``generate_complete_system(request, model) -> dict``
     - Generate a complete diagram (no ``**kwargs`` on the base signature)
   * - ``generate_fallback_element(request) -> dict``
     - Return a minimal valid element when all else fails

Shared Concrete Methods
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Description
   * - ``generate_modification()``
     - Default modification handler using LLM
   * - ``predict_with_retry()``
     - Free-text LLM call with jittered exponential backoff. Routed through
       the user's BYOK client when one is active for this request.
   * - ``predict_structured()``
     - OpenAI Structured Outputs call validated against a Pydantic schema.
       Omits ``temperature`` and passes ``reasoning_effort`` for gpt-5 /
       o-series models.
   * - ``predict_two_pass_structured()``
     - Two-pass generation: free-text reasoning, then a structured pass.
       Skips the reasoning pass for raw requests shorter than
       ``_TWO_PASS_MIN_LENGTH`` (250 chars).
   * - ``predict_two_pass()``
     - Older free-text two-pass variant (reasoning then JSON)
   * - ``validate_and_refine()``
     - LLM self-critique loop for output validation
   * - ``self_correct()`` / ``parse_validate_or_correct()``
     - Re-prompt the model with the validation error and re-parse
   * - ``repair_json_response()`` / ``parse_and_validate_with_repair()``
     - Last-resort JSON repair via LLM
   * - ``apply_single_layout()``
     - Layout positioning for single elements
   * - ``apply_system_layout()``
     - Layout positioning for complete systems
   * - ``_error_response()``
     - Standard error format with ``retryable`` flag

Retry Strategy
~~~~~~~~~~~~~~

- **Handler retry:** ``predict_with_retry()`` runs ``1 + max_retries``
  attempts (default 1 retry) with jittered exponential backoff.
- **Shared-client retry:** underneath that, the shared server LLM's SDK client
  is patched once (``src/utilities/llm_retry.py``) to retry transient upstream
  failures — 429 and 5xx — up to ``MAX_ATTEMPTS = 4`` with a bounded backoff
  (~5 s worst case). Non-429 4xx responses fail fast.
- **Fallback:** graceful degradation through multiple levels (primary LLM →
  self-correct / JSON repair → type-specific fallback → error response with
  ``retryable``).

ClassDiagramHandler
-------------------

**Location:** ``src/diagram_handlers/types/class_diagram_handler.py``

Generates UML class diagrams with classes, attributes, methods, and relationships.

Features
~~~~~~~~

- **Domain Pattern Injection:** Detects 10 pre-defined domain patterns (ecommerce,
  hospital, university, etc.) and injects expert knowledge into the prompt.
- **Two-Pass Generation:** Pass 1 produces free-text reasoning; Pass 2 produces
  structured JSON.
- **Validation Loop:** LLM self-critique to catch structural issues.
- **Impact Analysis:** For modifications, analyzes which elements are affected.
- **Incremental Fallback:** If full generation fails, generates class-by-class.
- **Schema-Enforced Naming:** Class names capped at 30 chars, attribute/method names
  at 50 chars via Pydantic ``max_length``. Prevents runaway LLM names.
- **Literal Action Types:** Modification actions (``add_class``, ``modify_attribute``,
  etc.) are ``Literal`` types — the LLM cannot hallucinate invalid actions.
- **Enum Type Safety:** The LLM prompt requires every referenced enum type
  (e.g. ``OrderStatus``) to be created as an Enumeration in the same response.

Domain Patterns (currently disabled)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Domain and state pattern hints are defined in ``src/domain_patterns.py`` and
``src/state_patterns.py`` but are **not currently injected** into the LLM prompt.
They were disabled because current-generation models produce good diagrams
without them, while the pattern injection biased the LLM toward hardcoded
templates.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Domain
     - Sample Keywords
     - Core Classes
   * - ``ecommerce``
     - shop, store, order, cart
     - Product, Customer, Order, Cart, Payment
   * - ``library``
     - book, library, isbn, borrow
     - Book, Member, Loan, Author, Category
   * - ``hospital``
     - patient, doctor, medical
     - Patient, Doctor, Appointment, Prescription
   * - ``university``
     - student, course, enrollment
     - Student, Course, Professor, Enrollment
   * - ``banking``
     - account, bank, transaction
     - BankAccount, Customer, Transaction
   * - ``social_media``
     - post, user, follow, like
     - User, Post, Comment, Like, Follow
   * - ``hotel``
     - hotel, room, booking, guest
     - Hotel, Room, Guest, Booking
   * - ``restaurant``
     - restaurant, menu, order, table
     - Restaurant, MenuItem, Table, Order
   * - ``inventory``
     - inventory, warehouse, stock
     - InventoryItem, Warehouse, StockMovement
   * - ``project_management``
     - project, task, sprint, team
     - Project, Task, Team, Sprint, User

To re-enable pattern injection, import ``get_pattern_hint`` /
``get_state_pattern_hint`` in ``src/execution/model_operations.py``, compute the
hint from the clean ``operation_request`` text, and pass it to the handler as
``domain_hint=...``. See the source files for the full pattern data.

StateMachineHandler
-------------------

**Location:** ``src/diagram_handlers/types/state_machine_handler.py``

Generates UML state machine diagrams with states, transitions, guards, and actions.

Features
~~~~~~~~

- **State Pattern Injection:** 8 behavioral patterns (order processing,
  authentication, task management, etc.)
- **Specialized Validation:** Checks for initial state, final state, and orphan
  states (no incoming or outgoing transitions).
- **Fallback:** Generates a minimal 3-state machine (initial -> active -> final).

State Patterns
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Pattern
     - State Flow
   * - ``order_processing``
     - new -> payment_pending -> confirmed -> shipped -> delivered
   * - ``authentication``
     - idle -> authenticating -> authenticated -> expired
   * - ``document_workflow``
     - draft -> review -> approved / rejected
   * - ``task_management``
     - todo -> in_progress -> review -> done
   * - ``booking_reservation``
     - requested -> confirmed -> checked_in -> completed
   * - ``user_registration``
     - initiated -> email_verification -> profile_setup -> active
   * - ``payment_processing``
     - initiated -> processing -> authorized -> captured
   * - ``support_ticket``
     - open -> assigned -> in_progress -> resolved -> closed

ObjectDiagramHandler
--------------------

**Location:** ``src/diagram_handlers/types/object_diagram_handler.py``

Generates UML object (instance) diagrams based on a ClassDiagram reference.

Features
~~~~~~~~

- **Class Reference Required:** Always resolves the ClassDiagram to use as a
  reference catalog for valid class names and attribute types.
- **Reference Catalog Extraction:** Builds a structured catalog of available
  classes, their attributes, and relationships.
- **Heuristic Value Generator:** Generates realistic attribute values based on
  the attribute type and name (e.g., ``email`` gets ``"user@example.com"``).

AgentDiagramHandler
-------------------

**Location:** ``src/diagram_handlers/types/agent_diagram_handler.py``

Generates BESSER conversational agent diagrams with states, transitions, and
agent components (intents, LLMs, RAG databases, tools, skills, workspaces, GUIs).

Elements
~~~~~~~~

- **State:** Named state with ``replies[]`` (actions run on entry) and
  ``fallbackBodies[]`` (actions run when no intent matches). Each action has a
  ``text`` and a ``replyType`` (see below).
- **Initial:** Starting pseudo-element
- **Transition:** Links states; ``condition`` is ``when_intent_matched`` (with
  the intent name), ``when_no_intent_matched``, or ``auto``
- **Components:** Intents (``trainingPhrases[]``), LLMs, RAG databases, tools,
  skills, workspaces, and GUIs. They have no canvas bounds.

.. note::

   Components are read from the editor's ``components`` section. Older projects
   keep them in ``elements`` (intents on the canvas) or in a legacy top-level
   ``agentComponents`` map; all three are merged when reading
   (``agent_model_elements()`` in ``src/utilities/model_context.py``, in the
   order ``elements``, ``agentComponents``, ``components``; later sections win
   on duplicate ids). The layout engine reserves an
   intent row on the canvas only for old-format models that already have
   ``AgentIntent`` elements.

Reply Types
~~~~~~~~~~~

``ReplyType`` in ``src/schemas/agent_diagram.py`` is the single source of truth
for both the schemas and the handler prompts.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - ``replyType``
     - Purpose
     - Key fields
   * - ``text``
     - Scripted text reply (default)
     - ``text``
   * - ``llm``
     - LLM-generated reply
     - ``system_message``, ``llm_name``, ``inputPromptMode``,
       ``customInputPrompt``, ``storeInSession``, ``sendReply``
   * - ``llm_chat``
     - LLM reply with conversation history
     - same as ``llm``
   * - ``rag``
     - RAG knowledge-base lookup
     - ``ragDatabaseName``, ``llm_name``
   * - ``db_reply``
     - Database query
     - ``dbSelectionType`` (``default``/``custom``), ``dbCustomName``,
       ``dbQueryMode`` (``llm_query``/``sql``), ``dbOperation``
       (``any``/``select``/``insert``/``update``/``delete``),
       ``dbSqlQuery``, ``llm_name``
   * - ``code``
     - Custom Python function
     - ``text`` must be a complete ``def <name>(session):`` function (bare
       code is wrapped automatically)
   * - ``web_crawl_llm``
     - Crawl a URL, then reply via LLM
     - ``initial_url``
   * - ``ws_markdown``, ``ws_html``, ``ws_speech``
     - WebSocket Markdown, HTML, or text-to-speech reply
     - ``ws_message``
   * - ``ws_options``
     - WebSocket option buttons
     - ``ws_options`` (newline-separated)
   * - ``ws_location``
     - WebSocket GPS location
     - ``ws_latitude``, ``ws_longitude``
   * - ``ws_file``, ``ws_image``, ``ws_dataframe``, ``ws_plotly``
     - WebSocket file, image, dataframe, or Plotly chart
     - none
   * - ``gui_reply``
     - Show a GUI page
     - ``guiId`` (the ``gui_id`` of an ``AgentGUI`` component)

Component Specs
~~~~~~~~~~~~~~~

``SystemAgentSpec`` carries ``states`` and ``transitions`` plus one list per
component type: ``intents``, ``llms``, ``ragElements``, ``tools``, ``skills``,
``workspaces``, and ``guis``.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Spec
     - List
     - Fields (defaults)
   * - ``AgentIntentSpec``
     - ``intents``
     - ``intentName``, ``intentDescription``, ``trainingPhrases``
   * - ``AgentLLMSpec``
     - ``llms``
     - ``name``, ``provider`` (``openai``), ``num_previous_messages`` (1),
       ``global_context``
   * - ``AgentRagSpec``
     - ``ragElements``
     - ``name``, ``llm_name``, ``llm_prompt``, ``k`` (4),
       ``embedding_provider`` (``openai``)
   * - ``AgentToolSpec``
     - ``tools``
     - ``name``, ``description``, ``code`` (Python function source)
   * - ``AgentSkillSpec``
     - ``skills``
     - ``name``, ``content``, ``description``
   * - ``AgentWorkspaceSpec``
     - ``workspaces``
     - ``name``, ``path``, ``description``, ``writable`` (true),
       ``max_read_bytes`` (200000)
   * - ``AgentGUISpec``
     - ``guis``
     - ``gui_id``, ``persist`` (true), ``width``, ``is_form`` (false)

Modification Actions
~~~~~~~~~~~~~~~~~~~~

``AgentModification.action`` is a ``Literal``, so the LLM cannot produce any
other action. Component actions take the component name in ``target.name`` and
their fields in ``changes``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Action
     - Target / changes
   * - ``add_state``
     - ``target.stateName``; ``changes.replies``
   * - ``modify_state``, ``modify_intent``
     - ``target.stateName`` / ``target.intentName``; ``changes.name``
   * - ``add_intent``
     - ``target.intentName``; ``changes.trainingPhrases``,
       ``changes.intentDescription``
   * - ``add_transition``, ``remove_transition``
     - ``target.sourceStateName``, ``target.targetStateName``;
       ``changes.condition``, ``changes.intentName``
   * - ``add_state_body``
     - ``target.stateName``; ``changes.text``, ``changes.replyType`` and the
       reply type's fields
   * - ``add_intent_training_phrase``
     - ``target.intentName``; ``changes.trainingPhrase``
   * - ``add_rag_element``
     - ``target.name``; ``changes.llm_name``, ``changes.llm_prompt``,
       ``changes.k``, ``changes.embedding_provider``
   * - ``add_llm``
     - ``target.name``; ``changes.provider``,
       ``changes.num_previous_messages``, ``changes.global_context``
   * - ``add_tool``
     - ``target.name``; ``changes.description``, ``changes.code``
   * - ``add_skill``
     - ``target.name``; ``changes.content``, ``changes.description``
   * - ``add_workspace``
     - ``target.name``; ``changes.path``, ``changes.description``,
       ``changes.writable``
   * - ``add_gui``
     - ``target.name``; ``changes.gui_id``, ``changes.persist``,
       ``changes.is_form``, ``changes.width``
   * - ``remove_element``
     - ``target.stateName`` or ``target.intentName``; no ``changes``. Removes
       a state with its bodies, fallback bodies, and connected transitions, or
       an intent with its training phrases. Other components (LLMs, RAG
       databases, tools, skills, workspaces, GUIs) cannot be removed with it.

.. code-block:: json

   {
     "action": "add_llm",
     "target": {"name": "gpt4"},
     "changes": {"provider": "openai", "num_previous_messages": 3}
   }

The model summary sent to the LLM (``_summarize_agent_diagram()`` in
``src/utilities/model_context.py``) lists states, every component type, and
transitions, so modifications can reference existing components by name.

Features
~~~~~~~~

- **Rich Normalization Pipeline:** 7 normalizers ensure well-formed output:

  1. Ensure all states have replies
  2. Ensure all intents have training phrases
  3. Remove duplicate transitions
  4. Validate source/target references
  5. Ensure unique element names
  6. Auto-insert initial transition if missing
  7. Fix orphan elements

GUINoCodeDiagramHandler
-----------------------

**Location:** ``src/diagram_handlers/types/gui_nocode_diagram_handler.py``

Generates GrapesJS-compatible GUI models for no-code web application design.

Features
~~~~~~~~

- **Auto-Generate Mode:** Creates one page per ClassDiagram class with CRUD
  forms and tables — no LLM call needed.
- **LLM Mode:** Generates customized pages with charts, dashboards, and
  complex layouts via LLM.
- **Per-Domain Design System** (``gui_design_system.py``): a set of design
  themes — ``government``, ``finance``, ``health``, ``startup``, ``default``
  — each a bundle of concrete tokens (palette, type scale, spacing, radius,
  shadow, font stack). ``stylesheet_rules(domain)`` emits GrapesJS CSS rule
  objects in the exact shape ``editor.loadProjectData`` consumes, so a
  generated app actually *looks* like its domain instead of shipping the same
  slate-blue skin every time. ``block_exemplars(domain)`` supplies proven,
  editable-safe HTML composition patterns built from reusable ``.ds-*``
  component classes.
- **HTML → GrapesJS converter** (``gui_html_converter.py``): turns rich themed
  markup written by the LLM into the nested component-definition tree the GUI
  editor loads. Tag identity is preserved (an ``<h2>`` stays ``tagName:"h2"``
  so the frontend's ``markTextEditable`` can make it double-click editable);
  ``<script>``/``<style>``, ``on*`` handlers and external ``href``/``src``
  values are stripped.
- **Widget splicing:** the LLM writes only the page *chrome*, leaving a
  ``<!--WIDGET:kind-->`` marker where data should go. The server splices the
  real, data-bound widget (table, bar/pie/line/radar chart, metric card,
  form, dashboard) into that slot from typed Python builders. LLM markup can
  never masquerade as a widget — ``data-gjs-type``/``data-source`` attributes
  are dropped on parse.
- **Class Metadata Injection:** Extracts class attributes and types to
  auto-populate form fields, table columns, and chart axes.

.. note::

   Generated apps are served under a strict Content-Security-Policy that
   blocks external stylesheets, webfonts and remote images. Every font stack
   in the design system is therefore system-fonts-only, and all imagery in
   the exemplars is a CSS gradient or inline SVG.

GUINoCodeDiagram Generation Modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a ClassDiagram exists in the project, the agent asks which mode to use
and offers the choice as two quick actions:

1. **Fast & deterministic:** one screen per class with data tables and method
   buttons. No LLM call.
2. **AI-Generated (experimental):** personalized screens with navigation,
   styling, and realistic content.

If the user message already contains a customization hint (``chart``,
``dashboard``, ``custom``, ``sidebar``, ``metric``, ``kpi``, ``landing``,
``hero``, ``theme``, ``color``, ``dark``, ``tailored``, …) the AI-generated
path is taken without asking. The choice is stored as a pending flow and
resolved by ``confirmation.handle_pending_gui_choice``.

BPMNDiagramHandler
------------------

**Location:** ``src/diagram_handlers/types/bpmn_diagram_handler.py``

Generates BPMN process diagrams — start/end events, tasks, exclusive /
parallel / inclusive gateways and sequence flows — optionally grouped into
pools (participants) and lanes (roles within a pool).

Features
~~~~~~~~

- **Reasoning-prompt completeness rules:** the two-pass reasoning prompt
  explicitly forbids silently merging several described decision points into
  one gateway — a failure mode found by statistical probing, not by a single
  manual test.
- **Deterministic post-generation repair** (``_validate_and_refine``): pure
  Python fixes for structural invariants the LLM violates despite the system
  prompt — ``_connect_orphaned_nodes`` reattaches nodes with no incoming
  flow, ``_normalize_pool_refs`` and ``_infer_missing_lane_owners`` fix
  pool/lane membership, ``_unique_id`` de-duplicates ids.
- **Reference validation on modification** (``_validate_mod_refs``): a
  modification naming a node that does not exist is rejected rather than
  applied to an arbitrary substitute.
- **Positions are not generated here.** The editor's injector lays out the
  process (and any pools/lanes) and routes the flows itself. Flow *type*
  (message vs. sequence) is likewise derived on the editor side from pool
  membership — the agent never sets it.

.. note::

   Pools and lanes are **generation-only**. ``generate_modification`` does not
   yet support ``add_pool`` / ``add_lane`` actions.

UserProfileDiagramHandler
-------------------------

**Location:** ``src/diagram_handlers/types/user_profile_handler.py``

Generates BESSER **User Profile** models (diagram type ``UserDiagram``). A
user-profile model describes a *target user* as a set of class-instance boxes
drawn from a fixed metamodel (``User``, ``Personal_Information``,
``Competence``, ``Language``, …). Each attribute row is a matching
**criterion** carrying a comparison operator (``age >= 18``, ``level == B2``)
rather than a plain instance value.

It mirrors ``ObjectDiagramHandler``, with two differences:

1. **The reference catalog is bundled, not transmitted.** The editor does not
   send the metamodel, so the handler loads it from disk via
   ``utilities.user_metamodel.load_user_metamodel()`` (cached; degrades to an
   empty catalog if the resource is missing). A curated
   ``load_user_metamodel_semantics()`` supplies element and attribute
   descriptions for the prompt.
2. **Attributes carry an inferred operator** from ``<``, ``<=``, ``==``,
   ``>=``, ``>``.

The handler also understands the metamodel's structure: it builds an
association graph, knows which classes are singletons, computes each class's
path to the ``User`` root, and assembles the required intermediate boxes and
links so a generated profile is always structurally connected.

.. note::

   Step 6 of the checklist below is currently **unfinished for this type**:
   ``UserDiagram`` is not in ``_TARGET_DIAGRAM_TYPES`` and its vocabulary is not
   in ``_SYSTEM_PROMPT``, so the classifier can never return it as a
   ``target_diagram_type``. Profile requests still work because
   ``KEYWORD_TARGETS`` resolves "user profile" / "persona" / "target user" at
   Level 1, but a request that avoids that vocabulary will not reach this
   handler.

QuantumCircuitDiagramHandler
----------------------------

**Location:** ``src/diagram_handlers/types/quantum_circuit_diagram_handler.py``

Generates quantum circuit diagrams in Quirk JSON format.

Features
~~~~~~~~

- **Gate Mapping:** 60+ quantum gate symbol mappings (H, X, Y, Z, CNOT, SWAP,
  Toffoli, etc.)
- **Algorithm Detection:** Recognizes named algorithms (Grover's search, QFT,
  Bell state, teleportation) and generates optimized circuits.
- **Quirk Output:** Produces column-based circuit representation compatible with
  the Quirk quantum circuit simulator.

Layout Engine
-------------

**Location:** ``src/diagram_handlers/core/layout_engine.py``

Deterministic canvas positioning engine. LLMs never emit element positions —
the layout engine runs after every generation.

Algorithm
~~~~~~~~~

1. Collect existing element positions from the model
2. Compute dynamic canvas bounds (expand for large diagrams)
3. Calculate ideal grid shape (approximately square)
4. Assign grid positions, snapped to 20px grid
5. Check collision against all previously placed rectangles
6. Fall back to extending grid if no free position found

Parameters
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Parameter
     - Default Value
     - Description
   * - Canvas X range
     - -900 to 900
     - Horizontal bounds (expanded dynamically by element count)
   * - Canvas Y range
     - -500 to 500
     - Vertical bounds (expanded dynamically by element count)
   * - ``H_GAP``
     - 100px
     - Horizontal gap between elements
   * - ``V_GAP``
     - 80px
     - Vertical gap between elements
   * - ``REL_EXTRA_GAP``
     - 60px
     - Additional gap between classes joined by a relationship
   * - ``MARGIN``
     - 40px
     - Minimum clearance from any occupied rectangle
   * - ``GRID_SNAP``
     - 20px
     - Coordinates snap to multiples of this value

Class diagrams get a dedicated layered layout rather than a plain grid:
``layout_class_system()`` runs a Sugiyama pipeline (build graph → remove
cycles → assign layers → minimize crossings → assign coordinates) so
inheritance and association structure is legible. Per-type entry points
exist for each handler — ``layout_class_single`` / ``_system``,
``layout_object_*``, ``layout_state_*``, ``layout_agent_*``,
``layout_user_*`` — dispatched by the public ``apply_layout()``.

DiagramHandlerFactory
---------------------

**Location:** ``src/diagram_handlers/registry/factory.py``

Registry that maps diagram type strings to handler instances.

Handlers are registered by listing the **class** in the module-level
``HANDLER_CLASSES`` tuple; the factory instantiates each one and keys the
registry on its own ``get_diagram_type()`` return value, so the token can
never drift from the handler that owns it.

.. code-block:: python

   HANDLER_CLASSES = (
       ClassDiagramHandler,
       ObjectDiagramHandler,
       StateMachineHandler,
       AgentDiagramHandler,
       GUINoCodeDiagramHandler,
       QuantumCircuitDiagramHandler,
       BPMNDiagramHandler,
       UserProfileDiagramHandler,
   )


   class DiagramHandlerFactory:
       def __init__(self, llm):
           self.llm = llm
           self._handlers = {}
           for handler_class in HANDLER_CLASSES:
               handler = handler_class(llm)
               self._handlers[handler.get_diagram_type()] = handler

       def get_handler(self, diagram_type: str) -> Optional[BaseDiagramHandler]:
           """Return handler for type, or None if unsupported."""

       def get_supported_types(self) -> list[str]:
           """Return all registered diagram type strings."""

       def is_supported(self, diagram_type: str) -> bool:
           """Check if a diagram type has a registered handler."""

Adding a New Diagram Type
~~~~~~~~~~~~~~~~~~~~~~~~~

A diagram type has to be registered in **every** one of these places. Missing
one produces a stale-list bug that manual testing tends not to catch, because
testing from inside the new diagram's own tab never exercises routing or
discoverability.

1. **Schemas** — add ``src/schemas/<type>.py`` (single-element spec,
   complete-system spec, modification actions) and export them from
   ``src/schemas/__init__.py``.
2. **Handler** — add ``src/diagram_handlers/types/<type>_diagram_handler.py``
   extending ``BaseDiagramHandler`` and implementing the 5 abstract methods:
   ``get_diagram_type``, ``get_system_prompt``, ``generate_single_element``,
   ``generate_complete_system``, ``generate_fallback_element``. Override
   ``generate_modification`` if the default LLM path is not enough.
3. **Factory** — add the class to ``HANDLER_CLASSES`` in
   ``src/diagram_handlers/registry/factory.py``.
4. **Protocol** — add the token to ``SUPPORTED_DIAGRAM_TYPES`` in
   ``src/protocol/types.py``.
5. **Metadata** — add display metadata in
   ``src/diagram_handlers/registry/metadata.py``.
6. **Classifier** — add the token to ``_TARGET_DIAGRAM_TYPES`` in
   ``src/unified_classifier.py`` and teach ``_SYSTEM_PROMPT`` the type's
   vocabulary, so a request naming it routes to the right intent.
7. **Orchestrator** — add ``KEYWORD_TARGETS`` entries, an
   ``_IMPLICIT_PATTERNS`` regex, and append the type to ``FALLBACK_PRIORITY``
   in ``src/orchestrator/workspace_orchestrator.py``.
8. **Capability copy** — ``src/state_bodies.py`` enumerates supported types in
   more than one place (the quick-response capability text and the global
   fallback prompt). Grep for an existing type's name to find them all.
9. **Suggestions** — add a suggestion list in ``src/suggestions.py`` and wire
   it into ``_DIAGRAM_SUGGESTION_HANDLERS``.
10. **Docs** — this page, the README table, and
    :doc:`websocket_protocol`'s supported-types list.
11. **Frontend** (separate repo) — the type must exist in the editor's own
    diagram-type union and be a valid ``activeDiagramType`` context value.

Layout, retry, structured output, two-pass generation, and validation are
inherited automatically from ``BaseDiagramHandler``.
