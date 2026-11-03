Orchestration
=============

The orchestration layer is responsible for planning multi-step operations and
resolving which diagram type should be targeted for a given user request.

.. contents:: On this page
   :local:
   :depth: 2

Request Planner
---------------

**Location:** ``src/orchestrator/request_planner.py``

The request planner converts a single user message into an ordered list of
operations. It uses either a heuristic approach or an LLM-based planner
depending on request complexity.

Decision Flow
~~~~~~~~~~~~~

The request planner uses a 3-tier approach to minimize LLM calls:

- **Tier 0 -- Fast heuristic regex patterns:** A bank of compiled regular
  expressions matches common request shapes (e.g., "create a web app for X",
  "generate Django", "create a GUI for this system", "add a state machine").
  Handles ~90% of simple requests with zero LLM calls.
- **Tier 1 -- Keyword-based fallback with discriminating patterns:** Diagram
  type is resolved via explicit keywords (``KEYWORD_TARGETS``) or discriminating
  regex patterns (``_IMPLICIT_PATTERNS``). Skips the LLM planner when the
  intent classifier already resolved a single target with no generation request.
  When no pattern matches a creation intent, escalates to Tier 2 instead of
  defaulting blindly.
- **Tier 2 -- LLM planner:** Only genuinely complex multi-step requests
  (multiple diagram types + generation in one message) invoke the LLM for
  decomposition.

``_should_use_llm_planner()`` now includes a fast-path that returns ``False``
when ``matched_intent`` is a single-target intent and ``inferred_target_count``
is 1, allowing Tier 0 and Tier 1 to handle the request without invoking the
LLM.

After planning, the result passes through **normalize operations**
(``_normalize_operations()``):

- Deduplicate identical operations
- Validate operation shapes
- Enforce ClassDiagram-first ordering (required by other handlers)

If neither tier produces valid operations, the heuristic fallback is used.

Operation Format
~~~~~~~~~~~~~~~~

Each operation is a dict with one of two types:

**Model operation:**

.. code-block:: json

   {
     "type": "model",
     "diagramType": "ClassDiagram",
     "mode": "complete_system",
     "request": "create a bookstore class diagram"
   }

``mode`` is one of ``complete_system`` or ``modify_model``
(``ALLOWED_MODEL_MODES``). ``request`` is a **focused sub-request** for that
one diagram, carrying enough domain detail for the handler to act on it alone
— not a bare "create a class diagram".

**Generation operation:**

.. code-block:: json

   {
     "type": "generation",
     "generatorType": "django",
     "config": { "project_name": "myapp" }
   }

Generator Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~

When a generation operation is planned, the planner checks whether the required
diagram types exist. Missing prerequisites are auto-injected as modeling
operations before the generation step.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Generator
     - Required Diagram Types
   * - ``web_app``
     - ClassDiagram, GUINoCodeDiagram
   * - ``react``
     - ClassDiagram, GUINoCodeDiagram
   * - ``django``
     - ClassDiagram
   * - ``backend``
     - ClassDiagram
   * - ``sql``
     - ClassDiagram
   * - ``sqlalchemy``
     - ClassDiagram
   * - ``python``
     - ClassDiagram
   * - ``java``
     - ClassDiagram
   * - ``pydantic``
     - ClassDiagram
   * - ``jsonschema``
     - ClassDiagram
   * - ``smartdata``
     - ClassDiagram
   * - ``rest_api``
     - ClassDiagram
   * - ``rdf``
     - ClassDiagram
   * - ``agent``
     - AgentDiagram
   * - ``qiskit``
     - QuantumCircuitDiagram

``export`` and ``deploy`` have no prerequisites. ``GENERATOR_PREREQUISITES``
(``src/routing/generation_signals.py``) is the source of truth, and it is
also injected verbatim into the Tier-2 planner prompt so the LLM planner
orders operations correctly.

.. note::

   ``react`` appears in ``GENERATOR_PREREQUISITES`` but is not a key of
   ``GENERATOR_KEYWORDS``, so it is not in ``ALLOWED_GENERATORS`` and no route
   can currently produce it. Treat that row as reserved, not reachable.

Example
~~~~~~~

User message: ``"create a bookstore class model and then generate django"``

Planned operations:

1. ``{ "type": "model", "diagramType": "ClassDiagram", "mode": "complete_system", "request": "create a bookstore class model" }``
2. ``{ "type": "generation", "generatorType": "django", "config": {} }``

Workspace Orchestrator
----------------------

**Location:** ``src/orchestrator/workspace_orchestrator.py``

Resolves which diagram type to target when the user does not specify one explicitly.

Three-Level Resolution
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Level 0: The classifier's target_diagram_type
     When the unified classifier named a target, it wins. Only diagrams
     the message explicitly NAMES (Level 1 keywords) are added as extra
     targets; Level 2 pattern hits never append a second diagram.
            │
            ▼ (classifier left it NULL)
   Level 1: Explicit keywords (KEYWORD_TARGETS)
     "class diagram" → ClassDiagram
     "object diagram" → ObjectDiagram
     "state machine" → StateMachineDiagram
     "agent diagram" → AgentDiagram
     "gui" → GUINoCodeDiagram
     "quantum circuit" → QuantumCircuitDiagram
     "bpmn" / "business process" → BPMN
     "user profile" / "persona" → UserDiagram
     "component diagram" → ComponentDiagram
     "deployment diagram" → DeploymentDiagram
            │
            ▼ (no keyword match)
   Level 2: Discriminating pattern rules (_IMPLICIT_PATTERNS)
     AND-based regex patterns requiring
     strong, unambiguous vocabulary
            │
            ▼ (no pattern match)
   Level 3: Context fallback (FALLBACK_PRIORITY)
     Active diagram type from WorkspaceContext, then the
     project snapshot in priority order:
       ClassDiagram > ObjectDiagram > StateMachineDiagram >
       AgentDiagram > GUINoCodeDiagram > QuantumCircuitDiagram >
       BPMN > UserDiagram > ComponentDiagram > DeploymentDiagram

Level 1: Keyword Matching
~~~~~~~~~~~~~~~~~~~~~~~~~~

Direct string matching against the user message (``KEYWORD_TARGETS``):

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Keyword Pattern
     - Resolved Type
   * - ``"class diagram"``, ``"class model"``, ``"domain model"``,
       ``"structural model"``, ``"structural diagram"``
     - ``ClassDiagram``
   * - ``"object diagram"``, ``"object model"``
     - ``ObjectDiagram``
   * - ``"state machine"``, ``"statemachine"``, ``"state diagram"``
     - ``StateMachineDiagram``
   * - ``"agent diagram"``, ``"agent model"``, ``"agent that"``,
       ``"an agent"``, ``"chatbot"``
     - ``AgentDiagram``
   * - ``"gui diagram"``, ``"a gui"``, ``"web ui"``
     - ``GUINoCodeDiagram``
   * - ``"quantum circuit"``, ``"quantum"``, ``"qubit"``, ``"grover"``, etc.
     - ``QuantumCircuitDiagram``
   * - ``"bpmn"``, ``"business process"``, ``"process diagram"``,
       ``"process model"``, ``"workflow diagram"``
     - ``BPMN``
   * - ``"user profile"``, ``"user model"``, ``"user diagram"``,
       ``"target user"``, ``"user persona"``, ``"persona"``
     - ``UserDiagram``
   * - ``"component diagram"``, ``"component model"``, ``"uml component"``
     - ``ComponentDiagram``
   * - ``"deployment diagram"``, ``"deployment model"``
     - ``DeploymentDiagram``

.. note::

   Component and Deployment keywords are deliberately limited to phrases that
   **name** the diagram. Generic architecture vocabulary ("system
   architecture", "software architecture", "subsystem", "service
   architecture") is ordinary class-diagram phrasing and resolves through the
   normal levels instead.

Level 2: Discriminating Pattern Rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When no explicit keyword matches, the system uses **discriminating regex
patterns** (``_IMPLICIT_PATTERNS``) that require strong, unambiguous signals
using AND-based logic.

.. note::

   This replaced an older **additive weight scoring** system where generic words
   like "model" (weight=1) + "system" (weight=1) + "application" (weight=1) could
   accumulate to score 3 for ClassDiagram — three vague words confidently picking
   a diagram type. The new system requires at least one domain-specific term or a
   co-occurrence of two related terms.

How patterns work:

- **Single strong signal**: ``"lifecycle"`` alone → StateMachineDiagram. No
  supporting evidence needed — the word is unambiguous.
- **Co-occurrence**: ``"state"`` + ``"transition"`` (within 40 characters) →
  StateMachineDiagram. Neither word alone is sufficient.
- **No match on generic words**: ``"system"``, ``"model"``, ``"application"``
  alone produce NO match. The request falls through to Level 3.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Diagram Type
     - Discriminating Signals
   * - ``QuantumCircuitDiagram``
     - Any of: ``quantum``, ``qubit``, ``qiskit``, ``grover``, ``shor``,
       ``hadamard``, ``cnot``, ``superposition``, ``entangle``, ``qft``,
       ``teleportation``, ``bell state``, ``gate``
   * - ``ObjectDiagram``
     - ``object instance``, ``instance of``, ``runtime object``, ``instances``
   * - ``BPMN``
     - ``bpmn``, ``business process``, ``process diagram/model/flow``,
       ``gateway``, ``sequence flow``, ``swimlane``, ``pool``, or a
       create/model/design verb near ``process``
   * - ``StateMachineDiagram``
     - ``lifecycle``, ``workflow state``, or ``state``/``status``
       co-occurring with ``transition``/``flow``/``event``/``process``
   * - ``UserDiagram``
     - ``user profile``, ``user persona``, ``target user``, ``user model``,
       ``persona``, ``audience profile``
   * - ``AgentDiagram``
     - ``multi-agent``, ``conversational agent``, ``chatbot``, or ``agent``
       co-occurring with ``intent``/``training``/``reply``/``response``
   * - ``GUINoCodeDiagram``
     - ``gui``, ``user interface``, ``wireframe``, ``no-code``, ``grapesjs``,
       or ``frontend``/``screen``/``page``/``layout``/``dashboard``
       co-occurring with ``design``/``create``/``build``
   * - ``ComponentDiagram`` / ``DeploymentDiagram``
     - the diagram-naming phrases above, including plurals and a ``uml``
       prefix (``components diagram``, ``uml deployment model``)
   * - ``ClassDiagram``
     - ``structural``, ``domain model``, ``business model``,
       ``system model``, or ``class``/``entity`` co-occurring with
       ``attribute``/``method``/``relationship``/``association``/``inheritance``

.. note::

   **Order matters.** Patterns are evaluated in list order, and BPMN is
   deliberately checked *before* StateMachineDiagram because the word
   "process" appears in both vocabularies. Quantum is checked first because
   its vocabulary is the most specific.

Level 3: Context Fallback
~~~~~~~~~~~~~~~~~~~~~~~~~~

If no discriminating pattern matches:

1. Use ``active_diagram_type`` from the ``WorkspaceContext``
2. Check ``project_snapshot`` for existing diagram types in priority order
3. Use ``diagram_type`` from the request header
4. Last resort: default to ``ClassDiagram``

.. note::

   When Level 2 produces no match and the matched intent is
   ``create_complete_system_intent``, the system now escalates to the **LLM
   planner** (Tier 2) rather than defaulting blindly to ClassDiagram. This
   ensures ambiguous creation requests like "build a system with states and
   processes" get LLM-resolved diagram types.

Execution Flow
--------------

The orchestration and execution layers work together:

.. mermaid::

   flowchart TD
       UM["User Message"] --> PARSE["parse_assistant_request()"]
       PARSE --> IC["Intent Classifier → State Body"]
       IC --> EPO["execute_planned_operations()"]
       EPO --> PLAN["plan_assistant_operations()"]
       PLAN --> H["Heuristic operations"]
       PLAN --> LLM["LLM planner (if complex)"]
       PLAN --> NORM["Normalize + deduplicate"]
       NORM --> LOOP{"For each operation"}
       LOOP -->|"type == model"| EMO["execute_model_operation()"]
       EMO --> R1["Resolve diagram type"]
       R1 --> R2["Resolve target model"]
       R2 --> R3["Build workspace context"]
       R3 --> R4["Dispatch to handler"]
       R4 --> R5["Apply layout"]
       R5 --> R6["Send reply"]
       LOOP -->|"type == generation"| HGR["handle_generation_request()"]
       HGR --> G1["Match generator type"]
       G1 --> G2["Parse inline config"]
       G2 --> G3["Return trigger payload"]

Common Preamble
---------------

Every state body starts with ``_common_preamble()``
(``src/app/state_bodies.py``), which runs these checks in order:

1. **Reconnect replay** — a ``replay_last_response`` action re-sends the
   buffered terminal reply and stops. It never re-runs generation or consumes
   a pending flow.
2. **Pending dialogs** — ``resume_pending()`` (``src/app/confirmation.py``)
   hands the message to each open dialog in ``pending_dialogs.PRECEDENCE``
   order until one consumes it (see `Pending Dialogs`_).
3. **Parse** the request into an ``AssistantRequest``
4. **File attachments** — ``handle_file_attachments()``
5. **Record** the user message in conversation memory, keyed on the stable
   payload ``sessionId`` so it survives reconnects
6. **Ask instead of guess** — if the cached classification set
   ``needs_clarification``, the preamble streams ``clarifying_question`` and
   stops rather than guessing a destructive mutation. It reads the cached
   verdict, so this costs no extra LLM call, and it never fires on a
   ``frontend_event``.

It returns the parsed ``AssistantRequest`` when the message should be handled
normally, or ``None`` when a pending flow, an attachment, or a clarifying
question already consumed it — in which case the state body short-circuits.

Pending Dialogs
~~~~~~~~~~~~~~~

A turn that ends on a question stores what the answer needs in one session
variable, ``PENDING_DIALOGS``, through ``src/common/pending_dialogs.py``
(``get`` / ``put`` / ``clear``; never ``session.get`` on a slot name). Several
dialogs can be open at once — the domain-mismatch "Update model + generate"
chain keeps its Spec-Driven stash while the rebuild asks replace-or-keep, and a
"design X and generate Y" plan keeps its paused generator while its model step
asks the same — so the order is fixed in ``PRECEDENCE``:

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Dialog
     - Question
     - Resume handler (``RESUME_HANDLERS``)
   * - ``GUI_CHOICE``
     - Basic CRUD pages or experimental AI design?
     - ``handle_pending_gui_choice``
   * - ``SYSTEM_CONFIRMATION``
     - Replace, keep or new tab; or confirm a destructive modify
     - ``handle_pending_system_confirmation``
   * - ``SMART_GEN``
     - Run the Spec-Driven Agent? (exact phrases confirm; the classifier may
       only cancel)
     - ``handle_pending_smart_gen_confirmation``
   * - ``GENERATOR``
     - Continue with the paused generator? (exact yes/ok/generate/no only,
       because the classifier has stamped a bare "ok" as ``decline_intent``)
     - ``handle_pending_plan_generation_confirmation``

A handler returns ``True`` when it consumed the message and ``False`` when the
message is not its answer, after either leaving its dialog open or abandoning
it; the next open dialog is then tried. The classifier is shown the same first
open dialog (``_pending_flow_context``). Answers no handler consumes — the
generator menu, generator settings, a non-exact reply to a Spec-Driven or
plan-pause question — reach ``handle_generation_request`` through routing: the
pending gate in ``json_intent_matches`` keeps them out of intent routing,
``should_route_to_generation`` sends them to the generation state, and
``decline_body`` hands a decline to it while a generator dialog is open.

A plan's model operations on independent diagram types run in parallel, but
open at most one replace/keep, GUI-choice or destructive-modify question
(``pending_dialogs.one_per_plan``): an operation whose question would come
second stops before asking and is queued in the open dialog's
``remaining_operations``, so it asks once that dialog is answered. A queued
destructive modify keeps the change it already computed, so asking later makes
no second LLM call. If the operation that claimed the dialog fails before
opening it, the queued operations run in plan order right away and the first
of them that has a question asks it.
