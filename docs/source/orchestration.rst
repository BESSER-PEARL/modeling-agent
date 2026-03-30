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
     "diagram_type": "ClassDiagram",
     "mode": "complete_system",
     "request_text": "create a bookstore class diagram"
   }

**Generation operation:**

.. code-block:: json

   {
     "type": "generation",
     "generator": "django",
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
   * - ``flutter``
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
   * - ``rest_api``
     - ClassDiagram
   * - ``agent``
     - AgentDiagram
   * - ``qiskit``
     - QuantumCircuitDiagram

Example
~~~~~~~

User message: ``"create a bookstore class model and then generate django"``

Planned operations:

1. ``{ "type": "model", "diagram_type": "ClassDiagram", "mode": "complete_system", "request_text": "create a bookstore class model" }``
2. ``{ "type": "generation", "generator": "django" }``

Workspace Orchestrator
----------------------

**Location:** ``src/orchestrator/workspace_orchestrator.py``

Resolves which diagram type to target when the user does not specify one explicitly.

Three-Level Resolution
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Level 1: Explicit keywords
     "class diagram" → ClassDiagram
     "object diagram" → ObjectDiagram
     "state machine" → StateMachineDiagram
     "agent diagram" → AgentDiagram
     "gui" → GUINoCodeDiagram
     "quantum circuit" → QuantumCircuitDiagram
            │
            ▼ (no keyword match)
   Level 2: Discriminating pattern rules
     AND-based regex patterns requiring
     strong, unambiguous vocabulary
            │
            ▼ (no pattern match)
   Level 3: Context fallback
     Active diagram type from WorkspaceContext
     Default priority: ClassDiagram > ObjectDiagram >
       StateMachine > Agent > GUI > Quantum

Level 1: Keyword Matching
~~~~~~~~~~~~~~~~~~~~~~~~~~

Direct string matching against the user message (``KEYWORD_TARGETS``):

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Keyword Pattern
     - Resolved Type
   * - ``"class diagram"``, ``"domain model"``, ``"structural model"``
     - ``ClassDiagram``
   * - ``"object diagram"``
     - ``ObjectDiagram``
   * - ``"state machine"``, ``"state diagram"``
     - ``StateMachineDiagram``
   * - ``"agent diagram"``, ``"an agent"``, ``"chatbot"``
     - ``AgentDiagram``
   * - ``"gui diagram"``, ``"a gui"``, ``"web ui"``
     - ``GUINoCodeDiagram``
   * - ``"quantum circuit"``, ``"quantum"``, ``"qubit"``, ``"grover"``, etc.
     - ``QuantumCircuitDiagram``

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
       ``hadamard``, ``cnot``, ``superposition``, ``entangle``, ``qft``
   * - ``ObjectDiagram``
     - ``object instance``, ``instance of``, ``runtime object``
   * - ``StateMachineDiagram``
     - ``lifecycle``, ``workflow state``, or ``state`` co-occurring with
       ``transition``/``flow``/``event``/``process``
   * - ``AgentDiagram``
     - ``multi-agent``, ``conversational agent``, ``chatbot``, or ``agent``
       co-occurring with ``intent``/``training``/``reply``
   * - ``GUINoCodeDiagram``
     - ``gui``, ``user interface``, ``wireframe``, or ``frontend``/``screen``/
       ``page``/``layout`` co-occurring with ``design``/``create``/``build``
   * - ``ClassDiagram``
     - ``structural``, ``domain model``, ``business model``, or ``class``/
       ``entity`` co-occurring with ``attribute``/``method``/``relationship``

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

Every state body starts with ``_common_preamble()`` which:

1. Checks for a pending GUI choice and handles it
2. Checks for a pending system confirmation and handles it
3. Parses the request into an ``AssistantRequest``
4. Handles file attachments (if present)

If any pending flow is resolved, the preamble returns a result directly and the
state body short-circuits.
