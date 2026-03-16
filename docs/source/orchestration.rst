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
- **Tier 1 -- Keyword-based fallback with intent-aware fast path:** Keyword
  detection determines diagram type + mode. Skips the LLM planner when the
  intent classifier already resolved a single target with no generation request.
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
   Level 2: Implicit semantic scoring
     Weighted token matching per diagram type
            │
            ▼ (no clear winner)
   Level 3: Context fallback
     Active diagram type from WorkspaceContext
     Default priority: ClassDiagram > ObjectDiagram >
       StateMachine > Agent > GUI > Quantum

Level 1: Keyword Matching
~~~~~~~~~~~~~~~~~~~~~~~~~~

Direct string matching against the user message:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Keyword Pattern
     - Resolved Type
   * - ``"class diagram"``
     - ``ClassDiagram``
   * - ``"object diagram"``
     - ``ObjectDiagram``
   * - ``"state machine"``, ``"state diagram"``
     - ``StateMachineDiagram``
   * - ``"agent diagram"``, ``"multi agent"``
     - ``AgentDiagram``
   * - ``"gui"``, ``"user interface"``, ``"no code"``
     - ``GUINoCodeDiagram``
   * - ``"quantum circuit"``, ``"quantum"``
     - ``QuantumCircuitDiagram``

Level 2: Semantic Scoring
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each diagram type has a set of token/weight pairs. The type with the highest
cumulative score wins. For example, ``ClassDiagram`` scores highly on tokens like
"class", "attribute", "method", "inheritance", while ``StateMachineDiagram``
scores on "state", "transition", "event", "guard".

Level 3: Context Fallback
~~~~~~~~~~~~~~~~~~~~~~~~~~

If no scoring threshold is met:

1. Use ``active_diagram_type`` from the ``WorkspaceContext``
2. Fall back to default priority order

Execution Flow
--------------

The orchestration and execution layers work together:

.. code-block:: text

   User Message
       │
       ▼
   parse_assistant_request()
       │
       ▼
   Intent Classifier → State Body
       │
       ▼
   execute_planned_operations()
       │
       ▼
   plan_assistant_operations()
       ├── Heuristic operations (always computed)
       ├── LLM planner (if complex request)
       └── Normalize + deduplicate
       │
       ▼
   For each operation:
       ├── type == "model" → execute_model_operation()
       │   ├── Resolve diagram type (workspace_orchestrator)
       │   ├── Resolve target model
       │   ├── Build workspace context
       │   ├── Dispatch to handler
       │   ├── Apply layout
       │   ├── Send reply
       │   └── Quality review
       │
       └── type == "generation" → handle_generation_request()
           ├── Match generator type
           ├── Parse inline config
           └── Return trigger payload

Common Preamble
---------------

Every state body starts with ``_common_preamble()`` which:

1. Checks for a pending GUI choice and handles it
2. Checks for a pending system confirmation and handles it
3. Parses the request into an ``AssistantRequest``
4. Handles file attachments (if present)

If any pending flow is resolved, the preamble returns a result directly and the
state body short-circuits.
