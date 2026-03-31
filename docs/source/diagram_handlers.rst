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
   ├── ClassDiagramHandler          # UML class diagrams
   ├── StateMachineHandler          # UML state machines
   ├── ObjectDiagramHandler         # UML object diagrams
   ├── AgentDiagramHandler          # BESSER agent diagrams
   ├── GUINoCodeDiagramHandler      # GrapesJS GUI models
   └── QuantumCircuitDiagramHandler # Quirk quantum circuits

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
   * - ``generate_complete_system(request, model, **kw) -> dict``
     - Generate a complete diagram
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
     - LLM call with exponential backoff and jitter
   * - ``predict_two_pass()``
     - Two-pass generation: free-text reasoning then structured JSON
   * - ``validate_and_refine()``
     - LLM self-critique loop for output validation
   * - ``repair_json_response()``
     - Last-resort JSON repair via LLM
   * - ``apply_single_layout()``
     - Layout positioning for single elements
   * - ``apply_system_layout()``
     - Layout positioning for complete systems
   * - ``_error_response()``
     - Standard error format with ``retryable`` flag

Retry Strategy
~~~~~~~~~~~~~~

- **Retry:** Exponential backoff with jitter (3 attempts)
- **Rate limiting:** Handled by OpenAI's API directly (429 responses caught by retry logic)
- **Fallback:** Graceful degradation through multiple levels (primary LLM → JSON repair → type-specific fallback → error response)

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
They were disabled because GPT-4.1 produces good diagrams without them, and the
pattern injection biased the LLM toward hardcoded templates.

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

Generates BESSER conversational agent diagrams with states, intents, and transitions.

Elements
~~~~~~~~

- **State:** Named state with ``replies[]`` (response texts)
- **Intent:** Named intent with ``trainingPhrases[]``
- **Initial:** Starting pseudo-element
- **Transition:** Links states via intents

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
- **Chart Color Palettes:** Provides consistent color schemes for data-bound
  charts and visualizations.
- **Class Metadata Injection:** Extracts class attributes and types to
  auto-populate form fields, table columns, and chart axes.

GUINoCodeDiagram Generation Modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a ClassDiagram exists in the project:

1. **Auto mode:** Deterministic page generation (no LLM).
2. **LLM mode:** Customized generation with user preferences.

If the user message contains customization hints (chart, dashboard, custom, etc.),
the LLM mode is used automatically. Otherwise, the agent asks which mode to use.

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
     - Horizontal bounds (expandable)
   * - Canvas Y range
     - -500 to 500
     - Vertical bounds (expandable)
   * - Horizontal gap
     - 60px
     - Minimum space between elements
   * - Vertical gap
     - 50px
     - Minimum space between rows
   * - Margin
     - 40px
     - Minimum clearance around elements
   * - Grid snap
     - 20px
     - Coordinates snap to 20px grid

DiagramHandlerFactory
---------------------

**Location:** ``src/diagram_handlers/registry/factory.py``

Registry that maps diagram type strings to handler instances.

.. code-block:: python

   class DiagramHandlerFactory:
       def __init__(self, llm):
           self._handlers = {
               "ClassDiagram":          ClassDiagramHandler(llm),
               "ObjectDiagram":         ObjectDiagramHandler(llm),
               "StateMachineDiagram":   StateMachineHandler(llm),
               "AgentDiagram":          AgentDiagramHandler(llm),
               "GUINoCodeDiagram":      GUINoCodeDiagramHandler(llm),
               "QuantumCircuitDiagram": QuantumCircuitDiagramHandler(llm),
           }

       def get_handler(self, diagram_type: str) -> Optional[BaseDiagramHandler]:
           """Return handler for type, or None if unsupported."""

       def get_supported_types(self) -> list[str]:
           """Return all registered diagram type strings."""

       def is_supported(self, diagram_type: str) -> bool:
           """Check if a diagram type has a registered handler."""

Adding a New Diagram Type
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create a new handler class extending ``BaseDiagramHandler`` in
   ``src/diagram_handlers/types/``
2. Implement the 4 required abstract methods
3. Register in ``DiagramHandlerFactory.__init__``
4. Add to ``SUPPORTED_DIAGRAM_TYPES`` in ``src/protocol/types.py``
5. Add display metadata in ``src/diagram_handlers/registry/metadata.py``

Layout, retry, two-pass generation, and validation are inherited
automatically from ``BaseDiagramHandler``.
