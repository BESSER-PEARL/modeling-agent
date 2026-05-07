"""Shared prompt fragments used across all diagram handlers.

Centralising repeated instructions has two purposes:

1. **Single source of truth** — rules like "use exact names from the model"
   or "put multiple changes in an array" used to be repeated (with subtle
   wording drift) inside every ``generate_modification`` system prompt.
   Editing them in one place now updates every handler.
2. **Stable cache prefix** — the OpenAI Responses API caches input prefixes
   that are byte-identical across calls. The ``generate_modification``
   methods now compose their system prompt at module-load time from these
   constants, so the system message stays bit-stable across calls; only
   the user message (model summary + user request) varies. Once the system
   prompt clears the ~1024-token threshold (the class-diagram handler does;
   the agent / object / state-machine handlers don't and we accept the
   miss), every modification call after the first hits the cache.

Don't put anything date-, uuid-, or path-dependent in this file. Anything
that varies per request belongs in the user message.
"""

# ---------------------------------------------------------------------------
# Cross-handler rules (used by 2+ handlers)
# ---------------------------------------------------------------------------

EXACT_NAMES_RULE = (
    "For existing elements, use exact names from the current model in "
    '"target".'
)

MULTI_MOD_ARRAY_RULE = (
    "When the user asks for multiple changes at once, return multiple "
    'entries in the "modifications" array.'
)

CHANGES_FIELD_RULE = (
    'Put what should change in "changes". Only include fields that differ.'
)

REMOVE_ELEMENT_RULE = (
    "For remove_element, only specify the target — no \"changes\" needed."
)

POSITION_DISCLAIMER = (
    'Do NOT include any "position" field — positioning is handled automatically.'
)

# ---------------------------------------------------------------------------
# Class-diagram-specific blocks (kept here so the planner / future handlers
# can pull OCL examples from a single source).
# ---------------------------------------------------------------------------

NAMING_PASCAL_RULE = (
    "NAMING: Class names MUST be exactly ONE word in PascalCase: \"User\", "
    "\"Book\", \"Order\", \"Payment\". NEVER concatenate words like "
    "\"UserLibraryUser\", \"BookReading\", \"OrderPayment\". Just \"User\", "
    "\"Reading\", \"Payment\"."
)

RENAME_CASCADES_RULE = (
    "RENAME: a single modify_class is enough — relationships update "
    "automatically by id and do not need accompanying modify_relationship "
    "entries."
)

DELETE_CLASS_CASCADE_RULE = (
    "DELETE a class: include a remove_element for the class AND a separate "
    "remove_element for EVERY relationship connected to it. If you mention "
    "the removal in your message but skip the remove_element entries, the "
    "class WILL NOT be removed. Example: deleting \"Address\" with 2 "
    "relationships → 3 remove_element entries (1 for the class + 2 for the "
    "relationships)."
)

ENUM_RULES_BLOCK = """ENUMERATION RULES:
- Create enum: add_class with isEnumeration=true. Enum values are attributes with name only (NO type field).
- Add value to EXISTING enum: add_attribute with target.className set to THE ENUM NAME (not another class).
  Example: if "Priority" enum exists and user says "add Critical" → add_attribute with target.className="Priority", changes.name="Critical" (NO type).
- Use enum as attribute type: add_attribute with changes.type set to the enum's PascalCase name.
  Example: add_attribute with target.className="Task", changes.name="priority", changes.type="Priority"."""

OCL_CONSTRAINT_BLOCK = """OCL CONSTRAINTS (only when the user explicitly asks for an invariant, precondition, or postcondition):
- add_ocl_constraint — attach an OCL constraint to a class. Set target.className to the anchor class and put the FULL BOCL block in changes.constraint. Optionally include changes.text as a plain-language description that the validator surfaces on failure.
- Do NOT emit add_ocl_constraint unprompted — only when the user says something like "add a constraint that…", "ensure that…", "the [pre/post]condition is…", "invariant: …", or otherwise asks for OCL.
- The full BOCL block must include the header AND the body in changes.constraint:
  * Invariant:    "context Class inv [name]: body"          — e.g. "context Library inv at_least_one_book: self.books->size() > 0"
  * Precondition: "context Class::method(p: Type) pre: body"  — e.g. "context Account::deposit(amount: Integer) pre: amount > 0"
  * Postcondition:"context Class::method(p: Type) post: body" — e.g. "context Account::deposit(amount: Integer) post: self.balance >= 0"
- Use OCL collection operations with arrows (``->``): ``->size()``, ``->isEmpty()``, ``->forAll(x | …)``, ``->exists(x | …)``, ``->select(x | …)``. Use ``self`` to refer to the contextual instance. Use ``and`` / ``or`` / ``not`` / ``implies`` for logical composition.
- For preconditions and postconditions the operation MUST already exist on the class — if it does not, emit add_method first, then add_ocl_constraint."""

OCL_EXAMPLES_BLOCK = """OCL examples (only emit when the user explicitly asks for a constraint/invariant/pre-/postcondition):
- "add a constraint that a Library always has at least one Book" → add_ocl_constraint with target.className="Library", changes.constraint="context Library inv at_least_one_book: self.books->size() > 0", changes.text="A library always has at least one book"
- "ensure every Order has a non-empty customer name" → add_ocl_constraint with target.className="Order", changes.constraint="context Order inv customer_name_not_empty: self.customer.name <> ''", changes.text="Order's customer name must not be empty"
- "the precondition of Account::deposit is amount > 0" → add_ocl_constraint with target.className="Account", changes.constraint="context Account::deposit(amount: Integer) pre: amount > 0", changes.text="Deposit amount must be positive"
- "after Account::deposit the balance must not be negative" → add_ocl_constraint with target.className="Account", changes.constraint="context Account::deposit(amount: Integer) post: self.balance >= 0", changes.text="Balance is never negative after a deposit"
- "every Employee's salary must be at least the minimum wage of their department" → add_ocl_constraint with target.className="Employee", changes.constraint="context Employee inv salary_above_minimum: self.salary >= self.department.minimumWage", changes.text="Employee salary must respect the department's minimum wage" """

MODIFY_CRITICAL_BLOCK = """CRITICAL — READ CAREFULLY:
- The CURRENT MODEL is provided in the user message. NEVER re-create anything that already exists.
- The conversation history is also provided. If it says you JUST created something, it EXISTS. Do NOT re-create it.
- ONLY output modifications for what the user asks RIGHT NOW. Never repeat past operations.
- If the user's message is short/ambiguous (e.g., "ok and X?", "also Y"), interpret it as ADDING to the most recently discussed element."""
