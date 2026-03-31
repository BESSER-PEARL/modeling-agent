Intent Recognition
==================

This document describes how the Modeling Agent recognizes user intent and routes
messages to the correct handler. It covers the full pipeline from raw user
input to state transition.

.. contents:: On this page
   :local:
   :depth: 2

Pipeline Overview
-----------------

Intent recognition uses a **3-stage pipeline** designed to be both fast and
accurate. Stages 1 and 3 are pure deterministic checks (regex/string matching)
that add **zero latency**. Only Stage 2 makes an LLM call.

.. code-block:: text

   User message
       │
       ▼
   ┌────────────────────────────────────┐
   │  Stage 1: Deterministic Pre-Filter │  ← zero latency
   │  _is_diagram_creation_request()    │
   │  _is_modeling_request()            │
   │  route_to_generation()             │
   └──────────────┬─────────────────────┘
                  │
                  ▼
   ┌────────────────────────────────────┐
   │  Stage 2: LLM Intent Classifier   │  ← GPT-4.1-mini call
   │  Classifies into 1 of 8 intents   │
   │  Confidence threshold: 0.55       │
   └──────────────┬─────────────────────┘
                  │
                  ▼
   ┌────────────────────────────────────┐
   │  Stage 3: Post-Classification      │  ← zero latency
   │  Guards & Safety Nets              │
   │  Catches LLM misclassifications    │
   └──────────────┬─────────────────────┘
                  │
                  ▼
           State Transition
           → Handler Execution


Stage 1: Deterministic Pre-Filters
-----------------------------------

Pre-filters run **before** the LLM classifier and are evaluated during
state transition routing (priority 2 in the transition system). They catch
common patterns that the LLM might misclassify.

**Location:** ``src/handlers/generation_handler.py``

_is_diagram_creation_request()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detects when the user says "generate" but means *create a diagram*, not
*generate source code*.

.. code-block:: python

   _is_diagram_creation_request("generate a class diagram")        # → True
   _is_diagram_creation_request("generate a class diagram for X")  # → True
   _is_diagram_creation_request("generate django")                 # → False
   _is_diagram_creation_request("generate python code")            # → False

Logic:

1. Message must contain a **diagram type token**: ``"class diagram"``,
   ``"state machine"``, ``"object diagram"``, ``"agent diagram"``,
   ``"gui diagram"``, ``"quantum circuit"``, etc.
2. Message must start with a **creation verb** (possibly prefixed by filler):
   ``generate``, ``create``, ``build``, ``design``, ``make``, ``model``, etc.
3. Also catches ``"I need a class diagram"``, ``"give me a state diagram"``.

_is_modeling_request()
~~~~~~~~~~~~~~~~~~~~~~

Detects modeling/creation requests that contain generator keywords (e.g.,
"create a web app for hotel booking" contains "web app" which is a
generator keyword).

.. code-block:: python

   _is_modeling_request("create a web app for hotel booking")  # → True
   _is_modeling_request("generate a class diagram")            # → True (via fast path)
   _is_modeling_request("generate django")                     # → False
   _is_modeling_request("generate python code")                # → False

Logic:

1. **Fast path**: delegates to ``_is_diagram_creation_request()``
2. Must have a **modeling verb**: ``create``, ``build``, ``design``, ``model``,
   ``make``, ``develop``, ``architect``, ``plan``, ``draft``
3. Must have a **domain qualifier**: ``for``, ``system``, ``management``,
   ``booking``, ``hotel``, ``library``, etc.
4. Must **not** have explicit generation phrases: ``"generate code"``,
   ``"run generator"``, ``"source code"``, ``"export"``, ``"deploy"``

.. note::

   ``"generate"`` is intentionally NOT in the modeling verbs list because
   ``"generate django"`` IS a code-generation request. The diagram-specific
   ``"generate a class diagram"`` case is handled by the fast path above.

route_to_generation()
~~~~~~~~~~~~~~~~~~~~~

The transition condition that decides whether to route to the generation state.
Called at **priority 2** in the transition system (after LLM intent matching).

.. code-block:: python

   # Returns True (route to generation):
   route_to_generation(session, request_with_message="generate django")
   route_to_generation(session, request_with_action="frontend_event")

   # Returns False (do NOT route to generation):
   route_to_generation(session, request_with_message="generate a class diagram")
   route_to_generation(session, request_with_message="create a web app for X")

Logic (in order):

1. Frontend events → always route to generation
2. Pending generator state → route to generation (user is in a config flow)
3. Mixed request (modeling + generation) → do NOT route (handled by workflow)
4. Modeling request → do NOT route
5. Diagram creation request → do NOT route
6. Generator keyword detected → route to generation


Stage 2: LLM Intent Classification
------------------------------------

The BESSER framework calls GPT-4.1-mini to classify the user message into
one of 8 predefined intents.

**Configuration:** ``src/agent_setup.py``

.. code-block:: python

   LLMIntentClassifierConfiguration(
       llm_name='gpt-4.1-mini',
       use_intent_descriptions=True,
       use_training_sentences=False,
       use_entity_descriptions=True,
       use_entity_synonyms=False,
   )

**Confidence threshold:** ``0.55`` (configurable in ``config.yaml``)

The 8 Intents
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Intent
     - State
     - When to Use
   * - ``hello_intent``
     - greetings
     - User greetings, starting conversation
   * - ``create_complete_system_intent``
     - create_complete_system
     - Create/build a complete system, diagram, or algorithm from scratch.
       **Includes** "generate a class diagram", "generate a state machine".
   * - ``modify_model_intent``
     - modify_model
     - Modify, add to, extend, remove from, or create a single element in
       an existing model
   * - ``modeling_help_intent``
     - modeling_help
     - Ask for general help, explanation, or guidance about modeling concepts
   * - ``describe_model_intent``
     - describe_model
     - Ask questions about the current model on the canvas
   * - ``uml_spec_intent``
     - uml_rag
     - Ask about the official UML specification document
   * - ``generation_intent``
     - generation
     - Generate source code, export, or deploy from an **existing** model.
       **Excludes** "generate a class diagram" (that's creation, not generation).
   * - ``workflow_intent``
     - workflow
     - Complete end-to-end workflow: model → validate → generate

**Intent Definition Location:** ``modeling_agent.py`` (lines 87–224)

Key Disambiguation Rules
~~~~~~~~~~~~~~~~~~~~~~~~

The intent descriptions include explicit disambiguation rules for the LLM:

- ``"generate a class diagram"`` = ``create_complete_system_intent`` (create a diagram)
- ``"generate django"`` = ``generation_intent`` (generate code)
- ``"generate python code"`` = ``generation_intent`` (generate code)
- ``"create a web app for X"`` = ``create_complete_system_intent`` (model a system)
- ``"I also want to store X"`` = ``modify_model_intent`` (extend existing model)


Stage 3: Post-Classification Guards
-------------------------------------

Even with good intent descriptions, the LLM can misclassify. Stage 3 provides
**safety nets inside the handlers** that catch mistakes.

**Location:** ``src/handlers/generation_handler.py`` → ``handle_generation_request()``

Guard 1: Modeling Request Redirect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a modeling request lands in the generation handler (e.g., "create a web app
for hotel booking" misclassified as ``generation_intent``), the handler detects
it and returns a helpful redirect message instead of triggering the generator.

.. code-block:: python

   # Inside handle_generation_request():
   if not pending_generator and (
       _is_modeling_request(request.message)
       or _is_diagram_creation_request(lower)
   ):
       return {"action": "assistant_message", "message": "It looks like you want
       to create a diagram..."}

Guard 2: Non-GUI Diagram Redirect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When no generator type is detected but the message mentions a specific diagram
type (e.g., "class diagram", "state diagram"), the handler redirects instead of
showing the generic generator selection menu.

Guard 3: GUI Hint Check
~~~~~~~~~~~~~~~~~~~~~~~~

When no generator type is detected and the message mentions GUI-specific terms
(``gui``, ``frontend``, ``no-code``), the handler suggests creating a GUI
diagram instead.


Transition Priority System
--------------------------

State transitions are checked in this order (first match wins):

**Location:** ``src/state_bodies.py`` → ``add_unified_transitions()``

.. code-block:: text

   Priority 1: Intent-matched JSON transitions (LLM classifier result)
       json_intent_matches(session, {'intent_name': 'X'})
       → Most accurate signal, highest priority

   Priority 2: Keyword-based generation route
       route_to_generation(session)
       → Catches generator keywords, frontend events, pending config

   Priority 3: Text-event intent transitions (backward compatibility)
       when_intent_matched(intent)

   Priority 4: Fallback transitions
       json_no_intent_matched(session)
       → Routes to fallback state

**Important:** Priority 2 is only reached when Priority 1 does NOT match. This
means:

- If the LLM correctly classifies "generate django" as ``generation_intent``,
  Priority 1 handles it and Priority 2 is never checked.
- If the LLM misclassifies "generate django" as something else, Priority 2
  catches it via ``route_to_generation()`` because "django" is a generator keyword.
- If the LLM misclassifies "generate a class diagram" as ``generation_intent``,
  Priority 1 routes to the generation state, but **Stage 3 guards** catch it.


Diagram Type Resolution
-----------------------

Once the intent is classified and the state body executes, the system needs to
determine **which diagram type** to target.

See :doc:`orchestration` for the full 3-level resolution system (explicit
keywords → discriminating patterns → context fallback).


Common Misclassification Patterns
----------------------------------

These are the known ambiguous patterns and how the pipeline handles them:

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - User Message
     - Risk
     - How It's Handled
   * - ``"generate a class diagram"``
     - LLM picks ``generation_intent``
     - Stage 1 pre-filter + Stage 3 safety net redirect to creation
   * - ``"create a web app for hotel"``
     - ``"web app"`` is a generator keyword
     - ``_is_modeling_request()`` catches it; Stage 3 safety net as backup
   * - ``"generate backend code"``
     - ``"backend"`` is a broad keyword
     - Word-boundary matching prevents false positives in non-generation context
   * - ``"generate sql"``
     - ``"sql"`` could match inside ``"sqlalchemy"``
     - Word-boundary matching for ambiguous short keywords
   * - ``"I also want to store emails"``
     - Could be ``create_complete_system``
     - Intent description explicitly maps "also want"/"also store" to ``modify_model``
   * - ``"create a class diagram and generate django"``
     - Mixed modeling + generation
     - ``_looks_like_mixed_modeling_and_generation()`` detects it → workflow state


Debugging Intent Recognition
------------------------------

When intent recognition goes wrong, check in this order:

1. **Check the LLM classification**: Enable debug logging to see what intent
   GPT-4.1-mini predicted and with what confidence.

2. **Check pre-filters**: Does ``_is_modeling_request()`` or
   ``_is_diagram_creation_request()`` return the expected value for the input?

3. **Check transition priority**: Is the message hitting Priority 1 (intent
   match) or Priority 2 (generation route)? If Priority 2, the LLM may have
   misclassified.

4. **Check handler safety nets**: If the message reaches the wrong handler,
   does the safety net catch it? Check the response ``action`` field.

5. **Check keyword detection**: Does ``detect_generator_type()`` return the
   expected value? Remember that word-boundary matching applies to ``"sql"``
   and ``"backend"``.

6. **Check diagram type resolution**: Does ``determine_target_diagram_type()``
   return the expected diagram type? Check explicit keywords first, then
   discriminating patterns, then context fallback.

.. code-block:: python

   # Quick diagnostic snippet:
   from handlers.generation_handler import (
       detect_generator_type,
       _is_modeling_request,
       _is_diagram_creation_request,
   )

   msg = "your test message here"
   print(f"Generator type:  {detect_generator_type(msg)}")
   print(f"Is modeling:     {_is_modeling_request(msg)}")
   print(f"Is diagram:      {_is_diagram_creation_request(msg.lower())}")
