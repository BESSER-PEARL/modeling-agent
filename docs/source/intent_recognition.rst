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

Intent recognition is driven by a **unified classifier**: exactly *one*
structured-output LLM call per user message, cached for the whole message
cycle. That single call returns both the state-level intent and every
sub-routing field any downstream state body needs (generation route,
generator type, target diagram type, model disposition, pending-flow
answer, …), so no second prompt is ever needed to refine the verdict.

.. code-block:: text

   User message
       │
       ▼
   ┌──────────────────────────────────────────────┐
   │  Stage 1: Deterministic short-circuits       │  ← zero latency
   │  frontend_event callbacks                    │
   │  [auto-fix] repair marker                    │
   └──────────────┬───────────────────────────────┘
                  │ (normal conversational message)
                  ▼
   ┌──────────────────────────────────────────────┐
   │  Stage 2: Unified classifier                 │  ← ONE classifier-tier
   │  classify_message() → UnifiedClassification  │    LLM call, cached
   │  intent + sub-routing fields in one object   │    per message
   └──────────────┬───────────────────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────────────────┐
   │  Stage 3: Post-validation & pending-flow gate│  ← zero latency
   │  _post_validate(), ActiveFlow suppression    │
   └──────────────┬───────────────────────────────┘
                  │
                  ▼
           State Transition
           → State body → Handler

**Location:** ``src/unified_classifier.py`` (classifier),
``src/session_helpers.py`` (transition conditions),
``src/state_bodies.py`` (transition wiring).


Stage 1: Deterministic Short-Circuits
--------------------------------------

Two message shapes never reach the LLM at all. Both are handled in
``get_or_classify()`` / ``classify_message()``
(``src/unified_classifier.py``).

frontend_event callbacks
~~~~~~~~~~~~~~~~~~~~~~~~

Payloads whose ``action`` is ``"frontend_event"`` (e.g. a
``generator_result`` echo after a generator finishes) are protocol events,
not prose. Classifying their text is both wasteful and wrong — a
generation-completion echo was once classified as ``hello_intent`` and
routed to greetings. They are pinned deterministically:

.. code-block:: python

   UnifiedClassification(
       intent="generation_intent",
       generation_route="other",
       reason="frontend_event callback — routed deterministically, no LLM call",
   )

A third shape never reaches the classifier at all: the
``replay_last_response`` control message, which
``_ensure_unified_classification`` returns on before classifying. It is pure
reconnect recovery and must not consume a turn.

The ``[auto-fix]`` repair marker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Messages the editor itself sends to repair a validation error start with
``AUTO_FIX_PREFIX``. They are pinned to ``modify_model_intent`` with
``model_disposition="extend_existing"`` and
``pending_flow_action="new_request"``, so a machine repair is never
mistaken for an answer to a question the assistant happened to be asking.


Stage 2: The Unified Classifier
--------------------------------

``classify_message(request, llm_provider, history, recent_smart_gen, pending_flow)``
makes one structured-output call on the **classifier tier**
(``MODEL_CLASSIFIER``, see :doc:`configuration`) at ``temperature=0.0``, with
:class:`UnifiedClassification` as the Pydantic response schema. It **never
raises** — on any failure it returns a safe ``fallback_intent``
classification and the agent degrades to its own fallback body.

The user block passed to the classifier carries the workspace context, the
recent conversation history (so referents like "the same", "it", "do that
for all" resolve), whether a smart generation just ran for this project, and
any pending question the assistant is currently awaiting an answer to.

Per-message caching
~~~~~~~~~~~~~~~~~~~

``get_or_classify(session, request, llm_provider)`` wraps the call in a
cache keyed on the BAF event id. The first transition condition or state
body to ask triggers the classification; every later caller on the same
message reads the cached object. One incoming WebSocket message therefore
consumes **exactly one** classification call.

The cache is primed by ``_ensure_unified_classification``, registered as a
transition condition that always returns ``False`` — it is a pure
side-effect hook, so it runs before any ``json_intent_matches`` condition
without ever causing a transition itself.

The 11 intents
~~~~~~~~~~~~~~

``_INTENT_NAMES`` in ``src/unified_classifier.py`` mirrors the
``new_intent`` declarations in ``modeling_agent.py``. Adding a state there
means mirroring it here.

.. list-table::
   :header-rows: 1
   :widths: 28 24 48

   * - Intent
     - State
     - When to Use
   * - ``hello_intent``
     - ``greetings_state``
     - An actual greeting / small-talk / thanks from the user
   * - ``create_complete_system_intent``
     - ``create_complete_system_state``
     - A NEW diagram or complete system from scratch.
       **Includes** "generate a class diagram" and "generate the GUI".
   * - ``modify_model_intent``
     - ``modify_model_state``
     - Add / remove / change elements of an existing diagram, including
       single-element creation
   * - ``modeling_help_intent``
     - ``modeling_help_state``
     - Conceptual modeling help, or how to run already-generated code
   * - ``describe_model_intent``
     - ``describe_model_state``
     - A question or advice request about the diagram on the canvas
   * - ``uml_spec_intent``
     - ``uml_rag_state``
     - The formal UML specification document (rare)
   * - ``generation_intent``
     - ``generation_state``
     - Source code in any stack, export, or deploy
   * - ``decline_intent``
     - ``decline_state``
     - User opts out ("nothing", "no thanks", "never mind")
   * - ``out_of_scope_intent``
     - ``out_of_scope_state``
     - A non-software artifact (an actual picture, a poem, a joke)
   * - ``meta_question_intent``
     - ``meta_question_state``
     - A question about the assistant itself — what it can do, why use it
   * - ``fallback_intent``
     - (state's own fallback)
     - Nothing fits; BAF's fallback body runs

**Intent declarations:** ``modeling_agent.py``.
**Authoritative rulebook:** ``unified_classifier._SYSTEM_PROMPT`` — the
one-line ``description=`` strings on each ``new_intent`` are summaries for
the local fallback classifier, not the rules the router actually follows.

Sub-routing fields
~~~~~~~~~~~~~~~~~~

The schema is deliberately wide so no downstream state body needs a second
LLM call:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Purpose
   * - ``generation_route``
     - ``smart`` / ``deterministic`` / ``modeling`` / ``other`` — which
       generation path to take (or that this is a misrouted modeling request)
   * - ``generator_type``
     - Which BESSER built-in to run when ``generation_route="deterministic"``
   * - ``refined_instructions``
     - The polished prompt handed to the Spec-Driven Agent when
       ``generation_route="smart"``
   * - ``provider``
     - Suggested LLM provider for a smart run (``anthropic`` / ``openai``);
       the frontend's BYOK selection can override it
   * - ``domain_mismatch`` / ``suggested_new_domain``
     - Guards a smart run whose request describes a different domain than the
       existing class diagram
   * - ``target_diagram_type``
     - Which diagram a create / modify / describe request is about. The
       ``_TARGET_DIAGRAM_TYPES`` literal currently lists 7 of the 8 supported
       tokens — ``UserDiagram`` is **missing**, so the classifier can never
       name a User Profile target and those requests fall through to the
       orchestrator's keyword cascade (which does resolve them). Adding the
       token here is the outstanding follow-up.
   * - ``model_disposition``
     - ``extend_existing`` / ``replace_existing`` / ``new_tab`` /
       ``reuse_for_generation`` / ``new_from_scratch``
   * - ``needs_clarification`` / ``clarifying_question``
     - Set when acting would require a guess; the agent asks instead
   * - ``pending_flow_action`` / ``pending_flow_answer``
     - Whether this message ANSWERS the assistant's pending question, and
       which valid answer it maps to
   * - ``reason``
     - One short sentence, used in logs and surfaced as a hint

Deterministic guards inside the classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A few decisions are too unreliable to leave to the LLM and are settled by
regex before or after the call:

- ``_names_unsupported_stack()`` — a message naming a language or framework
  BESSER has no deterministic generator for (rust, kotlin, rails, next.js,
  C++, C#, …) is forced onto the ``smart`` route. The LLM mapped
  "c classes" → ``java`` and "c++ classes" → ``python`` often enough that
  this had to become deterministic.
- ``_GITHUB_URL_RE`` + ``_GITHUB_CONTINUE_VERB_RE``
  (``src/handlers/generation_handler.py``) — "continue from
  github.com/owner/repo" must always produce a ``trigger_github_import``
  action, so it is matched deterministically rather than inferred.


Stage 3: Post-Validation and the Pending-Flow Gate
---------------------------------------------------

``_post_validate()`` (``src/unified_classifier.py``) runs on the returned
object before it is cached, correcting verdicts the rulebook says must hold
(for example rerouting a continue-from-GitHub message).

The pending-flow gate lives in ``json_intent_matches`` /
``json_no_intent_matched`` (``src/session_helpers.py``). When the assistant
is awaiting a reply to a question — replace/keep, the GUI generation-mode
choice, a smart-gen confirmation, a generator config prompt — the
classifier decides, *with that pending question in its context*, whether
the message answers it:

- ``pending_flow_action == "answer"`` (or no verdict at all) → intent
  matching is suppressed and the message stays in the current state, where
  ``_common_preamble()`` hands it to the flow handler.
- ``pending_flow_action == "new_request"`` → routing proceeds normally, so
  an unrelated instruction can never be swallowed by a half-finished flow.


Transition Priority System
--------------------------

State transitions are checked in this order (first match wins).

**Location:** ``src/state_bodies.py`` → ``add_unified_transitions()``

.. code-block:: text

   Priority 0: _ensure_unified_classification(session)
       Side-effect hook — primes the per-message classification cache.
       ALWAYS returns False, so it never transitions.

   Priority 1: Intent-matched JSON transitions
       json_intent_matches(session, {'intent_name': 'X'})
       → Reads the unified classifier's verdict (BAF's local classifier
         is consulted only if the unified call never ran)

   Priority 2: route_to_generation(session)
       → frontend_event callbacks and pending generator/smart-gen flows

   Priority 3: Text-event intent transitions (voice / plain-text events)
       when_intent_matched(intent)

   Priority 4: Fallback transitions
       json_no_intent_matched(session) → the state's fallback state

Priority 2 is deliberately cheap: ``should_route_to_generation()``
(``src/handlers/generation_handler.py``) makes **no** LLM call and runs
**no** text heuristics. It returns ``True`` only for ``frontend_event``
payloads and for messages arriving while a generator config or smart-gen
confirmation is pending. The older keyword safety nets
(``_is_modeling_request``, ``_is_diagram_creation_request``, phrase lists)
have been removed — the classifier's rulebook covers those cases directly.


The Local Fallback Classifier
------------------------------

The agent's *default* BAF intent classifier is the **local, free**
``SimpleIntentClassifier``, configured in
``agent_setup.init_intent_classifier_config()``:

.. code-block:: python

   SimpleIntentClassifierConfiguration(framework='tensorflow')

It is **not** an LLM. It trains at startup on the ``training_sentences``
declared with every intent in ``modeling_agent.py`` — which is why those
sentences are required, not optional. This keeps BAF's own per-message
prediction free instead of costing an LLM round-trip on every message.

It is consulted in only two situations:

1. **Exception path** — the unified classifier returned ``fallback_intent``
   (provider unavailable, call failed, empty message). Then
   ``json_intent_matches`` falls through to
   ``session.event.predicted_intent``.
2. **Text events** — voice messages (after speech-to-text) and plain-text
   events arrive without a JSON payload and route via
   ``when_intent_matched`` (transition priority 3).

.. note::

   The trade-off: BAF trains one Simple classifier per state at startup
   (plus a NER model), which is why the container takes minutes to open its
   WebSocket. See :doc:`deployment` for the health-check start period that
   accounts for it.


Diagram Type Resolution
-----------------------

Once the intent is classified and the state body executes, the system still
needs to decide **which diagram type** to target. The classifier's
``target_diagram_type`` is the primary signal; when it is ``None`` the
workspace orchestrator's keyword / pattern / context cascade resolves it.

See :doc:`orchestration` for the full three-level resolution system.


Debugging Intent Recognition
------------------------------

When routing goes wrong, check in this order:

1. **Read the classification.** Every classification logs its ``reason``
   field. That one sentence usually explains the whole routing decision.

2. **Check which classifier answered.** If the log shows a
   ``fallback_intent`` with a reason like "LLM provider unavailable" or
   "LLM returned no result", the unified call failed and BAF's local Simple
   classifier decided instead — expect lower accuracy.

3. **Check the pending-flow gate.** If the assistant had a question
   outstanding, look at ``pending_flow_action``. An ``answer`` verdict
   keeps the message in the current state by design; a ``new_request``
   verdict lets it route away.

4. **Check the rulebook, not the intent descriptions.** Routing rules live
   in ``unified_classifier._SYSTEM_PROMPT``. The ``description=`` strings on
   ``new_intent`` only feed the local fallback classifier.

5. **Check the deterministic guards.** ``_names_unsupported_stack()`` forces
   the smart route; the GitHub-continue regexes force an import. Neither
   can be overridden by the LLM verdict.

6. **Check diagram type resolution.** Does the classifier's
   ``target_diagram_type`` match what you expect? If it is ``None``, step
   through ``determine_target_diagram_type()`` (explicit keywords, then
   discriminating patterns, then context fallback).

.. code-block:: python

   # Quick diagnostic snippet (no agent runtime needed):
   from unified_classifier import _names_unsupported_stack
   from handlers.generation_handler import detect_generator_type

   msg = "your test message here"
   print(f"Generator keyword:  {detect_generator_type(msg)}")
   print(f"Unsupported stack:  {_names_unsupported_stack(msg)}")
