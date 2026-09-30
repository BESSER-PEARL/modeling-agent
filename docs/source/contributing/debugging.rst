Debugging & Common Pitfalls
===========================

This section covers how to debug intent recognition issues, routing problems,
and diagram type resolution, plus a list of known pitfalls.

.. contents:: On this page
   :local:
   :depth: 2


Debugging Intent Recognition
-----------------------------

When a user message is handled by the wrong state, check in this order:

Step 1: Read the classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every ``UnifiedClassification`` carries a ``reason`` field — one sentence
explaining the verdict — and it is logged. That is usually the whole answer.
Enable debug logging for more:

.. code-block:: python

   import logging
   logging.getLogger("besser").setLevel(logging.DEBUG)

Step 2: Check which classifier answered
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``fallback_intent`` verdict with a reason like "LLM provider unavailable" or
"LLM returned no result" means the unified call failed and BAF's local
``SimpleIntentClassifier`` decided instead — expect lower accuracy.

Step 3: Check the pending-flow gate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the assistant had a question outstanding, look at ``pending_flow_action``.
An ``"answer"`` verdict deliberately keeps the message in the current state so
``_common_preamble()`` can hand it to the flow handler; only
``"new_request"`` lets it route away. A message that seems "stuck" in a state
is usually this working as designed.

Step 4: Check transition priority
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Is the message hitting **Priority 1** (``json_intent_matches``, reading the
classifier verdict) or **Priority 2** (``route_to_generation``)? Priority 2
fires only for ``frontend_event`` payloads and pending generator / smart-gen
flows — it runs no LLM call and no text heuristics.

.. note::

   The old keyword pre-filters and cross-validation safety nets
   (``_is_modeling_request()``, ``_is_diagram_creation_request()``, the
   ``json_intent_matches`` keyword override) **no longer exist**. Routing
   rules now live in ``unified_classifier._SYSTEM_PROMPT``, which is the
   single place to change routing behavior.

Step 5: Check the deterministic guards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two decisions override the LLM verdict and cannot be changed by prompting:

.. code-block:: python

   from unified_classifier import _names_unsupported_stack
   from handlers.generation_handler import detect_generator_type

   msg = "your test message here"
   print(f"Generator keyword:  {detect_generator_type(msg)}")
   print(f"Unsupported stack:  {_names_unsupported_stack(msg)}")

``_names_unsupported_stack()`` forces the smart generation route; the
GitHub-continue regexes in ``generation_handler`` force an import.

Step 6: Check Keyword Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Does ``detect_generator_type()`` return the expected value? Remember:

- Word-boundary matching applies to ``"sql"`` and ``"backend"``
- Fuzzy regex patterns (``_FUZZY_PATTERNS``) are checked after exact keywords
- Dict ordering in ``GENERATOR_KEYWORDS`` matters (longer keywords first)

Step 7: Check Diagram Type Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Does ``determine_target_diagram_type()`` return the expected diagram type?

1. Check ``_collect_explicit_targets(msg)`` for keyword matches
2. Check ``_rank_implicit_targets(msg)`` for discriminating pattern matches
3. Check the context fallback (active diagram type from ``WorkspaceContext``)


Debugging Request Routing
--------------------------

Request Not Reaching the Right Handler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Check transition priority:

   - **Priority 1**: LLM intent match (``json_intent_matches``)
   - **Priority 2**: Keyword-based generation route (``route_to_generation``)
   - **Priority 3**: Text-event intent match (backward compatibility)
   - **Priority 4**: Fallback (``json_no_intent_matched``)

2. Check if ``pending_generator_type`` or ``pending_complete_system`` is set on
   the session — these **suppress** intent matching:

   .. code-block:: python

      session.get("pending_generator_type")   # "_awaiting_selection" suppresses
      session.get("pending_complete_system")   # Any truthy value suppresses
      session.get("pending_gui_choice")        # Any truthy value suppresses


Common Pitfalls
----------------

1. **"generate" is ambiguous**

   ``"generate django"`` = code generation, but ``"generate a class diagram"`` =
   diagram creation. Always test both when changing generation-related code.

2. **Substring matching**

   ``"sql"`` matches inside ``"sqlalchemy"``. Use ``_BOUNDARY_KEYWORDS`` for
   short keywords that might be substrings of other keywords.

3. **A pending flow can suppress intent matching**

   When the assistant is awaiting an answer, ``json_intent_matches()`` returns
   ``False`` for ALL intents *unless* the classifier labelled the message
   ``pending_flow_action="new_request"`` — so an answer stays in the current
   state for ``_common_preamble`` to handle, while an unrelated instruction
   still routes normally.

4. **Frontend context can be stale**

   After injecting a diagram, the next message from the frontend may carry the
   pre-injection model snapshot. The agent resolves the model from
   ``projectSnapshot`` (``activeModel`` is ignored), but the snapshot itself
   can still lag an injection.

5. **Dict ordering in GENERATOR_KEYWORDS matters**

   Keywords are checked in insertion order. ``"sqlalchemy"`` must come before
   ``"sql"`` to avoid the shorter keyword matching first.

6. **Routing rules live in one place now**

   There is no keyword pre-filter layer to add a special case to. Routing
   behavior changes belong in ``unified_classifier._SYSTEM_PROMPT``, or — for
   decisions the LLM is unreliable at — in an explicit deterministic guard
   like ``_names_unsupported_stack()``.

7. **Misclassification between non-generation intents**

   There is no override layer rescuing, say, ``modify_model_intent`` from
   being read as ``create_complete_system_intent``. Such misclassifications
   rely entirely on the classifier rulebook, so fix them there.
