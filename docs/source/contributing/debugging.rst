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

Step 1: Check the LLM Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enable debug logging to see what intent GPT-4.1-mini predicted and with what
confidence:

.. code-block:: python

   import logging
   logging.getLogger("besser").setLevel(logging.DEBUG)

Step 2: Check Pre-Filters
~~~~~~~~~~~~~~~~~~~~~~~~~~

Test whether the deterministic checks produce the expected result:

.. code-block:: python

   from handlers.generation_handler import (
       detect_generator_type,
       _is_modeling_request,
       _is_diagram_creation_request,
   )

   msg = "your test message here"
   print(f"Generator type:  {detect_generator_type(msg)}")
   print(f"Is modeling:     {_is_modeling_request(msg)}")
   print(f"Is diagram:      {_is_diagram_creation_request(msg.lower())}")

Step 3: Check Transition Priority
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Is the message hitting **Priority 1** (LLM intent match) or **Priority 2**
(generation route via keyword detection)?

If Priority 2, the LLM classifier likely misclassified the message. Check
``route_to_generation()`` with the same message.

Step 4: Check Handler Safety Nets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the message reaches the wrong handler, does the safety net catch it? Check
the response ``action`` field — if it's ``"assistant_message"`` with a redirect
message, the safety net fired.

Step 5: Check Cross-Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``json_intent_matches()`` function in ``src/session_helpers.py`` performs
cross-validation when the LLM says ``generation_intent``:

- If ``detect_generator_type()`` returns ``None`` → classification is overridden
- If ``_is_modeling_request()`` returns ``True`` → classification is overridden

This prevents misrouted modeling requests from reaching the generation handler.

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

3. **Pending state suppresses intent matching**

   When ``pending_generator_type`` is set, ``json_intent_matches()`` returns
   ``False`` for ALL intents, so the message stays in the current state for
   ``_common_preamble`` to handle.

4. **Frontend context can be stale**

   After injecting a diagram, the next message from the frontend may carry the
   pre-injection model snapshot. Don't assume ``activeModel`` is up-to-date.

5. **Dict ordering in GENERATOR_KEYWORDS matters**

   Keywords are checked in insertion order. ``"sqlalchemy"`` must come before
   ``"sql"`` to avoid the shorter keyword matching first.

6. **Pattern-based modeling detection uses "for" as a strong signal**

   ``_is_modeling_request()`` treats ``"verb … for <anything>"`` as modeling.
   This means ``"create a tool for generating code"`` would be classified as
   modeling. The ``_EXPLICIT_GENERATION_PHRASES`` veto list catches common
   false positives, but unusual phrasings may need new entries.

7. **Cross-validation only applies to generation_intent**

   The ``json_intent_matches()`` cross-validation in ``session_helpers.py``
   only overrides when the LLM predicts ``generation_intent``. Misclassifications
   between other intents (e.g. ``modify_model`` vs ``create_complete_system``)
   are not cross-validated — they rely solely on the LLM and intent descriptions.
