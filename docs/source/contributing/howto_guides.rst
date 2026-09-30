How-To Guides
=============

Step-by-step guides for the most common contribution scenarios: adding diagram
types, generators, intents, and modifying intent recognition.

.. contents:: On this page
   :local:
   :depth: 2


How to Add a New Diagram Type
-----------------------------

Adding a new diagram type touches every list below. Missing one produces a
stale-list bug that manual testing tends not to catch — testing from inside
the new diagram's own tab never exercises routing or discoverability, because
``context.activeDiagramType`` already gives the answer away. Test from a
*different* tab, with phrasing that does not literally name the type.

0. **Create the schemas** in ``src/schemas/<type>.py`` (single-element spec,
   complete-system spec, modification actions with ``Literal`` action names)
   and re-export them from ``src/schemas/__init__.py``.

1. **Create the handler** in ``src/diagram_handlers/types/``:

   .. code-block:: python

      class MyDiagramHandler(BaseDiagramHandler):
          def get_diagram_type(self) -> str:
              return "MyDiagram"

          def get_system_prompt(self) -> str:
              return "You are a MyDiagram expert..."

          def generate_single_element(self, user_request: str, existing_model=None, **kwargs) -> dict: ...
          def generate_complete_system(self, user_request: str, existing_model=None) -> dict: ...
          def generate_fallback_element(self, request: str) -> dict: ...

2. **Register** in ``src/diagram_handlers/registry/factory.py`` — add your
   handler class to the ``HANDLER_CLASSES`` tuple at module level.

3. **Add type** to ``SUPPORTED_DIAGRAM_TYPES`` in ``src/protocol/types.py``

4. **Add display metadata** in ``src/diagram_handlers/registry/metadata.py``

5. **Add explicit keywords** to ``KEYWORD_TARGETS`` in
   ``src/orchestrator/workspace_orchestrator.py``:

   .. code-block:: python

      KEYWORD_TARGETS = [
          ...
          ("my diagram", "MyDiagram"),
          ("my model", "MyDiagram"),
      ]

6. **Add discriminating pattern** to ``_IMPLICIT_PATTERNS`` in the same file:

   .. code-block:: python

      _IMPLICIT_PATTERNS.append(
          ("MyDiagram", re.compile(
              r"\b(?:strong_signal_word|another_signal"
              r"|word_a\b.{0,30}\bword_b)\b", re.I)),
      )

   See :doc:`../orchestration` for how discriminating patterns work.

   .. warning::

      Pattern order matters — the first match wins. Place a new pattern where
      its vocabulary will not be stolen by a broader one above it (BPMN sits
      above StateMachineDiagram precisely because "process" is in both).

7. **Append to** ``FALLBACK_PRIORITY`` in the same file, so the type can be
   resolved from the project snapshot when nothing else matches.

8. **Teach the classifier**: add the token to ``_TARGET_DIAGRAM_TYPES`` in
   ``src/unified_classifier.py`` and add the type's vocabulary to
   ``_SYSTEM_PROMPT``, so a request naming it reaches a modeling intent at
   all. Without this the message can fall through to the fallback body.

9. **Update the capability copy** in ``src/state_bodies.py`` — the
   ``_QUICK_RESPONSES`` capability text *and* ``global_fallback_body``'s
   prompt both enumerate supported types independently. Grep for an existing
   type's name (e.g. ``"quantum"``) to find every place.

10. **Add suggestions** in ``src/suggestions.py`` and wire them into
    ``_DIAGRAM_SUGGESTION_HANDLERS``.

11. **Add tests** in ``tests/test_diagram_handlers.py`` plus a dedicated
    ``tests/test_<type>.py``. Handler-level tests instantiate the handler with
    ``MyDiagramHandler(None)`` — the LLM argument is only needed by methods
    that actually call the model, so deterministic repair logic can be tested
    as a pure function against hand-built dicts.

12. **Update docs**: ``docs/source/diagram_handlers.rst``,
    ``docs/source/websocket_protocol.rst``, ``docs/source/getting_started.rst``
    and the README table.

13. **Frontend** (separate repo): the type must exist in the editor's own
    diagram-type union and be a valid ``activeDiagramType`` context value.


How to Add a New Generator
--------------------------

1. **Add keywords** to ``GENERATOR_KEYWORDS`` in
   ``src/handlers/generation_handler.py``:

   .. code-block:: python

      GENERATOR_KEYWORDS: Dict[str, List[str]] = {
          ...
          "my_gen": ["my generator", "generate my_gen"],
      }

   .. warning::

      Dict ordering matters. If your keyword is a substring of another
      (e.g. ``"sql"`` vs ``"sqlalchemy"``), place the longer keyword first.

2. **Add required fields** (if any) to ``GENERATOR_REQUIRED_FIELDS``:

   .. code-block:: python

      GENERATOR_REQUIRED_FIELDS["my_gen"] = ["setting1", "setting2"]

3. **Add inline config parsing** in ``parse_inline_generator_config()``

4. **Add prerequisites** to ``GENERATOR_PREREQUISITES`` in
   ``src/handlers/generation_handler.py`` (``request_planner.py`` imports it
   from there and injects it into the Tier-2 planner prompt):

   .. code-block:: python

      GENERATOR_PREREQUISITES["my_gen"] = ["ClassDiagram"]

5. **Add config prompt** in ``_build_config_prompt()`` and defaults in
   ``_normalize_defaults()``

6. **Mirror the name** in the two ``_DETERMINISTIC_GENERATOR_TYPES``
   ``Literal`` lists — ``src/unified_classifier.py`` and
   ``src/handlers/smart_generation_handler.py``. A generator the classifier
   cannot name is a generator it will never route to.

7. **Add tests** in ``tests/test_generation_handler.py``

8. **Update docs** in ``docs/source/usage.rst``


How to Add a New Intent
-----------------------

1. **Define the intent** in ``modeling_agent.py``:

   .. code-block:: python

      my_intent = agent.new_intent(
          name="my_intent",
          description="User wants to do X.",   # ONE line — see below
          training_sentences=[                  # REQUIRED
              "do x for me",
              "can you x this",
              "please x the model",
          ],
      )

   .. warning::

      ``training_sentences`` are **required**, not optional. The agent's
      default BAF classifier is the local ``SimpleIntentClassifier``, which
      trains on them at startup; an intent with none breaks it.

      Keep ``description`` to **one line**. It only feeds that local fallback
      classifier — it does not drive routing, and the long keyword essays
      these strings used to carry are gone. See
      :doc:`../intent_recognition`.

2. **Mirror the name** in ``_INTENT_NAMES`` in
   ``src/unified_classifier.py`` and write the real routing rule in
   ``_SYSTEM_PROMPT``. This is the step that actually makes the intent
   reachable — include positive examples, what it is *not*, and
   disambiguation against confusable intents.

3. **Create a state** in ``modeling_agent.py``:

   .. code-block:: python

      my_state = agent.new_state("my_state")

4. **Write the state body** in ``src/state_bodies.py``:

   .. code-block:: python

      def my_body(session: Session):
          request = _common_preamble(session)
          if request is None:
              return          # a pending flow or attachment consumed it
          # ... handle the intent ...
          reply_message(session, "Done!")

5. **Register** in ``register_all()`` (same file):

   - Add to ``states`` dict
   - Add to ``intents`` dict
   - Add to ``intent_map``
   - Add a ``(state_name, fallback_name)`` entry to the transition-wiring loop

6. **Add tests** for the state body logic and for the classifier verdict
   (``tests/test_unified_classifier.py``)


How to Modify Intent Recognition
---------------------------------

The intent recognition system has multiple layers. Choose the right one for
your change:

Fixing a Misclassification for a Specific Phrase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit the rule for that intent in ``_SYSTEM_PROMPT``
(``src/unified_classifier.py``). This is the highest-impact, lowest-risk
change, and it is the **only** place routing rules live.

.. code-block:: python

   _SYSTEM_PROMPT = (
       ...
       "modify_model_intent: user wants to ADD / REMOVE / CHANGE "
       "elements in an existing diagram. ... "
       'NEW: "your problematic phrase" is this intent because ...'
   )

.. warning::

   Do **not** put the rule in the ``description=`` string of
   ``agent.new_intent()`` in ``modeling_agent.py``. Those one-liners only feed
   BAF's local fallback classifier; they do not drive routing. Keep them short.

   Do add a representative ``training_sentence`` there, though — the local
   ``SimpleIntentClassifier`` trains on them at startup, and an intent with no
   training sentences breaks that classifier.

Adding a Deterministic Guard (Zero Latency)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The keyword pre-filter layer (``_is_modeling_request()``,
``_is_diagram_creation_request()``, and the ``json_intent_matches()``
cross-validation) has been **removed**. Routing rules belong in
``unified_classifier._SYSTEM_PROMPT``.

Reach for a deterministic guard only where the LLM is *measurably* unreliable
and the decision must never be guessed. Existing examples:

- ``_names_unsupported_stack()`` in ``src/unified_classifier.py`` — forces the
  smart generation route when a message names a language or framework BESSER
  has no generator for. Added because the classifier mapped "c classes" to
  ``java`` and "c++ classes" to ``python``.
- ``_GITHUB_URL_RE`` + ``_GITHUB_CONTINUE_VERB_RE`` in
  ``src/handlers/generation_handler.py`` — a continue-from-GitHub request must
  never be invented, missed, or swallowed.

Both run in ``_post_validate()`` or at the handler boundary and *override* the
LLM verdict, so keep them precision-first.

Adding a Generator Keyword
~~~~~~~~~~~~~~~~~~~~~~~~~~

Add to ``GENERATOR_KEYWORDS`` in ``src/handlers/generation_handler.py``.

If the keyword is short or ambiguous (≤6 chars), add it to
``_BOUNDARY_KEYWORDS`` for word-boundary matching to avoid substring collisions:

.. code-block:: python

   _BOUNDARY_KEYWORDS = {"sql", "backend", "your_short_keyword"}

Adding a Diagram Type Keyword
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- For **exact phrases**: add to ``KEYWORD_TARGETS`` in
  ``src/orchestrator/workspace_orchestrator.py``.
- For **discriminating patterns**: add to ``_IMPLICIT_PATTERNS`` (same file).

Changing Transition Routing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modify ``add_unified_transitions()`` in ``src/state_bodies.py``.

.. warning::

   Always test with both the intended phrase AND similar phrases that should
   NOT match. For example, when adding "backend" as a generator keyword,
   verify that "go back to the backend concept" does NOT trigger generation.
