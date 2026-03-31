How-To Guides
=============

Step-by-step guides for the most common contribution scenarios: adding diagram
types, generators, intents, and modifying intent recognition.

.. contents:: On this page
   :local:
   :depth: 2


How to Add a New Diagram Type
-----------------------------

Adding a new diagram type touches 8 places. Follow this checklist in order:

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

7. **Add tests** in ``tests/test_diagram_handlers.py``

8. **Update docs** in ``docs/source/diagram_handlers.rst``


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
   ``src/orchestrator/request_planner.py``:

   .. code-block:: python

      GENERATOR_PREREQUISITES["my_gen"] = ["ClassDiagram"]

5. **Add config prompt** in ``_build_config_prompt()``

6. **Add tests** in ``tests/test_generation_handler.py``

7. **Update docs** in ``docs/source/usage.rst``


How to Add a New Intent
-----------------------

1. **Define the intent** in ``modeling_agent.py``:

   .. code-block:: python

      my_intent = agent.new_intent(
          name="my_intent",
          description="When the user wants to do X. Keywords: ..."
      )

   .. warning::

      Intent descriptions are the **primary signal** for the LLM classifier.
      Include explicit positive examples, negative examples (what it's NOT),
      and disambiguation rules for confusable intents. See
      :doc:`../intent_recognition` for the full intent description guidelines.

2. **Create a state** in ``modeling_agent.py``:

   .. code-block:: python

      my_state = agent.new_state("my_state")

3. **Write the state body** in ``src/state_bodies.py``:

   .. code-block:: python

      def my_body(session: Session):
          request = parse_assistant_request(session)
          # ... handle the intent ...
          reply_message(session, "Done!")

4. **Register** in ``register_all()`` (same file):

   - Add to ``states`` dict
   - Add to ``intents`` dict
   - Add to ``intent_map``

5. **Add tests** for the state body logic


How to Modify Intent Recognition
---------------------------------

The intent recognition system has multiple layers. Choose the right one for
your change:

Fixing a Misclassification for a Specific Phrase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, update the **intent description** in ``modeling_agent.py`` with an
explicit example. This is the highest-impact, lowest-risk change.

.. code-block:: python

   description=(
       "... existing description ... "
       'NEW: "your problematic phrase" should match this intent because ...'
   )

Adding a Pre-Filter Guard (Zero Latency)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add logic to ``_is_modeling_request()`` or ``_is_diagram_creation_request()``
in ``src/handlers/generation_handler.py``.

These run **before** the LLM classifier and catch obvious patterns. They also
run **after** classification as safety nets (cross-validation in
``json_intent_matches()``).

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
