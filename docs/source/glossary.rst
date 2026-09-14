Glossary
========

.. glossary::
   :sorted:


   BESSER
      An open-source low-code platform for smart software modeling — *better
      software faster*. The Modeling Agent is part of the BESSER ecosystem.
      See `BESSER on GitHub <https://github.com/BESSER-PEARL>`_ and the
      `BESSER documentation <https://besser-pearl.github.io/BESSER/>`_.

   BESSER Agentic Framework
      The Python framework that provides the state machine, WebSocket platform,
      and intent classification infrastructure used by the Modeling Agent. Part
      of the BESSER platform. See
      `BESSER documentation <https://besser-pearl.github.io/BESSER/>`_.

   Apollon
      The third-party diagram editor library the Web Modeling Editor renders
      with. Its model format uses UUID-keyed ``elements`` and ``relationships``
      maps with explicit bounds — quite unlike the agent's simple
      :term:`SystemSpec`, which is why a :term:`Converter` sits between them.

   BPMN
      Business Process Model and Notation. The agent's storage-bucket token for
      this diagram type is ``BPMN``, not ``BPMNDiagram`` — the editor's
      converter sets the Apollon ``model.type`` itself.

   BUML
      BESSER UML — the internal metamodel representation used by BESSER generators.
      The backend converts Apollon JSON to BUML for code generation and export.

   BYOK
      Bring Your Own Key. A user-supplied OpenAI, Anthropic or Mistral API key,
      sent as a BAF session variable and routed through a *per-request* client
      held in a context var — never written to the shared LLM objects, so two
      concurrent users can never cross keys.

   Converter
      A frontend component that transforms the Modeling Agent's simple spec format
      (e.g., ``{ classes: [...], relationships: [...] }``) into the detailed Apollon
      model format (with UUIDs, positions, bounds). Pure function, no editor needed.

   GrapesJS
      The open-source web builder framework used for GUI NoCode diagrams. The
      Modeling Agent generates GrapesJS-compatible JSON for the GUI editor.

   Intent
      A classification label assigned to a user message (e.g., ``modify_model_intent``,
      ``generation_intent``). The intent determines which state body handles the
      request.

   JSON Mode
      An OpenAI API mode (``response_format={"type": "json_object"}``) where the
      model is constrained to return valid JSON, without a schema. Used by
      ``gpt_predict_json``. Diagram generation uses the stricter
      :term:`Structured Outputs` instead.

   Reasoning Effort
      The parameter gpt-5 and o-series models take in place of
      ``temperature``, which they reject. ``model_config.reasoning_effort_for()``
      decides which of the two a given model gets; the default effort is
      ``low``, since a structured diagram spec does not need deep
      chain-of-thought.

   Smart Generator
      The LLM-authored code-generation path, as opposed to BESSER's
      template-driven deterministic generators. Used when the request names a
      stack BESSER has no generator for, or a BESSER stack plus extras the
      template cannot produce. Reached via a ``trigger_smart_generator``
      action; it runs on the user's own key, so the agent asks for explicit
      confirmation first.

   Modifier
      A frontend component that applies modifications (add, rename, remove) to an
      existing Apollon model. Pure function — reads current model, returns updated
      model.

   Orchestrator
      The backend component (``src/orchestrator/``) that plans multi-step operations
      and resolves target diagram types from natural language.

   Quirk
      The quantum circuit simulator format used for ``QuantumCircuitDiagram``.
      Circuits are represented as column arrays of gate operations.

   RAG
      Retrieval-Augmented Generation — a technique that retrieves relevant document
      chunks from a vector store (ChromaDB) and includes them in the LLM prompt.
      The Modeling Agent uses RAG over the OMG UML 2.5.1 specification for answering
      UML questions.

   Redux
      A JavaScript state management library used by the frontend. Redux holds all
      project data (diagrams, models, tabs) as a single source of truth. The agent
      writes to Redux via converters; the editor reads from Redux to render.

   State Body
      A Python function that executes when the BESSER state machine enters a
      particular state. Each state body (e.g., ``modify_modeling_body``) handles
      a specific type of user request. Defined in ``src/state_bodies.py``.

   Structured Outputs
      An OpenAI API feature that constrains the LLM response to match a Pydantic
      schema exactly. The Modeling Agent uses this for all diagram generation,
      ensuring valid field names, types, and value ranges.

   SystemSpec
      The intermediate format returned by diagram handlers before frontend
      conversion. Contains classes, relationships, states, etc. in a simplified
      structure without UUIDs or positions.

   Text Mode
      An OpenAI API mode where the LLM returns free-form text. Used for Q&A,
      model descriptions, and reasoning passes (temperature 0.4, where the
      model accepts one).

   Unified Classifier
      ``src/unified_classifier.py`` — the agent's router. One
      structured-output LLM call per message returns the state-level intent
      *and* every sub-routing field downstream code needs, cached on the BAF
      event id. It replaced both BAF's per-message LLM intent classification
      and a second generation sub-router prompt.

   Simple Intent Classifier
      BAF's local, free TensorFlow intent classifier, trained at startup on
      the ``training_sentences`` declared with each intent. It is the agent's
      configured default classifier, but only decides routing on the
      exception path (the unified call failed) and for voice / plain-text
      events. Training it for every state is the main reason the agent takes
      minutes to boot.

   User Profile
      A BESSER model describing a *target user* as class-instance boxes drawn
      from a fixed bundled metamodel, whose attribute rows are matching
      **criteria** with a comparison operator (``age >= 18``) rather than
      plain instance values. Its diagram-type token is ``UserDiagram``.

   Widget Slot
      A ``<!--WIDGET:kind-->`` marker the LLM leaves in GUI page markup. The
      server splices a real, typed, data-bound widget (table, chart, metric
      card, form) into it — LLM markup can never masquerade as one.
