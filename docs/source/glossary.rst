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

   BUML
      BESSER UML — the internal metamodel representation used by BESSER generators.
      The backend converts Apollon JSON to BUML for code generation and export.

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
      An OpenAI API mode where the LLM returns valid JSON matching a Pydantic
      schema. Used for structured diagram generation (temperature 0.2).

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
      model descriptions, and reasoning passes (temperature 0.4).
