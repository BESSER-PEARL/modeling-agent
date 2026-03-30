Configuration
=============

The Modeling Agent is configured via ``config.yaml`` at the repository root.
Copy ``config_example.yaml`` to ``config.yaml`` and edit the values.

.. contents:: On this page
   :local:
   :depth: 2

config.yaml Reference
---------------------

The configuration file uses YAML format. Below is the full structure with
all supported keys.

.. code-block:: yaml

   agent:
     check_transitions_delay: 5

   nlp:
     language: en
     region: US
     timezone: Europe/Madrid
     pre_processing: True
     intent_threshold: 0.55
     openai:
       api_key: your-api-key

   platforms:
     websocket:
       host: localhost
       port: 8765
       streamlit:
         host: localhost
         port: 5000

Agent
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Key
     - Default
     - Description
   * - ``agent.check_transitions_delay``
     - ``5``
     - Delay (seconds) before checking state transitions

WebSocket Platform
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Default
     - Description
   * - ``platforms.websocket.host``
     - ``localhost``
     - Bind address for the WebSocket server
   * - ``platforms.websocket.port``
     - ``8765``
     - Port for the WebSocket server
   * - ``platforms.websocket.streamlit.host``
     - ``localhost``
     - Streamlit UI host (if enabled)
   * - ``platforms.websocket.streamlit.port``
     - ``5000``
     - Streamlit UI port (if enabled)

NLP / LLM
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Key
     - Default
     - Description
   * - ``nlp.language``
     - ``en``
     - Language code for NLP processing
   * - ``nlp.region``
     - ``US``
     - Region for locale-specific processing
   * - ``nlp.timezone``
     - ``Europe/Madrid``
     - Timezone for timestamp handling
   * - ``nlp.pre_processing``
     - ``True``
     - Enable input pre-processing
   * - ``nlp.intent_threshold``
     - ``0.55``
     - Minimum confidence for intent classification
   * - ``nlp.openai.api_key``
     - (required)
     - OpenAI API key (can also be set via ``OPENAI_API_KEY`` env var)

Environment Variables
---------------------

The agent also reads from ``.env`` via ``python-dotenv``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``OPENAI_API_KEY``
     - Alternative location for OpenAI API key

LLM Configuration
-----------------

The agent uses two LLM instances, both GPT-4.1-mini:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Instance
     - Mode
     - Temperature
     - Purpose
   * - ``gpt``
     - JSON
     - 0.2
     - Structured diagram JSON generation
   * - ``gpt_text``
     - Text
     - 0.4
     - Free-text reasoning and Q&A

Intent classification uses a separate GPT-4.1-mini instance configured in
``agent_setup.py``.

RAG Configuration
-----------------

The RAG system uses ChromaDB with the following defaults:

- **Vector store directory:** ``uml_vector_store/`` (auto-created)
- **Source documents:** ``uml_specs/formal-17-12-05.pdf`` (OMG UML 2.5.1)
- **Embedding model:** OpenAI text-embedding (via LangChain)
- **Chunk size:** Configured in ``agent_setup.py``

If RAG initialization fails (e.g., missing PDF), the agent continues without
RAG support. UML spec queries fall back to LLM-only responses.

Security Notes
--------------

- **Never** commit real API keys to the repository.
- Use ``config_example.yaml`` and ``.env.example`` as templates.
- The ``config.yaml`` file is listed in ``.gitignore``.
- In production, use environment variables or secrets management.
