Configuration
=============

The Modeling Agent is configured via ``config.yaml`` at the repository root.
Copy ``config.yaml.example`` to ``config.yaml`` and edit the values.

.. contents:: On this page
   :local:
   :depth: 2

config.yaml Reference
--------------------

WebSocket Platform
~~~~~~~~~~~~~~~~~~

.. code-block:: ini

   [websocket_platform]
   websocket.host = 0.0.0.0
   websocket.port = 8765
   streamlit.host = localhost
   streamlit.port = 5001

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Default
     - Description
   * - ``websocket.host``
     - ``0.0.0.0``
     - Bind address for the WebSocket server
   * - ``websocket.port``
     - ``8765``
     - Port for the WebSocket server
   * - ``streamlit.host``
     - ``localhost``
     - Streamlit UI host (if enabled)
   * - ``streamlit.port``
     - ``5001``
     - Streamlit UI port (if enabled)

API
~~~

.. code-block:: ini

   [api]
   api.server.url = http://localhost:3001

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Key
     - Default
     - Description
   * - ``api.server.url``
     - ``http://localhost:3001``
     - Backend API server URL

NLP / LLM
~~~~~~~~~~

.. code-block:: ini

   [nlp]
   nlp.language = en
   nlp.region = US
   nlp.timezone = Europe/Madrid
   nlp.pre_processing = True
   nlp.intent_threshold = 0.4

   nlp.openai.api_key = YOUR-API-KEY
   nlp.intent.openai.model_name = gpt-4o-mini

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
     - ``0.4``
     - Minimum confidence for intent classification
   * - ``nlp.openai.api_key``
     - (required)
     - OpenAI API key
   * - ``nlp.intent.openai.model_name``
     - ``gpt-4o-mini``
     - Model for intent classification

Database Monitoring
~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

   [db]
   db.monitoring = False
   db.monitoring.dialect = postgresql
   db.monitoring.host = localhost
   db.monitoring.port = 5432
   db.monitoring.database = DB-NAME
   db.monitoring.username = DB-USERNAME
   db.monitoring.password = DB-PASSWORD

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Key
     - Default
     - Description
   * - ``db.monitoring``
     - ``False``
     - Enable session monitoring to database
   * - ``db.monitoring.dialect``
     - ``postgresql``
     - Database dialect
   * - ``db.monitoring.host``
     - ``localhost``
     - Database host
   * - ``db.monitoring.port``
     - ``5432``
     - Database port
   * - ``db.monitoring.database``
     - (required if enabled)
     - Database name
   * - ``db.monitoring.username``
     - (required if enabled)
     - Database username
   * - ``db.monitoring.password``
     - (required if enabled)
     - Database password

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
- Use ``config.yaml.example`` and ``.env.example`` as templates.
- The ``config.yaml`` file is listed in ``.gitignore``.
- In production, use environment variables or secrets management.
