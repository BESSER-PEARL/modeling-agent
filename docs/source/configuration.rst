Configuration
=============

The Modeling Agent reads two kinds of configuration:

- ``config.yaml`` at the repository root — the BESSER Agentic Framework
  (BAF) settings: NLP, the WebSocket platform, and the server OpenAI key.
  Copy ``config_example.yaml`` to ``config.yaml`` and edit the values.
- **Environment variables** — everything the agent itself added on top:
  the per-call-site model routing table, the backend URL, and a handful of
  behavioral switches.

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
       # CORS: browser origins allowed to open a WebSocket to this agent.
       origins:
         - "https://editor.besser-pearl.org"
         - "http://localhost:3000"
         - "http://localhost:5173"
         - "http://localhost:8080"
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
     - Bind address for the WebSocket server (``0.0.0.0`` in Docker)
   * - ``platforms.websocket.port``
     - ``8765``
     - Port for the WebSocket server
   * - ``platforms.websocket.origins``
     - production hosts + localhost
     - CORS whitelist of browser origins. BAF 4.3.2 passes this to
       ``websockets.serve(origins=...)``. When the key is **absent**, any
       origin is accepted. The Docker entrypoint writes it, so containers
       are restricted by default; override the two production entries with
       ``BESSER_AGENT_WS_ORIGIN`` and ``BESSER_AGENT_WS_ORIGIN_ALT``.
   * - ``platforms.websocket.streamlit.host``
     - ``localhost``
     - Streamlit UI host (the agent runs with ``use_ui=False``)
   * - ``platforms.websocket.streamlit.port``
     - ``5000``
     - Streamlit UI port

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
     - Minimum confidence for BAF's local intent classifier. Note this
       does **not** gate the unified classifier, which is authoritative —
       see :doc:`intent_recognition`.
   * - ``nlp.openai.api_key``
     - (required)
     - The **server** OpenAI key. A user's own key (BYOK) is routed
       separately per request — see below.


Model Routing
-------------

**Location:** ``src/model_config.py``

Every LLM call in the agent belongs to a *tier*, and each tier is
independently env-overridable so a deployment (e.g. a gateway exposing
different model names) can re-point one tier without a code change. The
variable name is the tier name prefixed with ``BESSER_AGENT_MODEL_``.

.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - Constant
     - Default
     - Used by
   * - ``MODEL_CLASSIFIER``
     - ``gpt-4o-mini``
     - The unified classifier, the request planner, JSON repair /
       self-correction / name-extraction recovery, the memory summarizer,
       UML RAG, help and fallback streaming, and ``gpt_predict_json``.
       Also the default model of the shared ``gpt`` / ``gpt_text``
       instances.
   * - ``MODEL_GENERATION_LARGE``
     - ``gpt-5.6-terra``
     - Complete-system structured diagram generation — the one place where
       output quality *is* the product
   * - ``MODEL_GENERATION_GUI``
     - (falls back to ``MODEL_GENERATION_LARGE``)
     - GUI complete-system generation. Its own knob because design quality
       tracks the model's taste far more than diagram generation does.
   * - ``MODEL_GENERATION_SMALL``
     - ``gpt-5.6-luna``
     - Single-element and modification structured calls,
       ``describe_model`` streaming, and the file-conversion text path
   * - ``MODEL_REASONING``
     - ``gpt-5.6-terra``
     - The free-text design-reasoning pass of two-pass generation
   * - ``MODEL_VISION``
     - ``gpt-5``
     - File-conversion vision calls (image / PDF → diagram)
   * - ``MODEL_EMBEDDINGS``
     - ``text-embedding-3-small``
     - RAG embeddings. Pinned explicitly so a silent library default bump
       can never invalidate the persisted vector store.

Example: point the classifier tier at a different model without touching
code::

   BESSER_AGENT_MODEL_CLASSIFIER=gpt-4o
   BESSER_AGENT_MODEL_GENERATION_LARGE=gpt-5.6-sol

Temperature vs. reasoning_effort
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The gpt-5 family and the o-series reject an explicit ``temperature`` other
than the default — the API returns HTTP 400. Call sites must therefore
omit the parameter for those models and pass ``reasoning_effort`` instead.
``model_config`` exposes two helpers that every call site uses:

.. code-block:: python

   supports_custom_temperature(model) -> bool
       # False for models whose name starts with gpt-5, o1, o3 or o4

   reasoning_effort_for(model) -> str | None
       # None for models that accept a temperature (they reject the param);
       # otherwise MODEL_REASONING_EFFORT

The pattern appears in ``diagram_handlers/core/base_handler.py`` and
``llm/provider.py``:

.. code-block:: python

   if supports_custom_temperature(effective_model):
       parse_kwargs["temperature"] = temperature
   else:
       parse_kwargs["reasoning_effort"] = reasoning_effort_for(effective_model)

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``BESSER_AGENT_MODEL_REASONING_EFFORT``
     - ``low``
     - ``reasoning_effort`` passed for gpt-5 / o-series calls. ``low``
       keeps hidden reasoning small — structured diagram specs do not need
       deep chain-of-thought. ``minimal`` is rejected by some models.


Environment Variables
---------------------

.. list-table::
   :header-rows: 1
   :widths: 38 14 48

   * - Variable
     - Default
     - Description
   * - ``OPENAI_API_KEY``
     - —
     - Server OpenAI key. The Docker entrypoint writes it into the
       generated ``config.yaml`` as ``nlp.openai.api_key``.
   * - ``BESSER_AGENT_MODEL_*``
     - see above
     - Per-tier model overrides
   * - ``BESSER_BACKEND_URL``
     - ``http://localhost:3001``
     - Base URL of the BESSER backend. Used by the diagram-validation
       bridge (``src/handlers/validation_handler.py``) and the pilot
       telemetry collector (``src/telemetry.py``). Without it, every
       validation in a container fails with connection-refused.
   * - ``BESSER_AGENT_ALLOW_CUSTOM_BASE_URL``
     - unset (off)
     - Whether a BYOK request may specify its own API base URL. Set to
       ``"false"`` on the hosted deployment.
   * - ``BESSER_AGENT_STT_LANGUAGE``
     - auto-detect
     - Pins speech-to-text to a language (``en``, ``fr``, ``de``, …).
       Unset means Whisper auto-detects — a hard pin to English
       mis-transcribed non-English voice.
   * - ``BESSER_AGENT_COMPACT_SPEC``
     - ``1`` (on)
     - Use the compact class-diagram spec schema for generation.
   * - ``LOG_PROMPTS``
     - unset (off)
     - Log full LLM prompts. Leave off outside local debugging.
   * - ``ANONYMIZED_TELEMETRY`` / ``CHROMA_TELEMETRY_ENABLED``
     - ``False``
     - Set defensively at startup by ``modeling_agent.py`` to disable
       Chroma telemetry.


Bring Your Own Key (BYOK)
--------------------------

**Location:** ``src/byok.py``, plus the request boundary in
``patches/websocket_platform.py``.

A user can paste their own API key in the frontend. It arrives as BAF
session variables (``user_api_key``, ``user_api_provider``,
``user_api_model``, ``user_api_base``) and is *never* written to
``config.yaml`` or to the shared LLM objects.

Routing is driven by a ``contextvars.ContextVar`` (``current_byok``). The
WebSocket request boundary sets it just before ``agent.receive_event`` and
resets it after. Because ``loop.call_soon_threadsafe`` copies the calling
thread's context, the value propagates into that session's event-loop
thread for that turn only — so two concurrent users can never cross keys,
and the shared server LLM is never mutated.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Property
     - Value
   * - Supported providers
     - ``openai``, ``anthropic``, ``mistral``, ``nebius`` (Mistral and Nebius
       speak the OpenAI Chat Completions protocol at
       ``https://api.mistral.ai/v1`` and
       ``https://api.tokenfactory.nebius.com/v1/``)
   * - Routed call shapes
     - ``base_handler._predict_raw`` / ``predict_with_retry`` (generation),
       ``session_helpers.stream_llm_response`` (conversational reply, help,
       describe), ``base_handler.predict_structured`` and the intent
       classifier's ``LLMProvider.parse`` (``.parse()`` on a user's OpenAI
       key, JSON mode + schema validation for the other providers), and file
       attachments (``gpt_predict_json``; image/PDF vision with an OpenAI key)
   * - Not routed
     - The local BAF fallback classifier (no LLM), RAG embeddings, and image/PDF
       vision for non-OpenAI keys (they cannot call the OpenAI vision API)
   * - Per-request timeout
     - 300 s (without it the SDKs default to several minutes, letting one
       hung call stall a whole turn; 120 s cut long-spec reasoning passes short)
   * - Error handling
     - Provider call errors (auth, rate limit) propagate so the existing
       ``errors.classify_error`` taxonomy can surface them. Only
       configuration problems (unknown provider, missing SDK) raise
       ``BYOKError``.

Because BYOK bypasses any gateway, the agent's OpenAI-canonical per-call
model names are collapsed into two tiers and mapped to each provider's
equivalent (``_PROVIDER_TIER_MODELS`` in ``src/byok.py``). A model the user
explicitly chose is used for every call; only without one does ``small`` use
the provider's cheap sibling so routing and repair calls stay inexpensive on
the user's key.


Tunable Constants
-----------------

**Location:** ``src/agent_config.py`` — values that are not env-driven but
live in one place instead of being scattered across modules.

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Constant
     - Value
     - Description
   * - ``MAX_TABS``
     - ``5``
     - Maximum diagram tabs per type in a workspace
   * - ``MAX_USER_MESSAGE_CHARS``
     - ``12_000``
     - Hard cap applied at the protocol boundary, so a huge paste cannot
       reach memory or an LLM prompt untruncated
   * - ``GRACE_PERIOD_SECONDS``
     - ``300``
     - How long a disconnected session survives before the reaper closes it
   * - ``STREAM_BUFFER_THRESHOLD``
     - ``200``
     - Streaming chunk buffer size in characters
   * - ``LLM_TEMPERATURE`` / ``LLM_TEXT_TEMPERATURE``
     - ``0.2`` / ``0.4``
     - Structured vs. free-text temperature (ignored for gpt-5 / o-series)
   * - ``LLM_MAX_TOKENS_LARGE`` / ``_SMALL`` / ``_TEXT``
     - ``8192`` / ``2048`` / ``4096``
     - Completion budgets
   * - ``CONVERSATION_HISTORY_DEPTH``
     - ``10``
     - Recent messages fed verbatim each turn. The rolling summary in
       ``memory/conversation_memory.py`` covers everything older, so the
       agent remembers the whole session, not just this window.


RAG Configuration
-----------------

**Location:** ``agent_setup.init_rag()``

- **Vector store:** ChromaDB, persisted in ``uml_vector_store/`` (auto-created)
- **Source documents:** ``uml_specs/formal-17-12-05.pdf`` (OMG UML 2.5.1)
- **Embedding model:** ``MODEL_EMBEDDINGS`` (``text-embedding-3-small``)
- **Chunking:** ``RecursiveCharacterTextSplitter``, chunk size 1000, overlap 100
- **Retrieval:** ``k=4``, with 6 previous messages of context
- **Answering model:** ``MODEL_CLASSIFIER`` tier

If RAG initialization fails (missing store, missing key, import error), the
agent logs a warning and continues **without** RAG — UML spec queries fall
back to LLM-only responses.


Security Notes
--------------

- **Never** commit real API keys to the repository.
- Use ``config_example.yaml`` and ``.env.example`` as templates.
- ``config.yaml`` is listed in ``.gitignore``.
- The Docker entrypoint prints the generated config for debugging but
  **redacts** the ``api_key`` line, so the key never lands in container logs.
- A user's BYOK key lives only in the per-request context var and the BAF
  session; it is never logged or persisted.
- In production, use environment variables or secrets management.
