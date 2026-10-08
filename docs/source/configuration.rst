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


.. _model-routing:

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
       instances, which is what BPMN and User Profile complete-system
       generation run on (those handlers pass no model override).
   * - ``MODEL_GENERATION_LARGE``
     - ``gpt-5-mini``
     - Complete-system generation for class, state machine, object, agent
       and quantum circuit diagrams, where output quality *is* the product
   * - ``MODEL_GENERATION_GUI``
     - ``gpt-6-sol``
     - GUI complete-system generation. Its own knob because design quality
       tracks the model's taste far more than diagram generation does.
   * - ``MODEL_GENERATION_SMALL``
     - ``gpt-6-luna``
     - Single-element and modification structured calls,
       ``describe_model`` streaming, and the file-conversion text path
   * - ``MODEL_REASONING``
     - ``gpt-5-mini``
     - The free-text design-reasoning pass of two-pass generation
   * - ``MODEL_VISION``
     - ``gpt-4o``
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

The gpt-5 and gpt-6 families and the o-series reject an explicit
``temperature`` other than the default — the API returns HTTP 400. Call
sites must therefore omit the parameter for those models and pass
``reasoning_effort`` instead. They also take only ``max_completion_tokens``
(``max_tokens`` is a 400). Claude generation 5 and later (Opus 5.5,
Opus 5, Sonnet 5, Fable 5 / 5.1), Opus 4.7 / 4.8 and Mythos reject
``temperature`` / ``top_p`` / ``top_k`` as well; a user's Anthropic key sends
them an ``output_config.effort`` instead.
``model_config`` exposes two helpers that every call site uses:

.. code-block:: python

   supports_custom_temperature(model) -> bool
       # False for gpt-5 and later (gpt-6, ...), o1 / o3 / o4, and the
       # Claude models listed above

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
     - ``reasoning_effort`` passed for gpt-5 / gpt-6 / o-series calls. ``low``
       keeps hidden reasoning small — structured diagram specs do not need
       deep chain-of-thought, and it is a level every such model
       accepts (``minimal`` is rejected by gpt-5.5 and gpt-6, ``none`` by
       gpt-6-astra).


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
       bridge (``src/handlers/validation_handler.py``) and the opt-in
       study telemetry collector (``src/telemetry.py``). Without it, every
       validation in a container fails with connection-refused.
   * - ``BESSER_AGENT_ALLOW_CUSTOM_BASE_URL``
     - unset (off)
     - Whether a BYOK request may specify its own API base URL. Set to
       ``"false"`` on the hosted deployment. Set to ``1`` for a local run that
       uses PIA, Ollama or another OpenAI-compatible endpoint (they arrive as
       ``provider=openai`` plus ``user_api_base``); when off, such a request
       silently falls back to the shared server LLM.
   * - ``BESSER_AGENT_STT_LANGUAGE``
     - auto-detect
     - Pins speech-to-text to a language (``en``, ``fr``, ``de``, …).
       Unset means Whisper auto-detects — a hard pin to English
       mis-transcribed non-English voice.
   * - ``BESSER_AGENT_COMPACT_SPEC``
     - ``1`` (on)
     - Use the compact class-diagram spec schema for generation.
   * - ``BESSER_AGENT_COST_LOG_INTERVAL``
     - ``600``
     - Seconds between the INFO log lines with the token and cost totals
       (see :ref:`cost-and-model-routing`). ``0`` turns them off.
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

Because BYOK bypasses any gateway, the agent's per-call tiers are collapsed
into two and mapped to each provider's equivalent (``_PROVIDER_TIER_MODELS``
in ``src/byok.py``). A model the user explicitly chose is used for every call;
only without one does ``small`` use the provider's cheap sibling, so small
edits, routing and repair calls stay inexpensive on the user's key. The
mapping and its cost effect are in :ref:`cost-and-model-routing`.


.. _cost-and-model-routing:

Cost and Model Routing
----------------------

What a message costs is decided by which call sites it reaches and which
tier each one requests. This section is the reference for keeping that cost
down without losing output quality.

Calls per turn
~~~~~~~~~~~~~~

Counts are for the shared server key and a request that succeeds first time.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Turn type
     - LLM calls
   * - Every user message
     - One unified-classifier call (``MODEL_CLASSIFIER``), cached per BAF
       event so no transition condition repeats it. ``frontend_event``
       messages and ``[auto-fix]`` repair requests skip it.
   * - Greeting, help, "what can you do"
     - Canned replies cost nothing more. Free-form help and fallback replies
       stream one more ``MODEL_CLASSIFIER`` call; describing a model streams
       one ``MODEL_GENERATION_SMALL`` call.
   * - Single element or modification
     - One structured call on ``MODEL_GENERATION_SMALL``. JSON repair or
       self-correction adds a ``MODEL_CLASSIFIER`` call only when the output
       fails to parse or validate.
   * - Complete system: class diagram, state machine, GUI
     - One structured call (``MODEL_GENERATION_LARGE``, or
       ``MODEL_GENERATION_GUI`` for a GUI) below the two-pass gate; above it, a
       ``MODEL_REASONING`` pass first, so two calls.
   * - Complete system: object, agent, quantum circuit
     - One structured call on ``MODEL_GENERATION_LARGE``.
   * - Complete system: BPMN
     - Two calls, both on the ``MODEL_CLASSIFIER`` tier (the handler passes no
       model override). The gate is measured on the enriched prompt, so in
       practice it almost always runs both passes.
   * - Complete system: User Profile
     - One structured call on the ``MODEL_CLASSIFIER`` tier.
   * - Multi-step request
     - One extra ``MODEL_CLASSIFIER`` call for the request planner, only for a
       multi-clause message with several targets or a generation step; then
       one of the rows above per planned operation.
   * - Long conversation
     - One ``MODEL_CLASSIFIER`` summarizer call each time the verbatim window
       fills (16 messages), not every turn.

Tiers
~~~~~

Routing is by call site: each call names the tier it needs (see
:ref:`model routing <model-routing>` above for the overrides). Prices are USD
per million tokens from ``_COST_PER_1K`` in ``src/tracking/token_tracker.py``.

.. list-table::
   :header-rows: 1
   :widths: 24 18 22 36

   * - Tier
     - Default
     - Input / cached / output
     - Why this tier
   * - ``MODEL_CLASSIFIER``
     - ``gpt-4o-mini``
     - 0.15 / 0.075 / 0.60
     - Runs on every message with a small, schema-constrained output, so it
       is the cheapest model that routes reliably.
   * - ``MODEL_GENERATION_LARGE``
     - ``gpt-5-mini``
     - 0.25 / 0.025 / 2.00
     - Complete-system diagrams, where output quality is the product.
   * - ``MODEL_GENERATION_GUI``
     - ``gpt-6-sol``
     - 2.00 / 0.20 / 10.00
     - GUI design quality tracks the model's taste far more than diagram
       generation does.
   * - ``MODEL_GENERATION_SMALL``
     - ``gpt-6-luna``
     - 0.10 / 0.01 / 0.50
     - Single-element and modification calls: latency-sensitive and
       schema-constrained.
   * - ``MODEL_REASONING``
     - ``gpt-5-mini``
     - 0.25 / 0.025 / 2.00
     - The free-text design analysis of two-pass generation.
   * - ``MODEL_VISION``
     - ``gpt-4o``
     - 2.50 / 1.25 / 10.00
     - Image and PDF input for file conversion.
   * - ``MODEL_EMBEDDINGS``
     - ``text-embedding-3-small``
     - 0.02 / — / —
     - RAG vectors; pinned so the persisted store keeps matching.

The ``gpt-6-sol`` and ``gpt-6-luna`` cached rates are not in the price file
the table is sourced from; they assume the 10 % that ``gpt-6-astra`` lists.

Static prefix first
~~~~~~~~~~~~~~~~~~~

Both providers bill a repeated prompt prefix at the cached rate (OpenAI
caches automatically from 1024 prompt tokens; Anthropic needs a
``cache_control`` marker and caches from 1024 tokens on Sonnet 5 and from 4096
on Haiku 4.5). A cache hit needs the prefix to be byte-identical, so:

- put the static instructions (system prompt, JSON schema) first and the
  per-message content (history, workspace context, the user's message) last;
- never interpolate a timestamp, id or other per-request value into a system
  prompt.

The unified classifier follows this rule: its ~9k-token system prompt is a
separate system message ahead of the per-message user block. On a user's
Anthropic key, ``LLMProvider.parse`` sends that prompt plus the schema as a
``system`` block with ``cache_control: {"type": "ephemeral"}``; the first
message in a five-minute window writes it at 1.25x the input rate and later
ones read it at 0.1x.

Output caps, reasoning effort and the two-pass gate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Output caps** (``src/agent_config.py``): ``LLM_MAX_TOKENS_LARGE`` 8192 for
  complete-system schemas, ``LLM_MAX_TOKENS_SMALL`` 2048 for single-element and
  modification schemas, ``LLM_MAX_TOKENS_TEXT`` 4096 for free text, and
  ``GUI_COMPLETE_SYSTEM_MAX_TOKENS`` 16384 for multi-page GUI output. The
  classifier gets 800 tokens on a non-reasoning model and 4000 on a reasoning
  model, whose hidden reasoning draws on the same budget. A structured call
  that hits its cap is retried once with an instruction to be concise; a
  second truncation fails the call instead of falling back to per-element
  generation.
- **Reasoning effort**: ``BESSER_AGENT_MODEL_REASONING_EFFORT`` (``low``) is
  sent as ``reasoning_effort`` to gpt-5 / gpt-6 / o-series models and as
  ``output_config.effort`` to Claude models that take no sampling
  parameters. Hidden reasoning is billed as output, so raising it raises the
  cost of every generation call.
- **Two-pass gate**: the reasoning pass runs only when the raw user message
  is at least ``_TWO_PASS_MIN_LENGTH`` (250) characters. The handler must pass
  ``raw_request`` to ``predict_two_pass_structured``; without it the gate
  measures the enriched prompt (history plus workspace context), which is
  almost always longer, so the extra pass runs on trivial requests too. The
  class diagram and state machine handlers pass it; the BPMN and GUI
  handlers do not.

Retries
~~~~~~~

- **Shared server clients**: ``src/utilities/llm_retry.py`` is the only
  transport retry. The SDK's own retries are turned off (``max_retries=0``) so
  they do not nest under it. A 429 or 5xx is tried up to ``MAX_ATTEMPTS`` (4)
  times with 0.6 / 1.2 / 2.4 s backoff (about 5 s of sleep in the worst case,
  on top of the failed attempts' own duration). ``insufficient_quota``,
  ``invalid_api_key`` and other permanent codes fail on the first attempt.
- **Handler retries**: ``predict_with_retry`` and ``predict_structured`` make
  at most one more attempt, never for a rate-limit, authentication or
  bad-request error. A 5xx that outlasts the transport retry can therefore
  cost up to 8 HTTP requests for one logical call; it was 24 when the SDK
  retries were nested in.
- **No fan-out on provider errors**: a rate-limit, authentication,
  bad-request or repeated-truncation error reaches the user (for a rate limit
  on the shared key, the reply offers adding an own API key). It does not
  trigger the two-pass single-pass fallback or the class diagram's per-class
  fallback, which used to repeat the failing call up to 11 more times.
- **BYOK clients** are not wrapped: the SDK default of 2 retries applies, so a
  5xx costs at most 3 HTTP requests per attempt and 6 per logical call.

BYOK cost behaviour
~~~~~~~~~~~~~~~~~~~

With a user's key the user pays for every routed call. Without a chosen model
the tiers map as follows; a chosen model is used for every call.

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Provider
     - ``large``: GENERATION_LARGE, GENERATION_GUI, REASONING, VISION
     - ``small``: GENERATION_SMALL, CLASSIFIER
   * - ``openai``
     - ``gpt-5.5``
     - ``gpt-4o-mini``
   * - ``anthropic``
     - ``claude-sonnet-5``
     - ``claude-haiku-4-5``
   * - ``mistral``
     - ``mistral-large-latest``
     - ``mistral-small-latest``
   * - ``nebius``
     - ``Qwen/Qwen3-30B-A3B-Instruct-2507``
     - same model

The tier is read from the model the call site requests, not from whether
that model is a reasoning model. A small edit on an OpenAI key therefore runs
on ``gpt-4o-mini`` ($0.15 / $0.60) rather than ``gpt-5.5`` ($5 / $30), and on
an Anthropic key on ``claude-haiku-4-5`` rather than ``claude-sonnet-5``. BPMN
and User Profile complete-system generation request the classifier tier, so
they map to ``small``.

Where to see spend
~~~~~~~~~~~~~~~~~~

``src/tracking/token_tracker.py`` accumulates the tokens and estimated cost
of every recorded call for the whole process since it started. Every
``BESSER_AGENT_COST_LOG_INTERVAL`` seconds (600 by default) the next call logs
one INFO line, for example::

   [TokenTracker] totals since start: calls=412 prompt=1893120 (cached=1204480) completion=96210 est_cost=$0.8123

Real provider usage is recorded wherever the SDK returns it, cached prompt
tokens are priced at the cached rate, and a truncated call is counted. Only
BAF's own ``predict()``, which returns no usage, is estimated from text
length. The tracker supports per-session buckets, but no call site passes a
session id, so only the global totals are populated. The figures are
estimates: the provider's billing dashboard is authoritative.

Keeping ``_COST_PER_1K`` current
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a ``MODEL_*`` default changes, or a model joins the BYOK picker, add an
entry with ``prompt``, ``completion`` and ``cached`` rates per 1K tokens (the
per-million list price divided by 1000), taken from the price file BESSER
vendors in ``spec_driven_agent/providers/data/model_prices.json``. Dated and
gateway ids (``claude-haiku-4-5-20251001``, ``us.anthropic.claude-sonnet-5``)
are priced as their base entry; a different variant such as ``gpt-5.5-pro``
needs its own. A model with no entry is priced with placeholder rates and
logged once as a warning.


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
     - ``64_000``
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


Research Study Mode
-------------------

**Location:** ``src/telemetry.py``, with the label parsed in
``src/protocol/adapters.py``.

Study mode is an opt-in usage recording for facilitated research sessions. It
is off unless the editor tab was opened with a study link
(``?study=<label>``; ``?pilot=<label>`` is still accepted for links already
handed out). The editor then adds ``context.pilotParticipant`` to every
message. Without that field the agent records nothing.

For a tagged message, the agent posts one ``prompt`` event to
``{BESSER_BACKEND_URL}/besser_api/telemetry/event`` containing:

- the per-tab session id and the participant label (a short token such as
  ``P3``, validated against ``^[A-Za-z0-9_-]{1,16}$``, never a name or email);
- the message text, truncated to 2000 characters;
- what the agent did with it (the reply action, e.g. ``assistant_message``)
  and the active diagram type.

The post runs on a short-timeout background thread and every failure is
swallowed, so a reply is never delayed or broken by it. The agent keeps no
copy. The BESSER backend stores the event only when its own switch
(``BESSER_TELEMETRY_ENABLED``) is on; storage and retention are described in
the BESSER backend documentation. A participant stops the recording by
closing the tab.


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
