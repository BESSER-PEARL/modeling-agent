Deployment
==========

This document covers local development setup, Docker deployment, and production
architecture.

.. contents:: On this page
   :local:
   :depth: 2

Requirements
------------

- Python 3.11 (3.10 minimum)
- An OpenAI API key
- ``besser-agentic-framework[extras,llms,tensorflow] == 4.3.2`` — the version
  is pinned because ``patches/websocket_platform.py`` is vendored over it
  (see `Vendored BAF patch`_)

Local Development
-----------------

.. code-block:: bash

   # Create virtual environment
   python -m venv venv

   # Activate (Linux/macOS)
   source venv/bin/activate

   # Activate (Windows PowerShell)
   .\\venv\\Scripts\\Activate.ps1

   # Install dependencies
   pip install --upgrade pip
   pip install -r requirements.txt

   # Configure
   cp config_example.yaml config.yaml
   # Edit config.yaml with your OpenAI API key

   # Run
   python modeling_agent.py

The agent listens on ``ws://0.0.0.0:8765`` by default.

Docker
------

.. code-block:: bash

   # Build
   docker build -t modeling-agent .

   # Run
   docker run -p 8765:8765 --env-file .env modeling-agent

The image is ``python:3.11-slim``. Its entrypoint **generates
``config.yaml`` from the environment** at container start, so no config file
needs to be baked in or mounted. It prints the generated config for
debugging but redacts the ``api_key`` line, so the key never reaches the
container logs.

Minimum ``.env``:

.. code-block:: text

   OPENAI_API_KEY=sk-proj-...

Anything from the environment-variable table in :doc:`configuration` can be
added — most usefully ``BESSER_BACKEND_URL`` (without it, every diagram
validation inside a container fails with connection-refused) and the
``BESSER_AGENT_MODEL_*`` tier overrides.

Vendored BAF patch
~~~~~~~~~~~~~~~~~~

The Dockerfile copies ``patches/websocket_platform.py`` over the
pip-installed framework file:

.. code-block:: dockerfile

   COPY patches/websocket_platform.py \
       /usr/local/lib/python3.11/site-packages/baf/platforms/websocket/websocket_platform.py

Stock BAF 4.3.2 deletes the ``_connections`` slot unconditionally when a
WebSocket closes. Because the frontend appends a stable ``?user_id=``
parameter (so conversation memory survives reconnects), the two sockets a
browser tab opens — the assistant widget and the workspace drawer — share one
session key, and a closing socket would evict the live one, silently dropping
the agent's replies. The vendored file adds an ownership-guarded delete plus
reply-route-to-sender, and is also where the per-request BYOK context var is
set and reset.

**Re-vendor this file if the pinned BAF version changes.**

Production Architecture
-----------------------

The agent is **one of three services deployed together**. They share a Docker
network and sit behind a single nginx reverse proxy that terminates TLS:

.. code-block:: text

   nginx (:443, TLS)
     ├── /                → besser-wme-frontend   (:8080)  React/TypeScript SPA
     ├── /besser_api      → besser-wme-backend    (:9000)  BESSER backend + generators
     └── /agent           → besser-wme-modeling-agent (:8765)  this service (WebSocket)

Each container binds only to ``127.0.0.1`` on the host; nginx is the only
public listener.

Compose service
~~~~~~~~~~~~~~~

The agent's service definition, with the hardening the production stack uses:

.. code-block:: yaml

   besser-wme-modeling-agent:
     image: <registry>/modeling_agent:<tag>
     container_name: besser-wme-modeling-agent
     init: true
     env_file:
       - ./.env.modeling-agent
     environment:
       PYTHONUNBUFFERED: "1"
       PYTHONDONTWRITEBYTECODE: "1"
       BESSER_BACKEND_URL: http://besser-wme-backend:9000
       BESSER_AGENT_ALLOW_CUSTOM_BASE_URL: "false"
       LOG_PROMPTS: "false"
     ports:
       - "127.0.0.1:8765:8765"
     restart: unless-stopped
     stop_grace_period: 45s
     security_opt:
       - no-new-privileges:true
     cap_drop:
       - ALL
     pids_limit: 256
     cpus: 1.5
     mem_limit: 1536m
     networks:
       - besser_network

Note two deliberate settings:

- ``BESSER_BACKEND_URL`` points at the backend **service name** on the shared
  Docker network, not at localhost.
- ``BESSER_AGENT_ALLOW_CUSTOM_BASE_URL: "false"`` closes the SSRF surface of
  a BYOK request supplying its own API base URL. Enable it only where a
  private gateway or a local model endpoint genuinely needs to be reachable.

Nginx configuration
~~~~~~~~~~~~~~~~~~~

The WebSocket location needs the upgrade headers and a long read timeout —
a complete-system generation can take a while, and a short timeout drops the
connection mid-flight:

.. code-block:: nginx

   # Internal subrequest used to verify the session before the upgrade.
   location = /_besser_agent_auth {
       internal;
       proxy_pass http://127.0.0.1:9000/besser_api/github/auth/verify;
       proxy_pass_request_body off;
   }

   location = /agent {
       auth_request /_besser_agent_auth;
       proxy_pass http://127.0.0.1:8765/;
       proxy_http_version 1.1;
       proxy_set_header Upgrade $http_upgrade;
       proxy_set_header Connection "upgrade";
       proxy_set_header Host $host;
       proxy_read_timeout 86400;
   }

The ``auth_request`` gate lets nginx verify the user's session before
permitting the upgrade; the agent still validates message semantics itself.

CORS is enforced independently by the agent: ``platforms.websocket.origins``
in ``config.yaml`` whitelists the browser origins allowed to open a socket.

The Docker entrypoint writes the key, so a container is restricted by default.
Set the two allowed host origins per deployment:

.. code-block:: bash

   BESSER_AGENT_WS_ORIGIN=https://editor.besser-pearl.org
   BESSER_AGENT_WS_ORIGIN_ALT=https://<your-second-host>

The generated file also whitelists ``http://localhost`` on ports 8080, 5173 and
3000 for local development.

.. warning::

   If the ``origins`` key is absent from ``config.yaml``, BAF accepts a
   WebSocket from **any** origin — it is not a deny-by-default setting. If you
   mount your own ``config.yaml`` over the generated one, carry the key across.

Health Monitoring
-----------------

The image's healthcheck simply opens a TCP connection to the WebSocket port:

.. code-block:: dockerfile

   HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
       CMD python -c "import socket; s=socket.socket(); s.connect(('localhost', 8765)); s.close()" || exit 1

.. warning::

   The ``start-period`` must cover the **full** boot, not just process start.
   BAF trains a NER model plus one intent classifier per state before opening
   the socket; container start to listening socket takes about **3.5 minutes**.
   A shorter start-period marks a healthy container ``unhealthy`` during boot,
   which makes a real failure easy to dismiss as the slow boot.

- **WebSocket keep-alive:** handled by the BESSER framework.
- **Logs:** the agent logs to stdout. Use Docker log drivers (the production
  stack caps at ``max-size: 10m``, ``max-file: 3``) or ``journalctl`` under
  systemd.
- **Docker logs:** ``docker logs besser-wme-modeling-agent``.

Systemd Service
~~~~~~~~~~~~~~~

For a non-containerized install:

.. code-block:: ini

   [Unit]
   Description=Modeling Agent
   After=network.target

   [Service]
   Type=simple
   User=besser
   WorkingDirectory=/opt/modeling-agent
   ExecStart=/opt/modeling-agent/venv/bin/python modeling_agent.py
   Restart=on-failure
   RestartSec=5
   EnvironmentFile=/etc/modeling-agent/env

   [Install]
   WantedBy=multi-user.target

Scaling Considerations
----------------------

- **Rate limiting:** handled by the provider API. The shared LLM client is
  patched once at its network-call layer (``src/utilities/llm_retry.py``) to
  retry 429 and 5xx with bounded backoff — ``MAX_ATTEMPTS = 4``, ~5 s of
  worst-case added latency. Non-429 4xx responses fail fast rather than
  burning the backoff budget, because under concurrency a pile-up of workers
  each parked on a doomed retry is what wedges the service.
- **Per-message LLM cost:** routing costs exactly one classifier-tier call per
  message (``get_or_classify`` caches on the BAF event id), not one per
  transition condition.
- **Request parsing:** parsed requests are cached per-event via
  ``id(session.event)``, avoiding 3–5 redundant JSON parses per message.
- **Session reaping:** BAF keeps sessions (and their event-loop threads) alive
  after disconnect so a reconnect can resume. Over days of uptime those
  orphaned threads accumulate until the process hits the OS thread limit
  (``RuntimeError: can't start new thread``). A reaper thread started by
  ``modeling_agent.py`` runs every 10 minutes and closes any session with no
  tracked connection that has been disconnected for longer than
  ``GRACE_PERIOD_SECONDS`` (300 s). It also reaps conversation memories older
  than an hour.
- **Session continuity:** conversation memory is keyed on the v2 payload's
  ``sessionId``, which survives reconnects — see :doc:`architecture`.

Security
--------

- Never commit API keys to the repository. ``config.yaml`` is gitignored, and
  the Docker entrypoint redacts the key from its own debug output.
- Use environment variables or secrets management in production.
- Inside Docker the agent binds ``0.0.0.0``, but the published port is bound
  to ``127.0.0.1`` on the host so only nginx can reach it. Outside Docker,
  restrict with a firewall or bind ``127.0.0.1`` directly.
- WebSocket connections are unencrypted by default. Terminate TLS at nginx
  and serve the agent as ``wss://<host>/agent``.
- Set ``platforms.websocket.origins`` to the exact browser origins allowed to
  connect. When unset, **any** origin is accepted.
- Keep ``BESSER_AGENT_ALLOW_CUSTOM_BASE_URL`` off on a hosted deployment: it
  gates whether a user-supplied BYOK config may redirect calls to an arbitrary
  base URL.
- Leave ``LOG_PROMPTS`` off outside local debugging — it logs full prompts,
  which contain the user's model and message content.
- Run with ``no-new-privileges``, ``cap_drop: ALL``, and pid/cpu/memory limits
  as shown in the compose snippet above.
