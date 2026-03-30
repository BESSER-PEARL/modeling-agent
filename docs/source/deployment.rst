Deployment
==========

This document covers local development setup, Docker deployment, and production
architecture.

.. contents:: On this page
   :local:
   :depth: 2

Requirements
------------

- Python 3.11+ (3.10 minimum)
- OpenAI API key with GPT-4.1-mini access
- BESSER Agentic Framework (``besser-agentic-framework[extras,llms,tensorflow]``)

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

Environment variables in ``.env``:

.. code-block:: text

   OPENAI_API_KEY=sk-proj-...

Production Architecture
-----------------------

The typical production setup uses nginx as a reverse proxy:

.. code-block:: text

   nginx (:8080)
     ├── /           → static frontend files (React SPA)
     ├── /besser_api → Python backend (:9000)
     └── /agent      → WebSocket proxy → modeling agent (:8765)

Nginx Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: nginx

   server {
       listen 8080;

       location / {
           root /var/www/besser-editor/build;
           try_files $uri $uri/ /index.html;
       }

       location /besser_api {
           proxy_pass http://127.0.0.1:9000;
           proxy_set_header Host $host;
       }

       location /agent {
           proxy_pass http://127.0.0.1:8765;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_read_timeout 86400;
       }
   }

Systemd Service
~~~~~~~~~~~~~~~

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

Health Monitoring
-----------------

- **WebSocket ping:** The BESSER framework handles WebSocket keep-alive.
- **Logs:** The agent logs to stdout. Use ``journalctl`` with systemd or Docker
  log drivers for persistence.
- **Docker logs:** ``docker logs modeling-agent`` for troubleshooting.

Scaling Considerations
----------------------

- **Rate limiting:** Handled by OpenAI's API directly. The retry loop (exponential backoff with jitter, 3 attempts) catches 429 responses and surfaces them to users.
- **Request parsing:** Parsed requests are cached per-event via ``id(session.event)`` to avoid redundant JSON parsing within a single message cycle.
- **Session persistence:** BESSER sessions persist across WebSocket reconnects (``delete_session`` is not called on disconnect). The frontend persists the session ID in ``sessionStorage`` for continuity across drawer open/close.

Security
--------

- Never commit API keys to the repository.
- Use environment variables or secrets management in production.
- The agent binds to ``0.0.0.0`` by default — restrict with a firewall or
  change to ``127.0.0.1`` when behind nginx.
- WebSocket connections are unencrypted by default. Use nginx with TLS
  termination for production.
