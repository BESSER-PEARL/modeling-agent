Run the Modeling Agent locally
==============================

**Goal:** start a WebSocket service the editor can connect to. You need
Python **3.11**, Git, and a provider key that can access the configured models.
The Docker image uses Python 3.11. Documentation builds use Python 3.12.

Clone and install
-----------------

.. code-block:: console

   git clone --branch develop https://github.com/BESSER-PEARL/modeling-agent.git
   cd modeling-agent
   python -m venv .venv

Activate the environment for your shell:

.. code-block:: powershell

   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1

.. code-block:: bash

   # Linux or macOS
   source .venv/bin/activate

.. code-block:: console

   python -m pip install -r requirements.txt

Copy ``config_example.yaml`` to ``config.yaml``. Set ``nlp.openai.api_key``
to your server provider key and allow your editor origin under
``platforms.websocket.origins``. Keep ``config.yaml`` out of version control.

See :doc:`configuration` to change model tiers, use a compatible gateway,
or configure RAG. Default model names must be accessible to your provider.

Start and verify
----------------

.. code-block:: console

   python modeling_agent.py

Wait for startup to finish. The framework prepares its local classifiers
before opening the socket; this can take several minutes. The default socket
address is ``ws://localhost:8765``.

From another terminal, check that the port opens:

.. code-block:: console

   python -c "import socket; s = socket.create_connection(('localhost', 8765), timeout=5); print('Agent port is open'); s.close()"

This checks listening status. To check actual responses, connect the editor
or a client following :doc:`websocket_protocol` and ask it to create a class.

Connect the editor
------------------

Configure the editor's ``UML_BOT_WS_URL`` as ``ws://localhost:8765`` and allow
the editor's origin, normally ``http://localhost:8080``, in ``config.yaml``.
The editor and BESSER backend run separately. Use the
`editor setup guide <https://besser.readthedocs.io/projects/besser-web-modeling-editor/en/latest/overview/getting-started.html>`_
for those services.

For a deployment behind TLS and nginx, use :doc:`deployment`. It also
documents the vendored WebSocket patch applied by the Docker image.

Next steps
----------

* :doc:`end_to_end_flow`: follow a request through the service.
* :doc:`websocket_protocol`: request and response shapes.
* :doc:`contributing/dev_setup`: development setup, including the framework patch.
* :doc:`troubleshooting`: startup, connection, and provider problems.
