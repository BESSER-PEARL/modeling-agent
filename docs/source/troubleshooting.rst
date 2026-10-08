Troubleshooting
===============

The socket is not listening yet
-------------------------------

Wait for classifier preparation to finish and read the startup log. Training
can take several minutes. If startup exits, check the first exception and the
provider configuration. A successful TCP check establishes only that the port
is open; use a real request to verify the complete service.

The browser cannot connect
---------------------------

Check the editor's ``UML_BOT_WS_URL`` and the agent's host and port. Allow the
browser's exact origin in ``platforms.websocket.origins``. For a hosted editor,
use ``wss://`` and ensure the reverse proxy forwards WebSocket upgrade headers.
See :doc:`deployment`.

The provider rejects a request
------------------------------

Check the error's status, the provider key, and model access. Use
``BESSER_AGENT_MODEL_*`` overrides to select model names your provider exposes.
A modeling assistant provider and a Spec-Driven generation provider are
configured separately; changing one does not necessarily change the other.
See :doc:`configuration`.

Replies disappear after reconnecting
------------------------------------

The Docker image applies ``patches/websocket_platform.py`` to its pinned
framework. A source setup needs the same patch; see
:doc:`contributing/dev_setup`. Clients should send a stable session id and
use the documented ``turnId`` / ``replySeq`` replay protocol.

Report a problem
----------------

Open an `issue <https://github.com/BESSER-PEARL/modeling-agent/issues>`_ with
the request that reproduces it, expected result, actual response, and relevant
log excerpt. Include the agent commit, provider, and model names. Remove
credentials and private model data before attaching anything.
