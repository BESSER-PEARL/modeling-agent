Modeling Agent
==============

The conversational backend of the BESSER Web Modeling Editor. It turns a
message into diagram operations that the editor applies to your project.

.. container:: doc-lead

   **Here to use the assistant?** Open the
   `editor <https://editor.besser-pearl.org>`_ and follow its
   `assistant guide <https://besser.readthedocs.io/projects/besser-web-modeling-editor/en/latest/user-guide/ai-assistant.html>`_.
   These pages explain how to run, integrate, and extend the service.

Choose your starting point
--------------------------

.. container:: doc-path

   .. rubric:: 01 / Run the service

   Install the agent, configure a provider, and check that its socket opens.

   :doc:`Local setup <getting_started>`

.. container:: doc-path

   .. rubric:: 02 / Integrate a client

   Send a request and handle diagram actions and reconnects.

   :doc:`WebSocket protocol <websocket_protocol>`

.. container:: doc-path

   .. rubric:: 03 / Extend the agent

   Add a diagram handler or change routing, then validate the contract.

   :doc:`Contributor guide <contributing>`

How it fits together
--------------------

The editor displays models. This service creates and modifies them from
natural language. The BESSER backend generates application code; the
Spec-Driven Agent customises that code when requested. Starting this service
alone does not start the editor or a code-generation worker.

Related documentation
---------------------

This site is one of three BESSER documentation sites. The other two are:

* `BESSER docs <https://besser.readthedocs.io/en/latest/>`__: the B-UML modeling
  language, the Python library, and the code generators.
* `Web Modeling Editor docs
  <https://besser.readthedocs.io/projects/besser-web-modeling-editor/en/latest/>`__:
  using the browser editor, from projects and diagrams to the AI assistant and
  code generation.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Start and operate

   getting_started
   configuration
   deployment
   troubleshooting

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Understand and integrate

   end_to_end_flow
   architecture
   intent_recognition
   orchestration
   websocket_protocol
   schema
   diagram_handlers
   usage
   api
   glossary

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Maintain

   contributing
   releases

.. toctree::
   :hidden:
   :caption: Related documentation

   BESSER docs <https://besser.readthedocs.io/en/latest/>
   Web Modeling Editor docs <https://besser.readthedocs.io/projects/besser-web-modeling-editor/en/latest/>
