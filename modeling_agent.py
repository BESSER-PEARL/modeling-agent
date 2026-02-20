# Intelligent UML Modeling Assistant – slim entrypoint
# ---------------------------------------------------
# All business logic lives under ``src/``.  This file only:
#   1. Puts ``src/`` on sys.path
#   2. Creates the BESSER Agent and WebSocket platform
#   3. Initialises LLMs / RAG / DiagramHandlerFactory via ``agent_setup``
#   4. Populates the shared ``agent_context`` module
#   5. Declares states & intents
#   6. Wires state bodies and transitions via ``state_bodies.register_all``
#   7. Runs the agent

import logging
import os
import sys

# ── Make ``src/`` importable for bare-style imports ──────────────────────
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from besser.agent.core.agent import Agent
from besser.agent import nlp
from besser.agent.exceptions.logger import logger

import agent_context as ctx
from agent_setup import (
    init_llm,
    init_rag,
    init_diagram_factory,
    init_intent_classifier_config,
)
from routing.intents import GENERATION_INTENT_NAME
from state_bodies import register_all

# ── Logging ──────────────────────────────────────────────────────────────
logger.setLevel(logging.INFO)

# ── Disable Chroma telemetry ─────────────────────────────────────────────
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "False")

# ── Agent ────────────────────────────────────────────────────────────────
agent = Agent("uml_modeling_agent")
agent.load_properties("config.ini")
logger.info(f"Agent properties loaded from config.ini (name={agent.name})")

websocket_platform = agent.use_websocket_platform(use_ui=False)

# ── LLMs / RAG / Handlers ───────────────────────────────────────────────
gpt, gpt_text, gpt_predict_json = init_llm(agent)
uml_rag = init_rag(agent)
diagram_factory = init_diagram_factory(gpt)

ic_config = init_intent_classifier_config()
agent.set_default_ic_config(ic_config)

# ── Populate shared context ──────────────────────────────────────────────
ctx.agent = agent
ctx.gpt = gpt
ctx.gpt_text = gpt_text
ctx.gpt_predict_json = gpt_predict_json
ctx.uml_rag = uml_rag
ctx.diagram_factory = diagram_factory
ctx.openai_api_key = agent.get_property(nlp.OPENAI_API_KEY)

# ── States ───────────────────────────────────────────────────────────────
greetings_state = agent.new_state("greetings_state", initial=True)
create_single_element_state = agent.new_state("create_single_element_state")
create_complete_system_state = agent.new_state("create_complete_system_state")
modify_model_state = agent.new_state("modify_model_state")
modeling_help_state = agent.new_state("modeling_help_state")
uml_rag_state = agent.new_state("uml_rag_state")
generation_state = agent.new_state("generation_state")

# ── Intents ──────────────────────────────────────────────────────────────
hello_intent = agent.new_intent(
    name="hello_intent",
    description="The user greets you or wants to start a conversation",
)
create_single_element_intent = agent.new_intent(
    name="create_single_element_intent",
    description=(
        "The user wants to create exactly ONE single UML element. "
        'Examples: "create a class called User", "add a Person class", '
        '"make one state", "create an object instance". '
        "This is for creating ONE element only, NOT multiple elements or systems."
    ),
)
create_complete_system_intent = agent.new_intent(
    name="create_complete_system_intent",
    description=(
        "The user wants to create a complete system, diagram, or multiple "
        'classes/elements. Keywords: "create a library system", "create a '
        'class diagram for", "design an e-commerce", "generate a banking '
        'application", "build a system", "create a diagram for", "model a", '
        '"create classes for", "generate the gui", "create the gui", '
        '"generate the frontend", "create a frontend", "gui diagram", '
        '"generate a gui diagram", "build the frontend". '
        "This is for creating MULTIPLE elements, a complete model, or a "
        "GUI / frontend diagram — NOT for generating source code artifacts."
    ),
)
modify_model_intent = agent.new_intent(
    name="modify_model_intent",
    description=(
        "The user wants to modify, change, update, edit, add to, or connect "
        'elements in an EXISTING UML model. Keywords: "add relationship", '
        '"connect", "add inheritance", "add generalization", "relate", '
        '"modify class", "change attribute", "update method", "delete", '
        '"remove", "rename", "add association", "add composition", '
        '"add aggregation", "link classes"'
    ),
)
modeling_help_intent = agent.new_intent(
    name="modeling_help_intent",
    description="The user asks for help with UML modeling, design patterns, or modeling concepts",
)
uml_spec_intent = agent.new_intent(
    name="uml_spec_intent",
    description=(
        "The user asks theoretical questions about the official UML "
        "specification document, UML standards, or formal UML definitions. "
        'Keywords: "according to UML specification", "what does UML standard '
        'say", "UML 2.5 specification", "OMG specification", "formal UML '
        'definition". This is NOT for creating diagrams, only for asking '
        "about the UML specification document itself."
    ),
)
generation_intent = agent.new_intent(
    name=GENERATION_INTENT_NAME,
    description=(
        "The user wants to generate deployable source code or technical "
        "artifacts from an existing model. Generators include: django, "
        "backend, web_app, sql, sqlalchemy, jsonschema, qiskit, python, "
        "java, pydantic, agent. This is strictly for CODE GENERATION, "
        "NOT for creating or generating diagrams, models, GUIs, or "
        "frontends — those belong to the modeling/creation intents."
    ),
)

# ── Wire state bodies & transitions ─────────────────────────────────────
register_all(
    agent=agent,
    states={
        "greetings": greetings_state,
        "create_single_element": create_single_element_state,
        "create_complete_system": create_complete_system_state,
        "modify_model": modify_model_state,
        "modeling_help": modeling_help_state,
        "uml_rag": uml_rag_state,
        "generation": generation_state,
    },
    intents={
        "hello": hello_intent,
        "create_single_element": create_single_element_intent,
        "create_complete_system": create_complete_system_intent,
        "modify_model": modify_model_intent,
        "modeling_help": modeling_help_intent,
        "uml_spec": uml_spec_intent,
        "generation": generation_intent,
    },
)


# ── Backward-compatible re-export for tests ──────────────────────────────
# ``test_model_helpers.py`` imports ``_resolve_class_diagram`` from here.
from utilities.model_resolution import resolve_class_diagram as _resolve_class_diagram  # noqa: E402, F401


# ── Run ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent.run()
