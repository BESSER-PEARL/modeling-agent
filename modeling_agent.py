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
import threading

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
    init_stt,
    init_rag,
    init_diagram_factory,
    init_intent_classifier_config,
)
from routing.intents import GENERATION_INTENT_NAME
from state_bodies import register_all
from memory.conversation_memory import cleanup_stale_memories

# ── Logging ──────────────────────────────────────────────────────────────
logger.setLevel(logging.INFO)

# ── Disable Chroma telemetry ─────────────────────────────────────────────
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "False")

# ── Agent ────────────────────────────────────────────────────────────────
agent = Agent("uml_modeling_agent")
agent.load_properties("config.yaml")
logger.info(f"Agent properties loaded from config.yaml (name={agent.name})")

websocket_platform = agent.use_websocket_platform(use_ui=False)

# ── LLMs / RAG / Handlers ───────────────────────────────────────────────
gpt, gpt_text, gpt_predict_json = init_llm(agent)
stt = init_stt(agent)
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
describe_model_state = agent.new_state("describe_model_state")
uml_rag_state = agent.new_state("uml_rag_state")
generation_state = agent.new_state("generation_state")
workflow_state = agent.new_state("workflow_state")

# ── Intents ──────────────────────────────────────────────────────────────
hello_intent = agent.new_intent(
    name="hello_intent",
    description="The user greets you or wants to start a conversation",
)
create_single_element_intent = agent.new_intent(
    name="create_single_element_intent",
    description=(
        "The user wants to create exactly ONE single element in a diagram. "
        'Examples: "create a class called User", "add a Person class", '
        '"make one state", "create an object instance", '
        '"add a Hadamard gate to qubit 0", "put an X gate on qubit 2", '
        '"add a CNOT gate". '
        "This is for creating ONE element only, NOT multiple elements, "
        "systems, or complete algorithms."
    ),
)
create_complete_system_intent = agent.new_intent(
    name="create_complete_system_intent",
    description=(
        "The user wants to CREATE or BUILD a complete system, diagram, "
        "algorithm, or multiple elements from scratch. "
        'Keywords: "create a library system", "create a class diagram for", '
        '"design an e-commerce", "build a system", "create a diagram for", '
        '"model a", "create classes for", "generate the gui", "create the gui", '
        '"generate the frontend", "build the frontend", '
        '"create a web app for", "model a web application for", '
        '"design a web app for", "build a web app for". '
        "Quantum keywords: \"create Grover's algorithm\", "
        '"build a quantum circuit", "implement Shor algorithm", '
        '"make a Bell state", "create QFT circuit", '
        '"do the Deutsch-Jozsa algorithm", "build a teleportation circuit", '
        '"implement quantum fourier transform", "create a quantum algorithm", '
        '"do grovers search", "implement bernstein vazirani", '
        '"build a variational circuit", "create an entanglement circuit". '
        "This is for creating MULTIPLE elements, a complete model, a "
        "GUI / frontend diagram, a web application system, or a quantum "
        "algorithm — NOT for generating source code artifacts or "
        "describing/explaining existing models."
    ),
)
modify_model_intent = agent.new_intent(
    name="modify_model_intent",
    description=(
        "The user wants to modify, change, update, edit, add to, remove from, "
        "or connect elements in an EXISTING model or diagram. "
        'Keywords: "add relationship", "connect", "add inheritance", '
        '"modify class", "change attribute", "update method", "delete", '
        '"remove", "rename", "add association", "link classes". '
        'Quantum modifications: "add a gate to the circuit", '
        '"remove the measurement", "change gate on qubit 2", '
        '"add more qubits", "extend the circuit", '
        '"replace the X gate with a Y gate", "add a CNOT between q0 and q3".'
    ),
)
modeling_help_intent = agent.new_intent(
    name="modeling_help_intent",
    description=(
        "The user asks for GENERAL HELP, explanation, or guidance about "
        "modeling concepts, design patterns, or how things work. "
        'Keywords: "help me", "how do I", "explain", "what is", '
        '"tell me about", "how does X work", "what are best practices". '
        'Quantum help: "explain quantum gates", "how does superposition work", '
        '"what is entanglement", "explain Grover\'s algorithm concept", '
        '"how do quantum circuits work", "what is a Hadamard gate". '
        "This is for CONCEPTUAL help and explanations, NOT for creating, "
        "modifying, or describing existing models."
    ),
)
describe_model_intent = agent.new_intent(
    name="describe_model_intent",
    description=(
        "The user asks a QUESTION about their CURRENT model, diagram, or "
        "circuit, wanting to inspect, understand, or get information about "
        "what already EXISTS on the canvas. "
        "Keywords: 'how many classes', 'what attributes', 'describe my diagram', "
        "'list all classes', 'show my model', 'what relationships', "
        "'tell me about my model', 'summarize my model', 'what states do I have', "
        "'what is in my diagram', 'explain my model'. "
        "Quantum: 'what gates are in my circuit', 'describe my quantum circuit', "
        "'what algorithm does my circuit implement', 'how many qubits', "
        "'describe the quantum algorithm', 'what does this circuit do', "
        "'analyze my circuit'. "
        "This is for ASKING about an existing model — NOT for creating, "
        "building, modifying, or generating anything new."
    ),
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
        "artifacts FROM AN EXISTING model, or export/deploy their project. "
        "This requires an existing model to already be on the canvas. "
        "Generators include: django, backend, web_app, sql, sqlalchemy, "
        "jsonschema, qiskit, python, java, pydantic, agent. "
        'Keywords: "generate django", "generate sql", "generate web_app", '
        '"generate python code", "run the web_app generator", '
        '"generate code from my model". '
        "Export: 'export to json', 'export buml', 'download project'. "
        "Deploy: 'deploy to render', 'publish app', 'deploy application'. "
        "This is strictly for CODE GENERATION from existing models, EXPORT, "
        "or DEPLOYMENT — NOT for creating, modeling, designing, or building "
        "new diagrams, systems, models, GUIs, or frontends. "
        "If the user says 'create a web app for X' or 'model a web "
        "application for Y', that is a creation/modeling intent, NOT "
        "generation."
    ),
)
workflow_intent = agent.new_intent(
    name="workflow_intent",
    description=(
        "The user wants a COMPLETE END-TO-END workflow: create the model, "
        "validate it, and generate code, all in one go. "
        'Keywords: "create a complete web app for", "build and deploy", '
        '"create end to end", "full workflow for", "build a full system", '
        '"create everything for", "design and generate", "model and deploy", '
        '"create a complete application", "build from scratch and generate code". '
        "This is for when the user explicitly wants the FULL pipeline — "
        "modeling, validation, and code generation — handled automatically "
        "in a single conversational flow."
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
        "describe_model": describe_model_state,
        "uml_rag": uml_rag_state,
        "generation": generation_state,
        "workflow": workflow_state,
    },
    intents={
        "hello": hello_intent,
        "create_single_element": create_single_element_intent,
        "create_complete_system": create_complete_system_intent,
        "modify_model": modify_model_intent,
        "modeling_help": modeling_help_intent,
        "describe_model": describe_model_intent,
        "uml_spec": uml_spec_intent,
        "generation": generation_intent,
        "workflow": workflow_intent,
    },
)


# ── Backward-compatible re-export for tests ──────────────────────────────
# ``test_model_helpers.py`` imports ``_resolve_class_diagram`` from here.
from utilities.model_resolution import resolve_class_diagram as _resolve_class_diagram  # noqa: E402, F401


# ── Session memory cleanup ───────────────────────────────────────────────
def _start_memory_cleanup_timer():
    def _cleanup_loop():
        import time
        while True:
            time.sleep(3600)  # Every hour
            try:
                cleanup_stale_memories(max_age_seconds=86400)
            except Exception:
                pass
    t = threading.Thread(target=_cleanup_loop, daemon=True)
    t.start()

_start_memory_cleanup_timer()


# ── Run ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent.run()
