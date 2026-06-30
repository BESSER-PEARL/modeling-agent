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

from baf.core.agent import Agent
from baf import nlp
from baf.exceptions.logger import logger

import agent_context as ctx
from agent_setup import (
    init_llm,
    init_stt,
    init_rag,
    init_diagram_factory,
    init_intent_classifier_config,
)
from agent_config import GRACE_PERIOD_SECONDS
from routing.intents import GENERATION_INTENT_NAME
from state_bodies import register_all
from memory.conversation_memory import cleanup_stale_memories

# ── Logging ──────────────────────────────────────────────────────────────
logger.setLevel(logging.INFO)
logger.propagate = False
# Configure root logger so our src/ modules' info/debug output appears.
if not logging.root.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
else:
    logging.root.setLevel(logging.INFO)

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
ctx.stt = stt

# ── States ───────────────────────────────────────────────────────────────
greetings_state = agent.new_state("greetings_state", initial=True)
create_complete_system_state = agent.new_state("create_complete_system_state")
modify_model_state = agent.new_state("modify_model_state")
modeling_help_state = agent.new_state("modeling_help_state")
describe_model_state = agent.new_state("describe_model_state")
uml_rag_state = agent.new_state("uml_rag_state")
generation_state = agent.new_state("generation_state")

# ── Intents ──────────────────────────────────────────────────────────────
#
# Intent descriptions were historically 30-50 lines of keyword essays
# that BAF's built-in classifier used as prompt context. Since
# ``unified_classifier.py`` took over primary routing, those keyword
# blobs are no longer driving any behavior — the real rules live in
# ``unified_classifier._SYSTEM_PROMPT``. The one-liners below exist
# only so BAF's classifier can still pick a sensible intent if our
# unified call ever fails (rare safety-net path). Keep them SHORT.
#
# Training sentences are REQUIRED: the default intent classifier is now
# the local SimpleIntentClassifier (see agent_setup), which trains on
# them at startup — this keeps the BAF fallback (and the voice/text-event
# route) free instead of one LLM call per message. The unified classifier
# remains authoritative via ``json_intent_matches``.
hello_intent = agent.new_intent(
    name="hello_intent",
    description="User greets the assistant or makes small-talk.",
    training_sentences=[
        "hello",
        "hi there",
        "hey, how are you",
        "good morning",
        "hello assistant",
        "hi, are you there",
    ],
)
create_complete_system_intent = agent.new_intent(
    name="create_complete_system_intent",
    description="User wants to create a NEW diagram or complete system from scratch.",
    training_sentences=[
        "create a library management system",
        "design a class diagram for an online shop",
        "create a state machine for order processing",
        "build a complete hotel booking system",
        "make a new agent diagram for a pizza chatbot",
        "model a quantum circuit for grover's search",
    ],
)
modify_model_intent = agent.new_intent(
    name="modify_model_intent",
    description="User wants to add/remove/change elements in an existing diagram.",
    training_sentences=[
        "add an email attribute to the user class",
        "rename order to purchase",
        "remove the price attribute from product",
        "add a transition from idle to active",
        "change the multiplicity between customer and order",
        "delete the payment class",
    ],
)
modeling_help_intent = agent.new_intent(
    name="modeling_help_intent",
    description="User asks a conceptual question about modeling (not about their own diagram).",
    training_sentences=[
        "what is an association class",
        "how do I model inheritance",
        "explain the difference between composition and aggregation",
        "when should I use a state machine",
        "help me with uml modeling",
        "what does multiplicity mean",
    ],
)
describe_model_intent = agent.new_intent(
    name="describe_model_intent",
    description="User asks a question about the current diagram on their canvas.",
    training_sentences=[
        "describe my diagram",
        "what classes do I have",
        "explain my current model",
        "how many attributes does the user class have",
        "what does my circuit do",
        "summarize the diagrams in my project",
    ],
)
uml_spec_intent = agent.new_intent(
    name="uml_spec_intent",
    description="User asks about the formal UML specification document (rare).",
    training_sentences=[
        "what does the uml specification say about associations",
        "how does the uml spec define stereotypes",
        "according to the uml standard what is a classifier",
        "uml specification rules for state machines",
        "cite the uml spec section on multiplicity",
    ],
)
generation_intent = agent.new_intent(
    name=GENERATION_INTENT_NAME,
    description="User wants SOURCE CODE in any stack, or to export/deploy.",
    training_sentences=[
        "generate django",
        "generate python code from my model",
        "generate sql for my diagram",
        "give me pydantic classes",
        "export my project as json",
        "deploy the app to render",
    ],
)
# ── Wire state bodies & transitions ─────────────────────────────────────
register_all(
    agent=agent,
    states={
        "greetings": greetings_state,
        "create_complete_system": create_complete_system_state,
        "modify_model": modify_model_state,
        "modeling_help": modeling_help_state,
        "describe_model": describe_model_state,
        "uml_rag": uml_rag_state,
        "generation": generation_state,
    },
    intents={
        "hello": hello_intent,
        "create_complete_system": create_complete_system_intent,
        "modify_model": modify_model_intent,
        "modeling_help": modeling_help_intent,
        "describe_model": describe_model_intent,
        "uml_spec": uml_spec_intent,
        "generation": generation_intent,
    },
)


# ── Session & thread cleanup ─────────────────────────────────────────────
def _start_cleanup_timer():
    """Periodically reap disconnected sessions and stale conversation memories.

    The BESSER framework keeps sessions (and their event-loop threads) alive
    after WebSocket disconnect to allow reconnects.  Over days of uptime,
    orphaned threads accumulate and eventually hit the OS thread limit
    (``RuntimeError: can't start new thread``).

    This reaper runs every 10 minutes and closes any session whose WebSocket
    connection is no longer tracked by the platform, provided it has been
    disconnected for at least 5 minutes (grace period for brief reconnects).
    """
    import time as _time

    # Track when we first notice a session has no active connection
    _disconnected_since: dict[str, float] = {}
    _GRACE_PERIOD = GRACE_PERIOD_SECONDS  # Grace period before reaping a disconnected session

    def _cleanup_loop():
        while True:
            _time.sleep(600)  # Every 10 minutes
            try:
                # Snapshot dict keys to avoid RuntimeError from concurrent modification.
                # These are O(n) copies but sessions are few (tens, not thousands).
                try:
                    active_conn_ids = set(list(websocket_platform._connections.keys()))
                    all_session_ids = list(agent._sessions.keys())
                except RuntimeError:
                    # Dict changed during iteration — skip this cycle
                    continue
                now = _time.time()

                for sid in all_session_ids:
                    if sid in active_conn_ids:
                        _disconnected_since.pop(sid, None)
                        continue

                    if sid not in _disconnected_since:
                        _disconnected_since[sid] = now
                        continue

                    if now - _disconnected_since[sid] < _GRACE_PERIOD:
                        continue

                    # Grace period expired — verify session still exists before closing
                    if sid not in agent._sessions:
                        _disconnected_since.pop(sid, None)
                        continue

                    try:
                        agent.close_session(sid)
                        # logger.info(f"[Reaper] Closed orphaned session {sid}")
                    except (KeyError, RuntimeError):
                        pass  # Session was already removed by another thread
                    except Exception as exc:
                        logger.warning(f"[Reaper] Failed to close session {sid}: {exc}")
                    _disconnected_since.pop(sid, None)

                # Clean up tracker for sessions that no longer exist
                for sid in list(_disconnected_since):
                    if sid not in agent._sessions:
                        _disconnected_since.pop(sid, None)

            except Exception as exc:
                logger.warning(f"[Reaper] Session cleanup error: {exc}")

            try:
                # --- Reap stale conversation memories ---
                cleanup_stale_memories(max_age_seconds=3600)
            except Exception:
                pass

    t = threading.Thread(target=_cleanup_loop, daemon=True, name="session-reaper")
    t.start()

_start_cleanup_timer()


# ── Run ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent.run()
