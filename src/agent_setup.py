"""
Agent Setup
-----------
Initialization functions for LLMs, RAG, and the diagram handler factory.

Called once by ``modeling_agent.py`` during startup; returns the initialized
objects so the entrypoint can populate :mod:`agent_context`.
"""

import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

from baf import nlp
from baf.core.agent import Agent
from baf.nlp.intent_classifier.intent_classifier_configuration import LLMIntentClassifierConfiguration
from baf.nlp.llm.llm_openai_api import LLMOpenAI
from baf.nlp.speech2text.openai_speech2text import OpenAISpeech2Text

from agent_config import (
    LLM_TEMPERATURE,
    LLM_TEXT_TEMPERATURE,
    LLM_MAX_TOKENS_LARGE,
    LLM_MAX_TOKENS_TEXT,
    CONVERSATION_HISTORY_DEPTH,
)

logger = logging.getLogger(__name__)


def init_llm(agent: Agent) -> Tuple[LLMOpenAI, LLMOpenAI, Callable[[str], str]]:
    """Initialize the LLMs and return ``(gpt, gpt_text, gpt_predict_json)``."""
    # NOTE: Do NOT put response_format in the global parameters dict.
    # The BESSER framework's intent_classification() hardcodes
    # response_format and then unpacks **parameters — having it in both
    # causes "got multiple values for keyword argument 'response_format'".
    # num_previous_messages is only used by BESSER's .chat() method, not
    # by .predict() which is our main call path.  Keep at 1 (the minimum
    # the framework validates) to avoid accidental context bloat if .chat()
    # is ever invoked.  Conversation context is injected explicitly in
    # execution.py via the ConversationMemory module.
    gpt = LLMOpenAI(
        agent=agent,
        name='gpt-4.1-mini',
        parameters={
            'temperature': LLM_TEMPERATURE,
            'max_completion_tokens': LLM_MAX_TOKENS_LARGE,
        },
        num_previous_messages=CONVERSATION_HISTORY_DEPTH,
    )

    # Thin wrapper that enforces JSON mode for predict() calls only.
    _gpt_json_params: Dict[str, Any] = {
        'temperature': LLM_TEMPERATURE,
        'max_completion_tokens': LLM_MAX_TOKENS_LARGE,
        'response_format': {'type': 'json_object'},
    }

    def gpt_predict_json(prompt: str) -> str:
        """Call gpt.predict with JSON-object response_format."""
        return gpt.predict(prompt, parameters=_gpt_json_params)

    # Free-text LLM (help, greetings, RAG fallback) — JSON mode would break.
    # IMPORTANT: The BESSER framework registers LLMs by name in a dict, and
    # also uses name as the OpenAI model parameter.  Two LLMs with the same
    # name would overwrite each other, leaving the first un-initialised.
    # We register under a unique key ('gpt-4.1-mini-text') and immediately
    # correct the model name back to the real API model so OpenAI calls work.
    gpt_text = LLMOpenAI(
        agent=agent,
        name='gpt-4.1-mini-text',
        parameters={
            'temperature': LLM_TEXT_TEMPERATURE,
            'max_completion_tokens': LLM_MAX_TOKENS_TEXT,
        },
        num_previous_messages=20,
    )
    # Fix the model name used in API calls (registry key stays 'gpt-4.1-mini-text')
    gpt_text.name = 'gpt-4.1-mini'

    if gpt is None:
        raise RuntimeError("LLM initialization returned None")

    # Initialize the LLM provider abstraction (for structured outputs, streaming)
    try:
        from llm import get_provider
        get_provider(gpt, model_name='gpt-4.1-mini')
        logger.info("LLM provider abstraction initialized")
    except Exception as exc:
        logger.warning(f"LLM provider init failed (non-critical): {exc}")

    logger.info(
        f"LLMs initialized: gpt-4.1-mini (json, t={LLM_TEMPERATURE}), "
        f"gpt-4.1-mini (text, t={LLM_TEXT_TEMPERATURE})"
    )
    return gpt, gpt_text, gpt_predict_json


def init_rag(agent: Agent):
    """Initialize the UML Specification RAG.  Returns *None* on failure."""
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    try:
        from chromadb.config import Settings as ChromaSettings
    except Exception:
        ChromaSettings = None

    from baf.nlp.rag.rag import RAG

    try:
        chroma_kwargs: Dict[str, Any] = {
            'embedding_function': OpenAIEmbeddings(
                openai_api_key=agent.get_property(nlp.OPENAI_API_KEY),
            ),
            'persist_directory': 'uml_vector_store',
        }
        if ChromaSettings is not None:
            chroma_kwargs['client_settings'] = ChromaSettings(anonymized_telemetry=False)

        vector_store: Chroma = Chroma(**chroma_kwargs)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        uml_rag = RAG(
            agent=agent,
            vector_store=vector_store,
            splitter=splitter,
            llm_name='gpt-4.1-mini',
            k=4,
            num_previous_messages=6,
        )

        uml_rag.llm_prompt = (
            "You are a UML (Unified Modeling Language) specification expert. "
            "Based on the context retrieved from the UML specification documents, "
            "answer the user's question about UML concepts, notation, semantics, or best practices.\n\n"
            "If the context contains relevant information, use it to provide an accurate and detailed answer.\n"
            "If you don't find the answer in the context, say that you don't have that specific information "
            "in the UML specification documents, but you can provide general guidance based on your knowledge.\n\n"
            "Be precise and reference specific UML concepts when applicable. "
            "Use clear examples when helpful."
        )

        logger.info("UML RAG initialized successfully")
        return uml_rag
    except Exception as exc:
        logger.warning(f"Failed to initialize UML RAG: {exc}. RAG features will be disabled.")
        return None


def init_stt(agent: Agent) -> OpenAISpeech2Text:
    """Initialize and return OpenAI speech-to-text for voice messages."""
    stt = OpenAISpeech2Text(
        agent=agent,
        model_name='whisper-1',
        language='en',
    )
    logger.info("Speech-to-text initialized: whisper-1 (language=en)")
    return stt


def init_diagram_factory(gpt: LLMOpenAI):
    """Create and return the :class:`DiagramHandlerFactory`."""
    from diagram_handlers.registry.factory import DiagramHandlerFactory

    factory = DiagramHandlerFactory(gpt)
    logger.info(f"Diagram handlers initialized: {', '.join(factory.get_supported_types())}")
    return factory


def init_intent_classifier_config() -> LLMIntentClassifierConfiguration:
    """Return the default intent-classifier configuration.

    Uses gpt-4.1-mini. The configuration is designed to leverage
      intent and entity descriptions
    """
    return LLMIntentClassifierConfiguration(
        llm_name='gpt-4.1-mini',
        parameters={},
        use_intent_descriptions=True,
        use_training_sentences=False,
        use_entity_descriptions=True,
        use_entity_synonyms=False,
    )
