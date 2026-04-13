"""
Mutable runtime context — populated by ``modeling_agent.py`` at startup.

Other modules import these names and read them at **call time** (not import
time), so they are always populated when user messages arrive.

Usage::

    from src.agent_context import diagram_factory, gpt_predict_json
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from baf.core.agent import Agent
    from baf.nlp.llm.llm_openai_api import LLMOpenAI
    from baf.nlp.rag.rag import RAG
    from src.diagram_handlers.registry.factory import DiagramHandlerFactory

# Populated by modeling_agent.py during agent bootstrap.
agent: "Agent | None" = None
gpt: "LLMOpenAI | None" = None
gpt_text: "LLMOpenAI | None" = None
gpt_predict_json: Optional[Callable[[str], str]] = None
uml_rag: "RAG | None" = None
diagram_factory: "DiagramHandlerFactory | None" = None
openai_api_key: Optional[str] = None
stt = None  # OpenAISpeech2Text | None
