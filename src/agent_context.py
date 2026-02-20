"""
Mutable runtime context — populated by ``modeling_agent.py`` at startup.

Other modules import these names and read them at **call time** (not import
time), so they are always populated when user messages arrive.

Usage::

    from src.agent_context import diagram_factory, gpt_predict_json
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from besser.agent.core.agent import Agent
    from besser.agent.nlp.llm.llm_openai_api import LLMOpenAI
    from besser.agent.nlp.rag.rag import RAG
    from src.diagram_handlers.factory import DiagramHandlerFactory

# Populated by modeling_agent.py during agent bootstrap.
agent: "Agent | None" = None
gpt: "LLMOpenAI | None" = None
gpt_text: "LLMOpenAI | None" = None
gpt_predict_json = None          # callable(str) -> str
uml_rag: "RAG | None" = None
diagram_factory: "DiagramHandlerFactory | None" = None
openai_api_key: str | None = None
