"""Convenience re-export for OpenAI-compatible LLM adapter.

Preferred canonical import:
    from protocore.integrations.openai import OpenAILLMClient
"""
from __future__ import annotations

from .llm.openai_client import OpenAILLMClient, StructuredOutputValidationError

__all__ = ["OpenAILLMClient", "StructuredOutputValidationError"]
