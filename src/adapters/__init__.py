"""Adapters layer for external model providers.

Architectural boundary:
- Wraps vendor-specific SDKs and APIs (Jev typesafe-sdk, LiteLLM client).
- Standardizes diverse external responses into unified internal models (NoulResponse,
  ChoiceResponse, ScoreResponse, LLMResponse).
- Isolates network calls, retries, and rate limiting away from evaluation engine logic.
"""

from src.adapters.base import BaseAdapter
from src.adapters.jev_adapter import JevAdapter
from src.adapters.llm_adapter import LLMAdapter

__all__ = ["BaseAdapter", "JevAdapter", "LLMAdapter"]


