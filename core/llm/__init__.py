"""
core/llm — LLM provider package.

Re-exports LLMCaller and PROVIDER_REGISTRY so all existing imports
(from core.llm import LLMCaller) continue to work unchanged.

Internal structure:
  registry.py      — PROVIDER_REGISTRY dict
  caller.py        — LLMCaller dispatch class
  kobold.py        — KoboldCPP prompt builder / caller
  openai_compat.py — OpenAI-compat + Mistral conv backends
"""

from .registry import PROVIDER_REGISTRY
from .caller   import LLMCaller

__all__ = ["LLMCaller", "PROVIDER_REGISTRY"]
