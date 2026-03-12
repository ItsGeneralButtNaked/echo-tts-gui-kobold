"""
core/llm/caller.py — LLMCaller: thin dispatch layer over provider backends.

Holds all connection config and exposes a single chat() method.
Prompt building lives in kobold.py / openai_compat.py.
"""

import requests

from .registry     import PROVIDER_REGISTRY
from .kobold       import chat_kobold
from .openai_compat import chat_openai, chat_mistral_conv, fetch_mistral_file_b64
from core.logger import log


class LLMCaller:
    """
    Unified LLM call layer.

    memory_inject_fn : optional callable(system_prompt, user_text) -> str
        Hook for RAG + structured memory injection without coupling the LLM
        layer to any specific memory implementation.
    """

    def __init__(self):
        self.provider_id      = "koboldcpp"
        self.base_url         = PROVIDER_REGISTRY["koboldcpp"]["base_url"]
        self.api_key          = ""
        self.agent_id         = ""        # Mistral conv API only
        self.model            = ""        # Ollama / LM Studio / llama.cpp
        self.system_prompt    = ""
        self.conv_id          = None      # Mistral conversation tracking
        self.max_reply_tokens = 300
        self.max_history      = 10

        # Pluggable memory/RAG injection — set externally to avoid circular imports.
        # Signature: (system_prompt: str, user_text: str) -> str
        self.memory_inject_fn = None

    # ── serialise / deserialise ──────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "provider_id":      self.provider_id,
            "base_url":         self.base_url,
            "api_key":          self.api_key,
            "agent_id":         self.agent_id,
            "model":            self.model,
            "system_prompt":    self.system_prompt,
            "conv_id":          self.conv_id,
            "max_reply_tokens": self.max_reply_tokens,
            "max_history":      self.max_history,
        }

    def from_dict(self, d: dict):
        self.provider_id      = d.get("provider_id",      self.provider_id)
        self.base_url         = d.get("base_url",          self.base_url)
        self.api_key          = d.get("api_key",           self.api_key)
        self.agent_id         = d.get("agent_id",          self.agent_id)
        self.model            = d.get("model",             self.model)
        self.system_prompt    = d.get("system_prompt",     self.system_prompt)
        self.conv_id          = d.get("conv_id",           self.conv_id)
        self.max_reply_tokens = int(d.get("max_reply_tokens", self.max_reply_tokens))
        self.max_history      = int(d.get("max_history",   self.max_history))

    def reset_conv(self):
        """Reset Mistral conversation ID (forces new session on next call)."""
        self.conv_id = None

    @property
    def _style(self) -> str:
        return PROVIDER_REGISTRY.get(self.provider_id, {}).get("chat_style", "openai")

    # ── memory injection helper ──────────────────────────────────────────────

    def _inject(self, system_prompt: str, user_text: str) -> str:
        """Apply the registered memory/RAG injection hook if present."""
        if callable(self.memory_inject_fn):
            return self.memory_inject_fn(system_prompt, user_text)
        return system_prompt

    # ── public entry point ───────────────────────────────────────────────────

    def chat(self, user_text: str, history: list,
             image_b64: str = None, image_mime: str = "image/jpeg",
             search_context: str = "",
             pre_injected_sp: str = None):
        """
        Send user_text (and optionally a base64-encoded image) and return reply.

        search_context   : pre-fetched web search results to inject for this
                           turn only (empty = no search).
        pre_injected_sp  : if provided, skip the lazy memory_inject_fn call
                           inside the backend and use this string directly as
                           the system prompt.  The caller is responsible for
                           having already run injection (e.g. in parallel with
                           web search).  Pass None to use the normal lazy path.

        Returns:
            str  — plain text reply (most providers)
            dict — {"reply": str, "file_ids": [...]}  (Mistral conv API only)

        image_b64  : raw base64 string (no data-URI prefix), or None
        image_mime : MIME type, e.g. "image/jpeg"
        """
        has_image = bool(image_b64)
        style = self._style
        print(f"[LLM] provider={self.provider_id} style={style} "
              f"model={self.model!r} agent={self.agent_id!r} has_image={has_image}"
              f" pre_injected={'yes' if pre_injected_sp is not None else 'no'}")

        if style == "mistral_conv":
            return chat_mistral_conv(self, user_text,
                                     image_b64=image_b64, image_mime=image_mime,
                                     pre_injected_sp=pre_injected_sp)
        elif style == "kobold":
            return chat_kobold(self, user_text, history,
                               image_b64=image_b64, search_context=search_context,
                               pre_injected_sp=pre_injected_sp)
        else:
            return chat_openai(self, user_text, history,
                               image_b64=image_b64, image_mime=image_mime,
                               search_context=search_context,
                               pre_injected_sp=pre_injected_sp)

    # ── delegated helpers ────────────────────────────────────────────────────

    def fetch_mistral_file_b64(self, file_id: str) -> tuple[str, str]:
        """Download a Mistral file by ID and return (base64_str, mime_type)."""
        return fetch_mistral_file_b64(self, file_id)

    # ── status check ─────────────────────────────────────────────────────────

    def ping(self) -> tuple[bool, str]:
        """Return (online: bool, label: str)"""
        try:
            style = self._style
            if style == "mistral_conv":
                r = requests.get(
                    f"{self.base_url}/v1/agents/{self.agent_id}",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=4,
                )
                if r.status_code == 200:
                    return True, r.json().get("name", self.agent_id)
                return False, "Mistral API error"
            elif style == "kobold":
                r = requests.get(f"{self.base_url}/api/v1/model", timeout=4)
                return True, r.json().get("result", "KoboldCPP")
            elif self.provider_id == "ollama":
                r = requests.get(f"{self.base_url}/api/tags", timeout=4)
                models = [m["name"] for m in r.json().get("models", [])]
                label = (self.model if self.model in models
                         else (models[0] if models else "Ollama"))
                return True, label
            else:
                requests.get(f"{self.base_url}/v1/models", timeout=4)
                return True, self.model or "LLM"
        except Exception as e:
            log.debug("[LLM] ping failed (%s): %s", self.provider_id, e)
            return False, PROVIDER_REGISTRY.get(
                self.provider_id, {}).get("label", self.provider_id)
