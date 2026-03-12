"""
core/llm.py — Unified LLM provider registry and caller.

Single source of truth for all LLM interaction. Used by both
ecko_desktop.py and ecko_web.py.
"""

import re
import requests

from core.logger import log

# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

PROVIDER_REGISTRY: dict[str, dict] = {
    "koboldcpp": {
        "label":          "KoboldCPP (local)",
        "base_url":       "http://localhost:5001",
        "needs_api_key":  False,
        "needs_agent_id": False,
        "needs_model":    False,
        "chat_style":     "kobold",
    },
    "mistral": {
        "label":          "Mistral API",
        "base_url":       "https://api.mistral.ai",
        "needs_api_key":  True,
        "needs_agent_id": True,
        "needs_model":    False,
        "chat_style":     "mistral_conv",
    },
    # "lmstudio": {
    #     "label":          "LM Studio (local)",
    #     "base_url":       "http://localhost:1234",
    #     "needs_api_key":  False,
    #     "needs_agent_id": False,
    #     "needs_model":    True,
    #     "chat_style":     "openai",
    # },
    # "ollama": {
    #     "label":          "Ollama (local)",
    #     "base_url":       "http://localhost:11434",
    #     "needs_api_key":  False,
    #     "needs_agent_id": False,
    #     "needs_model":    True,
    #     "chat_style":     "openai",
    # },
    # "llamacpp": {
    #     "label":          "llama.cpp server (local)",
    #     "base_url":       "http://localhost:8080",
    #     "needs_api_key":  False,
    #     "needs_agent_id": False,
    #     "needs_model":    False,
    #     "chat_style":     "openai",
    # },
}


# ─────────────────────────────────────────────────────────────────────────────
# LLM CALLER
# ─────────────────────────────────────────────────────────────────────────────

class LLMCaller:
    """
    Unified LLM call layer.
    Holds all provider config and exposes a single chat() method.

    memory_inject_fn: optional callable(system_prompt, user_text) -> str
        Set this to hook in RAG or structured memory injection without
        coupling the LLM layer to any specific memory implementation.
    """

    def __init__(self):
        self.provider_id      = "koboldcpp"
        self.base_url         = PROVIDER_REGISTRY["koboldcpp"]["base_url"]
        self.api_key          = ""
        self.agent_id         = ""        # Mistral only
        self.model            = ""        # Ollama / LM Studio / llama.cpp
        self.system_prompt    = ""
        self.conv_id          = None      # Mistral conversation tracking (per-session)
        self.max_reply_tokens = 300       # cap reply length; lower = faster + less context
        self.max_history      = 10        # how many past turns to send (pairs = turns/2)

        # Pluggable memory injection — set externally to avoid circular imports.
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
             search_context: str = ""):
        """
        Send user_text (and optionally a base64-encoded image) and return reply.

        search_context : pre-fetched web search results block to inject as
                         system context for this turn only (empty = no search).

        Returns:
            str  — plain text reply (most providers)
            dict — {"reply": str, "file_ids": [str, ...]}  (Mistral conv API only)

        image_b64   : raw base64 string (no data-URI prefix), or None
        image_mime  : MIME type, e.g. "image/jpeg", "image/png", "image/webp"
        """
        has_image = bool(image_b64)
        style = self._style
        print(f"[LLM] provider={self.provider_id} style={style} model={self.model!r} agent={self.agent_id!r} has_image={has_image}")

        if style == "mistral_conv":
            return self._chat_mistral_conv(user_text,
                                           image_b64=image_b64, image_mime=image_mime)
        elif style == "kobold":
            return self._chat_kobold(user_text, history, image_b64=image_b64,
                                     search_context=search_context)
        else:
            return self._chat_openai(user_text, history,
                                     image_b64=image_b64, image_mime=image_mime,
                                     search_context=search_context)

    # ── Kobold ───────────────────────────────────────────────────────────────

    def _chat_kobold(self, user_text: str, history: list,
                     image_b64: str = None, image_mime: str = "image/jpeg",
                     search_context: str = "") -> str:
        prompt = ""
        if self.system_prompt:
            sp = self._inject(self.system_prompt, user_text)
            prompt += f"[SYSTEM]\n{sp}\n\n"

        for m in history[-self.max_history:]:
            role, content = m["role"], m["content"]
            if role == "system":
                prompt += f"[SYSTEM]\n{content}\n\n"
            elif role == "user":
                if isinstance(content, list):
                    text_parts = [c.get("text", "") for c in content
                                  if isinstance(c, dict) and c.get("type") == "text"]
                    prompt += f"User: {' '.join(text_parts)}\n"
                else:
                    prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"

        if image_b64:
            prompt += f"User:\n[img-1]\n{user_text}\nAssistant:"
        else:
            prompt += f"User: {user_text}\n"

        # Inject search results as an assistant data-handoff line immediately
        # before the model continues generating. This forces the model to treat
        # the results as its own recalled knowledge and answer from them directly,
        # rather than ignoring a [SYSTEM] block that appears before history.
        if search_context:
            prompt += f"Assistant: Here's what I found: {search_context.strip()}\n\nAssistant:"
        else:
            prompt += "Assistant:"

        payload = {
            "prompt":         prompt,
            "max_new_tokens": self.max_reply_tokens,
            "temperature":    0.7,
            "top_p":          0.9,
            # Stop tokens for common model families (Gemma 3, Mistral, Llama, etc.)
            "stop_sequence":  [
                "User:", "\nUser:",
                "\nAssistant:", "Assistant:",
                "<end_of_turn>", "<eos>",
                "<|eot_id|>", "<|end|>",
            ],
        }
        if image_b64:
            payload["images"] = [image_b64]

        r = requests.post(
            f"{self.base_url}/api/v1/generate",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        raw_text = r.json()["results"][0]["text"].strip()

        # Truncate at any hallucinated turn boundary that leaked past stop sequences.
        # Gemma 3 4B in particular tends to continue generating extra "Assistant:" lines.
        import re as _re
        turn_boundary = _re.search(
            r'\n(?:User:|Assistant:|<end_of_turn>|<eos>|<\|eot_id\|>|<\|end\|>)',
            raw_text,
        )
        if turn_boundary:
            raw_text = raw_text[:turn_boundary.start()].strip()

        return raw_text

    # ── OpenAI-compatible ────────────────────────────────────────────────────

    def _chat_openai(self, user_text: str, history: list,
                     image_b64: str = None, image_mime: str = "image/jpeg",
                     search_context: str = "") -> str:
        messages = []
        if self.system_prompt:
            sp = self._inject(self.system_prompt, user_text)
            print(f"[LLM] openai system_prompt len={len(sp)} inject_delta={len(sp)-len(self.system_prompt)}")
            messages.append({"role": "system", "content": sp})
        # Inject web search results as a system message so the model treats
        # them as authoritative context rather than user-supplied text.
        # Skip for Mistral — their agent has native web search on the agent panel;
        # injecting server-side results would conflict with it.
        if search_context and self.provider_id != "mistral":
            messages.append({"role": "system", "content":
                f"The following real-time web search results have been fetched. "
                f"Use them to answer the user accurately and naturally — "
                f"answer directly without narrating the search process:\n\n{search_context.strip()}"})

        for m in history[-self.max_history:]:
            if m["role"] in ("user", "assistant"):
                messages.append({"role": m["role"], "content": m["content"]})

        if image_b64:
            data_uri = f"data:{image_mime};base64,{image_b64}"
            user_content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]
        else:
            user_content = user_text
        messages.append({"role": "user", "content": user_content})

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": self.max_reply_tokens,
            "stream": False,
        }
        if self.model:
            payload["model"] = self.model

        # Ollama native /api/chat
        if self.provider_id == "ollama":
            if image_b64:
                payload["messages"][-1] = {
                    "role": "user", "content": user_text, "images": [image_b64]
                }
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
            r.raise_for_status()
            return (r.json().get("message") or {}).get("content", "...").strip()

        # Mistral vision (conv API has no vision support)
        if self.provider_id == "mistral":
            if not payload.get("model"):
                payload["model"] = "mistral-medium-latest"
            r = requests.post(f"{self.base_url}/v1/chat/completions",
                              headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()

        # Default: standard OpenAI-compat (LM Studio / llama.cpp / KoboldCPP vision)
        r = requests.post(f"{self.base_url}/v1/chat/completions",
                          headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    # ── Mistral conversations API ─────────────────────────────────────────────

    def _chat_mistral_conv(self, user_text: str,
                           image_b64: str = None,
                           image_mime: str = "image/jpeg") -> dict:
        """
        Returns {"reply": str, "file_ids": [str, ...]}
        file_ids are Mistral file IDs for any generated images in this turn.

        RAG and memory context is appended compactly after the user message on
        every turn so the model always has fresh relevant context without the
        large prefix that caused repetition on smaller models.
        """
        # Build the context block (RAG chunks + memory entries).
        # _inject("", user_text) returns only the injected blocks since we pass
        # an empty system prompt — result is like "\n\n[Memory]\n...\n\n[Relevant Context]\n..."
        # Inject RAG + memory context. Prepend before "User:" so the model
        # sees it as authoritative background, matching the original working behaviour.
        context = self._inject("", user_text).strip()
        print(f"[LLM] mistral_conv context_block len={len(context)} preview={repr(context[:200]) if context else '(empty)'}")
        if context:
            full_text = context + "\n\nUser: " + user_text
        else:
            full_text = user_text

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if image_b64:
            # Mistral conversations API vision: multipart content list
            data_uri = f"data:{image_mime};base64,{image_b64}"
            inputs = [{"role": "user", "content": [
                {"type": "text",      "text": full_text},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]}]
        else:
            inputs = [{"role": "user", "content": full_text}]

        if self.conv_id is None:
            r = requests.post(
                f"{self.base_url}/v1/conversations",
                headers=headers,
                json={"agent_id": self.agent_id, "inputs": inputs},
                timeout=180,
            )
        else:
            r = requests.post(
                f"{self.base_url}/v1/conversations/{self.conv_id}",
                headers=headers, json={"inputs": inputs}, timeout=180,
            )
            if r.status_code == 404:
                self.conv_id = None
                r = requests.post(
                    f"{self.base_url}/v1/conversations",
                    headers=headers,
                    json={"agent_id": self.agent_id, "inputs": inputs},
                    timeout=180,
                )

        r.raise_for_status()
        data = r.json()

        if self.conv_id is None:
            self.conv_id = (
                data.get("conversation_id") or data.get("id") or
                (data.get("conversation") or {}).get("id")
            )

        reply    = ""
        file_ids = []

        for entry in data.get("outputs", []):
            etype = entry.get("type", "")
            if etype == "message.output" or entry.get("role") == "assistant":
                content = entry.get("content", "")
                if isinstance(content, list):
                    for chunk in content:
                        if not isinstance(chunk, dict): continue
                        ctype = chunk.get("type", "")
                        if ctype == "text":
                            reply += chunk.get("text", "")
                        elif ctype == "tool_file" or chunk.get("file_id"):
                            fid = chunk.get("file_id") or chunk.get("id")
                            if fid: file_ids.append(fid)
                else:
                    reply = str(content)
            elif etype == "tool.execution":
                tool_out = entry.get("output") or {}
                if isinstance(tool_out, dict):
                    fid = tool_out.get("file_id") or tool_out.get("id")
                    if fid: file_ids.append(fid)
                for f in entry.get("files", []):
                    fid = f.get("file_id") or f.get("id") if isinstance(f, dict) else f
                    if fid: file_ids.append(fid)

        # Strip markdown links/images and bare URLs (Mistral conv sometimes returns them)
        reply = re.sub(r'!?\[.*?\]\(https?://\S+?\)', '', reply)
        reply = re.sub(r'https?://\S+', '', reply)
        reply = re.sub(
            r"(?i)(here(?:'s| is)(?: the| your)?\s+(?:link|image|picture|result|file)[^\n]*\n?)",
            '', reply,
        )
        reply = reply.strip()

        print(f"[MISTRAL CONV] reply={repr(reply[:80])} file_ids={file_ids}")
        return {"reply": reply or "...", "file_ids": file_ids}

    def fetch_mistral_file_b64(self, file_id: str) -> tuple[str, str]:
        """Download a Mistral file by ID and return (base64_str, mime_type)."""
        import base64 as _b64
        url = f"{self.base_url}/v1/files/{file_id}/content"
        r = requests.get(url, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=60)
        r.raise_for_status()
        mime = r.headers.get("Content-Type", "image/png").split(";")[0].strip()
        return _b64.b64encode(r.content).decode(), mime

    # ── Status check ─────────────────────────────────────────────────────────

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
                label = self.model if self.model in models else (models[0] if models else "Ollama")
                return True, label
            else:
                requests.get(f"{self.base_url}/v1/models", timeout=4)
                return True, self.model or "LLM"
        except Exception as e:
            log.debug("[LLM] ping failed (%s): %s", self.provider_id, e)
            return False, PROVIDER_REGISTRY.get(self.provider_id, {}).get("label", self.provider_id)
