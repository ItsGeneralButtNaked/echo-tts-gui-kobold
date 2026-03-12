"""
core/llm/openai_compat.py — OpenAI-compatible chat completion and Mistral
conversations API backends.

chat_openai   : standard /v1/chat/completions (LM Studio, llama.cpp, Ollama,
                Mistral vision fallback)
chat_mistral_conv : Mistral /v1/conversations stateful agent API
fetch_mistral_file_b64 : download a Mistral-generated file by ID
"""

import re
import requests


def chat_openai(caller, user_text: str, history: list,
                image_b64: str = None, image_mime: str = "image/jpeg",
                search_context: str = "",
                pre_injected_sp: str = None) -> str:
    """
    Send a chat/completions request to any OpenAI-compatible endpoint.

    caller          : LLMCaller instance
    pre_injected_sp : if not None, use directly as the system prompt and skip
                      the lazy caller._inject() call (already done in parallel).
    """
    messages = []
    if caller.system_prompt:
        if pre_injected_sp is not None:
            sp = pre_injected_sp
            print(f"[LLM] openai system_prompt len={len(sp)} "
                  f"inject_delta={len(sp)-len(caller.system_prompt)} (pre-injected)")
        else:
            sp = caller._inject(caller.system_prompt, user_text)
            print(f"[LLM] openai system_prompt len={len(sp)} "
                  f"inject_delta={len(sp)-len(caller.system_prompt)}")
        messages.append({"role": "system", "content": sp})

    # Inject web search results as a system message.
    # Skipped for Mistral — their agent has native web search; server-side
    # injection would conflict with it.
    if search_context and caller.provider_id != "mistral":
        messages.append({"role": "system", "content":
            "The following real-time web search results have been fetched. "
            "Use them to answer the user accurately and naturally — "
            f"answer directly without narrating the search process:\n\n"
            f"{search_context.strip()}"})

    for m in history[-caller.max_history:]:
        if m["role"] in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m["content"]})

    if image_b64:
        data_uri = f"data:{image_mime};base64,{image_b64}"
        user_content = [
            {"type": "text",      "text": user_text},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]
    else:
        user_content = user_text
    messages.append({"role": "user", "content": user_content})

    headers = {"Content-Type": "application/json"}
    if caller.api_key:
        headers["Authorization"] = f"Bearer {caller.api_key}"

    payload = {
        "messages":    messages,
        "temperature": 0.7,
        "max_tokens":  caller.max_reply_tokens,
        "stream":      False,
    }
    if caller.model:
        payload["model"] = caller.model

    # Ollama native /api/chat
    if caller.provider_id == "ollama":
        if image_b64:
            payload["messages"][-1] = {
                "role": "user", "content": user_text, "images": [image_b64]
            }
        r = requests.post(f"{caller.base_url}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        return (r.json().get("message") or {}).get("content", "...").strip()

    # Mistral vision (conv API has no vision support, use completions fallback)
    if caller.provider_id == "mistral":
        if not payload.get("model"):
            payload["model"] = "mistral-medium-latest"
        r = requests.post(f"{caller.base_url}/v1/chat/completions",
                          headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    # Default: standard OpenAI-compat (LM Studio / llama.cpp / KoboldCPP vision)
    r = requests.post(f"{caller.base_url}/v1/chat/completions",
                      headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def chat_mistral_conv(caller, user_text: str,
                      image_b64: str = None,
                      image_mime: str = "image/jpeg",
                      pre_injected_sp: str = None) -> dict:
    """
    Mistral stateful conversations API.

    Returns {"reply": str, "file_ids": [str, ...]}
    file_ids are Mistral file IDs for any generated images in this turn.

    RAG and memory context is appended compactly after the user message on
    every turn so the model always has fresh relevant context without the
    large prefix that caused repetition on smaller models.

    pre_injected_sp : if not None, the injection chain has already run; derive
                      the context block from it directly rather than calling
                      caller._inject() again.
    """
    # _inject("", user_text) returns only the injected blocks since we pass
    # an empty system prompt.  When pre_injected_sp is provided the caller
    # already ran the chain against the real system_prompt, so strip that
    # prefix to recover the blocks-only delta.
    if pre_injected_sp is not None:
        base = caller.system_prompt or ""
        context = pre_injected_sp[len(base):].strip() if pre_injected_sp.startswith(base) else pre_injected_sp.strip()
        print(f"[LLM] mistral_conv context_block len={len(context)} "
              f"preview={repr(context[:200]) if context else '(empty)'} (pre-injected)")
    else:
        context = caller._inject("", user_text).strip()
        print(f"[LLM] mistral_conv context_block len={len(context)} "
              f"preview={repr(context[:200]) if context else '(empty)'}")
    full_text = (context + "\n\nUser: " + user_text) if context else user_text

    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {caller.api_key}",
    }

    if image_b64:
        data_uri = f"data:{image_mime};base64,{image_b64}"
        inputs = [{"role": "user", "content": [
            {"type": "text",      "text": full_text},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]}]
    else:
        inputs = [{"role": "user", "content": full_text}]

    if caller.conv_id is None:
        r = requests.post(
            f"{caller.base_url}/v1/conversations",
            headers=headers,
            json={"agent_id": caller.agent_id, "inputs": inputs},
            timeout=180,
        )
    else:
        r = requests.post(
            f"{caller.base_url}/v1/conversations/{caller.conv_id}",
            headers=headers, json={"inputs": inputs}, timeout=180,
        )
        if r.status_code == 404:
            caller.conv_id = None
            r = requests.post(
                f"{caller.base_url}/v1/conversations",
                headers=headers,
                json={"agent_id": caller.agent_id, "inputs": inputs},
                timeout=180,
            )

    r.raise_for_status()
    data = r.json()

    if caller.conv_id is None:
        caller.conv_id = (
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
                    if not isinstance(chunk, dict):
                        continue
                    ctype = chunk.get("type", "")
                    if ctype == "text":
                        reply += chunk.get("text", "")
                    elif ctype == "tool_file" or chunk.get("file_id"):
                        fid = chunk.get("file_id") or chunk.get("id")
                        if fid:
                            file_ids.append(fid)
            else:
                reply = str(content)
        elif etype == "tool.execution":
            tool_out = entry.get("output") or {}
            if isinstance(tool_out, dict):
                fid = tool_out.get("file_id") or tool_out.get("id")
                if fid:
                    file_ids.append(fid)
            for f in entry.get("files", []):
                fid = (f.get("file_id") or f.get("id")) if isinstance(f, dict) else f
                if fid:
                    file_ids.append(fid)

    # Strip markdown links/images and bare URLs
    reply = re.sub(r'!?\[.*?\]\(https?://\S+?\)', '', reply)
    reply = re.sub(r'https?://\S+', '', reply)
    reply = re.sub(
        r"(?i)(here(?:'s| is)(?: the| your)?\s+(?:link|image|picture|result|file)[^\n]*\n?)",
        '', reply,
    )
    reply = reply.strip()

    print(f"[MISTRAL CONV] reply={repr(reply[:80])} file_ids={file_ids}")
    return {"reply": reply or "...", "file_ids": file_ids}


def fetch_mistral_file_b64(caller, file_id: str) -> tuple[str, str]:
    """Download a Mistral file by ID and return (base64_str, mime_type)."""
    import base64 as _b64
    url = f"{caller.base_url}/v1/files/{file_id}/content"
    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {caller.api_key}"},
        timeout=60,
    )
    r.raise_for_status()
    mime = r.headers.get("Content-Type", "image/png").split(";")[0].strip()
    return _b64.b64encode(r.content).decode(), mime
