"""
core/llm/kobold.py — KoboldCPP prompt builder and caller.

Handles the flat text prompt format used by KoboldCPP's /api/v1/generate
endpoint. Manages vision (img-1 placeholder), search pre-fill injection,
and turn-boundary truncation for models that ignore stop sequences.
"""

import re
import requests


def chat_kobold(caller, user_text: str, history: list,
                image_b64: str = None, image_mime: str = "image/jpeg",
                search_context: str = "",
                pre_injected_sp: str = None) -> str:
    """
    Build and submit a KoboldCPP generate request.

    caller          : LLMCaller instance (provides system_prompt, base_url,
                      max_history, max_reply_tokens, _inject)
    pre_injected_sp : if not None, use directly as the system prompt and skip
                      the lazy caller._inject() call (already done in parallel).
    """
    prompt = ""
    if caller.system_prompt:
        if pre_injected_sp is not None:
            sp = pre_injected_sp
        else:
            sp = caller._inject(caller.system_prompt, user_text)
        prompt += f"[SYSTEM]\n{sp}\n\n"

    for m in history[-caller.max_history:]:
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
        # KoboldCPP vision: [img-1] placeholder tells KoboldCPP where to
        # attach the image in the prompt; it substitutes the correct
        # model-specific token at inference time (llava-style).
        prompt += f"User:\n[img-1]\n{user_text}\n"
    else:
        prompt += f"User: {user_text}\n"

    # Inject search results as an assistant data-handoff line immediately
    # before the model continues generating. This forces the model to treat
    # results as its own recalled knowledge and answer from them directly,
    # rather than ignoring a [SYSTEM] block that appears before history.
    if search_context:
        prompt += f"Assistant: Here's what I found: {search_context.strip()}\n\nAssistant:"
    else:
        prompt += "Assistant:"

    payload = {
        "prompt":         prompt,
        "max_new_tokens": caller.max_reply_tokens,
        "temperature":    0.7,
        "top_p":          0.9,
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
        f"{caller.base_url}/api/v1/generate",
        json=payload,
        timeout=120,
    )

    # TODO: KoboldCPP image generation passthrough (low priority / high VRAM requirement)
    #
    # KoboldCPP exposes SD image gen via /sdapi/v1/txt2img (A1111-compatible) or
    # /api/extra/sd/generate when an SD model is loaded alongside the LLM.
    # The KoboldCPP UI handles this client-side — it detects the LLM's intent and
    # calls the SD endpoint itself. External clients get no passthrough currently.
    #
    # Proposed implementation when VRAM allows:
    #   1. Detect image gen intent in raw_text (e.g. *generate image: <desc>* tag
    #      embedded by the LLM via system prompt instruction)
    #   2. Extract the description from the tag and strip it from the reply text
    #   3. Optional conversion step: if [IMGGEN_PROMPT_STYLE: illustrious|sdxl] is set
    #      in the system prompt, run a second lightweight LLM call using a
    #      [IMGGEN_CONVERSION_PROMPT: ...] system prompt block (also in the char card)
    #      to convert natural language → booru-tag structured prompt before hitting SD.
    #      Skip conversion entirely for Flux (natural caption encoder, passthrough fine).
    #   4. POST to caller.base_url + /sdapi/v1/txt2img with the (converted) prompt
    #   5. Return {\"reply\": reply_text, \"gen_images\": [data_uri]} from this function
    #
    # Practical constraints that make this genuinely low priority:
    #   - Running base LLM + TTS + Illustrious/SDXL + conversion LLM + adetailers
    #     simultaneously requires ~24-48GB VRAM to stay fluid. Most setups must
    #     disable TTS just to run the LLM + SD at all.
    #   - The static image library (extras/image_lib.py) is faster, visually
    #     consistent (no face drift), and works on any hardware. Generation is a
    #     lottery without heavy LoRA + consistent seeds. The library approach was
    #     born from VRAM constraints and turns out to be architecturally superior
    #     for character consistency — don't implement this unless generation quality
    #     genuinely justifies the resource cost for the target user's hardware.
    r.raise_for_status()
    raw_text = r.json()["results"][0]["text"].strip()

    # Truncate at hallucinated turn boundaries that leaked past stop sequences.
    # Gemma 3 4B in particular continues with extra "Assistant:" lines.
    turn_boundary = re.search(
        r'\n(?:User:|Assistant:|<end_of_turn>|<eos>|<\|eot_id\|>|<\|end\|>)',
        raw_text,
    )
    if turn_boundary:
        raw_text = raw_text[:turn_boundary.start()].strip()

    return raw_text
