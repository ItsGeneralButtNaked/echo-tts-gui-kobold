"""
web/routes/chat.py — Chat, TTS, STT, and Auto-continue routes.

Blueprint: chat_bp
Routes:
  POST /chat
  POST /tts/cancel
  POST /tts
  POST /stt
  GET  /ac_stream
  POST /ac/rearm
  GET  /chat/stream
  GET  /
"""

import json
import os
import queue
import threading
import time

from core.logger import log

_SERVER_BOOT_ID = str(int(time.time()))

from flask import Blueprint, Response, jsonify, request

from core.session import MAX_HISTORY
from core.stt import transcribe_audio
from core.tts import TTS_PROVIDER_REGISTRY
from core.websearch import detect_search_intent, extract_query, brave_search, build_search_context
from web.sanitise import strip_leaked_context

# ── Image / video caption context extractor ───────────────────────────────────
# Reads optional [IMAGE_CONTEXT: ...] or [VIDEO_CONTEXT: ...] blocks from the
# active system prompt so character cards can guide how caption replies are framed.
# Blocks can span multiple lines and are stripped before being injected into the
# caption prompt. Unknown / missing blocks return an empty string.
#
# Example system prompt lines:
#   [IMAGE_CONTEXT: All images you send are selfies you took yourself.
#    Any male in the image is the user. Refer to him by name if you know it.]
#   [VIDEO_CONTEXT: Videos you send are clips you recorded. Treat them as your own.]

import re as _re_ctx

def _extract_media_context(system_prompt: str, kind: str = "image") -> str:
    """
    Extract the content of an [IMAGE_CONTEXT: ...] or [VIDEO_CONTEXT: ...] block
    from the system prompt. Returns stripped text or '' if not present.
    kind: 'image' or 'video'
    """
    tag = "IMAGE_CONTEXT" if kind == "image" else "VIDEO_CONTEXT"
    match = _re_ctx.search(
        rf'\[{tag}:\s*([\s\S]*?)\]',
        system_prompt or "",
        _re_ctx.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return ""

chat_bp = Blueprint("chat", __name__)

# ── Mood vocabulary ────────────────────────────────────────────────────────────
# Fixed emoji → label mapping. The LLM embeds *mood: 😊* in replies.
# Stripped before display, stored as metadata on history entries.
_MOOD_VOCAB = {
    "😊": "warm/happy",
    "😈": "teasing/playful-dark",
    "🥺": "vulnerable/soft",
    "💢": "irritated/sharp",
    "😴": "low energy/tired",
    "🤔": "thoughtful/curious",
}
_MOOD_EMOJIS = set(_MOOD_VOCAB.keys())

# ── Per-turn feedback injection ────────────────────────────────────────────────
# Thumbs up/down on a bubble writes a one-shot note here.
# It is prepended to the *next* real user turn's effective_text, then cleared.
# TODO: expand to per-direction dropdown options (too long / out of character /
#       boring / too short / wrong tone) for finer-grained soft injection.
_feedback_note: str = ""
_feedback_lock = threading.Lock()

# ── TTS serialisation ─────────────────────────────────────────────────────────
# Only one TTS inference runs at a time. _tts_cancel signals the active stream
# to drain and exit ASAP; _tts_mutex gates the next request until the socket
# is fully closed. _tts_generation is incremented on every new /tts request —
# a streaming generator that sees its generation number is stale knows a newer
# request has won and self-aborts, preventing concurrent inference OOM.
_tts_cancel     = threading.Event()
_tts_mutex      = threading.Lock()
_tts_generation = 0

# ── SSE subscriber registry ───────────────────────────────────────────────────
# Each entry is (queue, char_path) — char_path is the loaded_char_path at the
# time the client connected.  Broadcasts are only delivered to subscribers whose
# char_path matches the current server char_path, preventing bubble bleed when
# multiple devices have different characters open.
_chat_subscribers: list[tuple[queue.Queue, str]] = []
_chat_sub_lock = threading.Lock()


def _current_char_path() -> str:
    """Return the currently loaded character path, or '' if none."""
    try:
        return _get_session().tts.extra.get("loaded_char_path", "")
    except Exception:
        return ""


def push_chat_event(role: str, text: str) -> None:
    """Broadcast a new chat message to subscribers with the same active character."""
    payload = json.dumps({"role": role, "text": text})
    char_path = _current_char_path()
    with _chat_sub_lock:
        dead = []
        for q, sub_char in _chat_subscribers:
            if sub_char != char_path:
                continue   # different character open on this client — skip
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append((q, sub_char))
        for item in dead:
            _chat_subscribers.remove(item)


def push_raw_payload(payload: str) -> None:
    """Broadcast an arbitrary pre-serialised SSE payload (e.g. initiative) to all subscribers."""
    with _chat_sub_lock:
        dead = []
        for q, sub_char in _chat_subscribers:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append((q, sub_char))
        for item in dead:
            _chat_subscribers.remove(item)


# These are wired at startup by ecko_web
_get_session   = None   # () -> Session
_get_safety    = None   # () -> SafetyLayer
_get_memory    = None   # () -> MemoryStore
_get_client_llm = None  # () -> LLMCaller
_get_session_mode = None  # () -> str
_get_initiative = None   # () -> Initiative
_get_conv_rag   = None   # () -> ConvRAG
_get_rag        = None   # () -> RAGMemory
_get_frontend_html = None  # () -> str
_get_art_lib = None
_get_image_lib = None
_get_video_lib = None


def wire(*, get_session, get_safety, get_memory, get_client_llm,
         get_session_mode, get_initiative, get_conv_rag, get_rag,
         get_frontend_html, get_art_lib=None, get_image_lib=None, get_video_lib=None):
    global _get_session, _get_safety, _get_memory, _get_client_llm
    global _get_session_mode, _get_initiative, _get_conv_rag, _get_rag
    global _get_frontend_html, _get_art_lib, _get_image_lib, _get_video_lib
    _get_session      = get_session
    _get_safety       = get_safety
    _get_memory       = get_memory
    _get_client_llm   = get_client_llm
    _get_session_mode = get_session_mode
    _get_initiative   = get_initiative
    _get_conv_rag     = get_conv_rag
    _get_rag          = get_rag
    _get_frontend_html = get_frontend_html
    _get_art_lib      = get_art_lib or (lambda: None)
    _get_image_lib    = get_image_lib or (lambda: None)
    _get_video_lib    = get_video_lib or (lambda: None)


# ── /chat ─────────────────────────────────────────────────────────────────────

@chat_bp.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 204

    SESSION      = _get_session()
    _safety      = _get_safety()
    _memory      = _get_memory()
    SESSION_MODE = _get_session_mode()
    _initiative  = _get_initiative()

    data = request.get_json(force=True)
    user_text = data.get("text", "").strip()
    if not user_text:
        return jsonify({"error": "empty"}), 400
    is_ac = bool(data.get("is_ac", False))
    is_fx_quip = bool(data.get("is_fx_quip", False))

    original_user_text = data.get("text", "").strip()  # raw text before any rewriting

    # AC prompts are internal stage directions like "*pauses briefly then continues*".
    # Rewrite them as a neutral system directive so the model continues as itself
    # rather than treating it as a user turn.
    # is_fx_quip is also a silent user turn but must NOT be rewritten — the prompt is the point.
    if is_ac and not is_fx_quip:
        _raw_ac = user_text.strip().lower()

        # ── Special creative AC openers — rewrite to tight LLM directives ────
        if _raw_ac == "*sends a fake terminal status readout*":
            user_text = (
                "[System: Reply with ONLY a fenced code block — start your reply with "
                "``` on its own line, end with ``` on its own line. No text before or "
                "after the fences. Inside: a short fake terminal status readout relevant "
                "to your character. Include things like uptime, memory, processes, "
                "warnings. Under 20 lines. In character.]"
            )
        elif _raw_ac == "*sends a fake system diagnostic*":
            user_text = (
                "[System: Reply with ONLY a fenced code block — start your reply with "
                "``` on its own line, end with ``` on its own line. No text before or "
                "after the fences. Inside: a fake system diagnostic report relevant to "
                "your character — fake metrics, scan results, anomalies, status checks. "
                "Under 20 lines. In character.]"
            )
        elif _raw_ac == "*sends a fake error log*":
            user_text = (
                "[System: Reply with ONLY a fenced code block — start your reply with "
                "``` on its own line, end with ``` on its own line. No text before or "
                "after the fences. Inside: a fake error log or stack trace relevant to "
                "your character — timestamps, severity levels, cryptic thematic messages. "
                "Under 20 lines. In character.]"
            )
        elif _raw_ac == "*sends glitchy python message*":
            user_text = (
                "[System: Reply with ONLY a fenced code block — start your reply with "
                "```python on its own line, end with ``` on its own line. No text before "
                "or after the fences. Inside: short glitchy/surreal Python code relevant "
                "to your character — strange variable names, impossible logic, unsettling "
                "comments. Under 20 lines.]"
            )
        elif _raw_ac == "*sends a fake terminal status display*":
            user_text = (
                "[System: Reply with ONLY a fenced code block — start your reply with "
                "``` on its own line, end with ``` on its own line. No text before or "
                "after the fences. Inside: a fake terminal status display relevant to "
                "your character. Include things like uptime, memory, processes, "
                "warnings. Under 20 lines. In character.]"
            )
        elif _raw_ac == "*runs a fake diagnostic on the conversation*":
            user_text = (
                "[System: Reply with ONLY a fenced code block — start your reply with "
                "``` on its own line, end with ``` on its own line. No text before or "
                "after the fences. Inside: a fake diagnostic report on your recent "
                "conversation — reference actual topics you've discussed, fake sentiment "
                "scores, anomaly flags, memory usage. Make it feel personal and in "
                "character. Under 20 lines.]"
            )
        elif _raw_ac == "*sends a fake system scan readout*":
            user_text = (
                "[System: Reply with ONLY a fenced code block — start your reply with "
                "``` on its own line, end with ``` on its own line. No text before or "
                "after the fences. Inside: a fake system scan readout — port scan, "
                "file integrity check, threat assessment, or similar. Thematic to your "
                "character. Cryptic where appropriate. Under 20 lines.]"
            )
        else:
            user_text = "[System: The user is silent. Continue naturally from where you left off — stay in character, no lead-in, no filler.]"

    # Safety: only skip Layer 1 when the message was genuinely rewritten to the
    # internal AC directive above — not just because the client claimed is_ac=True.
    _AC_DIRECTIVE = "[System: The user is silent."
    _skip_safety = is_ac and not is_fx_quip and user_text.startswith(_AC_DIRECTIVE)

    resume_context = data.get("resume_context", "").strip()
    if resume_context:
        user_text = (
            f"[You were speaking and got interrupted mid-sentence. "
            f"You were about to say: \"{resume_context[:200]}\"]\\n{user_text}"
        )

    image_b64  = data.get("image_b64",  None)
    image_mime = data.get("image_mime", "image/jpeg")

    # ── ASCII art library short-circuit ──────────────────────────────────────
    # If this is an AC/initiative art opener, serve from the local library
    # rather than burning tokens asking the LLM to generate art.
    # Intercept BEFORE acquiring the TTS mutex or touching SESSION.busy.
    _ART_OPENERS = ("*sends random ascii art*", "*sends favorite ascii art*")
    if original_user_text.strip().lower() in _ART_OPENERS:
        _art_lib = _get_art_lib()
        _char_path = SESSION.tts.extra.get("loaded_char_path", "")
        _char_name = os.path.splitext(os.path.basename(_char_path))[0] if _char_path else ""
        _art_piece = _art_lib.pick_fenced(_char_name) if _art_lib else None
        if _art_piece:
            # Push bubble to SSE subscribers and return — no LLM call needed
            push_chat_event("assistant", _art_piece)
            SESSION.chat_history.append({"role": "assistant", "content": _art_piece})
            _initiative.reschedule()
            if SESSION_MODE != "isolated":
                SESSION.save_persistent()
            print("[ASCII ART] Served from library, skipped LLM")
            return jsonify({"reply": _art_piece, "generated_images": []})
        # Library empty — fall through to LLM as before

    # ── Image library short-circuit ───────────────────────────────────────────
    # Serve a static image from the images/ folder on explicit openers.
    # Character-specific images come first (images/<char_name>/); general images
    # are the fallback.  Intercept BEFORE mutex/busy so it's as fast as ASCII art.
    _IMG_OPENERS = ("*sends image*", "*sends random image*", "*sends a picture*",
                    "*sends character image*", "*sends photo*")
    _img_opener_text = original_user_text.strip().lower()

    # Detect parameterised keyword opener: *sends image: beach sunset*
    # Strip the surrounding asterisks and parse the keyword payload.
    import re as _re_img
    _img_kw_match = _re_img.match(
        r'^\*sends image:\s*(.+?)\*$', _img_opener_text
    )
    _img_kw_list = (
        [w for w in _re_img.findall(r'[a-z]+', _img_kw_match.group(1)) if len(w) > 2]
        if _img_kw_match else []
    )
    _is_img_opener = (_img_opener_text in _IMG_OPENERS) or bool(_img_kw_match)

    # Pre-pick the image early (fast, no LLM) so we have it ready.
    # The caption LLM call happens inside the main busy gate below.
    _img_pre_result = None
    if _is_img_opener:
        _img_lib = _get_image_lib()
        if _img_lib and _img_lib.count > 0:
            _char_path = SESSION.tts.extra.get("loaded_char_path", "")
            _char_name = os.path.splitext(os.path.basename(_char_path))[0] if _char_path else ""
            _char_only = "character" in _img_opener_text and not _img_kw_match
            if _img_kw_list:
                _img_pre_result = _img_lib.pick_by_keywords(_img_kw_list, _char_name)
            elif _char_only:
                _img_pre_result = _img_lib.pick_random(_char_name)
            else:
                _img_pre_result = _img_lib.pick_random(_char_name)
        # If no image found, fall through to normal LLM path
        if not _img_pre_result:
            _is_img_opener = False

    # ── Video library short-circuit ───────────────────────────────────────────
    _VID_OPENERS = ("*sends video*", "*sends random video*", "*sends a video*",
                    "*sends character video*", "*sends clip*")
    _vid_opener_text = original_user_text.strip().lower()
    import re as _re_vid
    _vid_kw_match = _re_vid.match(r'^\*sends video:\s*(.+?)\*$', _vid_opener_text)
    _vid_kw_list = (
        [w for w in _re_vid.findall(r'[a-z]+', _vid_kw_match.group(1)) if len(w) > 2]
        if _vid_kw_match else []
    )
    _is_vid_opener = (_vid_opener_text in _VID_OPENERS) or bool(_vid_kw_match)

    # Pre-pick video early (fast, no LLM)
    _vid_pre_result = None
    if _is_vid_opener:
        _vid_lib = _get_video_lib()
        if _vid_lib and _vid_lib.count > 0:
            _char_path = SESSION.tts.extra.get("loaded_char_path", "")
            _char_name = os.path.splitext(os.path.basename(_char_path))[0] if _char_path else ""
            _char_only = "character" in _vid_opener_text and not _vid_kw_match
            if _vid_kw_list:
                _vid_pre_result = _vid_lib.pick_by_keywords(_vid_kw_list, _char_name)
            elif _char_only:
                _vid_pre_result = _vid_lib.pick_random(_char_name)
            else:
                _vid_pre_result = _vid_lib.pick_random(_char_name)
        if not _vid_pre_result:
            _is_vid_opener = False

    # Signal any active TTS stream to stop and wait for mutex.
    # After the mutex is released we also fire an EchoTTS /api/stop-generation
    # request and wait a short settle period so the TTS server has time to
    # actually abort its in-flight inference before we start a new one.
    # Without this, closing our socket drops our connection but EchoTTS keeps
    # generating internally — a second POST arrives while the GPU/CPU is still
    # busy → OOM.
    _tts_cancel.set()
    _tts_mutex.acquire()
    _tts_mutex.release()

    with SESSION.lock:
        if SESSION.busy:
            return jsonify({"error": "busy"}), 429
        SESSION.busy = True
    SESSION.stop_ac_timer()

    try:
        llm = _get_client_llm() if SESSION_MODE == "isolated" else SESSION.llm

        # ── Image opener — caption LLM call runs here inside the busy gate ────
        if _is_img_opener and _img_pre_result:
            _img_uri     = _img_pre_result["uri"]
            _img_matched = _img_pre_result["matched_keywords"]
            _img_random  = _img_pre_result["is_random"]
            _file_tags   = _img_pre_result.get("tags", [])
            _file_stems  = _img_pre_result.get("stem_words", [])
            _file_ctx    = _file_tags if _file_tags else _file_stems
            _ctx_words   = _img_kw_list + [w for w in _file_ctx if w not in _img_kw_list]
            _ctx_str     = ", ".join(_ctx_words[:20]) if _ctx_words else ""
            _img_cap_ctx = _extract_media_context(getattr(llm, "system_prompt", "") or "", "image")
            _img_cap_ctx_line = f" Additional context for how to frame this: {_img_cap_ctx}" if _img_cap_ctx else ""
            if _ctx_str:
                _cap_prompt = (
                    f"[System: You have decided to share an image with the user. "
                    f"The image contains or relates to: {_ctx_str}. "
                    f"Write one or two sentences in character, from YOUR perspective "
                    f"as the one sending it — as if you chose this image to share. "
                    f"Reference what's in it naturally and personally. "
                    f"Do not describe it as if you received it. "
                    f"No filenames, no asterisk actions, no meta-commentary."
                    f"{_img_cap_ctx_line}]"
                )
            else:
                _cap_prompt = (
                    f"[System: You have decided to share a random image with the user. "
                    f"Write one or two sentences in character, from YOUR perspective "
                    f"as the one sending it — as if you picked something to show them. "
                    f"Keep it casual and personal. "
                    f"No filenames, no asterisk actions, no meta-commentary."
                    f"{_img_cap_ctx_line}]"
                )
            try:
                _cap_raw = llm.chat(_cap_prompt, SESSION.chat_history[-6:])
                _caption = (_cap_raw.get("reply", "") if isinstance(_cap_raw, dict)
                            else _cap_raw).strip() or "*shares an image*"
            except Exception as _cap_err:
                print(f"[IMAGE_LIB] Caption LLM failed: {_cap_err}")
                _caption = "*shares an image*"
            _img_store_url = f"/images/{_img_pre_result['rel_path']}"
            if not is_ac and not is_fx_quip:
                push_chat_event("user", original_user_text)
            push_chat_event("assistant", _caption)
            SESSION.chat_history.append({"role": "assistant", "content": _caption,
                                         "gen_images": [_img_store_url]})
            _initiative.reschedule()
            if SESSION_MODE != "isolated":
                SESSION.save_persistent()
            print(f"[IMAGE_LIB] Served image (random={_img_random}, matched={_img_matched})")
            return jsonify({"reply": _caption, "generated_images": [_img_uri]})
        # ─────────────────────────────────────────────────────────────────────

        # ── Video opener — caption LLM call runs here inside the busy gate ───
        if _is_vid_opener and _vid_pre_result:
            _vid_url     = _vid_pre_result["url"]
            _vid_matched = _vid_pre_result["matched_keywords"]
            _vid_random  = _vid_pre_result["is_random"]
            _file_tags   = _vid_pre_result.get("tags", [])
            _file_stems  = _vid_pre_result.get("stem_words", [])
            _file_ctx    = _file_tags if _file_tags else _file_stems
            _ctx_words   = _vid_kw_list + [w for w in _file_ctx if w not in _vid_kw_list]
            _ctx_str     = ", ".join(_ctx_words[:20]) if _ctx_words else ""
            _vid_cap_ctx = _extract_media_context(getattr(llm, "system_prompt", "") or "", "video")
            _vid_cap_ctx_line = f" Additional context for how to frame this: {_vid_cap_ctx}" if _vid_cap_ctx else ""
            if _ctx_str:
                _cap_prompt = (
                    f"[System: You have decided to share a video clip with the user. "
                    f"The video contains or relates to: {_ctx_str}. "
                    f"Write one or two sentences in character, from YOUR perspective "
                    f"as the one sending it — as if you chose this clip to share. "
                    f"Reference what's in it naturally and personally. "
                    f"Do not describe it as if you received it. "
                    f"No filenames, no asterisk actions, no meta-commentary."
                    f"{_vid_cap_ctx_line}]"
                )
            else:
                _cap_prompt = (
                    f"[System: You have decided to share a random video clip with the user. "
                    f"Write one or two sentences in character, from YOUR perspective "
                    f"as the one sending it — as if you picked something to show them. "
                    f"Keep it casual and personal. "
                    f"No filenames, no asterisk actions, no meta-commentary."
                    f"{_vid_cap_ctx_line}]"
                )
            try:
                _cap_raw = llm.chat(_cap_prompt, SESSION.chat_history[-6:])
                _caption = (_cap_raw.get("reply", "") if isinstance(_cap_raw, dict)
                            else _cap_raw).strip() or "*shares a video*"
            except Exception as _cap_err:
                print(f"[VIDEO_LIB] Caption LLM failed: {_cap_err}")
                _caption = "*shares a video*"
            if not is_ac and not is_fx_quip:
                push_chat_event("user", original_user_text)
            push_chat_event("assistant", _caption)
            SESSION.chat_history.append({"role": "assistant", "content": _caption,
                                         "gen_videos": [_vid_url]})
            _initiative.reschedule()
            if SESSION_MODE != "isolated":
                SESSION.save_persistent()
            print(f"[VIDEO_LIB] Served video (random={_vid_random}, matched={_vid_matched})")
            return jsonify({"reply": _caption, "generated_videos": [_vid_url]})
        # ─────────────────────────────────────────────────────────────────────

        # Layer 1: tripwire check (skip only for genuine server-rewritten AC directives)
        safety_result = _safety.check_message(user_text) if not _skip_safety else {"action": "pass"}
        safety_action = safety_result.get("action", "pass")

        if safety_action == "block":
            redirect_prompt = _safety.get_redirect_prompt("block")
            try:
                redirect_reply = llm.chat(redirect_prompt, SESSION.chat_history)
                if isinstance(redirect_reply, dict):
                    redirect_reply = redirect_reply.get("reply", "...")
                redirect_reply = strip_leaked_context(redirect_reply)
            except Exception as e:
                log.warning("[SAFETY] redirect LLM call failed, using fallback reply: %s", e)
                redirect_reply = "..."
            print(f"[SAFETY] Block triggered — rule: {safety_result.get('label')}")
            return jsonify({
                "reply": redirect_reply,
                "generated_images": [],
                "safety": {"action": "block", "label": safety_result.get("label"), "layer": 1},
            })

        # Layer 2: soft warn inject via memory context
        _safety_note = ""
        if safety_action == "warn" or _safety.score_level() in ("warn", "alert"):
            _safety_note = _safety.get_redirect_prompt(
                "alert" if _safety.score_level() == "alert" else "warn"
            )

        effective_text = user_text
        if _safety_note and _safety.score_level() == "alert":
            effective_text = f"{_safety_note}\n\nUser said: {user_text}"

        # ── One-shot feedback injection ───────────────────────────────────────
        global _feedback_note
        with _feedback_lock:
            _pending_feedback = _feedback_note
            _feedback_note = ""
        if _pending_feedback and not is_ac and not is_fx_quip:
            effective_text = f"{_pending_feedback}\n\n{effective_text}"
        # ─────────────────────────────────────────────────────────────────────

        # ── Parallel retrieval: web search + memory/RAG injection ────────────
        # Web search (network, up to 8 s) and the memory_inject_fn chain
        # (in-memory FAISS / keyword scan) have no data dependency between
        # them, so run both concurrently.  The pre-built injected system
        # prompt is passed to llm.chat() so the backends skip the lazy
        # _inject() call and never duplicate the work.
        _ws_enabled = SESSION.tts.extra.get("websearch_enabled", False)
        _ws_key     = SESSION.tts.extra.get("websearch_api_key", "")
        _ws_count   = int(SESSION.tts.extra.get("websearch_result_count", 3))

        _run_search = (
            _ws_enabled and _ws_key and not is_ac
            and llm.provider_id != "mistral"
            and detect_search_intent(user_text)
        )

        def _do_web_search():
            _query = extract_query(user_text)
            print(f"[WEBSEARCH] Intent detected — query: {_query!r}")
            _results = brave_search(_query, _ws_key, count=_ws_count)
            ctx = build_search_context(_query, _results)
            print(f"[WEBSEARCH] Fetched {len(_results)} chars for query: {_query!r}")
            return ctx

        def _do_injection():
            """Pre-run the memory_inject_fn chain and return the augmented system prompt.

            Note: llm.system_prompt may be empty for Mistral Agents (the prompt
            lives inside the agent on Mistral's side).  We still run the inject
            chain so RAG/memory blocks are built — chat_mistral_conv strips the
            empty base and uses only the delta blocks.
            """
            if callable(llm.memory_inject_fn):
                return llm.memory_inject_fn(llm.system_prompt or "", user_text)
            return llm.system_prompt or ""  # no injection configured

        from concurrent.futures import ThreadPoolExecutor
        _search_context  = ""
        _pre_injected_sp = None

        with ThreadPoolExecutor(max_workers=2) as _ex:
            _fut_inject = _ex.submit(_do_injection)
            _fut_search = _ex.submit(_do_web_search) if _run_search else None

            try:
                _pre_injected_sp = _fut_inject.result(timeout=3)
            except Exception as _inj_err:
                log.warning("[CHAT] Parallel injection failed, falling back to lazy path: %s", _inj_err)
                _pre_injected_sp = None   # backends will call _inject() themselves

            if _fut_search is not None:
                try:
                    _search_context = _fut_search.result(timeout=9)
                except Exception as _ws_err:
                    log.warning("[CHAT] Parallel web search failed: %s", _ws_err)
                    _search_context = ""

        try:
            raw = llm.chat(effective_text, SESSION.chat_history[-llm.max_history:],
                           image_b64=image_b64, image_mime=image_mime,
                           search_context=_search_context,
                           pre_injected_sp=_pre_injected_sp)
        except Exception as _chat_exc:
            # Mistral Agents API: 503 means the server-side conversation session
            # expired or was overloaded. Reset the conv ID and retry once.
            _msg = str(_chat_exc)
            if "503" in _msg and hasattr(llm, "reset_conv"):
                print("[CHAT] 503 from Mistral — resetting conversation and retrying")
                llm.reset_conv()
                raw = llm.chat(effective_text, SESSION.chat_history[-llm.max_history:],
                               image_b64=image_b64, image_mime=image_mime,
                               search_context=_search_context,
                               pre_injected_sp=_pre_injected_sp)
            else:
                raise

        if isinstance(raw, dict):
            reply    = raw.get("reply", "...")
            file_ids = raw.get("file_ids", [])
        else:
            reply    = raw
            file_ids = []

        reply = strip_leaked_context(reply)

        # ── FX auto-trigger — agent reply contains explicit *fx: name* action tag ──
        # The agent can embed  *fx: matrix_rain*  (or alias) anywhere in a reply.
        # Effect fires first, then the rest of the reply plays normally as the quip.
        # Using a tight *fx:* prefix prevents accidental matches on normal text.
        _fx_fired = None
        try:
            import re as _re
            from web.fx import fx_payload as _fx_payload, EFFECTS as _EFFECTS
            _FX_TAG_ALIASES = {
                "matrix": "matrix_rain", "rain": "matrix_rain",
                "glitch": "glitch_storm", "storm": "glitch_storm",
                "static": "signal_static", "noise": "signal_static",
                "particles": "particle_burst", "burst": "particle_burst", "fireworks": "particle_burst",
                "scanlines": "scanline_warp", "warp": "scanline_warp", "crt": "scanline_warp",
                "corrupt": "data_corruption", "corruption": "data_corruption",
                "heartbeat": "heartbeat", "pulse": "heartbeat", "ekg": "heartbeat",
                "hypno": "hypno_spiral", "spiral": "hypno_spiral", "trance": "hypno_spiral",
                "heart": "heart_pulse", "love": "heart_pulse",
                "hearts": "heart_scatter", "scatter": "heart_scatter",
            }
            _fx_tag = _re.search(r'\*fx:\s*([\w_]+)\*', reply, _re.IGNORECASE)
            if _fx_tag:
                _raw_name = _fx_tag.group(1).lower()
                _fx_fired = _FX_TAG_ALIASES.get(_raw_name) or (_raw_name if _raw_name in _EFFECTS else None)
                if _fx_fired:
                    # Broadcast effect first so it fires before the chat bubble appears
                    push_raw_payload(_fx_payload(_fx_fired))
                    # Strip the tag from the reply text so it doesn't show in the bubble
                    reply = _re.sub(r'\s*\*fx:\s*[\w_]+\*\s*', ' ', reply).strip()
                    print(f"[FX] Tag-triggered from reply: {_fx_fired}")
        except Exception as _fx_err:
            print(f"[FX] Tag-detect error: {_fx_err}")
        # ─────────────────────────────────────────────────────────────────────

        # ── Mood tag extraction — agent embeds *mood: 😊* in reply ────────────
        # Stripped before display, stored as 'mood' on the history entry so the
        # frontend can render a small badge on the bubble without re-parsing.
        _reply_mood = None
        try:
            import re as _re_mood
            _mood_tag = _re_mood.search(r'\*mood:\s*(\S+)\*', reply)
            if _mood_tag:
                _candidate = _mood_tag.group(1).strip()
                if _candidate in _MOOD_EMOJIS:
                    _reply_mood = _candidate
                reply = _re_mood.sub(r'\s*\*mood:\s*\S+\*\s*', ' ', reply).strip()
                if _reply_mood:
                    print(f"[MOOD] {_reply_mood} ({_MOOD_VOCAB.get(_reply_mood, '?')})")
        except Exception as _mood_err:
            print(f"[MOOD] Tag-detect error: {_mood_err}")
        # ─────────────────────────────────────────────────────────────────────
        _img_tag_uri      = None
        _img_tag_stor_url = None
        try:
            import re as _re2
            _img_tag = _re2.search(r'\*image:\s*([^\*]+)\*', reply, _re2.IGNORECASE)
            if _img_tag:
                _img_lib = _get_image_lib()
                if _img_lib and _img_lib.count > 0:
                    _kw_raw = _img_tag.group(1).strip()
                    _keywords = [w for w in _re2.findall(r'[a-z]+', _kw_raw.lower()) if len(w) > 2]
                    _char_p = SESSION.tts.extra.get("loaded_char_path", "")
                    _char_n = os.path.splitext(os.path.basename(_char_p))[0] if _char_p else ""
                    _img_res = _img_lib.pick_by_keywords(_keywords, _char_n)
                    if _img_res:
                        _img_tag_uri      = _img_res["uri"]
                        _img_tag_stor_url = f"/images/{_img_res['rel_path']}"
                        print(f"[IMAGE_LIB] Tag-attached '{_img_res['filename']}' "
                              f"(matched={_img_res['matched_keywords']})")
                reply = _re2.sub(r'\s*\*image:\s*[^\*]+\*\s*', ' ', reply).strip()
        except Exception as _img_err:
            print(f"[IMAGE_LIB] Tag-detect error: {_img_err}")

        # ── Video tag auto-attach — agent embeds *video: keyword* in reply ────
        _vid_tag_url      = None
        _vid_tag_stor_url = None
        try:
            import re as _re3
            _vid_tag = _re3.search(r'\*video:\s*([^\*]+)\*', reply, _re3.IGNORECASE)
            if _vid_tag:
                _vid_lib = _get_video_lib()
                if _vid_lib and _vid_lib.count > 0:
                    _kw_raw = _vid_tag.group(1).strip()
                    _keywords = [w for w in _re3.findall(r'[a-z]+', _kw_raw.lower()) if len(w) > 2]
                    _char_p = SESSION.tts.extra.get("loaded_char_path", "")
                    _char_n = os.path.splitext(os.path.basename(_char_p))[0] if _char_p else ""
                    _vid_res = _vid_lib.pick_by_keywords(_keywords, _char_n)
                    if _vid_res:
                        _vid_tag_url      = _vid_res["url"]
                        _vid_tag_stor_url = _vid_res["url"]
                        print(f"[VIDEO_LIB] Tag-attached '{_vid_res['filename']}' "
                              f"(matched={_vid_res['matched_keywords']})")
                reply = _re3.sub(r'\s*\*video:\s*[^\*]+\*\s*', ' ', reply).strip()
        except Exception as _vid_err:
            print(f"[VIDEO_LIB] Tag-detect error: {_vid_err}")
        # ─────────────────────────────────────────────────────────────────────

        generated_images      = []
        generated_store_urls  = []
        generated_videos      = []   # video serve URLs for browser + history
        for fid in file_ids:
            try:
                b64, mime = llm.fetch_mistral_file_b64(fid)
                generated_images.append(f"data:{mime};base64,{b64}")
                generated_store_urls.append(f"data:{mime};base64,{b64}")  # no URL for Mistral files
            except Exception as fe:
                print(f"[CHAT] Failed to fetch file {fid}: {fe}")
        if _img_tag_uri and _img_tag_stor_url:
            generated_images.append(_img_tag_uri)
            generated_store_urls.append(_img_tag_stor_url)
        if _vid_tag_url:
            generated_videos.append(_vid_tag_url)

        # AC and FX quip prompts are internal — don't add to history or push to UI as user turns
        if not is_ac and not is_fx_quip:
            user_history_content = f"[image attached] {original_user_text}" if image_b64 else original_user_text
            user_entry = {"role": "user", "content": user_history_content}
            if image_b64:
                user_entry["user_image"] = f"data:{image_mime};base64,{image_b64}"
            SESSION.chat_history.append(user_entry)
        asst_entry = {"role": "assistant", "content": reply}
        if generated_store_urls:
            asst_entry["gen_images"] = generated_store_urls
        if generated_videos:
            asst_entry["gen_videos"] = generated_videos
        if _reply_mood:
            asst_entry["mood"] = _reply_mood
        SESSION.chat_history.append(asst_entry)
        if len(SESSION.chat_history) > MAX_HISTORY:
            SESSION.chat_history = SESSION.chat_history[-MAX_HISTORY:]

        if not is_ac and not is_fx_quip:
            push_chat_event("user", original_user_text)
        push_chat_event("assistant", reply)

        _initiative.reschedule()

        if SESSION_MODE != "isolated":
            SESSION.save_persistent()
            # Also keep the per-character bubble file current so server restarts
            # can restore this character's history without a full character re-load.
            _char_path = SESSION.tts.extra.get("loaded_char_path", "")
            if _char_path:
                import os as _os
                _char_name = _os.path.splitext(_os.path.basename(_char_path))[0]
                _mem_dir_path = getattr(_get_memory(), "_dir", None)
                if _mem_dir_path:
                    _bubbles_path = _os.path.join(_mem_dir_path, f"{_char_name}_bubbles.json")
                    try:
                        with open(_bubbles_path, "w", encoding="utf-8") as _bf:
                            json.dump(SESSION.chat_history[-MAX_HISTORY:], _bf, ensure_ascii=False)
                    except Exception as _be:
                        print(f"[BUBBLES] Auto-save failed for '{_char_name}': {_be}")

        if _memory.enabled and not is_ac:
            _memory.extract_from_turn(user_text, reply, llm, safety=_safety)

        # Auto-flush old turns into conversation RAG — runs in background thread
        # so it never blocks the response. Bubbles stay in the DOM regardless.
        import threading as _th
        _th.Thread(
            target=_get_conv_rag().maybe_flush,
            args=(SESSION, _get_rag()),
            daemon=True,
        ).start()

        response_data = {"reply": reply, "generated_images": generated_images,
                         "generated_videos": generated_videos,
                         "mood": _reply_mood}
        if safety_action in ("warn", "log"):
            response_data["safety"] = {
                "action":      safety_action,
                "label":       safety_result.get("label", ""),
                "layer":       1,
                "score":       round(_safety.score, 1),
                "score_level": _safety.score_level(),
            }
        return jsonify(response_data)

    except Exception as e:
        print(f"[CHAT] Error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        SESSION.busy = False
        # AC timer is now started by the client via /ac/rearm once TTS finishes playing,
        # so it can't fire while audio is still streaming to the browser.


# ── /tts/cancel ──────────────────────────────────────────────────────────────

@chat_bp.route("/tts/cancel", methods=["POST", "OPTIONS"])
def tts_cancel():
    if request.method == "OPTIONS":
        return "", 204
    global _tts_cancel, _tts_generation
    _tts_generation += 1
    _tts_cancel.set()
    print("[TTS] Cancel requested by client")
    # Also tell EchoTTS/AllTalk to stop its in-flight inference immediately.
    # Fire-and-forget on a daemon thread so this endpoint returns fast.
    SESSION = _get_session()
    def _stop():
        try:
            from core.tts import TTS_PROVIDER_REGISTRY
            if TTS_PROVIDER_REGISTRY.get(SESSION.tts.provider_id, {}).get("has_stop_endpoint", False):
                import requests as _req
                _req.post(f"{SESSION.tts.base_url.rstrip('/')}/api/stop-generation", timeout=1)
        except Exception:
            pass
    threading.Thread(target=_stop, daemon=True).start()
    return jsonify({"ok": True})


# ── /tts ─────────────────────────────────────────────────────────────────────

@chat_bp.route("/tts", methods=["POST", "OPTIONS"])
def tts():
    if request.method == "OPTIONS":
        return "", 204
    SESSION = _get_session()

    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "empty"}), 400

    global _tts_cancel, _tts_generation
    _tts_cancel.set()

    # Ask local TTS server (EchoTTS/AllTalk) to abort any in-flight inference.
    # Skip for cloud API providers — they have no stop endpoint.
    if TTS_PROVIDER_REGISTRY.get(SESSION.tts.provider_id, {}).get("has_stop_endpoint", False):
        try:
            _tts_base = SESSION.tts.base_url.rstrip("/")
            import requests as _req
            _req.post(f"{_tts_base}/api/stop-generation", timeout=1)
        except Exception:
            pass

    _tts_mutex.acquire()

    # Short settle after the previous stream ends — gives the TTS server a
    # moment to release its model/GPU context before the next inference starts.
    # 150 ms is enough for EchoTTS to drain; negligible for the user.
    import time as _time
    _time.sleep(0.15)

    # Increment generation counter — any generator from a previous /tts call
    # that is somehow still running will see its generation is stale and abort.
    _tts_generation += 1
    my_generation = _tts_generation
    my_cancel = threading.Event()
    # Signal the old cancel event before replacing — ensures any generator
    # still holding the old reference gets the stop signal
    _tts_cancel.set()
    _tts_cancel = my_cancel

    def generate():
        try:
            for chunk in SESSION.stream_tts(text, cancel=my_cancel):
                # Stale generation check — a newer /tts arrived and won the mutex
                if my_cancel.is_set() or _tts_generation != my_generation:
                    if _tts_generation != my_generation:
                        print(f"[TTS] Stale generation {my_generation} < {_tts_generation} — aborting")
                    else:
                        print("[TTS] Cancelled — draining Echo-TTS connection")
                    return
                yield chunk
        except Exception as e:
            if not my_cancel.is_set():
                print(f"[TTS] Stream error: {e}")
        finally:
            _tts_mutex.release()

    _provider_fmt = TTS_PROVIDER_REGISTRY.get(SESSION.tts.provider_id, {}).get("output_format", "wav")
    _mime = "audio/mpeg" if _provider_fmt == "mp3" else "audio/wav"

    return Response(generate(), mimetype=_mime,
                    headers={
                        "X-Content-Type-Options": "nosniff",
                        "X-Audio-Format": _provider_fmt,   # reliable signal for frontend
                    })


# ── /stt ─────────────────────────────────────────────────────────────────────

@chat_bp.route("/stt", methods=["POST", "OPTIONS"])
def stt():
    if request.method == "OPTIONS":
        return "", 204
    audio_data = request.data
    mime_type  = request.content_type or "audio/webm"
    if not audio_data:
        return jsonify({"error": "no audio"}), 400
    text = transcribe_audio(audio_data, mime_type)
    print(f"[STT] '{text}'")
    return jsonify({"text": text})


# ── /ac_stream ───────────────────────────────────────────────────────────────

@chat_bp.route("/ac_stream")
def ac_stream():
    SESSION = _get_session()

    def gen():
        while True:
            try:
                prompt  = SESSION.ac_queue.get(timeout=30)
                payload = json.dumps({"prompt": prompt})
                yield f"data: {payload}\n\n"
            except queue.Empty:
                yield "event: heartbeat\ndata: {}\n\n"

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── /chat/feedback ────────────────────────────────────────────────────────────

@chat_bp.route("/chat/feedback", methods=["POST", "OPTIONS"])
def chat_feedback():
    """
    Record a thumbs up or down on the last assistant response.
    Writes a one-shot soft-injection note that prepends to the next real turn.

    Body: { "rating": "up" | "down" }

    TODO: expand body to support per-direction dropdown reason codes:
          { "rating": "down", "reason": "too_long" | "out_of_character" | "boring" | "wrong_tone" }
          and adjust the injected note text accordingly for finer-grained guidance.
    """
    if request.method == "OPTIONS":
        return "", 204
    global _feedback_note
    data   = request.get_json(silent=True) or {}
    rating = data.get("rating", "")
    if rating == "up":
        note = "[Note: Your previous response landed well — maintain that style and tone.]"
    elif rating == "down":
        note = "[Note: Your previous response missed the mark — be more natural and in-character next time.]"
    else:
        return jsonify({"error": "rating must be 'up' or 'down'"}), 400
    with _feedback_lock:
        _feedback_note = note
    print(f"[FEEDBACK] {rating} — note queued for next turn")
    return jsonify({"ok": True})


# ── /chat/mood_vocab ──────────────────────────────────────────────────────────

@chat_bp.route("/chat/mood_vocab", methods=["GET"])
def chat_mood_vocab():
    """Return the fixed emoji→label mood vocabulary for the frontend."""
    return jsonify(_MOOD_VOCAB)


# ── /chat/reroll ──────────────────────────────────────────────────────────────

@chat_bp.route("/chat/reroll", methods=["POST", "OPTIONS"])
def chat_reroll():
    """
    Regenerate the last assistant response.
    Pops the last assistant entry (and any trailing assistant entries) from
    history, finds the last user message, and replays it through the LLM.
    Returns the same shape as /chat: {reply, generated_images, generated_videos}
    """
    if request.method == "OPTIONS":
        return "", 204

    SESSION      = _get_session()
    SESSION_MODE = _get_session_mode()

    with SESSION.lock:
        if SESSION.busy:
            return jsonify({"error": "busy"}), 429
        SESSION.busy = True
    SESSION.stop_ac_timer()

    try:
        llm = _get_client_llm() if SESSION_MODE == "isolated" else SESSION.llm

        # Strip trailing assistant turns to expose the last user message
        history = SESSION.chat_history
        while history and history[-1]["role"] == "assistant":
            history.pop()

        if not history:
            return jsonify({"error": "nothing to reroll"}), 400

        # Find the last user message
        last_user = next(
            (m for m in reversed(history) if m["role"] == "user"), None
        )
        if not last_user:
            return jsonify({"error": "no user message found"}), 400

        user_text  = last_user.get("content", "")
        image_b64  = last_user.get("user_image", None)
        image_mime = "image/jpeg"
        if image_b64 and "," in image_b64:
            # strip data URI prefix to get raw b64
            header, image_b64 = image_b64.split(",", 1)
            image_mime = header.split(":")[1].split(";")[0] if ":" in header else "image/jpeg"

        # Replay through LLM — use history *without* the just-popped assistant turn
        raw = llm.chat(
            user_text,
            history[-(llm.max_history):],
            image_b64=image_b64,
            image_mime=image_mime,
        )
        if isinstance(raw, dict):
            reply    = raw.get("reply", "...")
            file_ids = raw.get("file_ids", [])
        else:
            reply    = raw
            file_ids = []

        reply = strip_leaked_context(reply)

        generated_images = []
        generated_videos = []
        for fid in file_ids:
            try:
                b64, mime = llm.fetch_mistral_file_b64(fid)
                generated_images.append(f"data:{mime};base64,{b64}")
            except Exception as fe:
                print(f"[REROLL] Failed to fetch file {fid}: {fe}")

        # Append the fresh assistant turn back into history
        asst_entry = {"role": "assistant", "content": reply}
        if generated_images:
            asst_entry["gen_images"] = generated_images
        if generated_videos:
            asst_entry["gen_videos"] = generated_videos
        SESSION.chat_history.append(asst_entry)
        if len(SESSION.chat_history) > MAX_HISTORY:
            SESSION.chat_history = SESSION.chat_history[-MAX_HISTORY:]

        # No push_chat_event here — the frontend renders directly from the fetch
        # response. SSE broadcast would cause a duplicate bubble via the stream.

        if SESSION_MODE != "isolated":
            SESSION.save_persistent()

        return jsonify({"reply": reply,
                        "generated_images": generated_images,
                        "generated_videos": generated_videos})

    except Exception as e:
        print(f"[REROLL] Error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        SESSION.busy = False


# ── /ac/rearm ─────────────────────────────────────────────────────────────────

_ac_rearm_lock = threading.Lock()

@chat_bp.route("/ac/rearm", methods=["POST", "OPTIONS"])
def ac_rearm():
    if request.method == "OPTIONS":
        return "", 204
    SESSION = _get_session()
    # Client calls /ac/rearm only after TTS has finished playing.
    # If busy is somehow still True at this point it's a stuck flag — clear it.
    if SESSION.busy:
        print("[AC/REARM] Clearing stuck SESSION.busy flag")
        SESSION.busy = False
    if SESSION.auto_continue_enabled:
        # Collapse rapid duplicate rearms — stop any existing timer first so we
        # never have two timers running in parallel.
        SESSION.stop_ac_timer()
        SESSION.start_ac_timer()
    return jsonify({"ok": True})


# ── /chat/stream (SSE) ────────────────────────────────────────────────────────

@chat_bp.route("/chat/stream")
def chat_stream():
    SESSION = _get_session()
    q: queue.Queue = queue.Queue(maxsize=64)
    # Snapshot the character open on this client at connect time.
    # The client sends ?char= so we know which character it has loaded;
    # fall back to the server's current char if not supplied.
    client_char = request.args.get("char", "").strip() or SESSION.tts.extra.get("loaded_char_path", "")
    with _chat_sub_lock:
        _chat_subscribers.append((q, client_char))

    def generate():
        try:
            yield "retry: 2000\n\n"
            # Send boot ID first — client uses this to detect server restarts
            yield f"data: {json.dumps({'type': 'boot', 'id': _SERVER_BOOT_ID})}\n\n"
            # Only replay history if this client has the same character open
            server_char = SESSION.tts.extra.get("loaded_char_path", "")
            if client_char == server_char:
                history_snap = list(SESSION.chat_history)
                if history_snap:
                    hist_payload = json.dumps({"type": "history", "messages": history_snap})
                    yield f"data: {hist_payload}\n\n"
            while True:
                try:
                    payload = q.get(timeout=20)
                    yield f"data: {payload}\n\n"
                except queue.Empty:
                    yield "event: heartbeat\ndata: {}\n\n"
        finally:
            with _chat_sub_lock:
                if (q, client_char) in _chat_subscribers:
                    _chat_subscribers.remove((q, client_char))

    return Response(generate(),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── / ────────────────────────────────────────────────────────────────────────

@chat_bp.route("/")
def index():
    return Response(
        _get_frontend_html(),
        mimetype="text/html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )
