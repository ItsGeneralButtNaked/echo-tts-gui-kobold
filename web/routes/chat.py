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
import queue
import threading
import time

from core.logger import log

_SERVER_BOOT_ID = str(int(time.time()))

from flask import Blueprint, Response, jsonify, request

from core.session import MAX_HISTORY
from core.stt import transcribe_audio
from core.websearch import detect_search_intent, extract_query, brave_search, build_search_context
from web.sanitise import strip_leaked_context

chat_bp = Blueprint("chat", __name__)

# ── TTS serialisation ─────────────────────────────────────────────────────────
# Only one TTS inference runs at a time. _tts_cancel signals the active stream
# to drain and exit ASAP; _tts_mutex gates the next request until the socket
# is fully closed.
_tts_cancel = threading.Event()
_tts_mutex  = threading.Lock()

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


def wire(*, get_session, get_safety, get_memory, get_client_llm,
         get_session_mode, get_initiative, get_conv_rag, get_rag,
         get_frontend_html):
    global _get_session, _get_safety, _get_memory, _get_client_llm
    global _get_session_mode, _get_initiative, _get_conv_rag, _get_rag
    global _get_frontend_html
    _get_session      = get_session
    _get_safety       = get_safety
    _get_memory       = get_memory
    _get_client_llm   = get_client_llm
    _get_session_mode = get_session_mode
    _get_initiative   = get_initiative
    _get_conv_rag     = get_conv_rag
    _get_rag          = get_rag
    _get_frontend_html = get_frontend_html


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
        generated_images = []
        for fid in file_ids:
            try:
                b64, mime = llm.fetch_mistral_file_b64(fid)
                generated_images.append(f"data:{mime};base64,{b64}")
            except Exception as fe:
                print(f"[CHAT] Failed to fetch file {fid}: {fe}")

        # AC and FX quip prompts are internal — don't add to history or push to UI as user turns
        if not is_ac and not is_fx_quip:
            # Use original_user_text (before resume_context rewrite) so the bubble
            # shows what the user actually said, not the internal directive prefix.
            user_history_content = f"[image attached] {original_user_text}" if image_b64 else original_user_text
            user_entry = {"role": "user", "content": user_history_content}
            if image_b64:
                user_entry["user_image"] = f"data:{image_mime};base64,{image_b64}"
            SESSION.chat_history.append(user_entry)
        asst_entry = {"role": "assistant", "content": reply}
        if generated_images:
            asst_entry["gen_images"] = generated_images
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

        response_data = {"reply": reply, "generated_images": generated_images}
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
    global _tts_cancel
    _tts_cancel.set()
    print("[TTS] Cancel requested by client")
    # Also tell EchoTTS/AllTalk to stop its in-flight inference immediately.
    # Fire-and-forget on a daemon thread so this endpoint returns fast.
    SESSION = _get_session()
    def _stop():
        try:
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

    global _tts_cancel
    _tts_cancel.set()

    # Ask EchoTTS/AllTalk to abort any in-flight inference before we acquire
    # the mutex — this gives the TTS server the maximum lead time to stop, so
    # by the time we actually POST /v1/audio/speech below the previous job is
    # more likely to have cleared.  Fire-and-forget; failure is non-fatal.
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

    my_cancel = threading.Event()
    _tts_cancel = my_cancel

    def generate():
        try:
            for chunk in SESSION.stream_tts(text, cancel=my_cancel):
                if my_cancel.is_set():
                    print("[TTS] Cancelled — draining Echo-TTS connection")
                    return
                yield chunk
        except Exception as e:
            if not my_cancel.is_set():
                print(f"[TTS] Stream error: {e}")
        finally:
            _tts_mutex.release()

    return Response(generate(), mimetype="audio/wav",
                    headers={"X-Content-Type-Options": "nosniff"})


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


# ── /ac/rearm ─────────────────────────────────────────────────────────────────

_ac_rearm_lock = threading.Lock()

@chat_bp.route("/ac/rearm", methods=["POST", "OPTIONS"])
def ac_rearm():
    if request.method == "OPTIONS":
        return "", 204
    SESSION = _get_session()
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
