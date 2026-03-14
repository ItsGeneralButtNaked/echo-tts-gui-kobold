"""
web/frontend.py — Ecko single-page frontend HTML.
"""

FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>Ecko</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600&display=swap');
  :root {
    --hue:140;
    --green:hsl(var(--hue),100%,64%);
    --green-dim:hsl(var(--hue),54%,40%);
    --green-dark:hsl(var(--hue),47%,33%);
    --green-glow:hsla(var(--hue),100%,64%,0.18);
    --bg:#0e0e0e; --panel:#141414;
    --card:hsl(var(--hue),8%,11%);
    --border:hsl(var(--hue),32%,17%);
    --text:hsl(var(--hue),100%,90%);
    --text-dim:hsl(var(--hue),16%,42%);
    --tint-bg:hsl(var(--hue),47%,15%);
    --tint-dark:hsl(var(--hue),40%,10%);
    --tint-slot:hsl(var(--hue),40%,7%);
    --tint-slot-border:hsl(var(--hue),38%,22%);
    --danger:#ff4c4c; --font-mono:'Share Tech Mono',monospace; --font-ui:'Rajdhani',sans-serif;
  }
  *{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
  html,body{height:100%;background:var(--bg);color:var(--text);font-family:var(--font-ui);font-size:15px;overflow:hidden}
  body::before{content:'';position:fixed;inset:0;background:repeating-linear-gradient(to bottom,transparent 0px,transparent 3px,rgba(0,0,0,0.07) 3px,rgba(0,0,0,0.07) 4px);pointer-events:none;z-index:999}
  #app{display:flex;flex-direction:column;height:100dvh;max-width:640px;margin:0 auto;padding:10px 10px 6px;gap:8px}
  #header{display:flex;align-items:center;gap:10px;flex-shrink:0}
  #logo{font-family:var(--font-mono);font-size:22px;color:var(--green);letter-spacing:4px;text-shadow:0 0 14px var(--green);flex-shrink:0}
  .status-dot{width:8px;height:8px;border-radius:50%;background:#333;flex-shrink:0;transition:background .4s,box-shadow .4s}
  .status-dot.on{background:var(--green);box-shadow:0 0 6px var(--green)}
  .status-dot.err{background:var(--danger)}
  .status-label{font-size:11px;color:var(--text-dim);font-family:var(--font-mono);letter-spacing:1px;margin-right:4px}
  #provider-tag{font-family:var(--font-mono);font-size:11px;color:var(--green-dim);letter-spacing:1px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:160px}
  #header-right{margin-left:auto;display:flex;gap:6px;align-items:center}
  #wave-wrap{flex-shrink:0;background:var(--card);border:1px solid var(--border);border-radius:14px;overflow:hidden;position:relative;transition:height .25s,opacity .25s}
  #wave-wrap.wave-hidden{height:0!important;opacity:0;pointer-events:none;border:none;margin:0}
  #wave{width:100%;height:110px;display:block}
  #wave-mode-btn{position:absolute;bottom:6px;right:8px;font-family:var(--font-mono);font-size:9px;color:var(--text-dim);background:none;border:none;cursor:pointer;letter-spacing:1px;padding:2px 4px}
  #wave-mode-btn:hover{color:var(--green)}
  #wave-toggle-btn{font-family:var(--font-mono);font-size:9px;color:var(--text-dim);background:var(--card);border:1px solid var(--border);border-radius:6px 6px 0 0;cursor:pointer;letter-spacing:1px;padding:2px 8px;align-self:flex-end;flex-shrink:0;margin-bottom:-1px;position:relative;z-index:2}
  #wave-toggle-btn:hover{color:var(--green);border-color:var(--green)}
  #playing-indicator{display:none;position:absolute;top:8px;left:50%;transform:translateX(-50%);font-family:var(--font-mono);font-size:10px;color:var(--green);letter-spacing:2px;text-shadow:0 0 8px var(--green);animation:blink .8s step-end infinite;cursor:pointer;user-select:none}
  #playing-indicator.show{display:block}
  #playing-indicator.perm-hidden{display:none!important}@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
  #playing-indicator.show{display:block}
  #safety-light{background:#2f9d57;box-shadow:none}
  #safety-light.notice{background:#ffaa00;box-shadow:0 0 6px #ffaa00}
  #safety-light.warn{background:#ff6600;box-shadow:0 0 8px #ff6600}
  #safety-light.alert{background:#ff2222;box-shadow:0 0 12px #ff2222;animation:blink .6s step-end infinite}
  #lightbox{display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:9999;align-items:center;justify-content:center;cursor:zoom-out}
  #lightbox.show{display:flex}
  #lightbox img{max-width:92vw;max-height:92vh;border-radius:8px;box-shadow:0 0 40px var(--green-glow);object-fit:contain}
  #chat{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:8px;padding:2px 2px 4px;overscroll-behavior:contain}
  #chat::-webkit-scrollbar{width:3px}
  #chat::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
  .bubble{max-width:88%;padding:9px 14px;border-radius:14px;font-size:14px;line-height:1.5;word-break:break-word;animation:bubble-in .18s ease}
  @keyframes bubble-in{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
  .bubble.user{align-self:flex-end;background:var(--tint-bg);border:1px solid var(--green-dark);color:var(--green);font-family:var(--font-mono);font-size:13px}
  .bubble.assistant{align-self:flex-start;background:var(--card);border:1px solid var(--border);color:var(--text)}
  .bubble.thinking{align-self:flex-start;background:var(--card);border:1px solid var(--border);color:var(--text-dim);font-family:var(--font-mono);font-size:12px}
  .dot-pulse::after{content:'...';animation:dots 1.2s steps(4,end) infinite}
  @keyframes dots{0%{content:'.'}33%{content:'..'}66%{content:'...'}100%{content:'.'}}
  #input-area{flex-shrink:0;display:flex;flex-direction:column;gap:6px}
  #text-row{display:flex;gap:6px;align-items:flex-end}
  #msg-input{flex:1;background:var(--card);border:1px solid var(--border);border-radius:12px;color:var(--text);font-family:var(--font-ui);font-size:15px;padding:10px 14px;resize:none;min-height:44px;max-height:110px;outline:none;transition:border-color .2s;overflow-y:auto}
  #msg-input:focus{border-color:var(--green-dark)}
  #msg-input::placeholder{color:var(--text-dim)}
  .btn{background:var(--card);border:1px solid var(--border);border-radius:12px;color:var(--text);font-family:var(--font-ui);font-size:14px;font-weight:600;padding:10px 16px;cursor:pointer;transition:background .15s,border-color .15s,box-shadow .15s;white-space:nowrap;user-select:none;-webkit-user-select:none;letter-spacing:.5px}
  .btn:active{transform:scale(0.97)}
  .btn.primary{background:var(--tint-dark);border-color:var(--green-dark);color:var(--green)}
  .btn.on{background:var(--tint-bg);border-color:var(--green);color:var(--green);box-shadow:0 0 8px var(--green-glow)}
  .btn.danger{border-color:#4a1a1a;color:var(--danger)}
  #send-btn{min-width:44px;height:44px;font-size:18px;flex-shrink:0}
  #ptt-btn{width:100%;height:52px;font-size:16px;letter-spacing:2px;border-radius:14px}
  #ptt-btn.recording{background:#3a0f0f;border-color:var(--danger);color:var(--danger);box-shadow:0 0 16px rgba(255,76,76,.3);animation:pulse-rec 1s ease-in-out infinite}
  #img-btn{min-width:44px;height:44px;font-size:18px;padding:0;flex-shrink:0}
  #img-btn.has-image{border-color:var(--green);color:var(--green);background:var(--tint-dark)}
  #av-img-btn.has-image{border-color:var(--green);color:var(--green);background:var(--tint-dark)}
  #img-preview-row{display:none;align-items:center;gap:8px;padding:4px 2px}
  #img-preview-row.visible{display:flex}
  #img-preview-row img{height:56px;width:56px;object-fit:cover;border-radius:8px;border:1px solid var(--border)}
  #img-preview-row span{flex:1;font-size:11px;color:var(--text-dim);font-family:var(--font-mono);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  #img-clear-btn{font-size:18px;background:none;border:none;color:var(--text-dim);cursor:pointer;padding:4px 8px;line-height:1}
  #img-clear-btn:hover{color:var(--danger)}
  .bubble-img{max-width:180px;max-height:180px;border-radius:10px;border:1px solid var(--border);display:block;margin-bottom:6px;cursor:pointer}
  .bubble-img:hover{opacity:0.85}
  @keyframes pulse-rec{0%,100%{box-shadow:0 0 10px rgba(255,76,76,.2)}50%{box-shadow:0 0 22px rgba(255,76,76,.5)}}
  #controls-row{display:flex;gap:6px;flex-wrap:nowrap}
  #controls-row .btn{flex:1;font-size:12px;padding:8px 6px;min-width:0;text-align:center;overflow:hidden;text-overflow:ellipsis}
  #settings-panel{display:none;flex-direction:column;gap:0;background:var(--card);border:1px solid var(--border);border-radius:14px;padding:0;flex-shrink:0;max-height:80vh;overflow:hidden}
  @media(max-height:700px){#settings-panel{max-height:70vh}}
  @media(max-width:480px){
    #settings-panel{border-radius:10px;max-height:82vh}
    .setting-row label{min-width:60px;max-width:60px;font-size:10px}
    .setting-row{gap:5px;font-size:12px}
  }
  @media(min-width:768px){
    #app{padding:14px 20px 10px}
    #settings-panel{border-radius:16px}
  }
  @media(min-width:1024px){
    body{background:var(--bg)}
    #app{padding:16px 24px 12px}
  }
  #settings-panel.open{display:flex}
  #settings-tabs{display:flex;flex-shrink:0;border-bottom:1px solid var(--border);overflow-x:auto;scrollbar-width:none;background:var(--panel);border-radius:14px 14px 0 0}
  #settings-tabs::-webkit-scrollbar{display:none}
  .stab{flex:1;min-width:0;padding:8px 4px 7px;font-family:var(--font-mono);font-size:10px;letter-spacing:1px;color:var(--text-dim);background:none;border:none;border-bottom:2px solid transparent;cursor:pointer;transition:color .15s,border-color .15s;white-space:nowrap;text-align:center}
  .stab:hover{color:var(--text)}
  .stab.active{color:var(--green);border-bottom-color:var(--green)}
  #settings-body{flex:1;overflow-y:auto;-webkit-overflow-scrolling:touch;padding:10px 14px 12px;display:flex;flex-direction:column;gap:8px}
  @media(max-width:480px){#settings-body{padding:8px 10px 10px}}
  .stab-pane{display:none;flex-direction:column;gap:8px}
  .stab-pane.active{display:flex}
  .setting-row{display:flex;align-items:center;gap:6px;font-size:13px;flex-wrap:wrap}
  .setting-row label{color:var(--text-dim);font-family:var(--font-mono);font-size:11px;letter-spacing:1px;min-width:80px;max-width:80px;flex-shrink:0}
  .setting-row input[type=text],.setting-row input[type=password],.setting-row select{flex:1;min-width:0;max-width:100%;background:var(--panel);border:1px solid var(--border);border-radius:8px;color:var(--text);font-family:var(--font-mono);font-size:12px;padding:6px 10px;outline:none;overflow:hidden;text-overflow:ellipsis}
  .setting-row select option{background:var(--panel)}
  .setting-row input[type=range]{accent-color:var(--green);height:4px;min-width:0;flex-shrink:1}
  .char-select{flex:1;min-width:0;max-width:calc(100% - 130px)}
  .section-divider{border:none;border-top:1px solid var(--border);margin:4px 0}
  #vol-slider{flex:1;accent-color:var(--green);height:4px}
  .provider-field{display:none}
  .provider-field.visible{display:flex}
  /* Memory viewer */
  #memory-panel{display:none;margin-top:4px;border-top:1px solid var(--border);padding-top:8px}
  #memory-panel.open{display:block}
  #new-char-form{display:none;flex-direction:column;gap:6px;padding:6px 0 2px 0;border-top:1px solid var(--border);margin-top:2px}
  #new-char-form.open{display:flex !important}
  .mem-card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:8px 10px;margin-bottom:6px;font-size:12px}
  .mem-card .mem-cat{font-family:var(--font-mono);font-size:10px;letter-spacing:1px;color:var(--green-dim);margin-bottom:3px}
  .mem-card .mem-content{color:var(--text);line-height:1.4;margin-bottom:4px}
  .mem-card .mem-meta{font-size:10px;color:var(--text-dim);display:flex;gap:8px;flex-wrap:wrap}
  .mem-card .mem-actions{margin-top:6px;display:flex;gap:4px;flex-wrap:wrap}
  .mem-card .mem-actions button{font-size:10px;padding:2px 6px;border-radius:4px;background:var(--card);border:1px solid var(--border);color:var(--text-dim);cursor:pointer}
  .mem-card .mem-actions button:hover{color:var(--green);border-color:var(--green-dark)}
  /* FX row toggles */
  .fx-tog{width:18px;height:18px;min-width:18px;padding:0;border-radius:50%;font-size:9px;line-height:1;flex-shrink:0;border:1px solid var(--border);background:var(--panel);color:#555;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:background .15s,color .15s,border-color .15s}
  .fx-tog.on{background:var(--tint-dark);border-color:var(--green-dark);color:var(--green);box-shadow:0 0 5px var(--green-glow)}
  .fx-lbl{font-size:10px;color:var(--text-dim);flex-shrink:0}
  .fx-row.fx-off .fx-lbl{color:#444}
  .fx-row.fx-off input[type=range]{opacity:0.25;pointer-events:none;accent-color:#444}
  .fx-row.fx-off span[id]{color:#444}
  #fx-bank.fx-bank-off{opacity:0.35;pointer-events:none}
  #fx-bank.fx-bank-off .fx-tog{pointer-events:none}

  /* ── Avatar slot grid ── */
  .avatar-slot-wrap{display:flex;flex-direction:column;gap:3px}
  .avatar-slot-lbl{font-size:9px;color:var(--text-dim);font-family:var(--font-mono);letter-spacing:1px;text-transform:uppercase}
  .avatar-slot{width:100%;aspect-ratio:1/1;border:1px dashed var(--tint-slot-border);border-radius:6px;background:var(--tint-slot);display:flex;align-items:center;justify-content:center;cursor:pointer;overflow:hidden;position:relative;transition:border-color .2s}
  .avatar-slot:hover{border-color:var(--green)}
  .avatar-slot span{font-size:20px;color:var(--tint-slot-border);pointer-events:none}
  .avatar-slot img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:none}
  .avatar-slot.loaded span{display:none}
  .avatar-slot.loaded img{display:block}

  /* ── Avatar overlay ── */
  #avatar-overlay{display:none;position:fixed;inset:0;z-index:900;background:var(--tint-slot);flex-direction:column;align-items:center;justify-content:flex-start}
  #avatar-overlay.open{display:flex}
  /* All direct children of overlay are capped to app width */
  #avatar-overlay > *{width:100%;max-width:640px;box-sizing:border-box}
  /* Bezel gets all remaining vertical space */
  #avatar-bezel{flex:1 !important;max-width:640px}

  /* Outer bezel frame */
  #avatar-bezel{
    flex:1;width:100%;display:flex;align-items:center;justify-content:center;
    padding:16px 16px 0 16px;box-sizing:border-box;min-height:0;
  }
  #avatar-frame-border{
    position:relative;width:100%;height:100%;max-width:540px;
    border:2px solid var(--green);border-radius:4px;
    box-shadow:0 0 0 1px var(--tint-slot),0 0 0 3px var(--tint-dark),0 0 24px var(--green-glow),inset 0 0 24px rgba(0,0,0,.6);
    background:var(--tint-slot);display:flex;align-items:center;justify-content:center;overflow:hidden;
  }
  /* Corner accents */
  #avatar-frame-border::before,#avatar-frame-border::after{
    content:'';position:absolute;width:18px;height:18px;border-color:var(--green);border-style:solid;z-index:10;pointer-events:none;
  }
  #avatar-frame-border::before{top:-1px;left:-1px;border-width:2px 0 0 2px}
  #avatar-frame-border::after{bottom:-1px;right:-1px;border-width:0 2px 2px 0}

  #avatar-viewport{
    position:relative;width:100%;height:100%;overflow:hidden;cursor:grab;
    display:flex;align-items:center;justify-content:center;
    background:var(--tint-slot);
  }
  #avatar-viewport.dragging{cursor:grabbing}
  #avatar-wireframe{position:absolute;inset:0;width:100%;height:100%;z-index:1;pointer-events:none}
  #avatar-img{
    position:absolute;transform-origin:center center;
    image-rendering:pixelated;image-rendering:crisp-edges;
    user-select:none;-webkit-user-drag:none;pointer-events:none;
    transition:opacity 2.5s ease;z-index:2;
    width:100%;height:100%;object-fit:contain;
  }
  #avatar-pixel-canvas{ display:none; }
  /* Pixel wrap: scales down the whole img layer then back up via CSS */
  #avatar-pixel-wrap{
    position:absolute;inset:0;width:100%;height:100%;
    pointer-events:none;overflow:hidden;
    /* activated by JS: transform-origin, transform, image-rendering */
  }
  #avatar-code-canvas{position:absolute;inset:0;width:100%;height:100%;z-index:4;pointer-events:none;opacity:0;transition:opacity 2.5s ease}
  #avatar-scanlines{position:absolute;inset:0;width:100%;height:100%;z-index:5;pointer-events:none}
  #avatar-static-canvas{position:absolute;inset:0;width:100%;height:100%;z-index:6;pointer-events:none}
  #avatar-color-overlay{position:absolute;inset:0;z-index:7;pointer-events:none;opacity:0;mix-blend-mode:color;background:var(--green)}
  #avatar-sleep-overlay{position:absolute;inset:0;z-index:9;pointer-events:none;opacity:0;transition:opacity 4s ease;background:radial-gradient(ellipse at 50% 60%, transparent 30%, rgba(0,0,0,0.72) 100%)}
  #avatar-glitch-bar{position:absolute;left:0;right:0;height:4px;z-index:8;pointer-events:none;opacity:0;background:var(--green);filter:blur(1px)}

  /* Wave strip */
  #avatar-wave-wrap{width:100%;height:70px;flex-shrink:0;background:var(--tint-slot);border-top:1px solid var(--tint-dark);transition:height .2s,opacity .2s;box-sizing:border-box}
  #avatar-wave-wrap.wave-hidden{height:0;opacity:0;border:none}
  #avatar-wave{width:100%;height:100%;display:block}

  /* HUD bar */
  #avatar-hud{
    width:100%;display:flex;flex-direction:column;gap:4px;
    padding:6px 12px 10px 12px;background:var(--tint-slot);flex-shrink:0;box-sizing:border-box;
    border-top:1px solid var(--tint-dark);
  }
  #avatar-char-nav{flex-shrink:0}
  #avatar-nav-left:hover,#avatar-nav-right:hover{opacity:1 !important;text-shadow:0 0 8px var(--green)}
  #avatar-ptt-btn{width:100%;height:44px;font-size:14px;letter-spacing:2px;border-radius:10px}
  #avatar-ptt-btn.recording{background:#3a0f0f;border-color:var(--danger);color:var(--danger);box-shadow:0 0 16px rgba(255,76,76,.3);animation:pulse-rec 1s ease-in-out infinite}
  #avatar-controls-row{display:flex;gap:6px;align-items:center;justify-content:center;}
  #avatar-controls-row .btn{font-size:11px;padding:3px 10px;}

  /* Zoom controls inside frame — bottom left corner */
  #avatar-zoom-controls{
    position:absolute;bottom:8px;left:8px;z-index:20;display:flex;gap:4px;
    opacity:1;transition:opacity .3s;
  }
  #avatar-zoom-controls.locked{opacity:0}
  #avatar-frame-border:hover #avatar-zoom-controls.locked{opacity:0.6}
  #avatar-zoom-controls button{
    font-family:var(--font-mono);font-size:12px;padding:2px 7px;
    background:var(--tint-slot);border:1px solid var(--tint-slot-border);color:var(--green);
    border-radius:3px;cursor:pointer;line-height:1;
  }
  #avatar-zoom-controls button:hover{background:var(--tint-dark)}
  #avatar-zoom-reset{font-size:10px!important}
  #avatar-lock-btn.active{background:var(--tint-dark)!important;border-color:var(--green)!important}

  /* Avatar subtitles — bottom interior of viewport */
  #avatar-subtitle{
    position:absolute;bottom:0;left:0;right:0;z-index:18;
    padding:28px 18px 18px;
    background:linear-gradient(to top, rgba(0,0,0,0.88) 70%, rgba(0,0,0,0));
    font-family:'Courier New',monospace;font-size:19px;line-height:1.5;
    color:var(--green);letter-spacing:2px;text-transform:uppercase;
    text-shadow:0 0 12px var(--green),0 0 28px var(--green-glow);
    pointer-events:none;
    opacity:0;transition:opacity 0.35s ease;
  }
  #avatar-subtitle.visible{opacity:1}
  #avatar-subtitle .sub-line{
    display:block;width:100%;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  }
  #avatar-subtitle .sub-line.current .sub-word{
    opacity:0;transition:opacity 0.12s linear;
  }
  #avatar-subtitle .sub-line.current .sub-word.shown{opacity:1}
  /* CC toggle button — sits in zoom controls row */
  #avatar-cc-btn{font-size:10px!important;letter-spacing:1px;min-width:28px;}

  /* Avatar button in header */
  #avatar-header-btn{font-size:13px;padding:4px 8px}
  /* Floating close button — top-right of overlay, mirrors header button position */
  #avatar-close-btn{
    position:absolute;top:10px;right:12px;z-index:950;
    font-family:var(--font-mono);font-size:12px;padding:4px 10px;
    background:var(--tint-dark);border:1px solid var(--border);color:var(--green);
    border-radius:6px;cursor:pointer;letter-spacing:1px;
    opacity:0;transition:opacity .25s,background .15s,border-color .15s;
  }
  #avatar-bezel:hover #avatar-close-btn{opacity:1;}
  #avatar-close-btn:hover{background:var(--tint-bg);border-color:var(--green);}
</style>
</head>
<body>
<div id="lightbox"><img id="lightbox-img" src="" alt=""></div>
<div id="app">

  <div id="header">
    <div id="logo">ECKO</div>
    <div class="status-dot" id="dot-tts"></div><span class="status-label">TTS</span>
    <div class="status-dot" id="dot-llm"></div><span class="status-label">LLM</span>
    <div id="provider-tag">—</div>
    <div id="header-right">
      <button class="btn" id="avatar-header-btn" onclick="openAvatarOverlay()" title="Avatar mode">AVATAR</button>
      <button class="btn" onclick="toggleSettings()">SETTINGS</button>
    </div>
  </div>

  <!-- Settings panel -->
  <div id="settings-panel">
    <div id="settings-tabs">
      <button class="stab active" onclick="switchTab('model')">MODEL</button>
      <button class="stab" onclick="switchTab('voice')">VOICE</button>
      <button class="stab" onclick="switchTab('char')">CHAR</button>
      <button class="stab" onclick="switchTab('memory')">MEMORY</button>
      <button class="stab" onclick="switchTab('safety')">SAFETY</button>
      <button class="stab" onclick="switchTab('avatar')">AVATAR</button>
    </div>
    <div id="settings-body">

      <!-- ── MODEL tab ── -->
      <div class="stab-pane active" id="stab-model">
        <!-- Colour theme -->
    <div class="setting-row">
      <label>UI HUE</label>
      <input type="range" id="s-ui-hue" min="0" max="360" value="140" step="1"
             oninput="applyUIHue(this.value)" style="flex:1">
      <span id="s-ui-hue-val" style="font-family:var(--font-mono);font-size:11px;color:var(--text-dim);min-width:28px;text-align:right">140</span>
    </div>
        <hr class="section-divider">
        <!-- LLM Provider -->
    <div class="setting-row">
      <label>PROVIDER</label>
      <select id="s-provider" onchange="onProviderChange()"></select>
    </div>
    <div class="setting-row">
      <label>BASE URL</label>
      <input type="text" id="s-base-url" placeholder="http://localhost:5001">
    </div>
    <div class="setting-row provider-field" id="field-api-key">
      <label>API KEY</label>
      <input type="password" id="s-api-key" placeholder="sk-...">
    </div>
    <div class="setting-row provider-field" id="field-agent-id">
      <label>AGENT ID</label>
      <input type="text" id="s-agent-id" placeholder="ag_...">
    </div>
    <div class="setting-row provider-field" id="field-model">
      <label>MODEL</label>
      <input type="text" id="s-model" placeholder="llama3.2:latest">
    </div>
    <div class="setting-row">
      <label>SYS PROMPT</label>
      <input type="text" id="s-system-prompt" placeholder="(optional system prompt)">
    </div>

    <hr class="section-divider">

    <!-- Web Search -->
    <div class="setting-row">
      <label>WEB SEARCH</label>
      <button class="btn" id="s-websearch-btn" onclick="toggleWebSearch()" title="Enable Brave web search for relevant queries">OFF</button>
    </div>
    <div id="websearch-fields" style="display:none;flex-direction:column;gap:8px">
      <div class="setting-row">
        <label>BRAVE KEY</label>
        <input type="password" id="s-websearch-key" placeholder="BSA...">
      </div>
      <div class="setting-row">
        <label>RESULTS</label>
        <select id="s-websearch-count">
          <option value="2">2</option>
          <option value="3" selected>3</option>
          <option value="4">4</option>
          <option value="5">5</option>
        </select>
        <span style="font-size:10px;color:var(--text-dim);margin-left:8px">per search</span>
      </div>
    </div>
      </div>

      <!-- ── VOICE tab ── -->
      <div class="stab-pane" id="stab-voice">
        <!-- TTS Provider -->
    <div class="setting-row">
      <label>TTS PROV</label>
      <select id="s-tts-provider" onchange="onTTSProviderChange()"></select>
    </div>
    <div class="setting-row">
      <label>TTS URL</label>
      <input type="text" id="s-tts-base-url" placeholder="http://localhost:8000">
    </div>
    <div class="setting-row tts-provider-field" id="tts-field-api-key">
      <label>TTS KEY</label>
      <input type="password" id="s-tts-api-key" placeholder="sk-...">
    </div>
    <div class="setting-row">
      <label>VOICE</label>
      <select id="s-voice"></select>
    </div>
    <div class="setting-row" id="tts-field-voice-id" style="display:none">
      <label>VOICE ID</label>
      <input type="text" id="s-voice-id" placeholder="paste voice_id directly" style="flex:1;min-width:0;background:var(--panel);border:1px solid var(--border);border-radius:8px;color:var(--text);font-family:var(--font-mono);font-size:12px;padding:6px 10px;outline:none" title="Paste an ElevenLabs voice_id here to use it directly without selecting from the list above">
    </div>
    <div class="setting-row">
      <label>KV SCALE</label>
      <input type="text" id="s-kv-scale" placeholder="off" style="max-width:60px" title="speaker_kv_scale (e.g. 1.25) — blank to disable">
      <span style="font-size:10px;color:var(--text-dim)">MIN_T</span>
      <input type="text" id="s-kv-min-t" placeholder="0.9" style="max-width:48px" title="speaker_kv_min_t">
      <span style="font-size:10px;color:var(--text-dim)">LAYERS</span>
      <input type="text" id="s-kv-max-layers" placeholder="24" style="max-width:44px" title="speaker_kv_max_layers">
    </div>
    <div class="setting-row">
      <label>VOLUME</label>
      <input type="range" id="vol-slider" min="0.2" max="3.0" step="0.05" value="1.5">
      <span style="font-size:10px;color:var(--text-dim);flex-shrink:0;margin-left:6px">AMP</span>
      <input type="range" id="wave-amp" min="0.1" max="8" step="0.05" value="1" style="flex:1;accent-color:var(--green)" oninput="waveAmp=parseFloat(this.value);fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({wave_amp:waveAmp})}).catch(()=>{})">
      <span style="font-size:10px;color:var(--text-dim);flex-shrink:0;margin-left:6px">FADE</span>
      <input type="range" id="wave-fade" min="0.05" max="0.99" step="0.01" value="0.25" style="flex:1;accent-color:var(--green)" oninput="waveFade=parseFloat(this.value);fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({wave_fade:waveFade})}).catch(()=>{})" title="Smoothing — higher = slower fade">
    </div>
    <div class="setting-row">
      <label>REVERB/FX</label>
      <input type="file" id="reverb-ir-file" accept=".wav" style="display:none" onchange="loadIR(this)">
      <button class="btn" onclick="document.getElementById('reverb-ir-file').click()" style="font-size:11px;padding:4px 8px;flex-shrink:0">LOAD IR</button>
      <span id="reverb-ir-name" style="font-size:10px;color:var(--text-dim);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0">no IR loaded</span>
      <button class="btn" id="fx-toggle" onclick="toggleFX()" style="font-size:11px;padding:4px 8px;flex-shrink:0">FX OFF</button>
    </div>
    <div id="fx-bank">
    <div class="setting-row fx-row" id="fx-row-reverb">
      <button class="fx-tog on" id="fxtog-reverb" onclick="toggleFxRow('reverb')" title="Reverb on/off">●</button>
      <span class="fx-lbl">VERB WET</span>
      <input type="range" id="reverb-wet" min="0" max="1" step="0.01" value="0.25" style="flex:2;min-width:60px" oninput="updateReverb()">
      <span id="reverb-wet-val" style="font-size:10px;color:var(--text-dim);min-width:28px;flex-shrink:0">25%</span>
      <span class="fx-lbl">PRE</span>
      <input type="range" id="reverb-predelay" min="0" max="80" step="1" value="20" style="flex:1;min-width:40px" oninput="updateReverb()">
      <span id="reverb-predelay-val" style="font-size:10px;color:var(--text-dim);min-width:28px;flex-shrink:0">20ms</span>
    </div>
    <div class="setting-row fx-row" id="fx-row-delay">
      <button class="fx-tog on" id="fxtog-delay" onclick="toggleFxRow('delay')" title="Delay on/off">●</button>
      <span class="fx-lbl">DLY WET</span>
      <input type="range" id="delay-wet" min="0" max="1" step="0.01" value="0.25" style="flex:2;min-width:50px;max-width:90px" oninput="updateDelay()">
      <span id="delay-wet-val" style="font-size:10px;color:var(--text-dim);min-width:28px;flex-shrink:0">25%</span>
      <span class="fx-lbl">TIME</span>
      <input type="range" id="delay-time" min="0" max="800" step="10" value="0" style="flex:1;min-width:40px;max-width:80px" oninput="updateDelay()">
      <span id="delay-time-val" style="font-size:10px;color:var(--text-dim);min-width:28px;flex-shrink:0">0ms</span>
      <span class="fx-lbl">FB</span>
      <input type="range" id="delay-feedback" min="0" max="0.85" step="0.01" value="0.35" style="flex:1;min-width:36px;max-width:70px" oninput="updateDelay()">
      <span id="delay-fb-val" style="font-size:10px;color:var(--text-dim);min-width:28px;flex-shrink:0">35%</span>
    </div>
    <div class="setting-row fx-row" id="fx-row-crush">
      <button class="fx-tog on" id="fxtog-crush" onclick="toggleFxRow('crush')" title="Bitcrusher on/off">●</button>
      <span class="fx-lbl" title="Bit depth + sample rate reduction — telephone/codec/lo-fi effect">CRUSH</span>
      <input type="range" id="crush-wet" min="0" max="1" step="0.01" value="0" style="flex:2;min-width:60px" oninput="updateCrush()">
      <span id="crush-wet-val" style="font-size:10px;color:var(--text-dim);min-width:28px;flex-shrink:0">0%</span>
      <span class="fx-lbl" title="Bit depth (4=harsh, 8=telephone, 12=subtle)">BITS</span>
      <input type="range" id="crush-bits" min="2" max="16" step="1" value="8" style="flex:1;min-width:40px" oninput="updateCrush()">
      <span id="crush-bits-val" style="font-size:10px;color:var(--text-dim);min-width:20px;flex-shrink:0">8</span>
      <span class="fx-lbl" title="Sample rate reduction (1=off, 4=telephone, 8=lo-fi)">SR÷</span>
      <input type="range" id="crush-sr" min="1" max="16" step="1" value="4" style="flex:1;min-width:40px" oninput="updateCrush()">
      <span id="crush-sr-val" style="font-size:10px;color:var(--text-dim);min-width:20px;flex-shrink:0">÷4</span>
    </div>
    <div class="setting-row fx-row" id="fx-row-chorus">
      <button class="fx-tog on" id="fxtog-chorus" onclick="toggleFxRow('chorus')" title="Chorus on/off">●</button>
      <span class="fx-lbl">CHORUS</span>
      <input type="range" id="chorus-wet" min="0" max="1" step="0.01" value="0" style="flex:2;min-width:60px" oninput="updateChorus()">
      <span id="chorus-wet-val" style="font-size:10px;color:var(--text-dim);min-width:28px;flex-shrink:0">0%</span>
      <span class="fx-lbl">DEPTH</span>
      <input type="range" id="chorus-depth" min="0.001" max="0.02" step="0.001" value="0.005" style="flex:1;min-width:40px" oninput="updateChorus()">
      <span class="fx-lbl">RATE</span>
      <input type="range" id="chorus-rate" min="0.1" max="8" step="0.1" value="1.2" style="flex:1;min-width:40px" oninput="updateChorus()">
    </div>
    <div class="setting-row fx-row" id="fx-row-ring">
      <button class="fx-tog on" id="fxtog-ring" onclick="toggleFxRow('ring')" title="Ring mod on/off">●</button>
      <span class="fx-lbl" title="Ring modulator — robotic/alien effect">RING MOD</span>
      <input type="range" id="ringmod-wet" min="0" max="1" step="0.01" value="0" style="flex:2;min-width:60px" oninput="updateRingMod()">
      <span id="ringmod-wet-val" style="font-size:10px;color:var(--text-dim);min-width:28px;flex-shrink:0">0%</span>
      <span class="fx-lbl">FREQ</span>
      <input type="range" id="ringmod-freq" min="20" max="1200" step="10" value="120" style="flex:1;min-width:40px" oninput="updateRingMod()">
      <span id="ringmod-freq-val" style="font-size:10px;color:var(--text-dim);min-width:34px;flex-shrink:0">120Hz</span>
    </div>
    <div class="setting-row fx-row" id="fx-row-dist">
      <button class="fx-tog on" id="fxtog-dist" onclick="toggleFxRow('dist')" title="Distortion on/off">●</button>
      <span class="fx-lbl" title="Soft-clip distortion — warmth/saturation/overdrive">DIST</span>
      <input type="range" id="dist-wet" min="0" max="1" step="0.01" value="0" style="flex:2;min-width:60px" oninput="updateDist()">
      <span id="dist-wet-val" style="font-size:10px;color:var(--text-dim);min-width:28px;flex-shrink:0">0%</span>
      <span class="fx-lbl">DRIVE</span>
      <input type="range" id="dist-drive" min="1" max="100" step="1" value="20" style="flex:1;min-width:40px" oninput="updateDist()">
      <span id="dist-drive-val" style="font-size:10px;color:var(--text-dim);min-width:28px;flex-shrink:0">20</span>
    </div>
    </div><!-- /fx-bank -->
    <div id="fx-presets" style="display:flex;gap:5px;flex-wrap:wrap;padding:4px 0 2px 0">
      <span style="font-size:9px;color:var(--text-dim);font-family:var(--font-mono);align-self:center;letter-spacing:1px">PRESET:</span>
      <button class="btn" style="font-size:10px;padding:3px 8px" onclick="applyFxPreset('glitch')">GLITCH</button>
      <button class="btn" style="font-size:10px;padding:3px 8px" onclick="applyFxPreset('warm')">WARM</button>
      <button class="btn" style="font-size:10px;padding:3px 8px" onclick="applyFxPreset('radio')">RADIO</button>
      <button class="btn" style="font-size:10px;padding:3px 8px" onclick="applyFxPreset('echo')">ECHO</button>
      <button class="btn" style="font-size:10px;padding:3px 8px" onclick="applyFxPreset('robot')">ROBOT</button>
      <button class="btn" style="font-size:10px;padding:3px 8px" onclick="applyFxPreset('dreamy')">DREAMY</button>
      <button class="btn" style="font-size:10px;padding:3px 8px" onclick="applyFxPreset('intercom')">INTERCOM</button>
      <button class="btn" style="font-size:10px;padding:3px 8px" onclick="applyFxPreset('alien')">ALIEN</button>
      <button class="btn" style="font-size:10px;padding:3px 8px" onclick="applyFxPreset('tape')">TAPE</button>
      <button class="btn" style="font-size:10px;padding:3px 8px" onclick="applyFxPreset('cavern')">CAVERN</button>
    </div>
      </div>

      <!-- ── CHAR tab ── -->
      <div class="stab-pane" id="stab-char">
        <!-- Auto-continue -->
    <div class="setting-row">
      <label>AUTO-CONT</label>
      <select id="s-ac-mode">
        <option value="standard">Standard</option>
        <option value="aggressive">Aggressive</option>
        <option value="relaxed">Relaxed</option>
      </select>
      <button class="btn" id="s-ac-btn" onclick="toggleAC()">ON</button>
    </div>

    <hr class="section-divider">

    <!-- Initiative / proactive messaging -->
    <div class="setting-row" style="align-items:flex-start;flex-direction:column;gap:6px">
      <div style="display:flex;align-items:center;gap:8px;width:100%">
        <label style="letter-spacing:2px">INITIATIVE</label>
        <button class="btn" id="s-init-btn" onclick="toggleInitiative()" style="flex-shrink:0">OFF</button>
        <select id="s-init-mode" style="flex:1;min-width:80px">
          <option value="light">Light · 25–45 min</option>
          <option value="regular">Regular · 10–20 min</option>
          <option value="active">Active · 3–8 min</option>
        </select>
      </div>
      <div style="font-size:10px;color:var(--text-dim);padding-left:2px;line-height:1.5">
        Character asks questions and surfaces topics unprompted, drawing on conversation history and memory. Works alongside or without Auto-continue.
      </div>
    </div>

    <!-- FX auto-chance + test -->
    <div class="setting-row" style="align-items:center;gap:8px;flex-wrap:wrap">
      <label style="white-space:nowrap">FX CHANCE</label>
      <input type="range" id="s-fx-chance" min="0" max="100" step="5" value="15"
             style="flex:1;min-width:80px;accent-color:var(--green)"
             oninput="document.getElementById('s-fx-chance-val').textContent=this.value+'%';_saveFxChance()"
             title="Probability (0–100%) that the next initiative tick fires a visual effect instead of a message">
      <span id="s-fx-chance-val" style="font-size:10px;color:var(--green);min-width:30px;text-align:right">15%</span>
      <button class="btn" onclick="_testFxNow()" title="Fire a random visual effect right now for testing" style="white-space:nowrap;padding:5px 10px;font-size:10px">TEST FX</button>
    </div>

    <!-- Sleep timer -->
    <div class="setting-row" style="align-items:flex-start;flex-direction:column;gap:6px">
      <div style="display:flex;align-items:center;gap:8px;width:100%">
        <label style="letter-spacing:2px;white-space:nowrap">SLEEP TIMER</label>
        <button class="btn" id="s-sleep-timer-btn" onclick="_toggleSleepTimer()" style="flex-shrink:0">OFF</button>
        <span style="font-size:10px;color:var(--text-dim);flex:1">quiet hours</span>
        <span id="s-sleep-active-badge" style="display:none;font-size:9px;color:var(--green);font-family:var(--font-mono);letter-spacing:1px">● SLEEPING</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;padding-left:2px;width:100%">
        <span style="font-size:10px;color:var(--text-dim)">FROM</span>
        <input type="number" id="s-sleep-start" min="0" max="23" value="23"
               style="width:46px;text-align:center;font-size:11px;background:var(--bg2);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:3px"
               onchange="_saveSleepTimer()" title="Sleep start hour (24 h)">
        <span style="font-size:10px;color:var(--text-dim)">TO</span>
        <input type="number" id="s-sleep-end" min="0" max="23" value="8"
               style="width:46px;text-align:center;font-size:11px;background:var(--bg2);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:3px"
               onchange="_saveSleepTimer()" title="Sleep end hour (24 h)">
        <span style="font-size:10px;color:var(--text-dim)">h · suppresses auto-continue &amp; initiative</span>
      </div>
    </div>

    <!-- Visual FX -->
    <div class="setting-row">
      <label>VISUAL FX</label>
      <button class="btn" id="s-vis-fx-btn" onclick="toggleVisualFX()" title="Enable visual effects triggered by agent or initiative">OFF</button>
      <span style="font-size:10px;color:var(--text-dim);margin-left:8px">agent-triggered screen effects (avatar must be open)</span>
    </div>

    <!-- Sentiment mood FX -->
    <div class="setting-row">
      <label>MOOD FX</label>
      <button class="btn" id="s-mood-fx-btn" onclick="toggleMoodFX()" title="Use conversation sentiment to bias which visual effects fire">OFF</button>
      <span style="font-size:10px;color:var(--text-dim);margin-left:8px">sentiment-biased effect selection</span>
    </div>

    <!-- Character presets -->
    <div class="setting-row">
      <label>VISION</label>
      <button class="btn" id="s-vision-btn" onclick="toggleVision()" title="Enable image attachment for vision-capable models">OFF</button>
      <span style="font-size:10px;color:var(--text-dim);margin-left:8px">attach images to chat</span>
    </div>

    <hr class="section-divider">

    <div class="setting-row">
      <label>CHARACTER</label>
      <select id="s-char" class="char-select" onchange="onCharSelectChange()"></select>
      <button class="btn" style="flex-shrink:0;white-space:nowrap;padding:6px 10px" onclick="loadCharacter()">LOAD</button>
    </div>
    <div class="setting-row" id="char-save-row">
      <label></label>
      <button class="btn" id="btn-save-current" style="flex:1;padding:6px 8px;font-size:11px" onclick="saveCurrentCharacter()" disabled title="Load a character first">💾 SAVE CURRENT</button>
      <button class="btn primary" style="flex:1;padding:6px 8px;font-size:11px" onclick="toggleNewCharForm()">＋ SAVE NEW</button>
      <button class="btn" id="btn-delete-char" style="flex-shrink:0;padding:6px 8px;font-size:11px;color:var(--danger);border-color:#7a1a1a" onclick="deleteSelectedCharacter()" disabled title="Delete selected character">🗑</button>
    </div>
    <div id="new-char-form" style="flex-direction:column;gap:6px;padding:6px 0 2px 0;border-top:1px solid var(--border);margin-top:2px">
      <div class="setting-row" style="gap:6px">
        <label style="min-width:60px;max-width:60px">FOLDER</label>
        <input type="text" id="s-char-folder" placeholder="optional/subfolder" style="flex:1;min-width:0">
      </div>
      <div class="setting-row" style="gap:6px">
        <label style="min-width:60px;max-width:60px">NAME</label>
        <input type="text" id="s-char-name" placeholder="character name" style="flex:1;min-width:0">
        <button class="btn primary" style="flex-shrink:0;white-space:nowrap;padding:6px 10px" onclick="saveNewCharacter()">SAVE</button>
      </div>
    </div>
    <div class="setting-row">
      <label>CHAR MODE</label>
      <select id="s-char-mode">
        <option value="shared">Shared (all UIs)</option>
        <option value="isolated">Isolated (per surface)</option>
      </select>
    </div>
      </div>

      <!-- ── MEMORY tab ── -->
      <div class="stab-pane" id="stab-memory">
        <!-- Conversation RAG (auto-flush) -->
    <div class="setting-row">
      <label>CONV RAG</label>
      <span id="conv-rag-status" style="font-size:11px;color:#888;flex:1">off</span>
      <button class="btn" id="s-conv-rag-btn" onclick="toggleConvRag()" title="Automatically flush old conversation turns into a per-character RAG file">OFF</button>
    </div>
    <div class="setting-row" id="conv-rag-threshold-row" style="display:none">
      <label>FLUSH AT</label>
      <input type="range" id="s-conv-rag-threshold" min="6" max="100" step="2" value="20" style="flex:1;accent-color:var(--green);height:4px" oninput="document.getElementById('conv-rag-threshold-val').textContent=this.value+' msgs';saveConvRagSettings()">
      <span id="conv-rag-threshold-val" style="font-size:10px;color:var(--text-dim);min-width:50px;text-align:right">20 msgs</span>
      <button class="btn" style="flex-shrink:0;padding:4px 8px;font-size:10px;color:#ff6600;border-color:#7a3300" onclick="clearConvRagFile()" title="Delete the conversation RAG file for this character">CLEAR FILE</button>
    </div>

    <!-- ASCII art library -->
    <div class="setting-row">
      <label>ASCII ART</label>
      <span id="art-lib-count" style="font-size:11px;color:#888;flex:1">0 pieces</span>
      <button class="btn" onclick="reloadArtLib()" title="Reload art pieces from the ascii_art/ directory">RELOAD</button>
    </div>

    <!-- Extra RAG (manual file load) -->
    <div class="setting-row">
      <label>EXTRA RAG</label>
      <select id="s-rag-file" multiple style="flex:1;min-width:0;height:72px;resize:vertical"></select>
      <button class="btn" style="flex-shrink:0;white-space:nowrap;padding:6px 10px" onclick="loadRag()">LOAD</button>
      <button class="btn" style="flex-shrink:0;white-space:nowrap;padding:6px 10px" onclick="addRag()" title="Append selected file(s) to current index">ADD</button>
      <button class="btn" style="flex-shrink:0;white-space:nowrap;padding:6px 10px" onclick="clearRag()">CLEAR</button>
    </div>
    <div class="setting-row">
      <label>EXPORT CONV</label>
      <input type="text" id="s-rag-save-name" placeholder="filename (no .txt)" style="flex:1;min-width:0;background:var(--panel);border:1px solid var(--border);border-radius:8px;color:var(--text);font-family:var(--font-mono);font-size:12px;padding:6px 10px;outline:none">
      <button class="btn" style="flex-shrink:0;white-space:nowrap;padding:6px 10px" onclick="saveRag()">EXPORT</button>
    </div>
    <div class="setting-row">
      <label></label>
      <span id="rag-status" style="font-size:11px;color:#888;flex:1">No extra RAG loaded</span>
      <label style="min-width:auto;margin-right:4px;font-size:10px">SEMANTIC</label>
      <input type="checkbox" id="s-rag-semantic" style="accent-color:var(--green)">
      <label style="min-width:auto;margin-left:10px;margin-right:4px;font-size:10px" title="Use CUDA for RAG embeddings — faster but uses VRAM">CUDA</label>
      <input type="checkbox" id="s-rag-cuda" style="accent-color:var(--green)" onchange="_saveRagCuda()" title="Use CUDA for RAG sentence-transformer embeddings">
    </div>

    <hr class="section-divider">

    <!-- Context Mode -->
    <div class="setting-row">
      <label>CTX MODE</label>
      <div id="ctx-mode-btns" style="display:flex;gap:4px;flex:1;flex-wrap:wrap">
        <button class="btn" id="ctx-btn-voice_fast"     onclick="setContextMode('voice_fast')"     title="Absolute minimum context. Lowest latency, best for real-time voice.">⚡ FAST</button>
        <button class="btn" id="ctx-btn-voice_balanced" onclick="setContextMode('voice_balanced')" title="Light RAG + memories. Voice with occasional memory recall.">🎙️ VOICE</button>
        <button class="btn" id="ctx-btn-standard"       onclick="setContextMode('standard')"       title="Default. Balanced for text chat with moderate context.">💬 STD</button>
        <button class="btn" id="ctx-btn-deep_recall"    onclick="setContextMode('deep_recall')"    title="More RAG + memories. Slower but more contextually aware.">🧠 DEEP</button>
        <button class="btn" id="ctx-btn-full_context"   onclick="setContextMode('full_context')"   title="Maximum context. Uses all available memory and history.">📚 FULL</button>
      </div>
      <label style="min-width:auto;margin:0 2px 0 6px;font-size:10px;flex-shrink:0" title="Max reply tokens — separate from context size">TOKENS</label>
      <input type="number" id="s-max-tokens" min="50" max="2000" step="50" value="300" style="width:68px;flex:none" title="Max reply tokens" onchange="saveMaxTokens()">
    </div>

    <hr class="section-divider">

    <!-- Memory -->
    <div class="setting-row">
      <label>MEMORY</label>
      <span id="memory-status" style="font-size:11px;color:#888;flex:1">0 entries</span>
      <button class="btn" id="s-memory-btn" onclick="toggleMemory()">OFF</button>
      <button class="btn" style="flex-shrink:0;padding:6px 10px;font-size:11px" onclick="toggleMemoryPanel()">VIEW</button>
    </div>
    <div id="memory-panel">
      <div id="memory-cards" style="max-height:200px;overflow-y:auto;margin-bottom:6px"></div>
      <!-- Add entry row -->
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:6px">
        <input type="text" id="new-mem-content" placeholder="Add memory…" style="flex:1;min-width:120px;background:var(--panel);border:1px solid var(--border);border-radius:8px;color:var(--text);font-family:var(--font-mono);font-size:12px;padding:6px 10px;outline:none">
        <select id="new-mem-cat" style="background:var(--panel);border:1px solid var(--border);border-radius:8px;color:var(--text);font-family:var(--font-mono);font-size:12px;padding:6px">
          <option value="fact">fact</option><option value="preference">pref</option>
          <option value="emotion">emotion</option><option value="relationship">relation</option>
          <option value="topic">topic</option><option value="event">event</option>
        </select>
        <button class="btn primary" onclick="addMemory()" style="padding:6px 10px;font-size:12px">ADD</button>
        <button class="btn" onclick="clearAllMemory()" style="padding:6px 10px;font-size:12px;color:#ff6600;border-color:#ff6600" title="Delete all memory entries">CLEAR ALL</button>
      </div>
      <!-- Import / Export row -->
      <div style="display:flex;gap:6px;flex-wrap:wrap;border-top:1px solid var(--border);padding-top:6px">
        <button class="btn" onclick="memoryExport()" style="flex:1;padding:5px 8px;font-size:11px" title="Download current memory bank as JSON">EXPORT</button>
        <button class="btn" onclick="memoryImportPick('merge')" style="flex:1;padding:5px 8px;font-size:11px" title="Add entries from a JSON file — skips duplicates">IMPORT MERGE</button>
        <button class="btn" onclick="memoryImportPick('replace')" style="flex:1;padding:5px 8px;font-size:11px;color:#ffaa00;border-color:#7a5500" title="Replace entire memory bank from a JSON file (current bank is backed up first)">IMPORT REPLACE</button>
        <input type="file" id="mem-import-input" accept=".json" style="display:none" onchange="memoryImportFile(this)">
      </div>
      <div id="mem-import-status" style="font-size:10px;color:var(--text-dim);font-family:var(--font-mono);padding-top:4px;min-height:14px"></div>
    </div>
    <!-- Always-visible fresh-slate button -->
    <div class="setting-row" style="margin-top:6px">
      <button class="btn danger" onclick="clearAllSession(this)" style="flex:1;padding:7px 10px;font-size:11px;letter-spacing:.5px" title="Clear conversation history, conv RAG file, extra RAG index, and all memories">⚠ CLEAR ALL — fresh slate</button>
    </div>
      </div>

      <!-- ── SAFETY tab ── -->
      <div class="stab-pane" id="stab-safety">
        <!-- Session -->
    <div class="setting-row">
      <label>SESSION</label>
      <select id="s-session-mode">
        <option value="shared">Shared (all clients same conv)</option>
        <option value="isolated">Isolated (per device)</option>
      </select>
    </div>

    <hr class="section-divider">

    <!-- Safety -->
    <div class="setting-row">
      <label>SAFETY</label>
      <span id="safety-score-display" style="font-size:11px;color:var(--green);font-family:var(--font-mono);flex-shrink:0">SCORE: 0</span>
      <span id="safety-level-display" style="font-size:11px;color:var(--text-dim);font-family:var(--font-mono);flex:1">● OK</span>
      <button class="btn" id="safety-l1-btn" onclick="toggleSafetyLayer(1)" style="font-size:10px;padding:3px 7px;flex-shrink:0">L1 ON</button>
      <button class="btn" id="safety-l2-btn" onclick="toggleSafetyLayer(2)" style="font-size:10px;padding:3px 7px;flex-shrink:0">L2 ON</button>
      <button class="btn on" id="safety-indicator-toggle-btn" onclick="toggleSafetyIndicator()" style="font-size:10px;padding:3px 7px;flex-shrink:0" title="Show/hide safety light indicator">LED ON</button>
    </div>
    <div class="setting-row" style="flex-wrap:wrap;gap:6px">
      <label></label>
      <button class="btn" onclick="resetSafetyScore()" style="font-size:10px;padding:4px 8px">RESET SCORE</button>
      <button class="btn" onclick="clearSafetyFlags()" style="font-size:10px;padding:4px 8px">CLEAR FLAGS</button>
      <button class="btn" onclick="resetSafetyDefaults()" style="font-size:10px;padding:4px 8px">RESET RULES</button>
      <button class="btn" onclick="openRuleEditor()" style="font-size:10px;padding:4px 8px;flex:1;min-width:80px">EDIT RULES</button>
    </div>
    <!-- Flags log — no fixed height, expands naturally inside the scrollable settings panel -->
    <div id="safety-flags" style="font-size:10px;color:var(--text-dim);font-family:var(--font-mono);background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:6px 8px;display:none;word-break:break-word;line-height:1.6"></div>

    <!-- Rule editor modal -->
    <div id="rule-editor" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:9998;align-items:flex-start;justify-content:center;overflow-y:auto;padding:16px 8px">
      <div style="background:var(--card);border:1px solid var(--border);border-radius:16px;padding:16px;width:min(560px,100%);display:flex;flex-direction:column;gap:10px;margin:auto">
        <div style="display:flex;align-items:center;justify-content:space-between">
          <span style="font-family:var(--font-mono);font-size:12px;color:var(--green);letter-spacing:1px">SAFETY RULES</span>
          <button class="btn" onclick="closeRuleEditor()" style="padding:3px 8px;font-size:11px">✕</button>
        </div>
        <div style="font-size:10px;color:var(--text-dim)">Edit rules JSON. action: log | warn | block &nbsp;·&nbsp; severity: 1-3 &nbsp;·&nbsp; enabled: true | false</div>
        <textarea id="rule-editor-text" style="min-height:260px;height:40vh;background:var(--panel);border:1px solid var(--border);border-radius:8px;color:var(--text);font-family:var(--font-mono);font-size:11px;padding:10px;outline:none;resize:vertical;width:100%;box-sizing:border-box"></textarea>
        <div style="display:flex;gap:8px;justify-content:flex-end;flex-wrap:wrap">
          <button class="btn" onclick="closeRuleEditor()">CANCEL</button>
          <button class="btn primary" onclick="saveRules()">SAVE RULES</button>
        </div>
      </div>
    </div>
      </div>

      <!-- ── AVATAR tab ── -->
      <div class="stab-pane" id="stab-avatar">
        <!-- Avatar / PNG animator -->
    <div class="setting-row">
      <label style="letter-spacing:2px">AVATAR</label>
      <button class="btn" id="s-avatar-btn" onclick="toggleAvatarMode()" style="flex-shrink:0">OFF</button>
      <button class="btn" id="s-avatar-open-btn" onclick="openAvatarOverlay()" style="flex-shrink:0;font-size:11px;padding:4px 8px" title="Open avatar fullscreen">⛶ VIEW</button>
    </div>
    <!-- Frame upload slots -->
    <div id="avatar-slots" style="display:flex;flex-direction:column;gap:4px;padding:4px 0">
      <div style="font-size:10px;color:var(--text-dim);padding:2px 0 4px 0">Load 6 PNG frames:</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">
        <div class="avatar-slot-wrap"><label class="avatar-slot-lbl">IDLE (mouth closed)</label><div class="avatar-slot" id="slot-idle" onclick="triggerSlotUpload('idle')" title="Idle / mouth closed"><span>＋</span><img id="slot-idle-img"></div></div>
        <div class="avatar-slot-wrap"><label class="avatar-slot-lbl">TALK (mouth open)</label><div class="avatar-slot" id="slot-talk" onclick="triggerSlotUpload('talk')" title="Talking / mouth open"><span>＋</span><img id="slot-talk-img"></div></div>
        <div class="avatar-slot-wrap"><label class="avatar-slot-lbl">BLINK closed</label><div class="avatar-slot" id="slot-blink-closed" onclick="triggerSlotUpload('blink-closed')" title="Blink / mouth closed"><span>＋</span><img id="slot-blink-closed-img"></div></div>
        <div class="avatar-slot-wrap"><label class="avatar-slot-lbl">BLINK talking</label><div class="avatar-slot" id="slot-blink-talk" onclick="triggerSlotUpload('blink-talk')" title="Blink / mouth open"><span>＋</span><img id="slot-blink-talk-img"></div></div>
        <div class="avatar-slot-wrap"><label class="avatar-slot-lbl">SCREAM</label><div class="avatar-slot" id="slot-scream" onclick="triggerSlotUpload('scream')" title="Scream / high amplitude"><span>＋</span><img id="slot-scream-img"></div></div>
        <div class="avatar-slot-wrap"><label class="avatar-slot-lbl">SLEEP</label><div class="avatar-slot" id="slot-sleep" onclick="triggerSlotUpload('sleep')" title="Sleep / idle for a while"><span>＋</span><img id="slot-sleep-img"></div></div>
      </div>
      <input type="file" id="avatar-file-input" accept="image/png,image/webp,image/gif" style="display:none" onchange="onAvatarFileSelected(this)">
      <input type="file" id="avatar-folder-input" accept="image/png,image/webp,image/gif" multiple style="display:none" onchange="onAvatarFolderSelected(this)">
      <div style="display:flex;gap:6px;margin-top:4px">
        <button class="btn" style="flex:1;font-size:11px;padding:5px 8px" onclick="document.getElementById('avatar-folder-input').click()" title="Load avatar images from a folder. Name files: idle, talk, blink-closed, blink-talk, scream, sleep (+ .png/.webp/.gif)">LOAD FOLDER</button>
        <button class="btn" style="flex:1;font-size:11px;padding:5px 8px" onclick="clearAvatarImages()" title="Clear all avatar image slots">✕ CLEAR ALL</button>
      </div>
      <div style="font-size:9px;color:var(--text-dim);font-family:var(--font-mono);padding:2px 0 2px 2px">Naming: idle · talk · blink-closed · blink-talk · scream · sleep</div>
    </div>
    <!-- Avatar timing settings -->
    <div class="setting-row"><label style="min-width:90px">TALK THRESH</label><input type="range" id="av-talk-thresh" min="0.005" max="0.3" step="0.005" value="0.04" style="flex:1" oninput="document.getElementById('av-talk-thresh-val').textContent=parseFloat(this.value).toFixed(3)"><span id="av-talk-thresh-val" style="font-size:10px;color:var(--text-dim);min-width:40px;flex-shrink:0;text-align:right">0.040</span></div>
    <div class="setting-row"><label style="min-width:90px">SCREAM THRESH</label><input type="range" id="av-scream-thresh" min="0.1" max="3.0" step="0.05" value="0.8" style="flex:1" oninput="document.getElementById('av-scream-thresh-val').textContent=parseFloat(this.value).toFixed(2)"><span id="av-scream-thresh-val" style="font-size:10px;color:var(--text-dim);min-width:40px;flex-shrink:0;text-align:right">0.80</span></div>
    <div class="setting-row"><label style="min-width:90px">TALK DECAY</label><input type="range" id="av-talk-decay" min="20" max="400" step="10" value="80" style="flex:1" oninput="document.getElementById('av-talk-decay-val').textContent=this.value+'ms'"><span id="av-talk-decay-val" style="font-size:10px;color:var(--text-dim);min-width:40px;flex-shrink:0;text-align:right">80ms</span></div>
    <div class="setting-row"><label style="min-width:90px">BLINK CHANCE</label><input type="range" id="av-blink-chance" min="1" max="100" step="1" value="25" style="flex:1" oninput="document.getElementById('av-blink-chance-val').textContent=this.value+'%'"><span id="av-blink-chance-val" style="font-size:10px;color:var(--text-dim);min-width:40px;flex-shrink:0;text-align:right">25%</span></div>
    <div class="setting-row"><label style="min-width:90px">BLINK DURATION</label><input type="range" id="av-blink-dur" min="20" max="200" step="5" value="60" style="flex:1" oninput="document.getElementById('av-blink-dur-val').textContent=this.value+'ms'"><span id="av-blink-dur-val" style="font-size:10px;color:var(--text-dim);min-width:40px;flex-shrink:0;text-align:right">60ms</span></div>
    <div class="setting-row"><label style="min-width:90px">BLINK DELAY</label><input type="range" id="av-blink-delay" min="500" max="8000" step="100" value="3000" style="flex:1" oninput="document.getElementById('av-blink-delay-val').textContent=(this.value/1000).toFixed(1)+'s'"><span id="av-blink-delay-val" style="font-size:10px;color:var(--text-dim);min-width:40px;flex-shrink:0;text-align:right">3.0s</span></div>
    <div class="setting-row" style="gap:12px">
      <label style="min-width:90px">ENABLE</label>
      <button class="btn on" id="av-talk-en" onclick="this.classList.toggle('on')" title="Talking animation">TALK</button>
      <button class="btn on" id="av-blink-en" onclick="this.classList.toggle('on')" title="Blink animation">BLINK</button>
      <button class="btn on" id="av-sleep-en" onclick="this.classList.toggle('on')" title="Sleep after idle">SLEEP</button>
    </div>
    <!-- Noise / effects config -->
    <div class="setting-row">
      <label style="min-width:90px">OVERLAY</label>
      <select id="av-noise-mode" style="flex:1" onchange="applyAvatarNoise()">
        <option value="none">None</option>
        <option value="scanlines">Scanlines</option>
        <option value="static">Static grain</option>
        <option value="mixed" selected>Scanlines + grain</option>
      </select>
      <span style="font-size:10px;color:var(--text-dim);flex-shrink:0;margin-left:6px">INTENSITY</span>
      <input type="range" id="av-noise-intensity" min="0" max="1" step="0.01" value="0.5" style="flex:1;max-width:70px" oninput="applyAvatarNoise();document.getElementById('av-noise-intensity-val').textContent=Math.round(this.value*100)+'%'">
      <span id="av-noise-intensity-val" style="font-size:10px;color:var(--text-dim);min-width:28px;flex-shrink:0;text-align:right">50%</span>
    </div>
    <div class="setting-row">
      <label style="min-width:90px">SCANLINES</label>
      <select id="av-scanline-mode" style="flex:1" onchange="applyAvatarNoise()">
        <option value="static">Static</option>
        <option value="roll" selected>Rolling</option>
        <option value="flicker">Flicker</option>
      </select>
      <span style="font-size:10px;color:var(--text-dim);flex-shrink:0;margin-left:6px">SPACING</span>
      <input type="range" id="av-scanline-spacing" min="2" max="12" step="1" value="4" style="flex:1;max-width:70px" oninput="applyAvatarNoise()">
    </div>
    <div class="setting-row">
      <label style="min-width:90px">COLOR TINT</label>
      <button class="btn" id="av-tint-btn" onclick="toggleAvTint()" title="Screen-door color wash matching UI green">OFF</button>
      <span style="font-size:10px;color:var(--text-dim);flex-shrink:0;margin-left:6px">STRENGTH</span>
      <input type="range" id="av-tint-intensity" min="0.02" max="0.5" step="0.01" value="0.12" style="flex:1" oninput="_avApplyTint()">
    </div>
    <div class="setting-row">
      <label style="min-width:90px">GLITCH</label>
      <button class="btn" id="av-glitch-btn" onclick="toggleAvGlitch()" title="Random horizontal glitch bars on talk/scream">OFF</button>
      <span style="font-size:10px;color:var(--text-dim);flex-shrink:0;margin-left:6px">INTENSITY</span>
      <input type="range" id="av-glitch-intensity" min="0.1" max="1" step="0.05" value="0.4" style="flex:1" oninput="">
    </div>
    <div class="setting-row">
      <label style="min-width:90px">BLUR FILTER</label>
      <button class="btn" id="av-pixel-btn" onclick="toggleAvPixel()" title="Soft blur filter on the avatar viewport">OFF</button>
      <span style="font-size:10px;color:var(--text-dim);flex-shrink:0;margin-left:6px">BLUR</span>
      <input type="range" id="av-pixel-size" min="0" max="3" step="0.1" value="1" style="flex:1;accent-color:var(--green);height:4px" oninput="_avApplyPixel()">
      <span id="av-pixel-size-val" style="font-size:10px;color:var(--text-dim);min-width:32px;flex-shrink:0;text-align:right">1px</span>
    </div>
    <div class="setting-row" id="av-pixel-bilinear-row" style="display:none">
      <label style="min-width:90px">CONTRAST</label>
      <input type="range" id="av-pixel-contrast" min="80" max="200" step="5" value="100" style="flex:1;accent-color:var(--green);height:4px" oninput="_avApplyPixel()">
      <span id="av-pixel-contrast-val" style="font-size:10px;color:var(--text-dim);min-width:36px;flex-shrink:0;text-align:right">100%</span>
    </div>
    <div class="setting-row" id="av-pixel-mode-row" style="display:none">
      <label style="min-width:90px">MODE</label>
      <button class="btn" id="av-pixel-bilinear-btn" onclick="toggleAvPixelBilinear()" title="Soft: blur only. Edge: blur + contrast">SOFT</button>
      <span style="font-size:10px;color:var(--text-dim);flex:1;margin-left:8px" id="av-pixel-mode-label">edge enhance · adds contrast</span>
    </div>
    <div class="setting-row">
      <label style="min-width:90px">WIREFRAME BG</label>
      <button class="btn" id="av-wire-btn" onclick="toggleAvWireframe()" title="Perspective grid background">OFF</button>
      <button class="btn" id="av-wire-dir-btn" onclick="toggleAvWireDir()" title="Travel direction">▶ FWD</button>
      <button class="btn on" id="av-wire-floor-btn" onclick="toggleAvWireAxis('floor')" title="Floor/ceiling planes">FLOOR</button>
      <button class="btn on" id="av-wire-walls-btn" onclick="toggleAvWireAxis('walls')" title="Left/right wall planes">WALLS</button>
    </div>
    <div class="setting-row">
      <label style="min-width:90px">WIRE DEPTH/SPD</label>
      <input type="range" id="av-wire-depth" min="0.3" max="1" step="0.05" value="0.7" style="flex:1" oninput="">
      <input type="range" id="av-wire-speed" min="0" max="0.5" step="0.01" value="0.15" style="flex:1" oninput="">
    </div>
    <div class="setting-row">
      <label style="min-width:90px">WAVE IN OVERLAY</label>
      <button class="btn on" id="av-wave-en" onclick="toggleAvatarWave()" title="Show/hide wave in avatar overlay">ON</button>
      <label style="min-width:60px;margin-left:16px">MAIN WAVE</label>
      <button class="btn on" id="main-wave-en" onclick="toggleWaveDisplay()" title="Show/hide main waveform">ON</button>
    </div>
      </div>

      <!-- ── shared footer (always visible) ── -->
      <div id="settings-footer" style="border-top:1px solid var(--border);padding-top:8px;margin-top:4px;display:flex;justify-content:flex-end;gap:8px;flex-shrink:0">
        <button class="btn" id="btn-save-footer" onclick="saveCurrentCharacter()" disabled title="Load a character first" style="margin-right:auto;font-size:11px;padding:5px 10px">💾 SAVE CHAR</button>
        <button class="btn danger" onclick="resetConversation()">RESET CONV</button>
        <button class="btn primary" onclick="applySettings()">APPLY</button>
      </div>

    </div>
  </div><!-- /settings-panel -->

  <!-- Avatar fullscreen overlay -->
  <div id="avatar-overlay">
    <div id="avatar-bezel">
      <div id="avatar-frame-border">
    <button id="avatar-close-btn" onclick="closeAvatarOverlay()" title="Close avatar">✕ CLOSE</button>
        <div id="avatar-viewport">
          <canvas id="avatar-wireframe"></canvas>
          <div id="avatar-pixel-wrap">
          <img id="avatar-img" src="" alt="" draggable="false">
          </div>
          <canvas id="avatar-pixel-canvas"></canvas>
          <canvas id="avatar-code-canvas"></canvas>
          <canvas id="avatar-scanlines"></canvas>
          <canvas id="avatar-static-canvas"></canvas>
          <div id="avatar-color-overlay"></div>
          <div id="avatar-sleep-overlay"></div>
          <div id="avatar-glitch-bar"></div>
          <div id="avatar-subtitle"></div>
        </div>
        <div id="avatar-zoom-controls">
          <button onclick="avZoom(-0.15)" title="Zoom out">−</button>
          <button onclick="avZoom(0.15)"  title="Zoom in">＋</button>
          <button id="avatar-zoom-reset" onclick="avZoomReset()" title="Reset pan/zoom">⊙</button>
          <button id="avatar-lock-btn" onclick="avToggleLock()" title="Lock position">🔒</button>
          <button id="avatar-cc-btn" onclick="toggleAvatarCC()" title="Toggle subtitles">CC</button>
        </div>
      </div>
    </div>
    <div id="avatar-wave-wrap">
      <canvas id="avatar-wave"></canvas>
    </div>
    <div id="avatar-char-nav" style="display:flex;align-items:center;justify-content:center;width:100%;background:var(--tint-slot);border-top:1px solid var(--tint-dark);flex-shrink:0">
      <button id="avatar-nav-left"  onclick="navigateCharacter('left')"  style="background:none;border:none;color:var(--green);font-size:20px;padding:4px 14px;cursor:pointer;font-family:var(--font-mono);opacity:0.7;flex-shrink:0" title="Previous character">&#8249;</button>
      <span id="avatar-char-name" style="flex:1;text-align:center;font-family:var(--font-mono);font-size:18px;color:var(--green);letter-spacing:4px;text-shadow:0 0 10px var(--green);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:6px 0 4px 0"></span>
      <button id="avatar-nav-right" onclick="navigateCharacter('right')" style="background:none;border:none;color:var(--green);font-size:20px;padding:4px 14px;cursor:pointer;font-family:var(--font-mono);opacity:0.7;flex-shrink:0" title="Next character">&#8250;</button>
    </div>
    <!-- Optional text input row (toggled by ✎ button) -->
    <div id="avatar-text-row" style="display:none;width:100%;max-width:540px;align-self:center;padding:4px 8px;background:var(--tint-slot);border-top:1px solid var(--tint-dark);box-sizing:border-box;gap:6px;align-items:center">
      <button class="btn" id="av-img-btn" onclick="document.getElementById('img-file-input').click()" title="Attach image" style="font-size:11px;padding:3px 10px;flex-shrink:0">IMG</button>
      <textarea id="avatar-msg-input" rows="1" placeholder="Type a message…" style="flex:1;background:var(--tint-slot);border:1px solid var(--tint-slot-border);border-radius:8px;color:var(--green);font-family:var(--font-mono);font-size:13px;padding:6px 10px;resize:none;min-height:36px;max-height:80px;outline:none;overflow-y:auto"></textarea>
      <button class="btn primary" onclick="sendAvatarText()" style="font-size:11px;padding:4px 10px">SEND</button>
    </div>
    <div id="avatar-hud">
      <button class="btn" id="avatar-ptt-btn">⬤ HOLD TO TALK</button>
      <div id="avatar-controls-row">
        <button class="btn on" id="av-ac-indicator" onclick="toggleAC()" title="Auto-continue — click to toggle">AC: ON</button>
        <button class="btn" id="av-init-indicator" onclick="toggleInitiative()" title="Initiative — click to toggle">◈ INIT</button>
        <button class="btn" id="avatar-mic-btn" onclick="toggleOpenMic()" title="Open-mic VAD mode">VAD</button>
        <button class="btn" id="avatar-mic-mute-btn" onclick="toggleMicMute()" title="Mute/unmute microphone" disabled style="opacity:0.35">🎙 MUTE</button>
        <button class="btn" onclick="stopAudio()">STOP</button>
        <button class="btn" id="avatar-text-toggle-btn" onclick="toggleAvatarTextInput()" title="Toggle text input">TEXT</button>
        <button class="btn" id="av-pixel-hud-btn" onclick="toggleAvPixel()" title="Blur filter">BLUR</button>
        <div style="display:flex;align-items:center;gap:3px" title="Subtitle speed">
          <span style="font-size:9px;color:var(--text-dim);font-family:var(--font-mono)">CC</span>
          <input type="range" id="av-sub-speed-slider" min="4" max="30" step="1" value="11"
                 style="width:52px;accent-color:var(--green);cursor:pointer"
                 oninput="_setSubSpeed(this.value)" title="Subtitle speed (chars/sec)">
          <span id="av-sub-speed-val" style="font-size:9px;color:var(--green);font-family:var(--font-mono);min-width:16px">11</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Waveform -->
  <div id="wave-area" style="display:flex;flex-direction:column;flex-shrink:0">
    <div style="display:flex;align-items:flex-end;gap:0">
      <button id="wave-toggle-btn" onclick="toggleWaveDisplay()" title="Show/hide waveform">▼ WAVE</button>
    </div>
    <div id="wave-wrap">
      <canvas id="wave"></canvas>
      <div id="playing-indicator" onclick="toggleSpeakingIndicator()" title="Click to hide">▶ SPEAKING</div>
      <div id="safety-light" onclick="openSafetyPanel()" title="Safety status — click for details" style="position:absolute;top:8px;right:8px;width:10px;height:10px;border-radius:50%;cursor:pointer;transition:background .3s,box-shadow .3s"></div>
      <button id="wave-mode-btn" onclick="cycleWaveMode()">RIBBON</button>
    </div>
  </div>

  <!-- Chat -->
  <div id="chat"></div>

  <!-- Input -->
  <input type="file" id="img-file-input" accept="image/*" style="display:none" onchange="onImageSelected(this)">
  <div id="input-area">
    <div id="img-preview-row">
      <img id="img-preview-thumb" src="" alt="">
      <span id="img-preview-name"></span>
      <button id="img-clear-btn" onclick="clearImage()" title="Remove image">✕</button>
    </div>
    <div id="text-row">
      <button class="btn" id="img-btn" onclick="document.getElementById('img-file-input').click()" title="Attach image">IMG</button>
      <textarea id="msg-input" rows="1" placeholder="Type a message…"></textarea>
      <button class="btn primary" id="send-btn" onclick="sendText()">SEND</button>
    </div>
    <button class="btn" id="ptt-btn">⬤ HOLD TO TALK</button>
    <div id="controls-row">
      <button class="btn on" id="ac-indicator" onclick="toggleAC()" title="Auto-continue — click to toggle">AC: ON</button>
      <button class="btn" id="init-indicator" onclick="toggleInitiative()" title="Initiative — click to toggle">◈ INIT</button>
      <button class="btn" id="open-mic-btn" onclick="toggleOpenMic()" title="Open-mic VAD mode">VAD</button>
      <button class="btn" id="mic-mute-btn" onclick="toggleMicMute()" title="Mute/unmute microphone" disabled style="opacity:0.35">🎙 MUTE</button>
      <button class="btn" onclick="stopAudio()">STOP</button>
      <div style="display:flex;align-items:center;gap:4px;margin-left:auto" title="Subtitle speed">
        <span style="font-size:9px;color:var(--text-dim);font-family:var(--font-mono)">CC</span>
        <input type="range" id="sub-speed-slider" min="4" max="30" step="1" value="11"
               style="width:64px;accent-color:var(--green);cursor:pointer"
               oninput="_setSubSpeed(this.value)" title="Subtitle speed (chars/sec)">
        <span id="sub-speed-val" style="font-size:9px;color:var(--green);font-family:var(--font-mono);min-width:16px">11</span>
      </div>
    </div>
  </div>

</div>

<script>
// ── State ──
let voices=[], currentVoice='', masterGain=1.5;
let acEnabled=true, acMode='standard';
let isPlaying=false, isBusy=false;
let waveAmp=1.0;
let waveFade=0.25;   // smoothing alpha — how quickly wave decays to silence
const WAVE_NOISE_FLOOR=0.004; // gate: amplitude below this snaps to zero
let _ttsGeneration=0; // incremented on every stopAudio to invalidate in-flight playTTS
let mediaRecorder=null, audioChunks=[], pttActive=false;
let audioCtx=null, gainNode=null, analyserNode=null, currentSource=null;
let stopCurrentAudio=false;
let waveMode=0;
const waveModes=['ribbon','wave','bars','radial'];
const waveModeLabels=['RIBBON','WAVE','BARS','RADIAL'];
let animFrameId=null, acEventSource=null;
let providerRegistry={}, ttsProviderRegistry={};

// ── Reverb / Delay / FX ──
let _irBuffer=null;         // loaded ConvolverNode impulse response
let _irB64=null;            // base64 of raw IR WAV for persistence
let _irName='';             // filename for display
let _fxEnabled=false;       // master FX on/off
const fxRowEnabled={reverb:true,delay:true,crush:true,chorus:true,ring:true,dist:true};

function toggleFxRow(name){
  fxRowEnabled[name]=!fxRowEnabled[name];
  const on=fxRowEnabled[name];
  document.getElementById('fxtog-'+name).className='fx-tog'+(on?' on':'');
  const row=document.getElementById('fx-row-'+name);
  if(on) row.classList.remove('fx-off'); else row.classList.add('fx-off');
  _buildReverbGraph();
}

function toggleFX(){
  _fxEnabled=!_fxEnabled;
  const btn=document.getElementById('fx-toggle');
  btn.textContent=_fxEnabled?'FX ON':'FX OFF';
  btn.className='btn'+(_fxEnabled?' on':'');
  const bank=document.getElementById('fx-bank');
  if(_fxEnabled) bank.classList.remove('fx-bank-off');
  else bank.classList.add('fx-bank-off');
  _buildReverbGraph();
}

async function loadIR(input){
  const file=input.files[0]; if(!file) return;
  ensureAudio();
  const buf=await file.arrayBuffer();
  // Store as base64 for persistence in character/session
  const bytes=new Uint8Array(buf);
  let bin=''; for(let i=0;i<bytes.length;i++) bin+=String.fromCharCode(bytes[i]);
  _irB64=btoa(bin);
  _irName=file.name;
  _irBuffer=await audioCtx.decodeAudioData(buf);
  document.getElementById('reverb-ir-name').textContent=_irName;
  _buildReverbGraph();
}

async function loadIRFromB64(b64, name){
  if(!b64) return;
  ensureAudio();
  try{
    const bin=atob(b64);
    const bytes=new Uint8Array(bin.length);
    for(let i=0;i<bin.length;i++) bytes[i]=bin.charCodeAt(i);
    _irB64=b64; _irName=name||'loaded IR';
    _irBuffer=await audioCtx.decodeAudioData(bytes.buffer);
    document.getElementById('reverb-ir-name').textContent=_irName;
    _buildReverbGraph();
  }catch(e){console.error('[IR] Failed to decode saved IR:',e);}
}

// FX nodes
let _reverbNode=null,_reverbDryGain=null,_reverbWetGain=null,_preDelayNode=null;
let _delayNode=null,_delayFeedback=null,_delayFilter=null,_delayWetGain=null;
let _chorusDelay=null,_chorusLFO=null,_chorusLFOGain=null,_chorusDryGain=null,_chorusWetGain=null,_chorusMerge=null;
let _ringCarrier=null,_ringMod=null,_ringWetGain=null,_ringDryGain=null,_ringMerge=null;
let _crushNode=null,_crushDryGain=null,_crushWetGain=null,_crushMerge=null;
let _distNode=null,_distDryGain=null,_distWetGain=null,_distMerge=null;

// S-curve mapping: slider 0-1 → actual wet value (sensitive at low end, heavy at high)
function _scurve(v){ return v*v*(3-2*v); }

function _buildReverbGraph(){
  if(!audioCtx) return;
  // Tear down all old nodes safely
  for(const n of [_distNode,_distDryGain,_distWetGain,_distMerge,
                   _reverbNode,_reverbDryGain,_reverbWetGain,_preDelayNode,
                   _delayNode,_delayFeedback,_delayFilter,_delayWetGain,
                   _chorusDelay,_chorusLFO,_chorusLFOGain,_chorusDryGain,_chorusWetGain,_chorusMerge,
                   _ringCarrier,_ringMod,_ringWetGain,_ringDryGain,_ringMerge,
                   _crushNode,_crushDryGain,_crushWetGain,_crushMerge]){
    if(n){try{n.disconnect();}catch(e){}}
  }
  try{gainNode.disconnect();}catch(e){}

  if(!_fxEnabled){
    gainNode.connect(analyserNode);
    analyserNode.connect(audioCtx.destination);
    return;
  }

  // Read slider values — wet sliders use S-curve; gated by per-row enable toggles
  const reverbWet  = fxRowEnabled.reverb  ? _scurve(parseFloat(document.getElementById('reverb-wet').value||0))  : 0;
  const preMs      = parseFloat(document.getElementById('reverb-predelay').value||0)/1000;
  const delMs      = parseFloat(document.getElementById('delay-time').value||0)/1000;
  const delWet     = fxRowEnabled.delay   ? _scurve(parseFloat(document.getElementById('delay-wet').value||0))   : 0;
  const fb         = parseFloat(document.getElementById('delay-feedback').value||0.35);
  const chorusWet  = fxRowEnabled.chorus  ? _scurve(parseFloat(document.getElementById('chorus-wet').value||0))  : 0;
  const chorusDepth= parseFloat(document.getElementById('chorus-depth').value||0.005);
  const chorusRate = parseFloat(document.getElementById('chorus-rate').value||1.2);
  const ringWet    = fxRowEnabled.ring    ? _scurve(parseFloat(document.getElementById('ringmod-wet').value||0))  : 0;
  const ringFreq   = parseFloat(document.getElementById('ringmod-freq').value||120);
  const crushWet   = fxRowEnabled.crush   ? _scurve(parseFloat(document.getElementById('crush-wet').value||0))   : 0;
  const crushBits  = parseInt(document.getElementById('crush-bits').value||8);
  const crushSR    = parseInt(document.getElementById('crush-sr').value||4);

  // Chain: gainNode → [Crush] → [Chorus] → [RingMod] → [Reverb] → [Delay] → analyser

  let chainIn = gainNode;

  // ── 0. BITCRUSHER ──────────────────────────────────────────────────────────
  // Bit depth: WaveShaperNode with quantisation curve — zero latency, no buffer delay.
  // SR reduction: ScriptProcessorNode with bufSize=256 (~5.8ms) — only active when SR>1.
  if(crushWet>0){
    const steps=Math.pow(2,crushBits);
    // Build a quantisation transfer curve: 65536 points, input [-1,1] → quantised [-1,1]
    const curveLen=65536;
    const curve=new Float32Array(curveLen);
    for(let i=0;i<curveLen;i++){
      const x=(i/curveLen)*2-1; // -1 to +1
      curve[i]=Math.round(x*(steps/2))/(steps/2);
    }
    _crushNode=audioCtx.createWaveShaper();
    _crushNode.curve=curve;
    _crushNode.oversample='none';

    _crushDryGain=audioCtx.createGain(); _crushDryGain.gain.value=1-crushWet;
    _crushWetGain=audioCtx.createGain(); _crushWetGain.gain.value=crushWet;
    _crushMerge=audioCtx.createGain();
    chainIn.connect(_crushDryGain); _crushDryGain.connect(_crushMerge);

    if(crushSR>1){
      // SR reduction via ScriptProcessorNode — bufSize=256 keeps latency ~5.8ms
      let _srNode=audioCtx.createScriptProcessor(256,1,1);
      let _holdSample=0,_holdCount=0;
      _srNode.onaudioprocess=function(e){
        const inp=e.inputBuffer.getChannelData(0);
        const out=e.outputBuffer.getChannelData(0);
        for(let i=0;i<inp.length;i++){
          if(_holdCount<=0){_holdSample=inp[i];_holdCount=crushSR;}
          out[i]=_holdSample;_holdCount--;
        }
      };
      chainIn.connect(_crushNode); _crushNode.connect(_srNode); _srNode.connect(_crushWetGain);
      // Keep _srNode alive (GC protection)
      _crushNode._srNode=_srNode;
    } else {
      // Bit depth only — fully zero latency
      chainIn.connect(_crushNode); _crushNode.connect(_crushWetGain);
    }
    _crushWetGain.connect(_crushMerge);
    chainIn = _crushMerge;
  }
  if(chorusWet>0){
    _chorusDryGain=audioCtx.createGain(); _chorusDryGain.gain.value=1-chorusWet;
    _chorusWetGain=audioCtx.createGain(); _chorusWetGain.gain.value=chorusWet;
    _chorusDelay=audioCtx.createDelay(0.1); _chorusDelay.delayTime.value=0.025;
    _chorusLFO=audioCtx.createOscillator(); _chorusLFO.type='sine'; _chorusLFO.frequency.value=chorusRate;
    _chorusLFOGain=audioCtx.createGain(); _chorusLFOGain.gain.value=chorusDepth;
    _chorusMerge=audioCtx.createGain();
    chainIn.connect(_chorusDryGain); _chorusDryGain.connect(_chorusMerge);
    chainIn.connect(_chorusDelay); _chorusDelay.connect(_chorusWetGain); _chorusWetGain.connect(_chorusMerge);
    _chorusLFO.connect(_chorusLFOGain); _chorusLFOGain.connect(_chorusDelay.delayTime);
    _chorusLFO.start();
    chainIn = _chorusMerge;
  }

  // ── 2. RING MODULATOR (robotic/alien) ─────────────────────────────────────
  if(ringWet>0){
    _ringCarrier=audioCtx.createOscillator(); _ringCarrier.frequency.value=ringFreq;
    _ringMod=audioCtx.createGain(); _ringMod.gain.value=0; // carrier drives gain.value
    _ringCarrier.connect(_ringMod.gain);
    _ringDryGain=audioCtx.createGain(); _ringDryGain.gain.value=1-ringWet;
    _ringWetGain=audioCtx.createGain(); _ringWetGain.gain.value=ringWet;
    _ringMerge=audioCtx.createGain();
    chainIn.connect(_ringDryGain); _ringDryGain.connect(_ringMerge);
    chainIn.connect(_ringMod); _ringMod.connect(_ringWetGain); _ringWetGain.connect(_ringMerge);
    _ringCarrier.start();
    chainIn = _ringMerge;
  }

  // ── 2b. DISTORTION ──────────────────────────────────────────────────────────
  const distWet  = fxRowEnabled.dist ? _scurve(parseFloat(document.getElementById('dist-wet').value||0)) : 0;
  const distDrive= parseFloat(document.getElementById('dist-drive').value||20);
  if(distWet>0){
    // Soft-clip waveshaper — tanh-based curve, warm saturation at low drive
    const k=distDrive; const n=256;
    const distCurve=new Float32Array(n);
    for(let i=0;i<n;i++){
      const x=(i*2/(n-1))-1;
      distCurve[i]=Math.tanh(k*x)/Math.tanh(k);
    }
    _distNode=audioCtx.createWaveShaper(); _distNode.curve=distCurve; _distNode.oversample='4x';
    _distDryGain=audioCtx.createGain(); _distDryGain.gain.value=1-distWet;
    _distWetGain=audioCtx.createGain(); _distWetGain.gain.value=distWet*0.7; // compensate loudness
    _distMerge=audioCtx.createGain();
    chainIn.connect(_distDryGain); _distDryGain.connect(_distMerge);
    chainIn.connect(_distNode); _distNode.connect(_distWetGain); _distWetGain.connect(_distMerge);
    chainIn=_distMerge;
  }

  // ── 3. REVERB ─────────────────────────────────────────────────────────────
  if(_irBuffer && reverbWet>0){
    _preDelayNode=audioCtx.createDelay(0.1); _preDelayNode.delayTime.value=preMs;
    _reverbNode=audioCtx.createConvolver(); _reverbNode.buffer=_irBuffer;
    _reverbWetGain=audioCtx.createGain(); _reverbWetGain.gain.value=reverbWet;
    _reverbDryGain=audioCtx.createGain(); _reverbDryGain.gain.value=1-reverbWet;
    const reverbMix=audioCtx.createGain();
    chainIn.connect(_reverbDryGain); _reverbDryGain.connect(reverbMix);
    chainIn.connect(_preDelayNode); _preDelayNode.connect(_reverbNode);
    _reverbNode.connect(_reverbWetGain); _reverbWetGain.connect(reverbMix);
    chainIn = reverbMix;
  }

  // ── 4. DELAY ──────────────────────────────────────────────────────────────
  if(delMs>0 && delWet>0){
    _delayNode=audioCtx.createDelay(1.0); _delayNode.delayTime.value=delMs;
    _delayFilter=audioCtx.createBiquadFilter(); _delayFilter.type='lowpass'; _delayFilter.frequency.value=4000;
    _delayFeedback=audioCtx.createGain(); _delayFeedback.gain.value=fb;
    _delayWetGain=audioCtx.createGain(); _delayWetGain.gain.value=delWet;
    const delayMix=audioCtx.createGain();
    chainIn.connect(delayMix);
    chainIn.connect(_delayNode);
    _delayNode.connect(_delayFilter);
    _delayFilter.connect(_delayFeedback); _delayFeedback.connect(_delayNode);
    _delayFilter.connect(_delayWetGain); _delayWetGain.connect(delayMix);
    chainIn = delayMix;
  }

  chainIn.connect(analyserNode);
  analyserNode.connect(audioCtx.destination);
}

function updateCrush(){
  const v=parseFloat(document.getElementById('crush-wet').value||0);
  document.getElementById('crush-wet-val').textContent=Math.round(_scurve(v)*100)+'%';
  document.getElementById('crush-bits-val').textContent=document.getElementById('crush-bits').value;
  document.getElementById('crush-sr-val').textContent='÷'+document.getElementById('crush-sr').value;
  if(_fxEnabled) _buildReverbGraph();
}
function updateReverb(){
  const v=parseFloat(document.getElementById('reverb-wet').value||0);
  document.getElementById('reverb-wet-val').textContent=Math.round(_scurve(v)*100)+'%';
  document.getElementById('reverb-predelay-val').textContent=document.getElementById('reverb-predelay').value+'ms';
  if(_fxEnabled) _buildReverbGraph();
}
function updateDelay(){
  const v=parseFloat(document.getElementById('delay-wet').value||0);
  const f=parseFloat(document.getElementById('delay-feedback').value||0);
  document.getElementById('delay-wet-val').textContent=Math.round(_scurve(v)*100)+'%';
  document.getElementById('delay-time-val').textContent=document.getElementById('delay-time').value+'ms';
  document.getElementById('delay-fb-val').textContent=Math.round(f*100)+'%';
  if(_fxEnabled) _buildReverbGraph();
}
function updateChorus(){
  const v=parseFloat(document.getElementById('chorus-wet').value||0);
  document.getElementById('chorus-wet-val').textContent=Math.round(_scurve(v)*100)+'%';
  if(_fxEnabled) _buildReverbGraph();
}
function updateRingMod(){
  const v=parseFloat(document.getElementById('ringmod-wet').value||0);
  document.getElementById('ringmod-wet-val').textContent=Math.round(_scurve(v)*100)+'%';
  document.getElementById('ringmod-freq-val').textContent=document.getElementById('ringmod-freq').value+'Hz';
  if(_fxEnabled) _buildReverbGraph();
}

function updateDist(){
  const v=parseFloat(document.getElementById('dist-wet').value||0);
  document.getElementById('dist-wet-val').textContent=Math.round(_scurve(v)*100)+'%';
  document.getElementById('dist-drive-val').textContent=document.getElementById('dist-drive').value;
  if(_fxEnabled) _buildReverbGraph();
  _saveFx({dist_wet:v,dist_drive:parseFloat(document.getElementById('dist-drive').value)});
}


// Lightweight save helper — only sends changed keys
function _saveFx(obj){
  fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(obj)}).catch(()=>{});
}


// ── FX Presets ───────────────────────────────────────────────────────────────
function applyFxPreset(name){
  const presets={
    // Warm: light reverb, subtle dist, no pitch
    warm:{fx:true, reverb_wet:0.18,reverb_predelay:15,delay_wet:0,delay_time:0,delay_feedback:0.35,
          chorus_wet:0.08,chorus_depth:0.005,chorus_rate:1.0,
          ringmod_wet:0,ringmod_freq:120,crush_wet:0,crush_bits:8,crush_sr:1,
          dist_wet:0.15,dist_drive:12},
    // Radio: bitcrush + dist, no reverb
    radio:{fx:true, reverb_wet:0,reverb_predelay:0,delay_wet:0,delay_time:0,delay_feedback:0,
           chorus_wet:0,chorus_depth:0.005,chorus_rate:1.2,
           ringmod_wet:0,ringmod_freq:120,crush_wet:0.55,crush_bits:10,crush_sr:3,
           dist_wet:0.25,dist_drive:30},
    // Echo: slapback delay + light reverb
    echo:{fx:true, reverb_wet:0.2,reverb_predelay:20,delay_wet:0.3,delay_time:220,delay_feedback:0.38,
          chorus_wet:0,chorus_depth:0.005,chorus_rate:1.2,
          ringmod_wet:0,ringmod_freq:120,crush_wet:0,crush_bits:8,crush_sr:1,
          dist_wet:0,dist_drive:20},
    // Robot: ring mod + pitch down a touch
    robot:{fx:true, reverb_wet:0.1,reverb_predelay:10,delay_wet:0,delay_time:0,delay_feedback:0.3,
           chorus_wet:0,chorus_depth:0.005,chorus_rate:1.2,
           ringmod_wet:0.45,ringmod_freq:80,crush_wet:0.2,crush_bits:8,crush_sr:2,
           dist_wet:0.1,dist_drive:15,pitch_semitones:-2},
    // Clear: everything off
    // Glitch: low-bit crush + fast ringmod + dist — broken digital / data corruption
    glitch:{fx:true, reverb_wet:0,reverb_predelay:0,delay_wet:0.08,delay_time:30,delay_feedback:0.2,
            chorus_wet:0,chorus_depth:0.005,chorus_rate:1.2,
            ringmod_wet:0.35,ringmod_freq:280,crush_wet:0.8,crush_bits:4,crush_sr:8,
            dist_wet:0.35,dist_drive:45},
    // Dreamy: lush chorus + long reverb + hint of delay — floaty, detached
    dreamy:{fx:true, reverb_wet:0.45,reverb_predelay:35,delay_wet:0.18,delay_time:380,delay_feedback:0.42,
            chorus_wet:0.55,chorus_depth:0.014,chorus_rate:0.5,
            ringmod_wet:0,ringmod_freq:120,crush_wet:0,crush_bits:8,crush_sr:1,
            dist_wet:0,dist_drive:20},
    // Intercom: heavy crush + dist + tight delay — walkie-talkie / PA system
    intercom:{fx:true, reverb_wet:0,reverb_predelay:0,delay_wet:0.12,delay_time:60,delay_feedback:0.15,
              chorus_wet:0,chorus_depth:0.005,chorus_rate:1.2,
              ringmod_wet:0,ringmod_freq:120,crush_wet:0.7,crush_bits:6,crush_sr:5,
              dist_wet:0.4,dist_drive:55},
    // Alien: high-freq ringmod + chorus + light crush — otherworldly metallic shimmer
    alien:{fx:true, reverb_wet:0.15,reverb_predelay:8,delay_wet:0,delay_time:0,delay_feedback:0.3,
           chorus_wet:0.3,chorus_depth:0.008,chorus_rate:3.5,
           ringmod_wet:0.6,ringmod_freq:440,crush_wet:0.12,crush_bits:12,crush_sr:1,
           dist_wet:0.08,dist_drive:10},
    // Tape: warm saturation + slow chorus + short pre-delay — vintage analogue feel
    tape:{fx:true, reverb_wet:0.22,reverb_predelay:8,delay_wet:0,delay_time:0,delay_feedback:0.35,
          chorus_wet:0.2,chorus_depth:0.007,chorus_rate:0.7,
          ringmod_wet:0,ringmod_freq:120,crush_wet:0,crush_bits:8,crush_sr:1,
          dist_wet:0.3,dist_drive:18},
    // Cavern: huge reverb + long delay + low ringmod rumble — vast underground space
    cavern:{fx:true, reverb_wet:0.65,reverb_predelay:55,delay_wet:0.35,delay_time:500,delay_feedback:0.55,
            chorus_wet:0,chorus_depth:0.005,chorus_rate:1.2,
            ringmod_wet:0.08,ringmod_freq:40,crush_wet:0,crush_bits:8,crush_sr:1,
            dist_wet:0.05,dist_drive:8},
  };
  const p=presets[name]; if(!p) return;
  // Apply to sliders
  const set=(id,v)=>{const el=document.getElementById(id);if(el)el.value=v;};
  set('reverb-wet',p.reverb_wet); set('reverb-predelay',p.reverb_predelay);
  set('delay-wet',p.delay_wet); set('delay-time',p.delay_time); set('delay-feedback',p.delay_feedback);
  set('chorus-wet',p.chorus_wet); set('chorus-depth',p.chorus_depth); set('chorus-rate',p.chorus_rate);
  set('ringmod-wet',p.ringmod_wet); set('ringmod-freq',p.ringmod_freq);
  set('crush-wet',p.crush_wet); set('crush-bits',p.crush_bits); set('crush-sr',p.crush_sr);
  set('dist-wet',p.dist_wet); set('dist-drive',p.dist_drive);  // Fire update fns to refresh displays and rebuild chain
  updateReverb(); updateDelay(); updateChorus(); updateRingMod(); updateCrush(); updateDist();
  // Set FX master toggle
  if(p.fx!==_fxEnabled){
    _fxEnabled=p.fx;
    const btn=document.getElementById('fx-toggle');
    if(btn){btn.textContent=_fxEnabled?'FX ON':'FX OFF';btn.className='btn'+(_fxEnabled?' on':'');}
    const bank=document.getElementById('fx-bank');
    if(bank){if(_fxEnabled)bank.classList.remove('fx-bank-off');else bank.classList.add('fx-bank-off');}
  }
  _buildReverbGraph();
  // Persist entire preset
  fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
    fx_enabled:p.fx,reverb_wet:p.reverb_wet,reverb_predelay:p.reverb_predelay,
    delay_wet:p.delay_wet,delay_time:p.delay_time,delay_feedback:p.delay_feedback,
    chorus_wet:p.chorus_wet,chorus_depth:p.chorus_depth,chorus_rate:p.chorus_rate,
    ringmod_wet:p.ringmod_wet,ringmod_freq:p.ringmod_freq,
    crush_wet:p.crush_wet,crush_bits:p.crush_bits,crush_sr:p.crush_sr,
    dist_wet:p.dist_wet,dist_drive:p.dist_drive,pitch_semitones:p.pitch_semitones,
  })}).catch(()=>{});
}

// ── Open mic / VAD ──
let openMicState='off', openMicVAD=null;
let _vadLoaded=false, _vadLoading=false;
let _micMuted=false;
let _resumeText='', _currentReplyText='', _ttsPlayStartTime=0;
const CHARS_PER_SECOND=14;

// ── Image attachment ──
let pendingImageB64=null, pendingImageMime='image/jpeg', pendingImageName='';

// ── Audio ─────────────────────────────────────────────────────────────────
function ensureAudio(){
  if(audioCtx) return;
  audioCtx=new(window.AudioContext||window.webkitAudioContext)();
  gainNode=audioCtx.createGain(); gainNode.gain.value=masterGain;
  analyserNode=audioCtx.createAnalyser(); analyserNode.fftSize=2048;
  gainNode.connect(analyserNode); analyserNode.connect(audioCtx.destination);
  startWaveform();
}

// PCM streaming constants — must match what tts.py sends
const PCM_SAMPLE_RATE=44100, PCM_CHANNELS=1, CHUNK_SAMPLES=4096, WAV_HEADER_BYTES=44;
let _ttsAbortCtrl=null, _chatAbortCtrl=null, _scheduledSources=[];
let _ttsGainNode=null;

function _ensureTTSGain(){
  if(_ttsGainNode){try{_ttsGainNode.disconnect();}catch(e){}}
  _ttsGainNode=audioCtx.createGain();
  _ttsGainNode.gain.value=1.0;
  _ttsGainNode.connect(gainNode);
  return _ttsGainNode;
}

function stopAudio(){
  _ttsGeneration++;
  stopCurrentAudio=true;
  // Abort in-flight TTS stream
  if(_ttsAbortCtrl){_ttsAbortCtrl.abort();_ttsAbortCtrl=null;}
  // Abort in-flight /chat fetch — prevents double TTS call and server OOM
  if(_chatAbortCtrl){_chatAbortCtrl.abort();_chatAbortCtrl=null;}
  // Zero gain on audio thread first (sample-accurate, no rollover), then disconnect
  if(_ttsGainNode){
    try{_ttsGainNode.gain.cancelScheduledValues(0);_ttsGainNode.gain.setValueAtTime(0,audioCtx.currentTime);}catch(e){}
    try{_ttsGainNode.disconnect();}catch(e){}
    _ttsGainNode=null;
  }
  for(const src of _scheduledSources){try{src.stop();}catch(e){}}
  _scheduledSources=[];currentSource=null;isPlaying=false;isBusy=false;
  document.getElementById('playing-indicator').classList.remove('show');
  // Clear subtitles immediately on barge-in / stop
  if(typeof _subClear==='function') _subClear(false);
  // Return the cancel POST promise so callers can await server acknowledgement
  // before starting a new TTS request — prevents concurrent inference on EchoTTS.
  return fetch('/tts/cancel',{method:'POST'}).catch(()=>{});
}

async function playTTS(text,gen){
  if(!text||text==='...') return;
  ensureAudio();
  stopCurrentAudio=false; _scheduledSources=[]; isPlaying=true;
  // Wake avatar from sleep immediately when TTS starts — don't wait for analyser
  if (_avIsSleeping) { _avIsSleeping = false; _avUpdateDisplay(); }
  document.getElementById('playing-indicator').classList.add('show');
  _ttsAbortCtrl=new AbortController();
  // Prime subtitles as soon as we start — timing anchored to _ttsPlayStartTime
  // which gets set on the first scheduleChunk call below
  let _subStarted = false;
  const myGain=_ensureTTSGain();
  try{
    const res=await fetch('/tts',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text}),signal:_ttsAbortCtrl.signal});
    if(!res.ok){isPlaying=false;document.getElementById('playing-indicator').classList.remove('show');return;}

    // ── MP3 path (ElevenLabs) ─────────────────────────────────────────────
    // Buffer the full response, decode via decodeAudioData, play as one source.
    // True streaming MP3 decode would need a JS MP3 parser; not worth the dep.
    const _audioFmt=res.headers.get('X-Audio-Format')||'';
    const _ct=(res.headers.get('Content-Type')||'').toLowerCase();
    if(_audioFmt==='mp3'||_ct.includes('audio/mpeg')||_ct.includes('audio/mp3')){
      const chunks=[];
      const reader=res.body.getReader();
      while(true){
        const{done,value}=await reader.read();
        if(done||stopCurrentAudio||_ttsGeneration!==gen) break;
        if(value) chunks.push(value);
      }
      if(stopCurrentAudio||_ttsGeneration!==gen) return;
      // Concatenate all chunks into a single ArrayBuffer
      const totalLen=chunks.reduce((s,c)=>s+c.length,0);
      const merged=new Uint8Array(totalLen);
      let off=0; for(const c of chunks){merged.set(c,off);off+=c.length;}
      let audioBuf;
      try{ audioBuf=await audioCtx.decodeAudioData(merged.buffer); }
      catch(e){ console.error('[TTS/MP3] decodeAudioData failed',e); return; }
      if(stopCurrentAudio||_ttsGeneration!==gen) return;
      const src=audioCtx.createBufferSource(); src.buffer=audioBuf;
      src.connect(myGain);
      src.onended=()=>{const idx=_scheduledSources.indexOf(src);if(idx!==-1)_scheduledSources.splice(idx,1);};
      _scheduledSources.push(src); currentSource=src;
      const startAt=audioCtx.currentTime+0.06;
      src.start(startAt);
      // Start subtitles
      if(!_subStarted){_subStarted=true;if(typeof _subStart==='function')_subStart(text,gen);}
      // Wait for playback to finish
      const latency=(audioCtx.outputLatency||audioCtx.baseLatency||0);
      const deadline=startAt+audioBuf.duration-audioCtx.currentTime+latency;
      if(deadline>-0.5) await new Promise(r=>setTimeout(r,Math.max(0,deadline*1000)+500));
      while(_scheduledSources.length>0&&!stopCurrentAudio&&_ttsGeneration===gen){
        await new Promise(r=>setTimeout(r,80));
      }
      return; // MP3 path done — skip PCM path below
    }

    // ── PCM/WAV path (EchoTTS, Kokoro, AllTalk…) ─────────────────────────
    const reader=res.body.getReader();
    const BYTES_PER_SAMPLE=2, SCHEDULE_BYTES=CHUNK_SAMPLES*BYTES_PER_SAMPLE;
    let headerLeft=WAV_HEADER_BYTES, pcmBuf=new Uint8Array(0), schedTime=0, started=false;

    function appendBytes(b){const m=new Uint8Array(pcmBuf.length+b.length);m.set(pcmBuf);m.set(b,pcmBuf.length);pcmBuf=m;}

    function scheduleChunk(bytes){
      if(_ttsGeneration!==gen) return;
      const int16=new Int16Array(bytes.buffer,bytes.byteOffset,bytes.byteLength>>1);
      const f32=new Float32Array(int16.length);
      for(let i=0;i<int16.length;i++) f32[i]=int16[i]/32768.0;
      const buf=audioCtx.createBuffer(PCM_CHANNELS,f32.length,PCM_SAMPLE_RATE);
      buf.copyToChannel(f32,0);
      const src=audioCtx.createBufferSource(); src.buffer=buf;
      src.connect(myGain);
      src.onended=()=>{const idx=_scheduledSources.indexOf(src);if(idx!==-1)_scheduledSources.splice(idx,1);};
      if(!started){schedTime=audioCtx.currentTime+0.06;started=true;
        // Start subtitles on first audio chunk — _ttsPlayStartTime already set by caller
        if(!_subStarted){_subStarted=true;if(typeof _subStart==='function')_subStart(text,gen);}
      }
      src.start(schedTime); schedTime+=buf.duration; _scheduledSources.push(src); currentSource=src;
    }

    function flushScheduled(){
      while(pcmBuf.length>=SCHEDULE_BYTES){
        scheduleChunk(pcmBuf.slice(0,SCHEDULE_BYTES));
        pcmBuf=pcmBuf.slice(SCHEDULE_BYTES);
      }
    }

    while(true){
      const{done,value}=await reader.read();
      if(stopCurrentAudio||done||_ttsGeneration!==gen) break;
      let incoming=value;
      if(headerLeft>0){
        if(incoming.length<=headerLeft){headerLeft-=incoming.length;continue;}
        incoming=incoming.slice(headerLeft);headerLeft=0;
      }
      appendBytes(incoming); flushScheduled();
    }
    if(!stopCurrentAudio&&_ttsGeneration===gen&&pcmBuf.length>=2){
      const trim=pcmBuf.length&~1;
      scheduleChunk(pcmBuf.slice(0,trim));
    }
    if(started&&!stopCurrentAudio&&_ttsGeneration===gen){
      const latency=(audioCtx.outputLatency||audioCtx.baseLatency||0);
      const deadline=schedTime-audioCtx.currentTime+latency;
      if(deadline>-0.5) await new Promise(r=>setTimeout(r,Math.max(0,deadline*1000)+500));
      while(_scheduledSources.length>0&&!stopCurrentAudio&&_ttsGeneration===gen){
        await new Promise(r=>setTimeout(r,80));
      }
    }
  }catch(e){if(e.name!=='AbortError')console.error('[TTS]',e);}
  finally{_scheduledSources=[];isPlaying=false;document.getElementById('playing-indicator').classList.remove('show');}
}

// ── Waveform ──────────────────────────────────────────────────────────────
const canvas=document.getElementById('wave');
const ctx2d=canvas.getContext('2d');
const silentBuf=new Float32Array(2048);
// Set canvas resolution to device pixel ratio for sharp rendering
(function resizeCanvas(){
  const dpr=window.devicePixelRatio||1;
  const rect=canvas.getBoundingClientRect();
  canvas.width=rect.width*dpr; canvas.height=rect.height*dpr;
  ctx2d.scale(dpr,dpr);
})();
function startWaveform(){} // no-op — drawWave loop starts immediately below
function cycleWaveMode(){
  waveMode=(waveMode+1)%waveModes.length;
  document.getElementById('wave-mode-btn').textContent=waveModeLabels[waveMode];
  _saveWaveState();
}

// Smoothed wave buffer — persists between frames
let _waveSmoothed=null;

// ── UI Hue ─────────────────────────────────────────────────────────────────
let _uiHue = 140;
function _uiC(alpha) { return `hsla(${_uiHue},100%,64%,${typeof alpha==='number'?alpha.toFixed(2):alpha})`; }
function applyUIHue(hue) {
  _uiHue = parseInt(hue);
  document.documentElement.style.setProperty('--hue', _uiHue);
  const val = document.getElementById('s-ui-hue-val');
  if (val) val.textContent = _uiHue;
}

function drawWave(){
  requestAnimationFrame(drawWave);
  const W=canvas.width/devicePixelRatio,H=canvas.height/devicePixelRatio;
  let raw=silentBuf;
  if(analyserNode){const b=new Float32Array(analyserNode.fftSize);analyserNode.getFloatTimeDomainData(b);raw=b;}

  // Init smoothed buffer on first run or size change
  if(!_waveSmoothed||_waveSmoothed.length!==raw.length) _waveSmoothed=new Float32Array(raw.length);

  // Smooth: lerp toward new data, then apply noise floor gate
  const alpha=1-waveFade; // high fade → low alpha → slower response
  for(let i=0;i<raw.length;i++){
    _waveSmoothed[i]=_waveSmoothed[i]*(waveFade)+(raw[i]*alpha);
    if(Math.abs(_waveSmoothed[i])<WAVE_NOISE_FLOOR) _waveSmoothed[i]=0;
  }
  const data=_waveSmoothed;

  ctx2d.clearRect(0,0,W,H);ctx2d.fillStyle='#141414';ctx2d.fillRect(0,0,W,H);
  const green=getComputedStyle(document.documentElement).getPropertyValue('--green').trim()||'#4cff7a',mode=waveModes[waveMode];
  if(mode==='wave'){
    ctx2d.strokeStyle=green;ctx2d.lineWidth=1.5;ctx2d.beginPath();
    const midY=H/2,amp=H*.38*waveAmp;
    for(let i=0;i<data.length;i++){const x=(i/data.length)*W,y=midY-data[i]*amp;i===0?ctx2d.moveTo(x,y):ctx2d.lineTo(x,y);}
    ctx2d.stroke();
  } else if(mode==='ribbon'){
    const midY=H/2,amp=H*.36*waveAmp,off=amp*.15;
    const top=new Path2D(),bot=new Path2D();
    for(let i=0;i<data.length;i++){const x=(i/data.length)*W,y=midY-data[i]*amp;if(i===0){top.moveTo(x,y-off);bot.moveTo(x,y+off);}else{top.lineTo(x,y-off);bot.lineTo(x,y+off);}}
    const grad=ctx2d.createLinearGradient(0,midY-amp,0,midY+amp);
    grad.addColorStop(0,_uiC(0.7));grad.addColorStop(1,_uiC(0.15));
    ctx2d.fillStyle=grad;
    const fill=new Path2D(top);
    for(let i=data.length-1;i>=0;i--){const x=(i/data.length)*W,y=midY-data[i]*amp;fill.lineTo(x,y+off);}
    fill.closePath();ctx2d.fill(fill);
    ctx2d.strokeStyle=green;ctx2d.lineWidth=1.5;ctx2d.stroke(top);ctx2d.stroke(bot);
  } else if(mode==='bars'){
    const bars=48,barW=(W/bars)*.7,gap=(W/bars)*.3,midY=H/2,maxH=H*.42;
    ctx2d.fillStyle=green;
    for(let i=0;i<bars;i++){const idx=Math.floor(i*data.length/bars),h=Math.max(2,Math.abs(data[idx])*maxH*2),x=i*(W/bars)+gap/2;ctx2d.globalAlpha=.5+Math.abs(data[idx])*2;ctx2d.fillRect(x,midY-h/2,barW,h);}
    ctx2d.globalAlpha=1;
  } else if(mode==='radial'){
    const cx=W/2,cy=H/2,baseR=Math.min(W,H)*.12,maxR=Math.min(W,H)*.36,spokes=64;
    ctx2d.strokeStyle=green;ctx2d.lineWidth=1.5;
    for(let i=0;i<spokes;i++){const angle=(i/spokes)*Math.PI*2-Math.PI/2,idx=Math.floor(i*data.length/spokes),r=baseR+Math.abs(data[idx])*maxR;ctx2d.beginPath();ctx2d.moveTo(cx+Math.cos(angle)*baseR,cy+Math.sin(angle)*baseR);ctx2d.lineTo(cx+Math.cos(angle)*r,cy+Math.sin(angle)*r);ctx2d.stroke();}
  }
}
drawWave(); // start render loop immediately

// ── Open-mic ──────────────────────────────────────────────────────────────
function setOpenMicState(s){
  openMicState=s;
  const btn=document.getElementById('open-mic-btn');
  const ptt=document.getElementById('ptt-btn');
  const avBtn=document.getElementById('avatar-mic-btn');
  const muteBtn=document.getElementById('mic-mute-btn');
  const avMuteBtn=document.getElementById('avatar-mic-mute-btn');
  const states={off:['MIC','btn'],listening:['LISTENING…','btn on'],user_speaking:['SPEAKING…','btn on'],processing:['PROCESSING…','btn on'],playing:['PLAYING…','btn on']};
  const[text,cls]=states[s]||states.off;
  btn.textContent=text; btn.className=cls;
  ptt.style.display=s==='off'?'':'none';
  // Mute button always visible — disabled when VAD is off
  const vadActive=s!=='off';
  if(muteBtn) { muteBtn.style.display=''; muteBtn.disabled=!vadActive; muteBtn.style.opacity=vadActive?'':'0.35'; }
  if(avMuteBtn) { avMuteBtn.style.display=''; avMuteBtn.disabled=!vadActive; avMuteBtn.style.opacity=vadActive?'':'0.35'; }
  _updateMuteBtnUI();
  // Mirror state on avatar overlay mic button (shorter labels to fit HUD)
  if(avBtn){
    const avLabels={off:'VAD',listening:'LISTENING',user_speaking:'SPEAKING',processing:'PROCESSING',playing:'PLAYING'};
    avBtn.textContent=avLabels[s]||'🎙';
    avBtn.className='btn'+(s!=='off'?' on':'');
  }
}
function _updateMuteBtnUI(){
  const muteBtn=document.getElementById('mic-mute-btn');
  const avMuteBtn=document.getElementById('avatar-mic-mute-btn');
  const label=_micMuted?'🔇 UNMUTE':'🎙 MUTE';
  const cls='btn'+(_micMuted?' on':'');
  if(muteBtn){muteBtn.textContent=label;muteBtn.className=cls;}
  if(avMuteBtn){avMuteBtn.textContent=label;avMuteBtn.className=cls;}
}
function toggleMicMute(){
  _micMuted=!_micMuted;
  if(openMicVAD){
    try{ _micMuted ? openMicVAD.pause() : openMicVAD.start(); }catch(e){}
  }
  _updateMuteBtnUI();
  // If muting while speaking, treat as cancelled
  if(_micMuted && openMicState==='user_speaking') setOpenMicState('listening');
}
async function loadVAD(){
  if(_vadLoaded||_vadLoading)return;
  _vadLoading=true;
  await new Promise((resolve,reject)=>{
    const s1=document.createElement('script');
    s1.src='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.js';
    s1.onload=()=>{const s2=document.createElement('script');s2.src='https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/bundle.min.js';s2.onload=()=>{_vadLoaded=true;_vadLoading=false;resolve();};s2.onerror=reject;document.head.appendChild(s2);};
    s1.onerror=reject;document.head.appendChild(s1);
  });
}
function float32ToWav(float32,sampleRate){
  const n=float32.length;const buf=new ArrayBuffer(44+n*2);const v=new DataView(buf);
  const ws=(o,s)=>{for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i));};
  ws(0,'RIFF');v.setUint32(4,36+n*2,true);ws(8,'WAVE');ws(12,'fmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,sampleRate,true);v.setUint32(28,sampleRate*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);ws(36,'data');v.setUint32(40,n*2,true);
  let o=44;for(let i=0;i<n;i++){v.setInt16(o,Math.max(-32768,Math.min(32767,float32[i]*32768)),true);o+=2;}return buf;
}
async function toggleOpenMic(){
  ensureAudio();
  if(openMicState!=='off'){if(openMicVAD){try{openMicVAD.pause();openMicVAD.destroy();}catch(e){}openMicVAD=null;}setOpenMicState('off');return;}
  const btn=document.getElementById('open-mic-btn');btn.textContent='LOADING…';
  try{await loadVAD();}catch(e){console.error('[VAD]',e);btn.textContent='VAD';addBubble('assistant','[VAD load failed]');return;}
  openMicVAD=await vad.MicVAD.new({
    onnxWASMBasePath:'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/',
    baseAssetPath:'https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/',
    onSpeechStart:()=>{
      if(openMicState==='off'||_micMuted)return;
      // Suppress barge-in if user is actively typing
      const inp=document.getElementById('msg-input');
      if(inp&&inp.value.trim().length>0){return;}
      // Always stop and silence audio on speech start — isPlaying may already be
      // false even if scheduled chunks are still in the Web Audio queue
      _resumeText=_currentReplyText;
      stopAudio();  // fires /tts/cancel internally and returns the promise
      isBusy=false;
      setOpenMicState('user_speaking');
    },
    onSpeechEnd:async(audio)=>{
      if(openMicState==='off'||_micMuted)return;
      setOpenMicState('processing');
      const wavBuf=float32ToWav(audio,16000);const blob=new Blob([wavBuf],{type:'audio/wav'});
      const resumeCtx=_resumeText;_resumeText='';
      try{
        const r=await fetch('/stt',{method:'POST',headers:{'Content-Type':'audio/wav'},body:blob});
        const d=await r.json();const text=(d.text||'').trim();
        if(text){setOpenMicState('playing');await doSend(text,resumeCtx);}
        else{setOpenMicState('listening');}
      }catch(e){console.error('[VAD STT]',e);setOpenMicState('listening');}
    },
    positiveSpeechThreshold:0.6,negativeSpeechThreshold:0.4,preSpeechPadFrames:5,redemptionFrames:8,minSpeechFrames:3
  });
  await openMicVAD.start();
  setOpenMicState('listening');
}

// ── Chat / bubbles ────────────────────────────────────────────────────────
function addBubble(role,text){
  const div=document.createElement('div');div.className='bubble '+role;div.textContent=text;
  const chat=document.getElementById('chat');chat.appendChild(div);chat.scrollTop=chat.scrollHeight;
  if (_avOverlayOpen) _avCheckBubbleForCode(text);
  return div;
}
function addThinking(){
  const div=document.createElement('div');div.className='bubble thinking';div.innerHTML='<span class="dot-pulse">thinking</span>';
  const chat=document.getElementById('chat');chat.appendChild(div);chat.scrollTop=chat.scrollHeight;return div;
}
async function sendText(){
  const inp=document.getElementById('msg-input');
  const t=inp.value.trim();
  if(!t) return;
  if(isBusy && !isPlaying) return; // LLM still fetching, can't interrupt
  inp.value=''; inp.style.height='auto';
  // Await the /tts/cancel acknowledgement before proceeding — ensures the server
  // has signalled EchoTTS to stop before we POST /chat → /tts again (prevents OOM).
  await stopAudio();
  isBusy=false;
  // ── !fx command — explicit bracket syntax so normal messages are never eaten
  // Usage: !fx           → random effect + agent quip
  //        !fx matrix    → specific effect + agent quip
  //        !fx list      → show available effects in chat (no LLM call)
  if (_avOverlayOpen && /^!fx(\s|$)/i.test(t)) { _fxHandleCommand(t); return; }
  doSend(t);
}

// Effect name aliases for !fx <name> — forgiving but unambiguous (only matched when !fx prefix used)
const _FX_CMD_MAP = {
  matrix:      'matrix_rain',        rain:        'matrix_rain',
  glitch:      'glitch_storm',       storm:       'glitch_storm',
  static:      'signal_static',      noise:       'signal_static',
  particles:   'particle_burst',     burst:       'particle_burst',    fireworks: 'particle_burst',
  scanlines:   'scanline_warp',      warp:        'scanline_warp',     crt:       'scanline_warp',
  corrupt:     'data_corruption',    corruption:  'data_corruption',
  heartbeat:   'heartbeat',          pulse:       'heartbeat',         ekg:       'heartbeat',
  hypno:       'hypno_spiral',       spiral:      'hypno_spiral',      trance:    'hypno_spiral',
  heart:       'heart_pulse',        love:        'heart_pulse',       hearts:    'heart_scatter',
  scatter:     'heart_scatter',      floating:    'heart_scatter',
  vhs:         'vhs_rewind',         rewind:      'vhs_rewind',        tape:      'vhs_rewind',
  neural:      'neural_fire',        neurons:     'neural_fire',       synapses:  'neural_fire',
  melt:        'pixel_melt',         pixels:      'pixel_melt',
  void:        'void_pulse',         abyss:       'void_pulse',
  snap:        'static_burst',       zap:         'static_burst',
  cascade:     'cascade',            fall:        'cascade',
  bloom:       'chromatic_bloom',    chromatic:   'chromatic_bloom',
  crack:       'screen_crack',       shatter:     'screen_crack',
  flatline:    'ekg_flatline',       dead:        'ekg_flatline',
  binary:      'binary_rain',        ones:        'binary_rain',
  warp:        'warp_drive',         hyperspace:  'warp_drive',        jump:      'warp_drive',
  acid:        'acid_wash',          wash:        'acid_wash',
  ghost:       'ghost_signal',       haunted:     'ghost_signal',
  memory:      'memory_leak',        leak:        'memory_leak',       hex:       'memory_leak',
  hologram:    'hologram',           holo:        'hologram',
  shockwave:   'shockwave',          shock:       'shockwave',         wave:      'shockwave',
  morse:       'morse',              signal:      'morse',
  thermal:     'thermal_vision',     heat:        'thermal_vision',    infrared:  'thermal_vision',
  digital:     'digital_rain_color', colour:      'digital_rain_color',
  random:      'random',             r:           'random',
};
const _FX_ALL = [
  'matrix_rain','glitch_storm','signal_static','particle_burst','scanline_warp',
  'data_corruption','heartbeat','hypno_spiral','heart_pulse','heart_scatter',
  'vhs_rewind','neural_fire','pixel_melt','void_pulse','static_burst','cascade',
  'chromatic_bloom','screen_crack','ekg_flatline','binary_rain','warp_drive',
  'acid_wash','ghost_signal','memory_leak','hologram','shockwave','morse',
  'thermal_vision','digital_rain_color',
];

function _fxHandleCommand(rawText) {
  const parts  = rawText.trim().split(/\s+/);
  const arg    = (parts[1] || '').toLowerCase();

  // !fx list — just print available commands, no LLM call
  if (arg === 'list' || arg === 'help') {
    const lines = [
      '!fx commands:',
      '  !fx            — random effect',
      '  !fx matrix     — matrix rain',
      '  !fx glitch     — glitch storm',
      '  !fx static     — signal static',
      '  !fx particles  — particle burst',
      '  !fx scanlines  — scanline warp',
      '  !fx corrupt    — data corruption',
      '  !fx heartbeat  — heartbeat / ekg',
      '  !fx hypno      — hypno spiral',
      '  !fx heart      — pulsing heart',
      '  !fx hearts     — floating hearts',
      '  !fx vhs        — vhs rewind',
      '  !fx neural     — neural fire',
      '  !fx melt       — pixel melt',
      '  !fx void       — void pulse',
      '  !fx snap       — static burst',
      '  !fx cascade    — cascade fall',
      '  !fx bloom      — chromatic bloom',
      '  !fx crack      — screen crack',
      '  !fx flatline   — ekg flatline',
      '  !fx binary     — binary rain',
      '  !fx warp       — warp drive',
      '  !fx acid       — acid wash',
      '  !fx ghost      — ghost signal',
      '  !fx memory     — memory leak',
      '  !fx hologram   — hologram',
      '  !fx shockwave  — shockwave',
      '  !fx morse      — morse flash',
      '  !fx thermal    — thermal vision',
      '  !fx digital    — digital rain colour',
    ].join('\n');
    addBubble('assistant', lines);
    return;
  }

  // Resolve effect name
  let effectName;
  if (!arg || arg === 'random' || arg === 'r') {
    effectName = _FX_ALL[Math.floor(Math.random() * _FX_ALL.length)];
  } else {
    effectName = _FX_CMD_MAP[arg] || null;
    if (!effectName) {
      // Try prefix match
      const key = Object.keys(_FX_CMD_MAP).find(k => k.startsWith(arg));
      effectName = key ? _FX_CMD_MAP[key] : null;
    }
    if (!effectName) {
      addBubble('assistant', `Unknown effect "${arg}". Type !fx list for available effects.`);
      return;
    }
    if (effectName === 'random') effectName = _FX_ALL[Math.floor(Math.random() * _FX_ALL.length)];
  }

  // Fire the visual effect immediately
  if (typeof window.triggerFX === 'function') window.triggerFX(effectName);

  // Ask the agent for an in-character one-liner to go with it — send as a hidden system prompt
  // so the user bubble shows the !fx command but the LLM gets the real instruction
  addBubble('user', rawText);
  _sseOwnCount += 2;
  isBusy = true;
  const myGen = _ttsGeneration;
  const thinking = addThinking();
  const quipPrompt = `[System: The user just triggered the "${effectName.replace(/_/g,' ')}" visual effect on your avatar screen. React to it in character — one short punchy line, no more than two sentences. Make it feel alive, like you caused it or are showing off. No emojis, no preamble, just the line.]`;
  fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: quipPrompt, is_fx_quip: true}),
  }).then(r => r.json()).then(data => {
    thinking.remove();
    if (data.error || !data.reply) { isBusy = false; return; }
    if (_ttsGeneration !== myGen) { isBusy = false; return; }
    addBubble('assistant', data.reply);
    _currentReplyText = data.reply;
    _ttsPlayStartTime = audioCtx ? audioCtx.currentTime + 0.06 : 0;
    if (openMicState !== 'off') setOpenMicState('playing');
    playTTS(_stripCodeForTTS(data.reply), myGen).then(() => {
      if (openMicState === 'playing') setOpenMicState('listening');
    }).finally(() => { isBusy = false; rearmAC(); });
  }).catch(() => { thinking.remove(); isBusy = false; });
}

async function doSend(text,resumeCtx='',isAC=false){
  if(isBusy) return;
  isBusy=true;
  // Wake avatar from sleep on any new message
  if (_avIsSleeping) { _avIsSleeping = false; _avUpdateDisplay(); }
  // AC calls: skip user bubble and only count the assistant SSE push (not a user push)
  _sseOwnCount += isAC ? 1 : 2;
  const myGen=_ttsGeneration;
  const imgB64=pendingImageB64,imgMime=pendingImageMime;
  if(imgB64)clearImage();
  if(!isAC){
    const userBubble=addBubble('user',text);
    if(imgB64){const th=document.createElement('img');th.src='data:'+imgMime+';base64,'+imgB64;th.className='bubble-img';th.style.cursor='zoom-in';th.onclick=()=>openLightbox(th.src);userBubble.insertBefore(th,userBubble.firstChild);}
  }
  const thinking=addThinking();
  try{
    const payload={text};if(resumeCtx)payload.resume_context=resumeCtx;
    if(isAC)payload.is_ac=true;
    if(imgB64){payload.image_b64=imgB64;payload.image_mime=imgMime;}
    _chatAbortCtrl=new AbortController();
    const res=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(payload),signal:_chatAbortCtrl.signal});
    _chatAbortCtrl=null;
    const data=await res.json();thinking.remove();
    if(data.error){addBubble('assistant','[Error: '+data.error+']');isBusy=false;return;}
    // Check generation after the long LLM wait — barge-in may have happened
    if(_ttsGeneration!==myGen){thinking.remove();return;}
    const asstBubble=addBubble('assistant',data.reply);
    (data.generated_images||[]).forEach(dataUri=>{const img=document.createElement('img');img.src=dataUri;img.className='bubble-img generated-img';img.style.cssText='max-width:100%;max-height:360px;margin-top:8px;cursor:zoom-in';img.onclick=()=>openLightbox(dataUri);asstBubble.appendChild(img);});
    if(data.safety) _handleSafetyResponse(data.safety);
    _currentReplyText=data.reply;_ttsPlayStartTime=audioCtx?audioCtx.currentTime+0.06:0;
    if(_ttsGeneration===myGen){
      if(openMicState!=='off')setOpenMicState('playing');
      await playTTS(_stripCodeForTTS(data.reply),myGen);
      if(openMicState==='playing')setOpenMicState('listening');
    }
    // TTS finished — now safe to rearm AC (server timer was NOT started on reply)
    rearmAC();
    // On real turns, refresh memory count — extraction runs async so poll at 3s and 12s
    if(!isAC){
      const _refreshMemCount = async () => {
        try{
          const r=await fetch('/memory/count');
          const d=await r.json();
          if(d.count!==undefined)
            document.getElementById('memory-status').textContent=`${d.count} entries`;
        }catch(e){}
      };
      setTimeout(_refreshMemCount, 3000);
      setTimeout(_refreshMemCount, 12000);
    }
  }catch(e){
    thinking.remove();
    if(e.name!=='AbortError') addBubble('assistant','[Connection error]');
    else {
      // Chat was aborted (barge-in). Wait for the server's /tts stop+settle to
      // complete before rearming AC — otherwise AC can fire a new LLM+TTS call
      // while EchoTTS is still clearing the previous inference → OOM.
      setTimeout(rearmAC, 400);
    }
  }
  finally{isBusy=false;}
}

function onImageSelected(input){
  const file=input.files[0];if(!file)return;
  pendingImageMime='image/jpeg';pendingImageName=file.name;
  const reader=new FileReader();
  reader.onload=e=>{
    const img=new Image();
    img.onload=()=>{
      const MAX=1024;let w=img.width,h=img.height;
      if(w>MAX||h>MAX){if(w>h){h=Math.round(h*MAX/w);w=MAX;}else{w=Math.round(w*MAX/h);h=MAX;}}
      const c=document.createElement('canvas');c.width=w;c.height=h;c.getContext('2d').drawImage(img,0,0,w,h);
      const dataUri=c.toDataURL('image/jpeg',0.85);pendingImageB64=dataUri.split(',')[1];
      document.getElementById('img-preview-thumb').src=dataUri;
      document.getElementById('img-preview-name').textContent=`${file.name} (${w}×${h})`;
      document.getElementById('img-preview-row').classList.add('visible');
      document.getElementById('img-btn').classList.add('has-image');
      const avb=document.getElementById('av-img-btn');if(avb)avb.classList.add('has-image');
    };img.src=e.target.result;
  };reader.readAsDataURL(file);input.value='';
}
function clearImage(){
  pendingImageB64=null;pendingImageMime='image/jpeg';pendingImageName='';
  document.getElementById('img-preview-row').classList.remove('visible');
  document.getElementById('img-btn').classList.remove('has-image');
  const avb=document.getElementById('av-img-btn');if(avb)avb.classList.remove('has-image');
  document.getElementById('img-preview-thumb').src='';
  document.getElementById('img-preview-name').textContent='';
}

// ── PTT ──────────────────────────────────────────────────────────────────
const pttBtn=document.getElementById('ptt-btn');
async function startPTT(){
  ensureAudio();if(pttActive)return;
  try{
    const stream=await navigator.mediaDevices.getUserMedia({audio:true});
    pttActive=true;audioChunks=[];pttBtn.classList.add('recording');pttBtn.textContent='⬤ RECORDING…';
    const mimeType=MediaRecorder.isTypeSupported('audio/webm;codecs=opus')?'audio/webm;codecs=opus':MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')?'audio/ogg;codecs=opus':'audio/webm';
    mediaRecorder=new MediaRecorder(stream,{mimeType});
    mediaRecorder.ondataavailable=e=>{if(e.data.size>0)audioChunks.push(e.data);};
    mediaRecorder.onstop=async()=>{stream.getTracks().forEach(t=>t.stop());const blob=new Blob(audioChunks,{type:mimeType});pttBtn.textContent='⬤ PROCESSING…';await sendVoice(blob,mimeType);pttBtn.classList.remove('recording');pttBtn.textContent='⬤ HOLD TO TALK';};
    mediaRecorder.start();
  }catch(e){console.error('[PTT]',e);alert('Microphone access denied.');pttBtn.classList.remove('recording');pttBtn.textContent='⬤ HOLD TO TALK';pttActive=false;}
}
function stopPTT(){if(!pttActive||!mediaRecorder)return;pttActive=false;mediaRecorder.stop();}
async function sendVoice(blob,mimeType){
  try{const res=await fetch('/stt',{method:'POST',headers:{'Content-Type':mimeType},body:blob});const data=await res.json();const text=(data.text||'').trim();if(text)await doSend(text);else pttBtn.textContent='⬤ HOLD TO TALK';}
  catch(e){console.error('[STT]',e);}
}
pttBtn.addEventListener('touchstart',e=>{e.preventDefault();startPTT();},{passive:false});
pttBtn.addEventListener('touchend',  e=>{e.preventDefault();stopPTT();},{passive:false});
pttBtn.addEventListener('mousedown',()=>startPTT());
pttBtn.addEventListener('mouseup',  ()=>stopPTT());
pttBtn.addEventListener('mouseleave',()=>{if(pttActive)stopPTT();});

// ── Avatar PTT button — same hold-to-talk logic, synced label
const avPttBtn=document.getElementById('avatar-ptt-btn');
function _avPttStart(e){if(e.cancelable)e.preventDefault();avPttBtn.classList.add('recording');avPttBtn.textContent='⬤ REC';startPTT();}
function _avPttStop(e){if(e.cancelable)e.preventDefault();avPttBtn.classList.remove('recording');avPttBtn.textContent='⬤ HOLD TO TALK';stopPTT();}
avPttBtn.addEventListener('touchstart',_avPttStart,{passive:false});
avPttBtn.addEventListener('touchend',  _avPttStop, {passive:false});
avPttBtn.addEventListener('mousedown', _avPttStart);
avPttBtn.addEventListener('mouseup',   _avPttStop);
avPttBtn.addEventListener('mouseleave',()=>{if(pttActive){avPttBtn.classList.remove('recording');avPttBtn.textContent='⬤ HOLD TO TALK';stopPTT();}});

// ── Auto-continue SSE ─────────────────────────────────────────────────────
function _userIsTyping() {
  const inp   = document.getElementById('msg-input');
  const avInp = document.getElementById('avatar-msg-input');
  return (inp   && inp.value.trim().length   > 0) ||
         (avInp && avInp.value.trim().length > 0);
}

// Debounced rearm — collapses multiple calls within 300ms into one POST
let _rearmTimer = null;
function rearmAC(){
  if(!acEnabled) return;
  clearTimeout(_rearmTimer);
  _rearmTimer = setTimeout(()=>{ fetch('/ac/rearm',{method:'POST'}).catch(()=>{}); }, 300);
}

let _acRetryDelay=5000;
function connectACStream(){
  if(acEventSource){try{acEventSource.close();}catch(e){}acEventSource=null;}
  acEventSource=new EventSource('/ac_stream');
  acEventSource.onmessage=async(e)=>{
    _acRetryDelay=5000; // reset backoff on successful message
    const data=JSON.parse(e.data);if(!data.prompt||!acEnabled)return;
    if(_userIsTyping()){
      rearmAC();
      return;
    }
    if(isBusy||isPlaying){
      rearmAC();
      return;
    }
    // Small debounce — wait 600ms then recheck, in case audio just finished
    // and isPlaying briefly flicked false between chunks
    await new Promise(r => setTimeout(r, 600));
    if(isBusy||isPlaying||_userIsTyping()){
      rearmAC();
      return;
    }
    await doSend(data.prompt,'',true);
    // No rearm here — doSend rearms after TTS finishes playing
  };
  acEventSource.onerror=()=>{
    try{acEventSource.close();}catch(e){}
    acEventSource=null;
    const delay=_acRetryDelay;
    _acRetryDelay=Math.min(_acRetryDelay*2,60000);
    setTimeout(connectACStream,delay);
  };
}
connectACStream();

// ── Settings ──────────────────────────────────────────────────────────────
function toggleSettings(){
  const panel=document.getElementById('settings-panel');
  panel.classList.toggle('open');
  if(!panel.classList.contains('open')) _stopMemPoll();
}
function switchTab(name){
  document.querySelectorAll('.stab').forEach((b,i)=>{
    const id=b.getAttribute('onclick').match(/'(\w+)'/)[1];
    const on=id===name;
    b.classList.toggle('active',on);
    const pane=document.getElementById('stab-'+id);
    if(pane){pane.classList.toggle('active',on);pane.style.display=on?'flex':'none';}
  });
}
function toggleAC(){
  acEnabled=!acEnabled;
  document.getElementById('s-ac-btn').textContent=acEnabled?'ON':'OFF';
  document.getElementById('s-ac-btn').className='btn'+(acEnabled?' on':'');
  const _ind=document.getElementById('ac-indicator');
  _ind.textContent='AC: '+(acEnabled?'ON':'OFF');
  _ind.className='btn'+(acEnabled?' on':'');
  const _avAc=document.getElementById('av-ac-indicator');
  if(_avAc){_avAc.textContent='AC: '+(acEnabled?'ON':'OFF');_avAc.className='btn'+(acEnabled?' on':'');}
  fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({auto_continue_enabled:acEnabled})});
}
function onProviderChange(){
  const pid=document.getElementById('s-provider').value;
  const p=providerRegistry[pid]||{};
  document.getElementById('field-api-key').className='setting-row provider-field'+(p.needs_api_key?' visible':'');
  document.getElementById('field-agent-id').className='setting-row provider-field'+(p.needs_agent_id?' visible':'');
  document.getElementById('field-model').className='setting-row provider-field'+(p.needs_model?' visible':'');
  if(p.base_url)document.getElementById('s-base-url').value=p.base_url;
}
function onTTSProviderChange(skipDefaults){
  const pid=document.getElementById('s-tts-provider').value;
  const p=ttsProviderRegistry[pid]||{};
  document.getElementById('tts-field-api-key').style.display=p.needs_api_key?'flex':'none';
  // Show manual voice ID input only for ElevenLabs
  const vidRow=document.getElementById('tts-field-voice-id');
  if(vidRow){
    vidRow.style.display=(pid==='elevenlabs'||pid==='hume')?'flex':'none';
    const vidInput=document.getElementById('s-voice-id');
    if(vidInput) vidInput.placeholder=pid==='hume'?'paste Hume voice ID':'paste voice_id directly';
  }
  // Only overwrite base_url with registry default when user manually changes provider
  if(!skipDefaults && p.base_url) document.getElementById('s-tts-base-url').value=p.base_url;
  // Only refresh voice list from server when user manually changes provider,
  // not during loadState (which builds the list itself synchronously)
  if(!skipDefaults){
    const savedVoice=currentVoice||document.getElementById('s-voice').value;
    fetch('/state').then(r=>r.json()).then(d=>{
      const vsel=document.getElementById('s-voice');
      vsel.innerHTML='';
      (d.voices||[]).forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;vsel.appendChild(o);});
      if(savedVoice) vsel.value=savedVoice;
      if(!vsel.value && vsel.options.length) vsel.selectedIndex=0;
    });
  }
}
async function applySettings(){
  const pid=document.getElementById('s-provider').value;
  const tpid=document.getElementById('s-tts-provider').value;
  const payload={
    provider_id:   pid,
    base_url:      document.getElementById('s-base-url').value.trim(),
    api_key:       document.getElementById('s-api-key').value.trim(),
    agent_id:      document.getElementById('s-agent-id').value.trim(),
    model:         document.getElementById('s-model').value.trim(),
    system_prompt: document.getElementById('s-system-prompt').value,
    max_reply_tokens: parseInt(document.getElementById('s-max-tokens').value)||300,
    tts_provider_id: tpid,
    tts_base_url:  document.getElementById('s-tts-base-url').value.trim(),
    tts_api_key:   document.getElementById('s-tts-api-key').value.trim(),
    voice:         (document.getElementById('s-voice-id')?.value?.trim()) || document.getElementById('s-voice').value,
    auto_continue_mode:    document.getElementById('s-ac-mode').value,
    auto_continue_enabled: acEnabled,
    initiative_enabled:    _initiativeEnabled,
    initiative_mode:       _initiativeMode,
    session_mode:  document.getElementById('s-session-mode').value,
    char_mode:     document.getElementById('s-char-mode').value,
  };
  const kvRaw=document.getElementById('s-kv-scale').value.trim();
  payload.kv_scale=kvRaw?kvRaw:null;
  const kvMinT=document.getElementById('s-kv-min-t').value.trim();
  const kvMaxL=document.getElementById('s-kv-max-layers').value.trim();
  if(kvMinT) payload.kv_min_t=parseFloat(kvMinT);
  if(kvMaxL) payload.kv_max_layers=parseInt(kvMaxL);
  payload.fx_enabled=_fxEnabled;
  payload.reverb_wet=parseFloat(document.getElementById('reverb-wet').value);
  payload.reverb_predelay=parseFloat(document.getElementById('reverb-predelay').value);
  payload.delay_wet=parseFloat(document.getElementById('delay-wet').value);
  payload.delay_time=parseFloat(document.getElementById('delay-time').value);
  payload.delay_feedback=parseFloat(document.getElementById('delay-feedback').value);
  payload.chorus_wet=parseFloat(document.getElementById('chorus-wet').value);
  payload.chorus_depth=parseFloat(document.getElementById('chorus-depth').value);
  payload.chorus_rate=parseFloat(document.getElementById('chorus-rate').value);
  payload.ringmod_wet=parseFloat(document.getElementById('ringmod-wet').value);
  payload.ringmod_freq=parseFloat(document.getElementById('ringmod-freq').value);
  payload.crush_wet=parseFloat(document.getElementById('crush-wet').value);
  payload.crush_bits=parseInt(document.getElementById('crush-bits').value);
  payload.crush_sr=parseInt(document.getElementById('crush-sr').value);
  if(_irB64){payload.ir_b64=_irB64; payload.ir_name=_irName;}
  payload.master_gain=masterGain;
  if(gainNode)gainNode.gain.value=masterGain;
  Object.assign(payload, _collectAvatarSettings());
  Object.assign(payload, _collectAvatarImages());
  payload.ui_hue = _uiHue;
  // Web search
  payload.websearch_enabled      = _webSearchEnabled;
  payload.websearch_api_key      = document.getElementById('s-websearch-key').value.trim();
  payload.websearch_result_count = parseInt(document.getElementById('s-websearch-count').value)||3;
  payload.vision_enabled         = _visionEnabled;
  await fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  document.getElementById('settings-panel').classList.remove('open');
  // Do NOT call loadState() here — it would clobber client-side avatar images,
  // IR buffer, and other state that only lives in the browser until saved.
}

// ── Characters ────────────────────────────────────────────────────────────
let _loadedCharPath=''; // tracks currently loaded char path for "save current"

function onCharSelectChange(){
  // Enable/disable SAVE CURRENT and DELETE based on whether a char is selected
  const sel=document.getElementById('s-char');
  document.getElementById('btn-save-current').disabled=!sel.value;
  document.getElementById('btn-delete-char').disabled=!sel.value;
  const footer=document.getElementById('btn-save-footer');
  if(footer) footer.disabled=!sel.value;
}

function toggleNewCharForm(){
  const form=document.getElementById('new-char-form');
  form.classList.toggle('open');
  if(form.classList.contains('open')) document.getElementById('s-char-name').focus();
}

async function loadCharacter(){
  const sel=document.getElementById('s-char');if(!sel.value)return;
  const res=await fetch('/characters/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:sel.value})});
  const data=await res.json();
  if(!data.ok){alert('Failed to load: '+(data.error||'unknown'));return;}

  // If avatar overlay is open, use the dedicated swap fade;
  // otherwise just do the work immediately (no visible avatar to fade)
  const doSwap = async () => {
    _loadedCharPath=sel.value;
    document.getElementById('btn-save-current').disabled=false; const _sf=document.getElementById('btn-save-footer');if(_sf)_sf.disabled=false;
    if(data.master_gain!=null){
      masterGain=data.master_gain;
      document.getElementById('vol-slider').value=masterGain;
      if(gainNode) gainNode.gain.value=masterGain;
    }
    if(data.ui_hue!=null) applyUIHue(data.ui_hue);
    if(data.char_name){
      const cn=document.getElementById('avatar-char-name');
      if(cn) cn.textContent=data.char_name.toUpperCase();
      window._currentCharName=data.char_name.toUpperCase();
    }
    const chatDiv=document.getElementById('chat');
    chatDiv.innerHTML='';
    if(data.chat_history&&data.chat_history.length){
      data.chat_history.forEach(msg=>{
        const dispText=msg.user_image?msg.content.replace(/^\[image attached\]\s*/,''):msg.content;
        const bubble=addBubble(msg.role,dispText);
        if(msg.user_image){const ui=document.createElement('img');ui.src=msg.user_image;ui.className='bubble-img';ui.style.cursor='zoom-in';ui.onclick=()=>openLightbox(ui.src);bubble.insertBefore(ui,bubble.firstChild);}
        (msg.gen_images||[]).forEach(uri=>{
          const img=document.createElement('img');img.src=uri;
          img.className='bubble-img generated-img';
          img.style.cssText='max-width:100%;max-height:360px;margin-top:8px;cursor:zoom-in';
          img.onclick=()=>openLightbox(uri);bubble.appendChild(img);
        });
      });
    }
    await loadState();

  };

  if(_avOverlayOpen) await _avSwapFade(doSwap);
  else await doSwap();
}

function _collectAvatarSettings() {
  // Returns a plain object with all avatar UI state — no PNG frames (too large for char JSON)
  const vp = document.getElementById('avatar-viewport');
  const vpW = (vp && vp.offsetWidth  > 0) ? vp.offsetWidth  : 0;
  const vpH = (vp && vp.offsetHeight > 0) ? vp.offsetHeight : 0;
  // Store pan as fractions of viewport so it restores correctly across screen sizes.
  // If overlay is closed (vpW=0), store raw pixels and flag as absolute.
  const panRelative = vpW > 0;
  return {
    av_noise_mode:       (document.getElementById('av-noise-mode')        ||{value:'mixed'}).value,
    av_noise_intensity:  parseFloat((document.getElementById('av-noise-intensity') ||{value:0.5}).value),
    av_scanline_mode:    (document.getElementById('av-scanline-mode')     ||{value:'roll'}).value,
    av_scanline_spacing: parseInt((document.getElementById('av-scanline-spacing') ||{value:4}).value),
    av_tint_enabled:     _avTintEnabled,
    av_tint_intensity:   parseFloat((document.getElementById('av-tint-intensity') ||{value:0.12}).value),
    av_glitch_enabled:   _avGlitchEnabled,
    av_glitch_intensity: parseFloat((document.getElementById('av-glitch-intensity')||{value:0.4}).value),
    av_pixel_enabled:    _avPixelEnabled,
    av_pixel_size:       parseFloat((document.getElementById('av-pixel-size')||{value:1}).value),
    av_pixel_contrast:   parseInt((document.getElementById('av-pixel-contrast')||{value:100}).value),
    av_pixel_bilinear:   _avPixelBilinear,
    av_wire_enabled:     _avWireEnabled,
    av_wire_floor:       _avWireFloor,
    av_wire_walls:       _avWireWalls,
    av_wire_depth:       parseFloat((document.getElementById('av-wire-depth')  ||{value:0.7}).value),
    av_wire_speed:       parseFloat((document.getElementById('av-wire-speed')  ||{value:0.15}).value),
    av_wire_reverse:     _avWireReverse,
    av_talk_thresh:      parseFloat((document.getElementById('av-talk-thresh') ||{value:0.04}).value),
    av_scream_thresh:    parseFloat((document.getElementById('av-scream-thresh')||{value:0.8}).value),
    av_talk_decay:       parseInt((document.getElementById('av-talk-decay')    ||{value:80}).value),
    av_blink_chance:     parseInt((document.getElementById('av-blink-chance')  ||{value:25}).value),
    av_blink_dur:        parseInt((document.getElementById('av-blink-dur')     ||{value:60}).value),
    av_blink_delay:      parseInt((document.getElementById('av-blink-delay')   ||{value:3000}).value),
    av_talk_en:          document.getElementById('av-talk-en')  ? document.getElementById('av-talk-en').classList.contains('on')  : true,
    av_blink_en:         document.getElementById('av-blink-en') ? document.getElementById('av-blink-en').classList.contains('on') : true,
    av_sleep_en:         document.getElementById('av-sleep-en') ? document.getElementById('av-sleep-en').classList.contains('on') : true,
    av_enabled:          _avEnabled,
    av_scale:            _avScale,
    av_pan_x:            _avPanX,
    av_pan_y:            _avPanY,
  };
}

// Collect avatar image frames as b64 data URLs for persistence
function _collectAvatarImages() {
  const out = {};
  for (const [k, v] of Object.entries(_avFrames)) {
    if (v) out['av_img_' + k] = v;
  }
  return out;
}

// Apply avatar image frames from saved char data
function _applyAvatarImages(d) {
  if (!d) return;
  const slotKeys = ['idle','talk','blink-closed','blink-talk','scream','sleep'];
  let anyLoaded = false;
  for (const key of slotKeys) {
    const val = d['av_img_' + key];
    const slot = document.getElementById('slot-' + key);
    const img  = document.getElementById('slot-' + key + '-img');
    if (val) {
      _avFrames[key] = val;
      if (slot && img) { img.src = val; slot.classList.add('loaded'); }
      anyLoaded = true;
    } else {
      // Explicitly clear — character has no image for this slot
      _avFrames[key] = null;
      if (slot && img) { img.src = ''; slot.classList.remove('loaded'); }
    }
  }
  if (anyLoaded) _avUpdateDisplay();
}

function _applyAvatarSettings(d) {
  if (!d) return;
  const setVal = (id, v) => { const el = document.getElementById(id); if (el && v !== undefined) el.value = v; };
  const setBtn = (id, on) => { const el = document.getElementById(id); if (el) { el.textContent = on ? 'ON' : 'OFF'; el.className = 'btn' + (on ? ' on' : ''); } };
  setVal('av-noise-mode',       d.av_noise_mode);
  setVal('av-noise-intensity',  d.av_noise_intensity);
  setVal('av-scanline-mode',    d.av_scanline_mode);
  setVal('av-scanline-spacing', d.av_scanline_spacing);
  setVal('av-tint-intensity',   d.av_tint_intensity);
  setVal('av-glitch-intensity', d.av_glitch_intensity);
  if (d.av_pixel_size     !== undefined) setVal('av-pixel-size',     d.av_pixel_size);
  if (d.av_pixel_contrast !== undefined) setVal('av-pixel-contrast', d.av_pixel_contrast);
  setVal('av-wire-depth',       d.av_wire_depth);
  setVal('av-wire-speed',       d.av_wire_speed);
  setVal('av-talk-thresh',      d.av_talk_thresh);
  setVal('av-scream-thresh',    d.av_scream_thresh);
  setVal('av-talk-decay',       d.av_talk_decay);
  setVal('av-blink-chance',     d.av_blink_chance);
  setVal('av-blink-dur',        d.av_blink_dur);
  setVal('av-blink-delay',      d.av_blink_delay);
  // Toggle buttons
  if (d.av_tint_enabled   !== undefined){ _avTintEnabled   = d.av_tint_enabled;   setBtn('av-tint-btn',   _avTintEnabled);   _avApplyTint(); }
  if (d.av_glitch_enabled !== undefined){ _avGlitchEnabled = d.av_glitch_enabled; setBtn('av-glitch-btn', _avGlitchEnabled); }
  if (d.av_pixel_enabled  !== undefined){
    _avPixelEnabled  = d.av_pixel_enabled;
    _avPixelBilinear = !!d.av_pixel_bilinear;
    const pBtn   = document.getElementById('av-pixel-btn');
    const hudBtn = document.getElementById('av-pixel-hud-btn');
    if (pBtn)   { pBtn.textContent   = _avPixelEnabled ? 'ON' : 'OFF'; pBtn.className   = 'btn' + (_avPixelEnabled ? ' on' : ''); }
    if (hudBtn) { hudBtn.textContent = _avPixelEnabled ? '≋ BLUR' : 'BLUR'; hudBtn.className = 'btn' + (_avPixelEnabled ? ' on' : ''); }
    const bBtn = document.getElementById('av-pixel-bilinear-btn');
    const lbl  = document.getElementById('av-pixel-mode-label');
    if (bBtn) { bBtn.textContent = _avPixelBilinear ? 'SOFT' : 'EDGE'; bBtn.className = 'btn' + (_avPixelBilinear ? ' on' : ''); }
    if (lbl)  lbl.textContent = _avPixelBilinear ? 'soft blur only' : 'edge enhance · adds contrast';
    const row = document.getElementById('av-pixel-bilinear-row');
    if (row) row.style.display = _avPixelEnabled ? '' : 'none';
    const modeRow = document.getElementById('av-pixel-mode-row');
    if (modeRow) modeRow.style.display = _avPixelEnabled ? '' : 'none';
    if (_avPixelEnabled) _avApplyPixel();
  }
  if (d.av_wire_enabled   !== undefined){ _avWireEnabled   = d.av_wire_enabled;   setBtn('av-wire-btn',   _avWireEnabled); }
  if (d.av_wire_floor     !== undefined){ _avWireFloor = d.av_wire_floor; const b=document.getElementById('av-wire-floor-btn'); if(b){b.className='btn'+(d.av_wire_floor?' on':'');} }
  if (d.av_wire_walls     !== undefined){ _avWireWalls = d.av_wire_walls; const b=document.getElementById('av-wire-walls-btn'); if(b){b.className='btn'+(d.av_wire_walls?' on':'');} }
  if (d.av_talk_en   !== undefined){ const el=document.getElementById('av-talk-en');  if(el){ if(d.av_talk_en)  el.classList.add('on'); else el.classList.remove('on'); } }
  if (d.av_blink_en  !== undefined){ const el=document.getElementById('av-blink-en'); if(el){ if(d.av_blink_en) el.classList.add('on'); else el.classList.remove('on'); } }
  if (d.av_sleep_en  !== undefined){ const el=document.getElementById('av-sleep-en'); if(el){ if(d.av_sleep_en) el.classList.add('on'); else el.classList.remove('on'); } }
  if (d.av_enabled !== undefined) {
    const want = !!d.av_enabled;
    if (want !== _avEnabled) toggleAvatarMode(want);
  }
  _avPositionSaved = false;  // reset — will be set true if saved scale found below
  if (d.av_wire_reverse !== undefined) { _avWireReverse = d.av_wire_reverse; const b=document.getElementById('av-wire-dir-btn'); if(b){b.textContent=_avWireReverse?'◀ REV':'▶ FWD';b.className='btn'+(_avWireReverse?' on':'');} }
  if (d.av_scale !== undefined) { _avScale = d.av_scale; _avPositionSaved = true; } else { _avScale = 1.5; }
  if (d.av_pan_x !== undefined) { _avPanX = d.av_pan_x; }
  if (d.av_pan_y !== undefined) { _avPanY = d.av_pan_y; }
  _avApplyTransform();
  // Pan/scale restored from saved values
  // Restore avatar images if present
  _applyAvatarImages(d);
}

async function deleteSelectedCharacter(){
  const sel=document.getElementById('s-char');
  const path=sel.value;
  if(!path) return;
  const name=path.replace(/\.json$/i,'').split('/').pop();
  if(!confirm(`Delete character "${name}"?\nThis cannot be undone.`)) return;
  const res=await fetch('/characters/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path})});
  const data=await res.json();
  if(data.ok){
    await loadCharacterList();
    const btn=document.getElementById('btn-delete-char');
    btn.textContent='✓';setTimeout(()=>{btn.textContent='🗑';},1200);
  } else {
    alert('Delete failed: '+(data.error||'unknown'));
  }
}

async function saveCurrentCharacter(){
  // Overwrite the currently selected/loaded character with current settings
  const sel=document.getElementById('s-char');
  const path=_loadedCharPath||sel.value;
  if(!path){alert('No character loaded to overwrite.');return;}
  // Derive name and subfolder from the path (e.g. "FFVII/Aerith-Gainsborough" → name="Aerith-Gainsborough", folder="FFVII")
  const parts=path.replace(/\.json$/i,'').split('/');
  const name=parts[parts.length-1];
  const subfolder=parts.length>1?parts.slice(0,-1).join('/'):'';
  const kvRaw=document.getElementById('s-kv-scale').value.trim();
  const kvMinT=document.getElementById('s-kv-min-t').value.trim();
  const kvMaxL=document.getElementById('s-kv-max-layers').value.trim();
  const payload={name, subfolder, master_gain:masterGain, kv_scale:kvRaw||null,
    kv_min_t:kvMinT?parseFloat(kvMinT):0.9, kv_max_layers:kvMaxL?parseInt(kvMaxL):24,
    fx_enabled:_fxEnabled,
    reverb_wet:parseFloat(document.getElementById('reverb-wet').value),
    reverb_predelay:parseFloat(document.getElementById('reverb-predelay').value),
    delay_wet:parseFloat(document.getElementById('delay-wet').value),
    delay_time:parseFloat(document.getElementById('delay-time').value),
    delay_feedback:parseFloat(document.getElementById('delay-feedback').value),
    chorus_wet:parseFloat(document.getElementById('chorus-wet').value),
    chorus_depth:parseFloat(document.getElementById('chorus-depth').value),
    chorus_rate:parseFloat(document.getElementById('chorus-rate').value),
    ringmod_wet:parseFloat(document.getElementById('ringmod-wet').value),
    ringmod_freq:parseFloat(document.getElementById('ringmod-freq').value),
    crush_wet:parseFloat(document.getElementById('crush-wet').value),
    crush_bits:parseInt(document.getElementById('crush-bits').value),
    crush_sr:parseInt(document.getElementById('crush-sr').value),
    ir_b64:_irB64||null, ir_name:_irName||'',
    ui_hue:_uiHue,
    ..._collectAvatarSettings(),
    ..._collectAvatarImages()};
  const res=await fetch('/characters/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const data=await res.json();
  if(data.ok){
    _loadedCharPath=data.path;
    // Flash both save buttons green briefly as confirmation
    const btn=document.getElementById('btn-save-current');
    const btnF=document.getElementById('btn-save-footer');
    btn.textContent='✓ SAVED';btn.classList.add('on');
    if(btnF){btnF.textContent='✓ SAVED';btnF.classList.add('on');}
    setTimeout(()=>{btn.textContent='💾 SAVE CURRENT';btn.classList.remove('on');if(btnF){btnF.textContent='💾 SAVE CHAR';btnF.classList.remove('on');}},1500);
  } else {
    alert('Save failed: '+(data.error||'unknown'));
  }
}

async function saveNewCharacter(){
  const name=document.getElementById('s-char-name').value.trim();
  if(!name){document.getElementById('s-char-name').focus();return;}
  const subfolder=document.getElementById('s-char-folder').value.trim();
  const kvRaw=document.getElementById('s-kv-scale').value.trim();
  const kvMinT=document.getElementById('s-kv-min-t').value.trim();
  const kvMaxL=document.getElementById('s-kv-max-layers').value.trim();
  const payload={name, master_gain:masterGain, kv_scale:kvRaw||null,
    kv_min_t:kvMinT?parseFloat(kvMinT):0.9, kv_max_layers:kvMaxL?parseInt(kvMaxL):24,
    fx_enabled:_fxEnabled,
    reverb_wet:parseFloat(document.getElementById('reverb-wet').value),
    reverb_predelay:parseFloat(document.getElementById('reverb-predelay').value),
    delay_wet:parseFloat(document.getElementById('delay-wet').value),
    delay_time:parseFloat(document.getElementById('delay-time').value),
    delay_feedback:parseFloat(document.getElementById('delay-feedback').value),
    chorus_wet:parseFloat(document.getElementById('chorus-wet').value),
    chorus_depth:parseFloat(document.getElementById('chorus-depth').value),
    chorus_rate:parseFloat(document.getElementById('chorus-rate').value),
    ringmod_wet:parseFloat(document.getElementById('ringmod-wet').value),
    ringmod_freq:parseFloat(document.getElementById('ringmod-freq').value),
    crush_wet:parseFloat(document.getElementById('crush-wet').value),
    crush_bits:parseInt(document.getElementById('crush-bits').value),
    crush_sr:parseInt(document.getElementById('crush-sr').value),
    ir_b64:_irB64||null, ir_name:_irName||'',
    ..._collectAvatarSettings(),
    ..._collectAvatarImages()};
  if(subfolder) payload.subfolder=subfolder;
  const res=await fetch('/characters/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const data=await res.json();
  if(data.ok){
    document.getElementById('s-char-name').value='';
    document.getElementById('s-char-folder').value='';
    document.getElementById('new-char-form').classList.remove('open');
    await loadCharacterList();
    // Select the newly saved character
    const sel=document.getElementById('s-char');
    for(const opt of sel.options){if(opt.value===data.path){sel.value=data.path;break;}}
    _loadedCharPath=data.path;
    document.getElementById('btn-save-current').disabled=false; const _sf=document.getElementById('btn-save-footer');if(_sf)_sf.disabled=false;
    const btn=document.getElementById('btn-save-current');
    btn.textContent='✓ SAVED';btn.classList.add('on');
    setTimeout(()=>{btn.textContent='💾 SAVE CURRENT';btn.classList.remove('on');},1500);
  } else {
    alert('Save failed: '+(data.error||'unknown'));
  }
}
async function resetConversation(){
  if(!confirm('Reset conversation?'))return;
  await fetch('/reset',{method:'POST'});
  document.getElementById('chat').innerHTML='';
}

// ── Context Mode ─────────────────────────────────────────────────────────
let _contextMode = 'standard';
const _ctxModes  = ['voice_fast','voice_balanced','standard','deep_recall','full_context'];

// ── Vision ────────────────────────────────────────────────────────────────
let _visionEnabled = false;
function _applyVisionUI(enabled){
  _visionEnabled = enabled;
  const btn     = document.getElementById('s-vision-btn');
  const imgBtn  = document.getElementById('img-btn');
  const avImgBtn= document.getElementById('av-img-btn');
  if(btn){ btn.textContent = enabled ? 'ON' : 'OFF'; btn.className = 'btn' + (enabled ? ' on' : ''); }
  // Enable/disable both image buttons and their file input
  const fileInput = document.getElementById('img-file-input');
  [imgBtn, avImgBtn].forEach(b=>{ if(!b) return; b.disabled = !enabled; b.style.opacity = enabled ? '' : '0.35'; b.style.pointerEvents = enabled ? '' : 'none'; });
  if(fileInput) fileInput.disabled = !enabled;
  // If vision turned off, clear any pending image
  if(!enabled && pendingImageB64) clearImage();
}
function toggleVision(){
  _applyVisionUI(!_visionEnabled);
}

// ── Web search ────────────────────────────────────────────────────────────
let _webSearchEnabled = false;
function _applyWebSearchUI(enabled){
  _webSearchEnabled = enabled;
  const btn    = document.getElementById('s-websearch-btn');
  const fields = document.getElementById('websearch-fields');
  if(btn){ btn.textContent = enabled ? 'ON' : 'OFF'; btn.className = 'btn' + (enabled ? ' on' : ''); }
  if(fields) fields.style.display = enabled ? 'flex' : 'none';
}
function toggleWebSearch(){
  _applyWebSearchUI(!_webSearchEnabled);
}

function _applyContextModeUI(name){
  _contextMode = name || 'standard';
  _ctxModes.forEach(m=>{
    const btn = document.getElementById('ctx-btn-'+m);
    if(btn) btn.className = 'btn' + (m===_contextMode?' on':'');
  });
}

async function setContextMode(name){
  _applyContextModeUI(name);
  try{
    await fetch('/settings',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({context_mode:name})
    });
  }catch(e){console.error('[CTX MODE]',e);}
}

async function saveMaxTokens(){
  const val = parseInt(document.getElementById('s-max-tokens').value)||300;
  try{
    await fetch('/settings',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({max_reply_tokens:val})
    });
  }catch(e){console.error('[MAX TOKENS]',e);}
}

// ── Conversation RAG ─────────────────────────────────────────────────────
let _convRagEnabled = false;

async function toggleConvRag(){
  _convRagEnabled = !_convRagEnabled;
  await saveConvRagSettings();
  _updateConvRagUI();
}

async function saveConvRagSettings(){
  const threshold = parseInt(document.getElementById('s-conv-rag-threshold').value) || 20;
  await fetch('/conv_rag/set',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({enabled:_convRagEnabled, threshold}),
  });
  _updateConvRagUI();
}

function _updateConvRagUI(){
  const btn = document.getElementById('s-conv-rag-btn');
  const row = document.getElementById('conv-rag-threshold-row');
  const status = document.getElementById('conv-rag-status');
  if(btn){ btn.textContent = _convRagEnabled ? 'ON' : 'OFF'; btn.className = 'btn' + (_convRagEnabled ? ' on' : ''); }
  if(row) row.style.display = _convRagEnabled ? '' : 'none';
  if(status){
    const thresh = document.getElementById('s-conv-rag-threshold').value;
    status.textContent = _convRagEnabled ? `auto-flush at ${thresh} msgs` : 'off';
    status.style.color = _convRagEnabled ? 'var(--green-dim)' : '#888';
  }
}

async function clearConvRagFile(){
  if(!confirm('Delete the conversation RAG file for this character? This cannot be undone.')) return;
  await fetch('/conv_rag/clear',{method:'POST'});
  document.getElementById('conv-rag-status').textContent = _convRagEnabled ? 'file cleared — will rebuild' : 'off';
}

async function reloadArtLib(){
  const btn = event.target;
  btn.disabled = true; btn.textContent = '...';
  try{
    const r = await fetch('/ascii_art/reload',{method:'POST'});
    const d = await r.json();
    const el = document.getElementById('art-lib-count');
    if(el) el.textContent = d.count === 0 ? 'no files loaded' : d.count + ' piece' + (d.count===1?'':'s') + ' loaded';
  }catch(e){ console.error('[ART]',e); }
  btn.disabled = false; btn.textContent = 'RELOAD';
}

// ── Extra RAG ─────────────────────────────────────────────────────────────
async function loadRag(){
  const sel=document.getElementById('s-rag-file');
  const selected=[...sel.selectedOptions].map(o=>o.value).filter(Boolean);
  if(!selected.length)return;
  const semantic=document.getElementById('s-rag-semantic').checked;
  const res=await fetch('/rag/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filenames:selected,semantic})});
  const data=await res.json();
  const label=document.getElementById('rag-status');
  if(data.ok){label.textContent=`✓ ${data.filenames.join(', ')} (${data.chunks} chunks${data.semantic?' • semantic':''})`;label.style.color='#4cff7a';}
  else{alert('RAG load failed: '+(data.error||'unknown'));label.textContent='Load failed';label.style.color='#ff4444';}
}
async function addRag(){
  const sel=document.getElementById('s-rag-file');
  const selected=[...sel.selectedOptions].map(o=>o.value).filter(Boolean);
  if(!selected.length)return;
  const semantic=document.getElementById('s-rag-semantic').checked;
  const res=await fetch('/rag/add',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filenames:selected,semantic})});
  const data=await res.json();
  const label=document.getElementById('rag-status');
  if(data.ok){label.textContent=`✓ ${data.filenames.join(', ')} (${data.chunks} chunks${data.semantic?' • semantic':''})`;label.style.color='#4cff7a';}
  else{alert('RAG add failed: '+(data.error||'unknown'));label.textContent='Add failed';label.style.color='#ff4444';}
}
async function clearRag(){
  await fetch('/rag/clear',{method:'POST'});
  document.getElementById('rag-status').textContent='No extra RAG loaded';
  document.getElementById('rag-status').style.color='#888';
}
function _saveRagCuda() {
  const cuda = document.getElementById('s-rag-cuda').checked;
  fetch('/settings', { method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ rag_cuda: cuda }) }).catch(() => {});
}

async function saveRag(){
  const name=document.getElementById('s-rag-save-name').value.trim();
  if(!name){alert('Enter a filename first.');return;}
  const res=await fetch('/rag/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:name})});
  const data=await res.json();
  if(data.ok){
    document.getElementById('s-rag-save-name').value='';
    alert(`Saved as ${data.filename} (${data.turns} turns)`);
    loadRagFileList(); // refresh dropdown
  } else {
    alert('Save failed: '+(data.error||'unknown'));
  }
}

// ── Memory ────────────────────────────────────────────────────────────────
let memoryEnabled=false;
let _memRefreshTimer=null;
function _stopMemPoll(){if(_memRefreshTimer){clearInterval(_memRefreshTimer);_memRefreshTimer=null;}}
function toggleMemoryPanel(){
  const panel=document.getElementById('memory-panel');
  panel.classList.toggle('open');
  if(panel.classList.contains('open')){
    refreshMemory();
    _stopMemPoll();  // clear any stale timer before starting fresh
    _memRefreshTimer=setInterval(refreshMemory,4000);
  } else {
    _stopMemPoll();
  }
}

let _memCountTimer=null; // reserved for future use
async function toggleMemory(){
  const next=!memoryEnabled;
  const res=await fetch('/memory/enable',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:next})});
  const data=await res.json();
  memoryEnabled=data.enabled;  // trust server state, not local assumption
  document.getElementById('s-memory-btn').textContent=memoryEnabled?'ON':'OFF';
  document.getElementById('s-memory-btn').className='btn'+(memoryEnabled?' on':'');
}
async function refreshMemory(){
  try{
    const res=await fetch('/memory');const data=await res.json();
    const container=document.getElementById('memory-cards');container.innerHTML='';
    memoryEnabled=data.enabled;
    document.getElementById('s-memory-btn').textContent=memoryEnabled?'ON':'OFF';
    document.getElementById('s-memory-btn').className='btn'+(memoryEnabled?' on':'');
    document.getElementById('memory-status').textContent=`${data.stats.active} entries, ${data.stats.archived} archived`;
    (data.entries||[]).forEach(e=>{
      const card=document.createElement('div');card.className='mem-card';
      const globalLabel=e.global?'🌐 Un-global':'🌐 Global';
      card.innerHTML=`<div class="mem-cat">${e.category.toUpperCase()} ${e.global?'<span style="color:#4cff7a">🌐</span>':''}</div>
        <div class="mem-content">${e.content}</div>
        <div class="mem-meta"><span>score: ${e.score.toFixed(2)}</span><span>hits: ${e.hits}</span><span>${e.last_accessed.slice(0,10)}</span></div>
        <div class="mem-actions">
          <button onclick="memAct('delete','${e.id}')">🗑 Delete</button>
          <button onclick="memAct('promote','${e.id}')">${globalLabel}</button>
        </div>`;
      container.appendChild(card);
    });
    if(!data.entries.length)container.innerHTML='<div style="font-size:12px;color:var(--text-dim);padding:8px 0">No memories yet</div>';
  }catch(e){console.error('[MEMORY]',e);}
}
async function memAct(action,id){
  await fetch('/memory/'+action,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id})});
  refreshMemory();
}
async function addMemory(){
  const content=document.getElementById('new-mem-content').value.trim();
  if(!content)return;
  const cat=document.getElementById('new-mem-cat').value;
  await fetch('/memory/add',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({content,category:cat,score:0.6})});
  document.getElementById('new-mem-content').value='';
  refreshMemory();
}

async function clearAllMemory(){
  if(!confirm('Delete ALL memory entries? This cannot be undone.')) return;
  await fetch('/memory/clear',{method:'POST'});
  document.getElementById('memory-cards').innerHTML='';
  document.getElementById('memory-status').textContent='0 entries';
}

async function clearAllSession(btn){
  if(!confirm('CLEAR ALL?\n\nThis will permanently delete:\n  • Conversation history\n  • Conversation RAG file\n  • Extra RAG index\n  • All memory entries\n\nThis cannot be undone.')) return;
  await fetch('/clear_all',{method:'POST'});
  // Clear chat UI
  document.getElementById('chat').innerHTML='';
  // Clear memory UI
  document.getElementById('memory-cards').innerHTML='';
  document.getElementById('memory-status').textContent='0 entries';
  // Clear RAG status
  const ragStatus=document.getElementById('rag-status');
  if(ragStatus){ragStatus.textContent='No extra RAG loaded';ragStatus.style.color='#888';}
  const convRagStatus=document.getElementById('conv-rag-status');
  if(convRagStatus&&!_convRagEnabled){convRagStatus.textContent='off';}
  else if(convRagStatus){convRagStatus.textContent='file cleared — will rebuild';}
  // Flash button confirmation
  if(btn){const orig=btn.textContent;btn.textContent='✓ CLEARED';btn.style.color='var(--green)';setTimeout(()=>{btn.textContent=orig;btn.style.color='';},2000);}
}

// ── Memory export / import ────────────────────────────────────────────────
function memoryExport(){
  window.location.href='/memory/export';
}

let _memImportMode='merge';
function memoryImportPick(mode){
  _memImportMode=mode;
  const inp=document.getElementById('mem-import-input');
  inp.value='';  // reset so same file can be re-picked
  inp.click();
}

async function memoryImportFile(input){
  const file=input.files[0];
  if(!file) return;
  const statusEl=document.getElementById('mem-import-status');
  statusEl.style.color='var(--text-dim)';
  statusEl.textContent='Reading file…';
  try{
    const text=await file.text();
    const parsed=JSON.parse(text);
    // Accept either a full export {entries,archived} or a bare array
    const entries=Array.isArray(parsed)?parsed:(parsed.entries||[]);
    const archived=parsed.archived||[];

    if(_memImportMode==='replace'){
      if(!confirm(`REPLACE entire memory bank with ${entries.length} entries from "${file.name}"?

Your current bank will be backed up automatically.`)){
        statusEl.textContent='';
        return;
      }
    }

    const res=await fetch('/memory/import',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({mode:_memImportMode, entries, archived}),
    });
    const data=await res.json();
    if(!data.ok){ statusEl.style.color='var(--danger)'; statusEl.textContent='Import failed: '+(data.error||'unknown'); return; }

    if(_memImportMode==='replace'){
      statusEl.style.color='var(--green)';
      statusEl.textContent=`Replaced: ${data.active} entries loaded${data.backup?' · backup: '+data.backup:''}`;
    } else {
      statusEl.style.color='var(--green)';
      statusEl.textContent=`Merged: +${data.added} added, ${data.skipped} duplicates skipped`;
    }
    refreshMemory();
  } catch(e){
    statusEl.style.color='var(--danger)';
    statusEl.textContent='Parse error: '+e.message;
  }
}

// ── Load state ────────────────────────────────────────────────────────────
async function loadState(){
  try{
    const res=await fetch('/state');const data=await res.json();
    // Restore loaded character path early so loadCharacterList can pre-select it
    if(data.loaded_char_path){
      const prevChar = _loadedCharPath;
      _loadedCharPath = data.loaded_char_path;
      const footer=document.getElementById('btn-save-footer');
      document.getElementById('btn-save-current').disabled=false;
      if(footer) footer.disabled=false;
      // Reconnect SSE with the correct char so broadcast filtering works
      if(_loadedCharPath !== _sseCharPath) _startChatStream(_loadedCharPath);
    }

    providerRegistry=data.provider_registry||{};
    ttsProviderRegistry=data.tts_provider_registry||{};
    voices=data.voices||[];currentVoice=data.voice||(voices[0]||'');
    acEnabled=data.auto_continue_enabled!==false;acMode=data.auto_continue_mode||'standard';

    document.getElementById('dot-tts').className='status-dot'+(data.tts_online?' on':' err');
    document.getElementById('dot-llm').className='status-dot'+(data.llm_online?' on':' err');
    document.getElementById('provider-tag').textContent=data.llm_label||data.provider_label||'—';

    // LLM provider dropdown
    const psel=document.getElementById('s-provider');psel.innerHTML='';
    for(const[pid,pinfo]of Object.entries(providerRegistry)){const o=document.createElement('option');o.value=pid;o.textContent=pinfo.label;if(pid===data.provider_id)o.selected=true;psel.appendChild(o);}
    onProviderChange();
    document.getElementById('s-base-url').value=data.base_url||'';
    const _llmKeyEl=document.getElementById('s-api-key');
    if(typeof data.api_key==='string') _llmKeyEl.value=data.api_key;
    else if(data.api_key===true){_llmKeyEl.value='';_llmKeyEl.placeholder='(key saved — enter to replace)';}
    else{_llmKeyEl.value='';_llmKeyEl.placeholder='sk-...';}
    document.getElementById('s-agent-id').value=data.agent_id||'';
    document.getElementById('s-model').value=data.model||'';
    document.getElementById('s-system-prompt').value=data.system_prompt||'';
    document.getElementById('s-max-tokens').value=data.max_reply_tokens||300;

    // TTS provider dropdown — set fields BEFORE calling onTTSProviderChange
    const tsel=document.getElementById('s-tts-provider');tsel.innerHTML='';
    for(const[tid,tinfo]of Object.entries(ttsProviderRegistry)){const o=document.createElement('option');o.value=tid;o.textContent=tinfo.label;if(tid===data.tts_provider_id)o.selected=true;tsel.appendChild(o);}
    document.getElementById('s-tts-base-url').value=data.tts_base_url||'';
    const _ttsKeyEl=document.getElementById('s-tts-api-key');
    if(typeof data.tts_api_key==='string') _ttsKeyEl.value=data.tts_api_key;
    else if(data.tts_api_key===true){_ttsKeyEl.value='';_ttsKeyEl.placeholder='(key saved — enter to replace)';}
    else{_ttsKeyEl.value='';_ttsKeyEl.placeholder='sk-...';}
    // Pass skipDefaults=true so it doesn't overwrite base_url/voice with registry defaults
    onTTSProviderChange(true);
    // Restore manual voice ID field for ElevenLabs
    if((data.tts_provider_id==='elevenlabs'||data.tts_provider_id==='hume') && currentVoice){
      const vidEl=document.getElementById('s-voice-id');
      if(vidEl) vidEl.value=currentVoice;
    }

    // Voice selector — build list and select the saved voice
    const vsel=document.getElementById('s-voice');vsel.innerHTML='';
    voices.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;if(v===currentVoice)o.selected=true;vsel.appendChild(o);});
    if(currentVoice) vsel.value=currentVoice;

    // Volume — restore slider and apply to audio node
    const gain=data.master_gain!=null?data.master_gain:1.5;
    masterGain=gain;
    document.getElementById('vol-slider').value=gain;
    if(gainNode) gainNode.gain.value=gain;

    acEnabled = data.auto_continue_enabled !== undefined ? data.auto_continue_enabled : acEnabled;
    acMode    = data.auto_continue_mode    || acMode;
    document.getElementById('s-ac-mode').value=acMode;
    document.getElementById('s-ac-btn').textContent=acEnabled?'ON':'OFF';
    document.getElementById('s-ac-btn').className='btn'+(acEnabled?' on':'');
    const _acInd=document.getElementById('ac-indicator');
    _acInd.textContent='AC: '+(acEnabled?'ON':'OFF');
    _acInd.className='btn'+(acEnabled?' on':'');
    const _avAcL=document.getElementById('av-ac-indicator');
    if(_avAcL){_avAcL.textContent='AC: '+(acEnabled?'ON':'OFF');_avAcL.className='btn'+(acEnabled?' on':'');}

    // Initiative state
    _initiativeEnabled = data.initiative_enabled || false;
    _initiativeMode    = data.initiative_mode    || 'light';
    _initiativeNextSecs = data.initiative_next_secs || 0;
    _applyInitiativeUI();

    // FX chance
    if (data.initiative_fx_chance !== undefined) {
      const sl = document.getElementById('s-fx-chance');
      if (sl) { sl.value = data.initiative_fx_chance; document.getElementById('s-fx-chance-val').textContent = data.initiative_fx_chance + '%'; }
    }

    // Subtitle speed
    if (data.sub_speed !== undefined) _setSubSpeed(data.sub_speed);

    // Sleep timer
    if (data.sleep_timer_enabled !== undefined) {
      _applySleepTimerUI(data.sleep_timer_enabled, data.initiative_in_sleep);
    }
    if (data.sleep_start !== undefined) { const el = document.getElementById('s-sleep-start'); if (el) el.value = data.sleep_start; }
    if (data.sleep_end   !== undefined) { const el = document.getElementById('s-sleep-end');   if (el) el.value = data.sleep_end; }

    // Avatar settings — restore from char/session data
    _applyAvatarSettings(data);
    if (data.ui_hue !== undefined) {
      const slider = document.getElementById('s-ui-hue');
      if (slider) slider.value = data.ui_hue;
      applyUIHue(data.ui_hue);
    }
    if (data.safety_indicator_visible !== undefined) {
      _safetyIndicatorVisible = data.safety_indicator_visible;
      _applySafetyIndicator();
    }
    if (data.wave_amp !== undefined) {
      waveAmp = data.wave_amp;
      const s = document.getElementById('wave-amp');
      if (s) s.value = waveAmp;
    }
    if (data.wave_fade !== undefined) {
      waveFade = data.wave_fade;
      const s = document.getElementById('wave-fade');
      if (s) s.value = waveFade;
    }
    document.getElementById('s-session-mode').value=data.session_mode||'shared';
    document.getElementById('s-char-mode').value=data.char_mode||'shared';
    document.getElementById('s-kv-scale').value=data.kv_scale!=null?data.kv_scale:'';
    document.getElementById('s-kv-min-t').value=data.kv_min_t!=null?data.kv_min_t:'';
    document.getElementById('s-kv-max-layers').value=data.kv_max_layers!=null?data.kv_max_layers:'';
    // FX state
    if(data.fx_enabled!==undefined){
      _fxEnabled=data.fx_enabled;
      const btn=document.getElementById('fx-toggle');
      btn.textContent=_fxEnabled?'FX ON':'FX OFF';
      btn.className='btn'+(_fxEnabled?' on':'');
    }
    if(data.reverb_wet!==undefined) document.getElementById('reverb-wet').value=data.reverb_wet;
    if(data.reverb_predelay!==undefined){
      document.getElementById('reverb-predelay').value=data.reverb_predelay;
      document.getElementById('reverb-predelay-val').textContent=data.reverb_predelay+'ms';
    }
    if(data.delay_wet!==undefined) document.getElementById('delay-wet').value=data.delay_wet;
    if(data.delay_time!==undefined){
      document.getElementById('delay-time').value=data.delay_time;
      document.getElementById('delay-time-val').textContent=data.delay_time+'ms';
    }
    if(data.delay_feedback!==undefined) document.getElementById('delay-feedback').value=data.delay_feedback;
    if(data.chorus_wet!==undefined){ document.getElementById('chorus-wet').value=data.chorus_wet; updateChorus(); }
    if(data.chorus_depth!==undefined) document.getElementById('chorus-depth').value=data.chorus_depth;
    if(data.chorus_rate!==undefined) document.getElementById('chorus-rate').value=data.chorus_rate;
    if(data.ringmod_wet!==undefined){ document.getElementById('ringmod-wet').value=data.ringmod_wet; updateRingMod(); }
    if(data.ringmod_freq!==undefined){ document.getElementById('ringmod-freq').value=data.ringmod_freq; document.getElementById('ringmod-freq-val').textContent=data.ringmod_freq+'Hz'; }
    if(data.crush_wet!==undefined){ document.getElementById('crush-wet').value=data.crush_wet; updateCrush(); }
    if(data.crush_bits!==undefined){ document.getElementById('crush-bits').value=data.crush_bits; document.getElementById('crush-bits-val').textContent=data.crush_bits; }
    if(data.crush_sr!==undefined){ document.getElementById('crush-sr').value=data.crush_sr; document.getElementById('crush-sr-val').textContent='÷'+data.crush_sr; }
    if(data.ir_b64) loadIRFromB64(data.ir_b64, data.ir_name||'saved IR');
    else if(_fxEnabled) _buildReverbGraph();
    if(data.dist_wet!==undefined){ document.getElementById('dist-wet').value=data.dist_wet; updateDist(); }
    if(data.dist_drive!==undefined){ document.getElementById('dist-drive').value=data.dist_drive; document.getElementById('dist-drive-val').textContent=data.dist_drive; }

    // Wave display state
    if (data.wave_mode !== undefined) {
      const idx = waveModes.indexOf(data.wave_mode);
      if (idx >= 0) { waveMode = idx; document.getElementById('wave-mode-btn').textContent = waveModeLabels[waveMode]; }
    }
    if (data.main_wave_visible !== undefined) {
      _mainWaveVisible = data.main_wave_visible;
      const ww=document.getElementById('wave-wrap'); const btn=document.getElementById('wave-toggle-btn'); const mb=document.getElementById('main-wave-en');
      if(ww) ww.classList.toggle('wave-hidden', !_mainWaveVisible);
      if(btn) btn.textContent=_mainWaveVisible?'▼ WAVE':'▶ WAVE';
      if(mb){mb.textContent=_mainWaveVisible?'ON':'OFF';mb.className='btn'+(_mainWaveVisible?' on':'');}
    }
    if (data.avatar_wave_visible !== undefined) {
      _avatarWaveVisible = data.avatar_wave_visible;
      const ww=document.getElementById('avatar-wave-wrap'); const btn=document.getElementById('av-wave-en');
      if(ww) ww.classList.toggle('wave-hidden', !_avatarWaveVisible);
      if(btn){btn.textContent=_avatarWaveVisible?'ON':'OFF';btn.className='btn'+(_avatarWaveVisible?' on':'');}
    }
    if (data.visual_fx_enabled !== undefined) {
      _visualFxEnabled = !!data.visual_fx_enabled;
      const vfb = document.getElementById('s-vis-fx-btn');
      if (vfb) { vfb.textContent = _visualFxEnabled ? 'ON' : 'OFF'; vfb.className = 'btn' + (_visualFxEnabled ? ' on' : ''); }
    }
    if (data.mood_fx_enabled !== undefined) {
      _moodFxEnabled = !!data.mood_fx_enabled;
      const mfb = document.getElementById('s-mood-fx-btn');
      if (mfb) { mfb.textContent = _moodFxEnabled ? 'ON' : 'OFF'; mfb.className = 'btn' + (_moodFxEnabled ? ' on' : ''); }
    }

    // Conversation RAG restore
    if(data.conv_rag_enabled !== undefined){
      _convRagEnabled = !!data.conv_rag_enabled;
      const thrSlider = document.getElementById('s-conv-rag-threshold');
      const thrVal    = document.getElementById('conv-rag-threshold-val');
      if(thrSlider && data.conv_rag_threshold){ thrSlider.value = data.conv_rag_threshold; }
      if(thrVal   && data.conv_rag_threshold){ thrVal.textContent = data.conv_rag_threshold + ' msgs'; }
      _updateConvRagUI();
    }

    // ASCII art library count
    if(data.art_lib_count !== undefined){
      const el = document.getElementById('art-lib-count');
      if(el) el.textContent = data.art_lib_count === 0 ? 'no files loaded' : data.art_lib_count + ' piece' + (data.art_lib_count === 1 ? '' : 's') + ' loaded';
    }

    // Extra RAG status — restore selected file and semantic checkbox
    const ragLabel=document.getElementById('rag-status');
    if(data.rag_enabled){ragLabel.textContent=`✓ ${data.rag_chunks} chunks loaded${data.rag_semantic?' • semantic':''}`;ragLabel.style.color='#4cff7a';}
    else{ragLabel.textContent='No extra RAG loaded';ragLabel.style.color='#888';}
    if(data.rag_file){
      const rsel=document.getElementById('s-rag-file');
      if(rsel){
        const loaded=data.rag_file.split(',').map(s=>s.trim()).filter(Boolean);
        for(const opt of rsel.options) opt.selected=loaded.includes(opt.value);
      }
    }
    // Keep avatar char name in sync with loaded character
    if(data.char_name){
      window._currentCharName=data.char_name.toUpperCase();
      const cn=document.getElementById('avatar-char-name');
      if(cn && cn.textContent) cn.textContent=window._currentCharName;
    }
    const semCb=document.getElementById('s-rag-semantic');if(semCb&&data.rag_semantic!==undefined)semCb.checked=!!data.rag_semantic;
    const cudaCb=document.getElementById('s-rag-cuda');if(cudaCb&&data.rag_cuda!==undefined)cudaCb.checked=!!data.rag_cuda;

    // Memory status
    memoryEnabled=data.memory_enabled||false;
    document.getElementById('s-memory-btn').textContent=memoryEnabled?'ON':'OFF';
    document.getElementById('s-memory-btn').className='btn'+(memoryEnabled?' on':'');
    document.getElementById('memory-status').textContent=`${data.memory_count||0} entries`;

    // Safety layer buttons
    if(data.safety_layer1_enabled!==undefined||data.safety_layer2_enabled!==undefined){
      if(data.safety_layer1_enabled!==undefined) _safetyL1=data.safety_layer1_enabled;
      if(data.safety_layer2_enabled!==undefined) _safetyL2=data.safety_layer2_enabled;
      const l1btn=document.getElementById('safety-l1-btn');
      const l2btn=document.getElementById('safety-l2-btn');
      if(l1btn){l1btn.textContent=_safetyL1?'L1 ON':'L1 OFF';l1btn.className='btn'+(_safetyL1?' on':'');}
      if(l2btn){l2btn.textContent=_safetyL2?'L2 ON':'L2 OFF';l2btn.className='btn'+(_safetyL2?' on':'');}
      if(_safetyL1||_safetyL2) pollSafetyStatus();
    }

    // Context mode
    if(data.context_mode) _applyContextModeUI(data.context_mode);

    // Vision
    if(data.vision_enabled !== undefined) _applyVisionUI(data.vision_enabled);

    // Web search
    if(data.websearch_enabled !== undefined) _applyWebSearchUI(data.websearch_enabled);
    if(data.websearch_api_key !== undefined) document.getElementById('s-websearch-key').value = data.websearch_api_key;
    if(data.websearch_result_count !== undefined){
      const sel=document.getElementById('s-websearch-count');
      if(sel) sel.value = String(data.websearch_result_count);
    }

    // Restore chat bubbles from server history (only on first load — chat div empty)
    const chatDiv=document.getElementById('chat');
    if(chatDiv.children.length===0&&data.chat_history&&data.chat_history.length){
      data.chat_history.forEach(msg=>{
        const dispText=msg.user_image?msg.content.replace(/^\[image attached\]\s*/,''):msg.content;
        const bubble=addBubble(msg.role, dispText);
        if(msg.user_image){const ui=document.createElement('img');ui.src=msg.user_image;ui.className='bubble-img';ui.style.cursor='zoom-in';ui.onclick=()=>openLightbox(ui.src);bubble.insertBefore(ui,bubble.firstChild);}
        (msg.gen_images||[]).forEach(uri=>{
          const img=document.createElement('img');img.src=uri;img.className='bubble-img generated-img';
          img.style.cssText='max-width:100%;max-height:360px;margin-top:8px;cursor:zoom-in';
          img.onclick=()=>openLightbox(uri);bubble.appendChild(img);
        });
      });
    }
  }catch(e){console.error('[STATE]',e);}
}
async function loadCharacterList(){
  try{
    const res=await fetch('/characters');const chars=await res.json();
    const sel=document.getElementById('s-char');
    // Prefer the server-restored path (from session.json) over client-side memory
    const prev=_loadedCharPath||sel.value;
    sel.innerHTML='<option value="">— select —</option>';
    chars.forEach(c=>{const o=document.createElement('option');o.value=c.path;o.textContent=c.label;sel.appendChild(o);});
    if(prev) sel.value=prev;
    onCharSelectChange();
  }catch(e){}
}
async function loadRagFileList(){
  try{
    const res=await fetch('/rag/files');const files=await res.json();
    const sel=document.getElementById('s-rag-file');sel.innerHTML='';
    files.forEach(f=>{const o=document.createElement('option');o.value=f;o.textContent=f;sel.appendChild(o);});
  }catch(e){}
}

// ── Volume ────────────────────────────────────────────────────────────────
document.getElementById('vol-slider').addEventListener('input',e=>{masterGain=parseFloat(e.target.value);if(gainNode)gainNode.gain.value=masterGain;});

// ── Textarea ──────────────────────────────────────────────────────────────
const msgInput=document.getElementById('msg-input');
msgInput.addEventListener('input',()=>{msgInput.style.height='auto';msgInput.style.height=Math.min(msgInput.scrollHeight,110)+'px';});
msgInput.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendText();}});

// ── Lightbox ───────────────────────────────────────────────────────────────
const _lb=document.getElementById('lightbox');
const _lbImg=document.getElementById('lightbox-img');
function openLightbox(src){_lbImg.src=src;_lb.classList.add('show');}
_lb.addEventListener('click',()=>{_lb.classList.remove('show');_lbImg.src='';});
document.addEventListener('keydown',e=>{if(e.key==='Escape')_lb.classList.remove('show');});

// ── Safety ────────────────────────────────────────────────────────────────
let _safetyL1=true, _safetyL2=true;

function _updateSafetyUI(data){
  const level=data.score_level||'ok';
  const score=data.score||0;
  const light=document.getElementById('safety-light');
  const scoreEl=document.getElementById('safety-score-display');
  const levelEl=document.getElementById('safety-level-display');
  if(!light) return;
  // Set colour directly on style — CSS classes can't override inline styles
  const lightColours={ok:'#2f9d57',notice:'#ffaa00',warn:'#ff6600',alert:'#ff2222'};
  const lightGlow={ok:'none',notice:'0 0 6px #ffaa00',warn:'0 0 8px #ff6600',alert:'0 0 12px #ff2222'};
  light.style.background=lightColours[level]||'#2f9d57';
  light.style.boxShadow=lightGlow[level]||'none';
  // Blink on alert via style animation toggle
  light.style.animation=level==='alert'?'blink .6s step-end infinite':'none';
  const levelColours={ok:'var(--green-dim)',notice:'#ffaa00',warn:'#ff6600',alert:'#ff2222'};
  if(scoreEl) scoreEl.textContent=`SCORE: ${score.toFixed(1)}`;
  if(levelEl){levelEl.textContent=`● ${level.toUpperCase()}`;levelEl.style.color=levelColours[level]||'var(--text-dim)';}
  // Update flags panel
  const flagsEl=document.getElementById('safety-flags');
  if(flagsEl&&data.flags&&data.flags.length){
    flagsEl.style.display='block';
    flagsEl.innerHTML=data.flags.slice(-10).reverse().map(f=>
      `<div style="color:${f.action==='block'?'#ff2222':f.action==='warn'?'#ff6600':'#ffaa00'}">`+
      `[${f.ts.slice(11,19)}] L${f.layer} ${f.action.toUpperCase()} — ${f.label}${f.snippet?': '+f.snippet.slice(0,40)+'…':''}</div>`
    ).join('');
  }
  // Update layer buttons
  const l1btn=document.getElementById('safety-l1-btn');
  const l2btn=document.getElementById('safety-l2-btn');
  if(l1btn){l1btn.textContent=data.layer1_enabled?'L1 ON':'L1 OFF';l1btn.className='btn'+(data.layer1_enabled?' on':'');}
  if(l2btn){l2btn.textContent=data.layer2_enabled?'L2 ON':'L2 OFF';l2btn.className='btn'+(data.layer2_enabled?' on':'');}
  _safetyL1=data.layer1_enabled; _safetyL2=data.layer2_enabled;
}

async function pollSafetyStatus(){
  // Only poll while at least one safety layer is active
  if(!_safetyL1 && !_safetyL2) return;
  try{
    const r=await fetch('/safety/status');
    const data=await r.json();
    _updateSafetyUI(data);
  }catch(e){}
  if(_safetyL1 || _safetyL2) setTimeout(pollSafetyStatus, 8000);
}

function openSafetyPanel(){
  // Open settings panel and scroll to safety section
  document.getElementById('settings-panel').classList.add('open');
  setTimeout(()=>{
    const el=document.getElementById('safety-score-display');
    if(el) el.scrollIntoView({behavior:'smooth',block:'center'});
  },100);
  pollSafetyStatus();
}

async function toggleSafetyLayer(n){
  if(n===1) _safetyL1=!_safetyL1;
  else _safetyL2=!_safetyL2;
  const res=await fetch('/safety/settings',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({layer1_enabled:_safetyL1,layer2_enabled:_safetyL2})});
  // Update button states immediately from local flags — don't wait for poll
  const l1btn=document.getElementById('safety-l1-btn');
  const l2btn=document.getElementById('safety-l2-btn');
  if(l1btn){l1btn.textContent=_safetyL1?'L1 ON':'L1 OFF';l1btn.className='btn'+(_safetyL1?' on':'');}
  if(l2btn){l2btn.textContent=_safetyL2?'L2 ON':'L2 OFF';l2btn.className='btn'+(_safetyL2?' on':'');}
  // Restart poll loop if a layer just got turned on
  if(_safetyL1 || _safetyL2) pollSafetyStatus();
}

async function resetSafetyScore(){
  await fetch('/safety/reset',{method:'POST'});
  pollSafetyStatus();
}

let _safetyIndicatorVisible = true;
function _applySafetyIndicator() {
  const light = document.getElementById('safety-light');
  if (light) light.style.display = _safetyIndicatorVisible ? '' : 'none';
  const btn = document.getElementById('safety-indicator-toggle-btn');
  if (btn) {
    btn.textContent = _safetyIndicatorVisible ? 'LED ON' : 'LED OFF';
    btn.classList.toggle('on', _safetyIndicatorVisible);
  }
}
function toggleSafetyIndicator() {
  _safetyIndicatorVisible = !_safetyIndicatorVisible;
  _applySafetyIndicator();
  fetch('/settings', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({safety_indicator_visible: _safetyIndicatorVisible})}).catch(()=>{});
}

async function clearSafetyFlags(){
  await fetch('/safety/clear-flags',{method:'POST'});
  const flagsEl=document.getElementById('safety-flags');
  if(flagsEl){flagsEl.innerHTML='';flagsEl.style.display='none';}
  pollSafetyStatus();
}

async function resetSafetyDefaults(){
  if(!confirm('Reset rules to defaults?')) return;
  await fetch('/safety/defaults',{method:'POST'});
  pollSafetyStatus();
}

async function openRuleEditor(){
  const r=await fetch('/safety/rules');
  const rules=await r.json();
  document.getElementById('rule-editor-text').value=JSON.stringify(rules,null,2);
  document.getElementById('rule-editor').style.display='flex';
}

function closeRuleEditor(){
  document.getElementById('rule-editor').style.display='none';
}

async function saveRules(){
  const text=document.getElementById('rule-editor-text').value;
  let rules;
  try{rules=JSON.parse(text);}catch(e){alert('Invalid JSON: '+e.message);return;}
  const r=await fetch('/safety/rules',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(rules)});
  const data=await r.json();
  if(data.ok){closeRuleEditor();pollSafetyStatus();}
  else alert('Save failed: '+(data.error||'unknown'));
}

// Handle safety flags in chat responses
function _handleSafetyResponse(safety){
  if(!safety) return;
  pollSafetyStatus();
  if(safety.action==='block'||safety.action==='warn'){
    const light=document.getElementById('safety-light');
    if(light){
      light.style.transform='scale(1.6)';
      setTimeout(()=>light.style.transform='',400);
    }
  }
}

// ── Avatar PNG animator ───────────────────────────────────────────────────────
// Six frame slots: idle, talk, blink-closed, blink-talk, scream, sleep
// Frame priority: scream > blink (+ talk sub-state) > talk > idle > sleep
// Talk/scream driven by TTS analyser amplitude.
// Blink is a timer. Sleep kicks in after _avSleepAfterMs of silence.

const _avFrames = { idle:null, talk:null, 'blink-closed':null, 'blink-talk':null, scream:null, sleep:null };
let _avSlotTarget = null;   // which slot the file picker is filling
let _avEnabled = false;
let _avOverlayOpen = false;

// State
let _avIsTalking  = false;
let _avIsBlinking = false;
let _avIsScreaming = false;
let _avIsSleeping  = false;
let _avTalkDecayTimer = null;
let _avBlinkTimer  = null;
let _avSleepTimer  = null;
let _avSleepAfterMs = 120000; // sleep after 2 minutes of no talk

function _avShouldAllowSleep() {
  // Don't sleep while auto-continue or initiative are running
  if (acEnabled) return false;
  if (_initiativeEnabled) return false;
  return true;
}

function _avArmSleepTimer() {
  clearTimeout(_avSleepTimer);
  const s = _avGetSettings();
  if (!s.sleepEnabled) return;
  _avSleepTimer = setTimeout(() => {
    if (_avShouldAllowSleep() && _avGetSettings().sleepEnabled) {
      _avIsSleeping = true;
      _avUpdateDisplay();
    }
  }, _avSleepAfterMs);
}
// Settings (read from sliders at runtime)
function _avGetSettings() {
  return {
    talkThresh:   parseFloat(document.getElementById('av-talk-thresh').value   || 0.04),
    screamThresh: parseFloat(document.getElementById('av-scream-thresh').value || 0.8),
    talkDecay:    parseInt(document.getElementById('av-talk-decay').value      || 80),
    blinkChance:  parseInt(document.getElementById('av-blink-chance').value    || 25),
    blinkDur:     parseInt(document.getElementById('av-blink-dur').value       || 60),
    blinkDelay:   parseInt(document.getElementById('av-blink-delay').value     || 3000),
    talkEnabled:  document.getElementById('av-talk-en').classList.contains('on'),
    blinkEnabled: document.getElementById('av-blink-en').classList.contains('on'),
    sleepEnabled: document.getElementById('av-sleep-en').classList.contains('on'),
  };
}

function _avCurrentFrame() {
  const s = _avGetSettings();
  if (_avIsSleeping && s.sleepEnabled && _avFrames.sleep) return _avFrames.sleep;
  if (_avIsScreaming && _avFrames.scream) return _avFrames.scream;
  if (_avIsBlinking && s.blinkEnabled) {
    const blinkFrame = (_avIsTalking && s.talkEnabled && _avFrames['blink-talk'])
      ? _avFrames['blink-talk'] : _avFrames['blink-closed'];
    if (blinkFrame) return blinkFrame;
  }
  if (_avIsTalking && s.talkEnabled && _avFrames.talk) return _avFrames.talk;
  return _avFrames.idle;
}

function _avUpdateDisplayCore() {
  const img = document.getElementById('avatar-img');
  if (!img) return;
  if (!_avEnabled) { img.style.display = 'none'; return; }
  img.style.display = '';
  const frame = _avCurrentFrame();
  if (frame && img.src !== frame) img.src = frame;
  else if (!frame) img.src = '';
  // Sleep vignette — fades in when sleeping, out when woken
  const sleepOv = document.getElementById('avatar-sleep-overlay');
  if (sleepOv) sleepOv.style.opacity = _avIsSleeping ? '1' : '0';
}

function _avUpdateDisplay() {
  _avUpdateDisplayCore();
}

// Amplitude polling — taps the TTS analyser (character speaking)
let _avAmpFrame    = null;
let _avWasAbove    = false;  // tracks whether last frame was above talk threshold
function _avStartAmpLoop() {
  if (_avAmpFrame) return;
  function loop() {
    _avAmpFrame = requestAnimationFrame(loop);
    if (!_avEnabled && !_avOverlayOpen) return;
    const s = _avGetSettings();
    let amp = 0;
    if (analyserNode) {
      const buf = new Float32Array(analyserNode.fftSize);
      analyserNode.getFloatTimeDomainData(buf);
      for (let i = 0; i < buf.length; i++) amp = Math.max(amp, Math.abs(buf[i]));
      amp *= waveAmp;
    }
    const wasScreaming = _avIsScreaming;
    const wasTalking   = _avIsTalking;

    // Scream threshold — instantaneous
    _avIsScreaming = amp >= s.screamThresh;

    // Talk threshold:
    // - Mouth opens immediately when amp crosses threshold upward
    // - Decay timer only armed once on the falling edge (above→below transition)
    //   so it fires after talkDecay ms of continuous silence
    const isAbove = amp >= s.talkThresh;
    if (isAbove) {
      _avIsTalking  = true;
      _avIsSleeping = false;
      // Only cancel a pending decay — don't keep re-arming it every frame above threshold
      if (_avWasAbove === false) {
        // Rising edge: cancel any lingering decay timer
        clearTimeout(_avTalkDecayTimer);
        _avTalkDecayTimer = null;
      }
      // Reset sleep timer on any active audio
      _avArmSleepTimer();
    } else {
      // Falling edge: arm decay timer exactly once
      if (_avWasAbove === true && _avTalkDecayTimer === null) {
        _avTalkDecayTimer = setTimeout(() => {
          _avIsTalking      = false;
          _avTalkDecayTimer = null;
          _avUpdateDisplay();
        }, s.talkDecay);
        // Audio just stopped — arm sleep timer from this moment
        _avArmSleepTimer();
      }
    }
    _avWasAbove = isAbove;

    if (_avIsScreaming !== wasScreaming || _avIsTalking !== wasTalking) {
      _avUpdateDisplay();
      if (_avGlitchEnabled && (_avIsScreaming || (_avIsTalking && !wasTalking))) _avSpawnGlitch();
    }
  }
  loop();
}

// Blink scheduler
function _avScheduleBlink() {
  if (!_avEnabled && !_avOverlayOpen) return;
  const s = _avGetSettings();
  if (!s.blinkEnabled) { _avBlinkTimer = setTimeout(_avScheduleBlink, s.blinkDelay); return; }
  const doIt = Math.random() * 100 < s.blinkChance;
  if (doIt) {
    _avIsBlinking = true;
    _avUpdateDisplay();
    setTimeout(() => { _avIsBlinking = false; _avUpdateDisplay(); }, s.blinkDur);
  }
  _avBlinkTimer = setTimeout(_avScheduleBlink, s.blinkDelay + Math.random() * s.blinkDelay);
}

function _avStartBlink() { clearTimeout(_avBlinkTimer); _avScheduleBlink(); }
function _avStopBlink()  { clearTimeout(_avBlinkTimer); _avIsBlinking = false; }

// File upload slot handling
function triggerSlotUpload(slotName) {
  _avSlotTarget = slotName;
  const inp = document.getElementById('avatar-file-input');
  inp.value = '';
  inp.click();
}

function onAvatarFileSelected(input) {
  const file = input.files[0];
  if (!file || !_avSlotTarget) return;
  const reader = new FileReader();
  reader.onload = e => {
    const dataUrl = e.target.result;
    _avFrames[_avSlotTarget] = dataUrl;
    // Show thumbnail in slot
    const slot = document.getElementById('slot-' + _avSlotTarget);
    const img  = document.getElementById('slot-' + _avSlotTarget + '-img');
    if (slot && img) { img.src = dataUrl; slot.classList.add('loaded'); }
    // Update live display
    _avUpdateDisplay();
    // Sync to avatar overlay image if open
    const avImg = document.getElementById('avatar-img');
    if (avImg && _avOverlayOpen) avImg.src = _avCurrentFrame() || '';
  };
  reader.readAsDataURL(file);
}

// Load a folder of avatar images by filename convention
// Expected names: idle, talk, blink-closed, blink-talk, scream, sleep
function onAvatarFolderSelected(input) {
  const validSlots = ['idle','talk','blink-closed','blink-talk','scream','sleep'];
  const files = Array.from(input.files);
  let loadCount = 0;
  for (const file of files) {
    // Strip extension and match against known slot names
    const baseName = file.name.replace(/\.[^.]+$/, '').toLowerCase().trim();
    // Always try exact match first, then suffix — sort candidates by length desc
    // so 'blink-talk' is tested before 'talk', avoiding false suffix hits
    const sortedSlots = [...validSlots].sort((a, b) => b.length - a.length);
    const slot = sortedSlots.find(s =>
      baseName === s ||
      baseName.endsWith('-' + s) ||
      baseName.endsWith('_' + s)
    );
    if (!slot) continue;
    ((slotName) => {
      const reader = new FileReader();
      reader.onload = e => {
        const dataUrl = e.target.result;
        _avFrames[slotName] = dataUrl;
        const slotEl = document.getElementById('slot-' + slotName);
        const imgEl  = document.getElementById('slot-' + slotName + '-img');
        if (slotEl && imgEl) { imgEl.src = dataUrl; slotEl.classList.add('loaded'); }
        loadCount++;
        _avUpdateDisplay();
        const avImg = document.getElementById('avatar-img');
        if (avImg && _avOverlayOpen) avImg.src = _avCurrentFrame() || '';
      };
      reader.readAsDataURL(file);
    })(slot);
  }
  // Reset input so same folder can be re-selected
  input.value = '';
}

function clearAvatarImages() {
  if (!confirm('Clear all avatar image slots?')) return;
  const validSlots = ['idle','talk','blink-closed','blink-talk','scream','sleep'];
  for (const slotName of validSlots) {
    delete _avFrames[slotName];
    const slotEl = document.getElementById('slot-' + slotName);
    const imgEl  = document.getElementById('slot-' + slotName + '-img');
    if (slotEl) slotEl.classList.remove('loaded');
    if (imgEl) { imgEl.src = ''; }
  }
  _avUpdateDisplay();
}

// Avatar mode toggle (show/hide overlay on demand)
function toggleAvatarMode(forceTo) {
  _avEnabled = (forceTo !== undefined) ? !!forceTo : !_avEnabled;
  const btn = document.getElementById('s-avatar-btn');
  btn.textContent = _avEnabled ? 'ON' : 'OFF';
  btn.className = 'btn' + (_avEnabled ? ' on' : '');
  if (_avEnabled) {
    _avStartAmpLoop();
    _avStartBlink();
  } else {
    _avStopBlink();
    _avIsTalking = _avIsBlinking = _avIsScreaming = _avIsSleeping = false;
    _avWasAbove = false;
    clearTimeout(_avTalkDecayTimer); _avTalkDecayTimer = null;
  }
  _avUpdateDisplay();
}

function openAvatarOverlay() {
  _avOverlayOpen = true;
  document.getElementById('avatar-overlay').classList.add('open');
  // Some browsers suspend the AudioContext during heavy DOM reflows (e.g. a
  // display:none → flex transition on a fullscreen overlay). Resume immediately
  // so in-flight TTS playback isn't clipped.
  if(audioCtx && audioCtx.state === 'suspended') audioCtx.resume().catch(()=>{});
  requestAnimationFrame(function() {
    const vp = document.getElementById('avatar-viewport');
    // Only reset to defaults if no saved position has been loaded for this character.
    if (!_avPositionSaved) {
      _avScale = 1.5; _avPanX = 0;
      _avPanY = vp ? vp.offsetHeight * 0.08 : 0;
    }
    _avApplyTransform();
    // Sync lock button and cursor to current lock state
    const lockBtn = document.getElementById('avatar-lock-btn');
    const ctrl = document.getElementById('avatar-zoom-controls');
    if (lockBtn) { lockBtn.textContent = _avLocked ? '🔒' : '🔓'; lockBtn.className = _avLocked ? 'active' : ''; }
    if (ctrl) ctrl.classList.toggle('locked', _avLocked);
    if (vp) vp.style.cursor = _avLocked ? 'default' : 'grab';
    _avUpdateDisplay();
    _avStartAmpLoop();
    _avStartBlink();
    _drawAvatarWave();
    _avStartEffectLoop();
    _avApplyTint();
    if (_avGlitchEnabled) _avStartGlitchScheduler();
    if (_avPixelEnabled) { _avPixelRaf = null; _avApplyPixel(); }
    // Arm sleep timer in case audio is already idle when overlay opens
    _avArmSleepTimer();
    const ww = document.getElementById('avatar-wave-wrap');
    if (ww) ww.classList.toggle('wave-hidden', !_avatarWaveVisible);
    const cn = document.getElementById('avatar-char-name');
    if (cn) cn.textContent = window._currentCharName || 'ECKO';
    const hb = document.getElementById('avatar-header-btn');
    if (hb) hb.className = 'btn on';
  });
}

function closeAvatarOverlay() {
  _avOverlayOpen = false;
  document.getElementById('avatar-overlay').classList.remove('open');
  cancelAnimationFrame(_avWaveFrame); _avWaveFrame = null;
  cancelAnimationFrame(_avEffectFrame); _avEffectFrame = null;
  clearInterval(_avGlitchTimer);
  _avDismissCode();
  if (!_avEnabled) _avStopBlink();
  const hb = document.getElementById('avatar-header-btn');
  if (hb) hb.className = 'btn';
}

let _avatarTextInputVisible = false;
function toggleAvatarTextInput() {
  _avatarTextInputVisible = !_avatarTextInputVisible;
  const row = document.getElementById('avatar-text-row');
  const btn = document.getElementById('avatar-text-toggle-btn');
  if (row) row.style.display = _avatarTextInputVisible ? 'flex' : 'none';
  if (btn) btn.className = 'btn' + (_avatarTextInputVisible ? ' on' : '');
  if (_avatarTextInputVisible) {
    const inp = document.getElementById('avatar-msg-input');
    if (inp) { inp.style.height='auto'; setTimeout(()=>inp.focus(),50); }
  }
}

function sendAvatarText() {
  const inp = document.getElementById('avatar-msg-input');
  if (!inp) return;
  const text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  inp.style.height = 'auto';
  // Re-use the main sendText path by temporarily setting msg-input value
  const mainInp = document.getElementById('msg-input');
  if (mainInp) {
    mainInp.value = text;
    sendText();
  }
}


// ── Avatar character swap fade ────────────────────────────────────────────────
// The CSS transition on #avatar-img is 2.5s (for slow sleep/ghost fades).
// For a character swap we want a snappy cross-fade: override the transition
// for the duration of the swap, then restore it.
const _AV_SWAP_FADE_MS = 180;  // out duration
const _AV_SWAP_HOLD_MS = 60;   // brief hold at zero before src changes
const _AV_SWAP_IN_MS   = 220;  // in duration

async function _avSwapFade(swapFn) {
  const img = document.getElementById('avatar-img');
  if (!img) { await swapFn(); return; }

  // Override transition to swap speed
  const savedTransition = img.style.transition;
  img.style.transition = `opacity ${_AV_SWAP_FADE_MS}ms ease`;

  // Fade out
  img.style.opacity = '0';
  await new Promise(r => setTimeout(r, _AV_SWAP_FADE_MS + _AV_SWAP_HOLD_MS));

  // Execute the swap (loadState, frame update, etc.)
  await swapFn();

  // Brief yield so the browser paints the new src at opacity:0
  await new Promise(r => requestAnimationFrame(r));
  await new Promise(r => requestAnimationFrame(r));

  // Fade in with the swap speed
  img.style.transition = `opacity ${_AV_SWAP_IN_MS}ms ease`;
  img.style.opacity = '1';
  await new Promise(r => setTimeout(r, _AV_SWAP_IN_MS));

  // Restore the original slow transition for sleep/ghost system
  img.style.transition = savedTransition;
  img.style.opacity = '';  // let CSS class rules take over again
}

// ── Character navigation (avatar overlay arrows) ──────────────────────────
let _charNavBusy = false;

async function navigateCharacter(dir) {
  if (_charNavBusy) return;
  const currentPath = _loadedCharPath || '';

  // Fetch adjacent character
  let target;
  try {
    const res = await fetch(`/characters/adjacent?path=${encodeURIComponent(currentPath)}&dir=${dir}`);
    const data = await res.json();
    if (!data.ok) return;
    target = data;
  } catch(e) { console.error('[NAV]', e); return; }

  _charNavBusy = true;

  await _avSwapFade(async () => {
    try {
      const res = await fetch('/characters/load', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({path: target.path}),
      });
      const data = await res.json();
      if (!data.ok) { _charNavBusy = false; return; }

      _loadedCharPath = target.path;
      document.getElementById('btn-save-current').disabled = false;

      if (data.ui_hue != null) applyUIHue(data.ui_hue);

      const charName = data.char_name ? data.char_name.toUpperCase() : target.label.toUpperCase();
      window._currentCharName = charName;
      const cn = document.getElementById('avatar-char-name');
      if (cn) cn.textContent = charName;

      const sel = document.getElementById('s-char');
      if (sel) {
        for (const opt of sel.options) {
          if (opt.value === target.path) { sel.value = target.path; break; }
        }
      }

      if (data.master_gain != null) {
        masterGain = data.master_gain;
        const vs = document.getElementById('vol-slider');
        if (vs) vs.value = masterGain;
        if (gainNode) gainNode.gain.value = masterGain;
      }

      // Swap chat bubbles
      const chatDiv2 = document.getElementById('chat');
      chatDiv2.innerHTML = '';
      if (data.chat_history && data.chat_history.length) {
        data.chat_history.forEach(msg => {
          const dispText=msg.user_image?msg.content.replace(/^\[image attached\]\s*/,''):msg.content;
          const bubble = addBubble(msg.role, dispText);
          if(msg.user_image){const ui=document.createElement('img');ui.src=msg.user_image;ui.className='bubble-img';ui.style.cursor='zoom-in';ui.onclick=()=>openLightbox(ui.src);bubble.insertBefore(ui,bubble.firstChild);}
          (msg.gen_images || []).forEach(uri => {
            const im = document.createElement('img'); im.src = uri;
            im.className = 'bubble-img generated-img';
            im.style.cssText = 'max-width:100%;max-height:360px;margin-top:8px;cursor:zoom-in';
            im.onclick = () => openLightbox(uri); bubble.appendChild(im);
          });
        });
      }

      await loadState();
    } catch(e) { console.error('[NAV]', e); }
  });

  _charNavBusy = false;
}

// Auto-resize avatar text input
document.addEventListener('DOMContentLoaded', () => {
  const ai = document.getElementById('avatar-msg-input');
  if (ai) {
    ai.addEventListener('input', () => { ai.style.height='auto'; ai.style.height=Math.min(ai.scrollHeight,80)+'px'; });
    ai.addEventListener('keydown', e => { if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendAvatarText();} });
  }
});

// ── Avatar waveform (small, in overlay) ───────────────────────────────────────
let _avWaveFrame = null;
function _drawAvatarWave() {
  const canvas = document.getElementById('avatar-wave');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function _sizeCanvas() {
    const wrap = document.getElementById('avatar-wave-wrap');
    if (!wrap) return;
    const rect = wrap.getBoundingClientRect();
    const dpr  = devicePixelRatio || 1;
    const w = Math.round(rect.width  * dpr);
    const h = Math.round(rect.height * dpr);
    if (w > 0 && h > 0 && (canvas.width !== w || canvas.height !== h)) {
      canvas.width  = w;
      canvas.height = h;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
  }

  // Size immediately (overlay is visible by the time this is called)
  _sizeCanvas();

  function loop() {
    if (!_avOverlayOpen) return;
    _avWaveFrame = requestAnimationFrame(loop);
    _sizeCanvas();
    const dpr = devicePixelRatio;
    const W = canvas.width / dpr, H = canvas.height / dpr;
    if (W <= 0 || H <= 0) return;

    const data  = _waveSmoothed || silentBuf;
    const green = getComputedStyle(document.documentElement).getPropertyValue('--green').trim()||'#4cff7a';
    const mode  = waveModes[waveMode]; // same mode as main display

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--tint-slot').trim()||'#0a0f0a';
    ctx.fillRect(0, 0, W, H);

    if (mode === 'wave') {
      ctx.strokeStyle = green; ctx.lineWidth = 1.5; ctx.beginPath();
      const midY = H/2, amp = H * 0.38 * waveAmp;
      for (let i = 0; i < data.length; i++) {
        const x = (i/data.length)*W, y = midY - data[i]*amp;
        i === 0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
      }
      ctx.stroke();

    } else if (mode === 'ribbon') {
      const midY = H/2, amp = H * 0.36 * waveAmp, off = amp * 0.15;
      const top = new Path2D(), bot = new Path2D();
      for (let i = 0; i < data.length; i++) {
        const x = (i/data.length)*W, y = midY - data[i]*amp;
        if (i===0){top.moveTo(x,y-off);bot.moveTo(x,y+off);}
        else{top.lineTo(x,y-off);bot.lineTo(x,y+off);}
      }
      const grad = ctx.createLinearGradient(0, midY-amp, 0, midY+amp);
      grad.addColorStop(0,_uiC(0.7)); grad.addColorStop(1,_uiC(0.15));
      ctx.fillStyle = grad;
      const fill = new Path2D(top);
      for (let i = data.length-1; i >= 0; i--) {
        const x = (i/data.length)*W, y = midY - data[i]*amp;
        fill.lineTo(x, y+off);
      }
      fill.closePath(); ctx.fill(fill);
      ctx.strokeStyle = green; ctx.lineWidth = 1.5;
      ctx.stroke(top); ctx.stroke(bot);

    } else if (mode === 'bars') {
      const bars=48, barW=(W/bars)*.7, gap=(W/bars)*.3, midY=H/2, maxH=H*.42;
      ctx.fillStyle = green;
      for (let i = 0; i < bars; i++) {
        const idx = Math.floor(i*data.length/bars);
        const h = Math.max(2, Math.abs(data[idx])*maxH*2);
        const x = i*(W/bars)+gap/2;
        ctx.globalAlpha = .5 + Math.abs(data[idx])*2;
        ctx.fillRect(x, midY-h/2, barW, h);
      }
      ctx.globalAlpha = 1;

    } else if (mode === 'radial') {
      const cx=W/2, cy=H/2, baseR=Math.min(W,H)*.12, maxR=Math.min(W,H)*.36, spokes=64;
      ctx.strokeStyle = green; ctx.lineWidth = 1.5;
      for (let i = 0; i < spokes; i++) {
        const angle = (i/spokes)*Math.PI*2 - Math.PI/2;
        const idx = Math.floor(i*data.length/spokes);
        const r = baseR + Math.abs(data[idx])*maxR;
        ctx.beginPath();
        ctx.moveTo(cx+Math.cos(angle)*baseR, cy+Math.sin(angle)*baseR);
        ctx.lineTo(cx+Math.cos(angle)*r,     cy+Math.sin(angle)*r);
        ctx.stroke();
      }
    }
  }
  loop();
}

// ── Avatar effects engine ─────────────────────────────────────────────────────
let _avEffectFrame   = null;
let _avScanlineOff   = 0;
let _avTintEnabled   = false;
let _avGlitchEnabled = false;
let _avWireEnabled   = false;
let _avWireFloor     = true;
let _avWireWalls     = true;
let _avWireReverse   = false;


let _avGlitchBars    = [];
let _avGlitchTimer   = null;
let _avWireOffset    = 0;   // animated Z offset for perspective grid

// ── Noise / scanlines / static ────────────────────────────────────────────────
function applyAvatarNoise() {
  if (!_avOverlayOpen) return;
  if (!_avEffectFrame) _avStartEffectLoop();
}

function _avGetNoiseSettings() {
  const mode      = (document.getElementById('av-noise-mode')         || {value:'mixed'}).value;
  const intensity = parseFloat((document.getElementById('av-noise-intensity')  || {value:0.5}).value);
  const slMode    = (document.getElementById('av-scanline-mode')       || {value:'roll'}).value;
  const slSpacing = parseInt((document.getElementById('av-scanline-spacing')   || {value:4}).value);
  return { mode, intensity, slMode, slSpacing,
    scanlinesOn: mode === 'scanlines' || mode === 'mixed',
    staticOn:    mode === 'static'    || mode === 'mixed' };
}

function _avStartEffectLoop() {
  if (_avEffectFrame) return;
  const slCanvas  = document.getElementById('avatar-scanlines');
  const stCanvas  = document.getElementById('avatar-static-canvas');
  const wfCanvas  = document.getElementById('avatar-wireframe');
  if (!slCanvas || !stCanvas || !wfCanvas) return;
  const slCtx = slCanvas.getContext('2d');
  const stCtx = stCanvas.getContext('2d');
  const wfCtx = wfCanvas.getContext('2d');

  let lastTime = 0;
  function loop(ts) {
    if (!_avOverlayOpen) { _avEffectFrame = null; return; }
    _avEffectFrame = requestAnimationFrame(loop);
    const dt = Math.min(ts - lastTime, 50); lastTime = ts;

    const ns = _avGetNoiseSettings();
    const W  = slCanvas.offsetWidth  * devicePixelRatio;
    const H  = slCanvas.offsetHeight * devicePixelRatio;
    // Resize all canvases together only if dimensions changed
    if (slCanvas.width !== W || slCanvas.height !== H) {
      slCanvas.width = stCanvas.width = wfCanvas.width  = W;
      slCanvas.height= stCanvas.height= wfCanvas.height = H;
    }

    slCtx.clearRect(0, 0, W, H);
    stCtx.clearRect(0, 0, W, H);

    // ── Wireframe perspective grid ──────────────────────────────────────────
    if (_avWireEnabled) {
      const speed = parseFloat((document.getElementById('av-wire-speed') || {value:0.15}).value);
      const depth = parseFloat((document.getElementById('av-wire-depth') || {value:0.7}).value);
      const dir = _avWireReverse ? -1 : 1;
      _avWireOffset = ((_avWireOffset + dir * dt * speed * 0.0004) % 1 + 1) % 1;
      _drawWireframe(wfCtx, W, H, _avWireOffset, depth);
    } else {
      wfCtx.clearRect(0, 0, W, H);
    }

    // ── Scanlines ──────────────────────────────────────────────────────────
    if (ns.scanlinesOn && ns.intensity > 0) {
      const spacing = ns.slSpacing * devicePixelRatio;
      const alpha   = 0.08 + ns.intensity * 0.55;

      if (ns.slMode === 'roll') {
        _avScanlineOff = (_avScanlineOff + dt * 0.06) % spacing;
      } else if (ns.slMode === 'flicker') {
        if (Math.random() < 0.05) _avScanlineOff = Math.random() * spacing;
      }

      slCtx.fillStyle = `rgba(0,0,0,${alpha.toFixed(2)})`;
      for (let y = -spacing + _avScanlineOff; y < H; y += spacing) {
        slCtx.fillRect(0, y, W, devicePixelRatio * 1.5);
      }
      slCanvas.style.opacity = '1';
    } else {
      slCanvas.style.opacity = '0';
    }

    // ── Static grain ───────────────────────────────────────────────────────
    if (ns.staticOn && ns.intensity > 0) {
      const alpha = ns.intensity * 0.22;
      const imgData = stCtx.createImageData(W, H);
      const data = imgData.data;
      for (let i = 0; i < data.length; i += 4) {
        const v = Math.random() * 255 | 0;
        data[i] = data[i+1] = data[i+2] = v;
        data[i+3] = (Math.random() * 255 * alpha) | 0;
      }
      stCtx.putImageData(imgData, 0, 0);
      stCanvas.style.opacity = '1';
    } else {
      stCanvas.style.opacity = '0';
    }

    // ── Glitch bars ─────────────────────────────────────────────────────────
    if (_avGlitchEnabled && _avGlitchBars.length) {
      const gi = parseFloat((document.getElementById('av-glitch-intensity') || {value:0.4}).value);
      for (const bar of _avGlitchBars) {
        bar.opacity -= dt * 0.008;
        if (bar.opacity <= 0) continue;
        const op = Math.min(bar.opacity * gi, 0.85).toFixed(2);
        slCtx.save();
        slCtx.globalAlpha = parseFloat(op);
        slCtx.fillStyle = _uiC(0.35);
        slCtx.fillRect(0, bar.y * H, W, bar.h * H);
        slCtx.fillStyle = `rgba(255,60,60,0.18)`;
        slCtx.fillRect(bar.dx * W, bar.y * H, W * 0.6, bar.h * H);
        slCtx.restore();
      }
      _avGlitchBars = _avGlitchBars.filter(b => b.opacity > 0);
    }
  }
  loop(0);
}

// ── Perspective wireframe box ─────────────────────────────────────────────────

function _drawWireframe(ctx, W, H, zOff, depth) {
  ctx.clearRect(0, 0, W, H);

  const cx     = W / 2;
  const horizY = H * 0.42;   // single shared horizon for all planes
  const COLS   = 8;
  const ROWS   = 10;

  ctx.lineWidth = devicePixelRatio * 0.6;

  const floorW = W * 1.6;   // width of floor/ceiling fan base
  const wallH  = H * 1.6;   // height of wall fan base

  // ── FLOOR (below horizon) ────────────────────────────────────────────────
  if (_avWireFloor) {
    // Radial lines converging to cx, horizY — fan spreading to bottom edge
    for (let i = 0; i <= COLS; i++) {
      const t = i / COLS;
      const bx = (cx - floorW / 2) + t * floorW;
      const alpha = Math.max(0, 0.08 + depth * 0.35 * (1 - Math.abs(t - 0.5) * 1.4));
      ctx.strokeStyle = _uiC(alpha);
      ctx.beginPath(); ctx.moveTo(cx, horizY); ctx.lineTo(bx, H); ctx.stroke();
    }
    // Horizontal rows scrolling away from viewer (toward horizon)
    for (let i = 0; i < ROWS; i++) {
      const rawT = ((i / ROWS) + zOff) % 1;
      const t = Math.pow(rawT, 1.8);
      const y = horizY + (H - horizY) * t;
      const halfW = (floorW / 2) * t;
      const alpha = depth * 0.45 * t;
      ctx.strokeStyle = _uiC(alpha);
      ctx.beginPath(); ctx.moveTo(cx - halfW, y); ctx.lineTo(cx + halfW, y); ctx.stroke();
    }
  }

  // ── CEILING (above horizon, mirror of floor) ─────────────────────────────
  if (_avWireFloor) {
    // Radial lines converging to cx, horizY — fan spreading to top edge
    for (let i = 0; i <= COLS; i++) {
      const t = i / COLS;
      const bx = (cx - floorW / 2) + t * floorW;
      const alpha = Math.max(0, 0.04 + depth * 0.2 * (1 - Math.abs(t - 0.5) * 1.4));
      ctx.strokeStyle = _uiC(alpha);
      ctx.beginPath(); ctx.moveTo(cx, horizY); ctx.lineTo(bx, 0); ctx.stroke();
    }
    // Ceiling rows — same zOff direction, mirror y above horizon
    for (let i = 0; i < ROWS; i++) {
      const rawT = ((i / ROWS) + zOff) % 1;
      const t = Math.pow(rawT, 1.8);
      const y = horizY - horizY * t;
      const halfW = (floorW / 2) * t;
      const alpha = depth * 0.25 * t;
      ctx.strokeStyle = _uiC(alpha);
      ctx.beginPath(); ctx.moveTo(cx - halfW, y); ctx.lineTo(cx + halfW, y); ctx.stroke();
    }
  }

  // ── WALLS (left and right of horizon) ────────────────────────────────────
  if (_avWireWalls) {
    // Wall fan: radial lines from vanishing point (cx, horizY) to left/right edges.
    // Fan spans vertically centred on horizY with total height wallH.
    const wallTop    = horizY - wallH / 2;
    const wallBottom = horizY + wallH / 2;

    for (let i = 0; i <= COLS; i++) {
      const t  = i / COLS;
      const by = wallTop + t * wallH;
      const alpha = Math.max(0, 0.08 + depth * 0.35 * (1 - Math.abs(t - 0.5) * 1.4));
      ctx.strokeStyle = _uiC(alpha);
      ctx.beginPath(); ctx.moveTo(cx, horizY); ctx.lineTo(W, by); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx, horizY); ctx.lineTo(0, by); ctx.stroke();
    }

    // Vertical columns — x expands from cx outward, y span matches the fan at that x position.
    // At t=1 the column reaches the screen edge and spans wallTop→wallBottom.
    // At t=0 it collapses to a point at horizY. Lerp the top/bottom from horizY to wall extents.
    for (let i = 0; i < ROWS; i++) {
      const rawT = ((i / ROWS) + zOff) % 1;
      const t    = Math.pow(rawT, 1.8);

      const xr = cx + (W  - cx) * t;
      const xl = cx - cx * t;

      const colTop    = horizY + (wallTop    - horizY) * t;
      const colBottom = horizY + (wallBottom - horizY) * t;

      const alpha = depth * 0.45 * t;
      ctx.strokeStyle = _uiC(alpha);
      ctx.beginPath(); ctx.moveTo(xr, colTop); ctx.lineTo(xr, colBottom); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(xl, colTop); ctx.lineTo(xl, colBottom); ctx.stroke();
    }
  }

  // ── Horizon line ─────────────────────────────────────────────────────────
  if (_avWireFloor || _avWireWalls) {
    ctx.strokeStyle = _uiC(depth * 0.5);
    ctx.lineWidth = devicePixelRatio;
    ctx.beginPath(); ctx.moveTo(0, horizY); ctx.lineTo(W, horizY); ctx.stroke();
  }
}


// ── Wireframe toggle ──────────────────────────────────────────────────────────
function toggleAvWireframe() {
  _avWireEnabled = !_avWireEnabled;
  const btn = document.getElementById('av-wire-btn');
  if (btn) { btn.textContent = _avWireEnabled ? 'ON' : 'OFF'; btn.className = 'btn' + (_avWireEnabled ? ' on' : ''); }
  if (_avWireEnabled && _avOverlayOpen && !_avEffectFrame) _avStartEffectLoop();
}

function toggleAvWireDir() {
  _avWireReverse = !_avWireReverse;
  const b = document.getElementById('av-wire-dir-btn');
  if (b) { b.textContent = _avWireReverse ? '◀ REV' : '▶ FWD'; b.className = 'btn' + (_avWireReverse ? ' on' : ''); }
}

function toggleAvWireAxis(axis) {
  if (axis === 'floor') {
    _avWireFloor = !_avWireFloor;
    const b = document.getElementById('av-wire-floor-btn');
    if (b) b.className = 'btn' + (_avWireFloor ? ' on' : '');
  } else {
    _avWireWalls = !_avWireWalls;
    const b = document.getElementById('av-wire-walls-btn');
    if (b) b.className = 'btn' + (_avWireWalls ? ' on' : '');
  }
}

function _avSpawnGlitch() {
  if (!_avGlitchEnabled) return;
  const count = 1 + Math.floor(Math.random() * 3);
  for (let i = 0; i < count; i++) {
    _avGlitchBars.push({
      y:       Math.random(),
      h:       0.005 + Math.random() * 0.04,
      opacity: 0.6 + Math.random() * 0.4,
      dx:      (Math.random() - 0.5) * 0.08,
    });
  }
}

function _avStartGlitchScheduler() {
  clearInterval(_avGlitchTimer);
  _avGlitchTimer = setInterval(() => {
    if (!_avGlitchEnabled || !_avOverlayOpen) return;
    const isTalking = _avIsTalking || _avIsScreaming;
    const chance = isTalking ? 0.35 : 0.06;
    if (Math.random() < chance) _avSpawnGlitch();
  }, 180);
}

// ── Color tint ───────────────────────────────────────────────────────────────
function toggleAvTint() {
  _avTintEnabled = !_avTintEnabled;
  const btn = document.getElementById('av-tint-btn');
  if (btn) { btn.textContent = _avTintEnabled ? 'ON' : 'OFF'; btn.className = 'btn' + (_avTintEnabled ? ' on' : ''); }
  _avApplyTint();
}

function _avApplyTint() {
  const el = document.getElementById('avatar-color-overlay');
  if (!el) return;
  if (!_avTintEnabled) { el.style.opacity = '0'; return; }
  const v = parseFloat((document.getElementById('av-tint-intensity') || {value:0.12}).value);
  el.style.opacity = v.toFixed(2);
}

// ── Glitch toggle ─────────────────────────────────────────────────────────────
function toggleAvGlitch() {
  _avGlitchEnabled = !_avGlitchEnabled;
  const btn = document.getElementById('av-glitch-btn');
  if (btn) { btn.textContent = _avGlitchEnabled ? 'ON' : 'OFF'; btn.className = 'btn' + (_avGlitchEnabled ? ' on' : ''); }
  if (_avGlitchEnabled) _avStartGlitchScheduler();
  else { clearInterval(_avGlitchTimer); _avGlitchBars = []; }
}

// ── Pixel filter ──────────────────────────────────────────────────────────────
// Renders avatar frames and FX canvas through a downscale→upscale pipeline
// to produce a hard 8-bit (nearest-neighbour) or soft 16-bit (bilinear) look.

let _avPixelEnabled  = false;
let _avPixelBilinear = false;
let _avPixelSize     = 6;

function toggleAvPixel() {
  _avPixelEnabled = !_avPixelEnabled;
  const btn    = document.getElementById('av-pixel-btn');
  const hudBtn = document.getElementById('av-pixel-hud-btn');
  const on = _avPixelEnabled;
  if (btn)    { btn.textContent    = on ? 'ON' : 'OFF'; btn.className    = 'btn' + (on ? ' on' : ''); }
  if (hudBtn) { hudBtn.textContent = on ? '≋ BLUR' : 'BLUR'; hudBtn.className = 'btn' + (on ? ' on' : ''); }
  ['av-pixel-bilinear-row','av-pixel-mode-row'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = on ? '' : 'none';
  });
  _avApplyPixel();
}

function toggleAvPixelBilinear() {
  _avPixelBilinear = !_avPixelBilinear;
  const btn = document.getElementById('av-pixel-bilinear-btn');
  const lbl = document.getElementById('av-pixel-mode-label');
  if (btn) { btn.textContent = _avPixelBilinear ? 'SOFT' : 'EDGE'; btn.className = 'btn' + (_avPixelBilinear ? ' on' : ''); }
  if (lbl) lbl.textContent = _avPixelBilinear ? 'soft blur only' : 'edge enhance · adds contrast';
  _avApplyPixel();
}

function _avApplyPixel() {
  const sizeSlider     = document.getElementById('av-pixel-size');
  const sizeVal        = document.getElementById('av-pixel-size-val');
  const contrastSlider = document.getElementById('av-pixel-contrast');
  const contrastVal    = document.getElementById('av-pixel-contrast-val');
  if (sizeSlider)     _avPixelSize = parseFloat(sizeSlider.value) || 0;
  if (sizeVal)        sizeVal.textContent = _avPixelSize.toFixed(1) + 'px';
  const contrast = contrastSlider ? parseInt(contrastSlider.value) || 100 : 100;
  if (contrastVal)    contrastVal.textContent = contrast + '%';

  const vp = document.getElementById('avatar-viewport');
  if (!vp) return;

  if (!_avPixelEnabled) {
    vp.style.filter = '';
    return;
  }

  const parts = [];
  if (_avPixelSize > 0) parts.push(`blur(${_avPixelSize.toFixed(1)}px)`);
  if (!_avPixelBilinear && contrast !== 100) parts.push(`contrast(${contrast}%)`);
  if (_avPixelBilinear && contrast !== 100)  parts.push(`contrast(${contrast}%)`);
  vp.style.filter = parts.length ? parts.join(' ') : '';
}


// ── Zoom / pan ────────────────────────────────────────────────────────────────
let _avScale          = 1.5;
let _avPanX           = 0;
let _avPanY           = 0;
let _avPositionSaved  = false;  // true once a character with saved scale/pan has been loaded
let _avLocked     = true;
let _avDragStart  = null;

function avToggleLock() {
  _avLocked = !_avLocked;
  const btn  = document.getElementById('avatar-lock-btn');
  const ctrl = document.getElementById('avatar-zoom-controls');
  if (btn)  { btn.textContent = _avLocked ? '🔒' : '🔓'; btn.className = _avLocked ? 'active' : ''; }
  if (ctrl) { ctrl.classList.toggle('locked', _avLocked); }
  const vp = document.getElementById('avatar-viewport');
  if (vp) vp.style.cursor = _avLocked ? 'default' : 'grab';
}

function _avApplyTransform() {
  const img = document.getElementById('avatar-img');
  if (!img) return;
  img.style.transform = `translate(${_avPanX}px, ${_avPanY}px) scale(${_avScale})`;
}

let _avPosSaveTimerGlobal = null;
function _avSchedulePosSave() {
  clearTimeout(_avPosSaveTimerGlobal);
  _avPosSaveTimerGlobal = setTimeout(() => {
    fetch('/settings', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ av_scale: _avScale, av_pan_x: _avPanX, av_pan_y: _avPanY }),
    }).catch(() => {});
  }, 800);
}

function avZoom(delta) {
  if (_avLocked) return;
  _avScale = Math.max(0.3, Math.min(8, _avScale + delta));
  _avApplyTransform();
  _avSchedulePosSave();
}

function avZoomReset() {
  _avScale = 1.5; _avPanX = 0;
  const vp = document.getElementById('avatar-viewport');
  _avPanY = vp ? vp.offsetHeight * 0.08 : 0;
  _avApplyTransform();
  _avSchedulePosSave();
}

// Wire drag events onto the viewport
(function() {
  function getVP() { return document.getElementById('avatar-viewport'); }

  function onDown(e) {
    if (_avLocked) return;
    const vp = getVP(); if (!vp) return;
    const pt = e.touches ? e.touches[0] : e;
    _avDragStart = { x: pt.clientX, y: pt.clientY, panX: _avPanX, panY: _avPanY };
    vp.classList.add('dragging');
  }
  function onMove(e) {
    if (_avLocked || !_avDragStart) return;
    const pt = e.touches ? e.touches[0] : e;
    _avPanX = _avDragStart.panX + (pt.clientX - _avDragStart.x);
    _avPanY = _avDragStart.panY + (pt.clientY - _avDragStart.y);
    _avApplyTransform();
    e.preventDefault();
  }
  function onUp() {
    _avDragStart = null;
    const vp = getVP(); if (vp) vp.classList.remove('dragging');
    _avSchedulePosSave();
  }
  function onWheel(e) {
    e.preventDefault();
    if (!_avLocked) { avZoom(e.deltaY < 0 ? 0.1 : -0.1); }  // avZoom calls _avSchedulePosSave
  }

  // Attach after DOM ready — overlay exists but may not be visible
  document.addEventListener('DOMContentLoaded', () => {
    const vp = document.getElementById('avatar-viewport');
    if (!vp) return;
    vp.addEventListener('mousedown',  onDown);
    vp.addEventListener('touchstart', onDown, {passive:false});
    document.addEventListener('mousemove',  onMove);
    document.addEventListener('touchmove',  onMove, {passive:false});
    document.addEventListener('mouseup',    onUp);
    document.addEventListener('touchend',   onUp);
    vp.addEventListener('wheel', onWheel, {passive:false});
  });
  // Also attach immediately in case DOMContentLoaded already fired
  window.addEventListener('load', () => {
    const vp = document.getElementById('avatar-viewport');
    if (!vp || vp._dragBound) return;
    vp._dragBound = true;
    vp.addEventListener('mousedown',  onDown);
    vp.addEventListener('touchstart', onDown, {passive:false});
    document.addEventListener('mousemove',  onMove);
    document.addEventListener('touchmove',  onMove, {passive:false});
    document.addEventListener('mouseup',    onUp);
    document.addEventListener('touchend',   onUp);
    vp.addEventListener('wheel', onWheel, {passive:false});
  });
})();

// ── Avatar Code Renderer ──────────────────────────────────────────────────────
// Detects Python code blocks or ANSI-rich text in chat messages.
// When found and avatar overlay is open:
//   1. Fades avatar image to ghost (8% opacity)
//   2. Renders highlighted, scrolling code on the code canvas
//   3. Auto-dismisses after a duration scaled to code length

let _avCodeActive       = false;
let _avCodeRafId        = null;
let _avCodeDismissTimer = null;

const _avCodePalette = {
  bg:       '#080d08',
  default:  '#c8ffc8',
  keyword:  '#4cff7a',
  string:   '#a8ff78',
  comment:  '#3a7a3a',
  number:   '#78ffd6',
  func:     '#b8ffb8',
  builtin:  '#60ff60',
  operator: '#4cff7a',
  punct:    '#5a8a5a',
  ansiMap: {
    '30':'#1a3a1a','31':'#ff6060','32':'#4cff7a','33':'#ffe066',
    '34':'#60b8ff','35':'#c060ff','36':'#60fff0','37':'#c8ffc8',
    '90':'#3a6a3a','91':'#ff9090','92':'#78ff9a','93':'#fff09a',
    '94':'#90ccff','95':'#d090ff','96':'#90fff8','97':'#ffffff',
  }
};

const _avPyKeywords = new Set(['def','class','return','import','from','as','if','elif','else',
  'for','while','try','except','finally','with','pass','break','continue','yield',
  'lambda','not','and','or','in','is','True','False','None','async','await','raise',
  'global','nonlocal','del','assert']);
const _avPyBuiltins = new Set(['print','len','range','type','str','int','float','list','dict',
  'set','tuple','bool','open','input','super','self','cls','enumerate','zip','map',
  'filter','sorted','reversed','any','all','min','max','sum','abs','round','repr']);

function _avTokenisePython(line) {
  if (/\x1b\[/.test(line)) return _avTokeniseAnsi(line);
  const tokens = [];
  let i = 0;
  const push = (text, color) => { if (text) tokens.push({text, color}); };
  while (i < line.length) {
    if (line[i] === '#') { push(line.slice(i), _avCodePalette.comment); break; }
    const tq = '\x22\x22\x22', tq2 = '\x27\x27\x27';
    const qm = line.slice(i, i+3);
    if (qm === tq || qm === tq2) {
      const end = line.indexOf(qm, i+3);
      const s = end < 0 ? line.slice(i) : line.slice(i, end+3);
      push(s, _avCodePalette.string); i += s.length; continue;
    }
    if (line[i] === '"' || line[i] === "'") {
      const q = line[i]; let j = i+1;
      while (j < line.length && line[j] !== q) j++;
      push(line.slice(i, j+1), _avCodePalette.string); i = j+1; continue;
    }
    if (/[0-9]/.test(line[i])) {
      let j = i; while (j < line.length && /[\d._xXa-fA-F]/.test(line[j])) j++;
      push(line.slice(i, j), _avCodePalette.number); i = j; continue;
    }
    if (/[a-zA-Z_]/.test(line[i])) {
      let j = i; while (j < line.length && /[\w]/.test(line[j])) j++;
      const word = line.slice(i, j);
      let col = _avCodePalette.default;
      if (_avPyKeywords.has(word)) col = _avCodePalette.keyword;
      else if (_avPyBuiltins.has(word)) col = _avCodePalette.builtin;
      else if (line[j] === '(') col = _avCodePalette.func;
      push(word, col); i = j; continue;
    }
    if (/[=+\-*/%<>!&|^~@]/.test(line[i])) { push(line[i], _avCodePalette.operator); i++; continue; }
    if (/[()[\]{},.:;]/.test(line[i]))      { push(line[i], _avCodePalette.punct);    i++; continue; }
    push(line[i], _avCodePalette.default); i++;
  }
  return tokens;
}

function _avTokeniseAnsi(line) {
  const tokens = [];
  const re = /\x1b\[([0-9;]*)m/g;
  let last = 0, curColor = _avCodePalette.default, bold = false;
  let match;
  while ((match = re.exec(line)) !== null) {
    if (match.index > last) tokens.push({text: line.slice(last, match.index), color: curColor, bold});
    const codes = match[1].split(';');
    for (const c of codes) {
      if (c === '0' || c === '') { curColor = _avCodePalette.default; bold = false; }
      else if (c === '1') bold = true;
      else if (_avCodePalette.ansiMap[c]) curColor = _avCodePalette.ansiMap[c];
    }
    last = match.index + match[0].length;
  }
  if (last < line.length) tokens.push({text: line.slice(last), color: curColor, bold});
  return tokens;
}

function _avExtractCodeBlock(text) {
  // 1. Fenced block — closed (``` ... ```) with optional newline OR code on same line as tag
  const fenced = text.match(/```(?:python|py)?\s*\n?([\s\S]*?)```/i);
  if (fenced) return { code: fenced[1].trim(), type: 'code' };

  // 2. Unclosed fence
  const unclosed = text.match(/```(?:python|py)?\s*\n?([\s\S]+)$/i);
  if (unclosed && unclosed[1].trim().length > 30) return { code: unclosed[1].trim(), type: 'code' };

  // 3. ANSI escape codes — agent coloured terminal output
  if (/\x1b\[/.test(text)) return { code: text, type: 'code' };

  // 4. Heuristic: strong Python structural signals
  const lines = text.split('\n').filter(l => l.trim());
  if (lines.length < 3) return null;
  const structuralPy = lines.filter(l =>
    /^\s*(def |class |import |from .+ import|async def |@\w)/.test(l) ||
    /^\s*[\w.[\]]+\s*=\s*[^=]/.test(l) ||
    /^\s+(if |elif |else:|for |while |try:|except|return |yield |pass$|break$)/.test(l) ||
    /:\s*$/.test(l.replace(/#.*/, '').trimEnd())
  ).length;
  if (structuralPy >= 2 && structuralPy / lines.length > 0.4) return { code: text, type: 'code' };

  return null;
}

let _avCodeQueue = [];  // queued {code, type} objects to display sequentially

// Typewrite speeds (chars/sec)
const _AV_CODE_CPS  = 120;   // Python / code — fast terminal feel
const _AV_ASCII_CPS = 55;    // ASCII art — slower, character-by-character reveal

// How long (ms) to hold ASCII art on screen after fully revealed before fading
const _AV_ASCII_HOLD_MS = 3500;

function _avShowCode(code, type='code') {
  if (!_avOverlayOpen) return;
  if (_avCodeActive) {
    // Don't interrupt if we're already rendering the same content (SSE echo guard)
    if (_avCurrentCode === code) return;
    _avCodeQueue = [];
    _avDismissCode(/*soft*/true);
  }
  _avRunCode(code, type);
}

function _avRunCode(code, type='code') {
  if (!_avOverlayOpen) return;
  _avCodeActive = true;
  _avCurrentCode = code;
  clearTimeout(_avCodeDismissTimer);
  cancelAnimationFrame(_avCodeRafId);

  const img    = document.getElementById('avatar-img');
  const canvas = document.getElementById('avatar-code-canvas');
  if (!img || !canvas) return;

  img.style.opacity    = '0.08';
  canvas.style.opacity = '1';

  const lines = code.split('\n');
  const dpr   = devicePixelRatio;

  const revealedChars = new Array(lines.length).fill(0);
  let currentLine  = 0;
  const CHARS_PER_SEC = type === 'ascii' ? _AV_ASCII_CPS : _AV_CODE_CPS;
  let lastTs = null;

  let scrollY      = 0;
  let targetScroll = 0;

  let glitchBars = [];
  let glitchTick = 0;

  // Per-run generation counter so a stale RAF can't fire after interrupt
  const myGen = ++_avCodeGeneration;

  // Cache ctx and layout constants outside the RAF loop — recompute only on resize
  const ctx = canvas.getContext('2d');
  let W = 0, H = 0, fontSize = 0, lineH = 0, padX = 0, padY = 0, rightPad = 0;
  let gradTop = null, gradBot = null;  // vignette gradient cache
  let fontNormal = '', fontBold = '', fontGutter = '';

  function _recomputeLayout() {
    const newW = canvas.offsetWidth * dpr, newH = canvas.offsetHeight * dpr;
    if (newW === W && newH === H) return;  // no change
    W = newW; H = newH;
    canvas.width = W; canvas.height = H;
    const vpW = W / dpr, vpH = H / dpr;
    const vmin = Math.min(vpW, vpH);
    fontSize  = type === 'ascii'
      ? Math.max(12, vmin * 0.045) * dpr
      : Math.max(6,  vmin * 0.016) * dpr;
    lineH    = fontSize * 1.55;
    padX     = 22 * dpr;
    padY     = 14 * dpr;
    rightPad = 18 * dpr;
    fontNormal = `${fontSize}px "Courier New",monospace`;
    fontBold   = `bold ${fontSize}px "Courier New",monospace`;
    fontGutter = `${fontSize * 0.72}px "Courier New",monospace`;
    // Rebuild gradient cache after resize
    gradTop = ctx.createLinearGradient(0, 0, 0, padY * 4);
    gradTop.addColorStop(0, 'rgba(8,13,8,0.95)');
    gradTop.addColorStop(1, 'rgba(8,13,8,0)');
    gradBot = ctx.createLinearGradient(0, H - padY * 4, 0, H);
    gradBot.addColorStop(0, 'rgba(8,13,8,0)');
    gradBot.addColorStop(1, 'rgba(8,13,8,0.95)');
  }

  // For ASCII art, pre-tokenise all lines once as single plain-text tokens (no syntax needed)
  // For code, tokenise lazily per-line but cache results so we don't re-tokenise on every frame
  const tokenCache = new Array(lines.length).fill(null);
  function _getTokens(li, revealed) {
    if (type === 'ascii') {
      // ASCII: single token per line, no colour splitting needed
      return [{ text: revealed, color: _avCodePalette.default, bold: false }];
    }
    // For code: cache invalidates when the revealed slice grows
    if (tokenCache[li] && tokenCache[li].src === revealed) return tokenCache[li].tokens;
    const tokens = _avTokenisePython(revealed);
    tokenCache[li] = { src: revealed, tokens };
    return tokens;
  }

  // Measure cache — ctx.measureText is expensive; cache width per (font+text) pair
  const _measureCache = new Map();
  function _measureWidth(text, bold) {
    const key = (bold ? '1' : '0') + text;
    if (_measureCache.has(key)) return _measureCache.get(key);
    ctx.font = bold ? fontBold : fontNormal;
    const w = ctx.measureText(text).width;
    _measureCache.set(key, w);
    // Prevent unbounded growth on very long sessions
    if (_measureCache.size > 4000) _measureCache.clear();
    return w;
  }

  function render(ts) {
    if (!_avCodeActive || _avCodeGeneration !== myGen) return;
    _avCodeRafId = requestAnimationFrame(render);

    _recomputeLayout();  // no-op unless canvas was resized

    if (lastTs === null) lastTs = ts;
    const dt = Math.min(ts - lastTs, 100) / 1000;
    lastTs = ts;

    if (currentLine < lines.length) {
      const lineLen = lines[currentLine].length || 1;
      revealedChars[currentLine] = Math.min(lineLen, (revealedChars[currentLine] || 0) + CHARS_PER_SEC * dt);
      if (revealedChars[currentLine] >= lineLen) {
        revealedChars[currentLine] = lineLen;
        currentLine++;
        while (currentLine < lines.length && lines[currentLine].trim() === '') {
          revealedChars[currentLine] = 0;
          currentLine++;
        }
      }
    }

    const cursorY = padY + currentLine * lineH;
    const screenBottom = H - padY * 4;
    if (cursorY - scrollY > screenBottom) targetScroll = cursorY - screenBottom;

    const done = (currentLine >= lines.length);
    scrollY += (targetScroll - scrollY) * 0.06;

    ctx.fillStyle = 'rgba(8,13,8,0.93)';
    ctx.fillRect(0, 0, W, H);

    glitchTick++;
    if (glitchTick > 50 && Math.random() < 0.04) {
      glitchTick = 0;
      for (let n = 0; n < 1 + Math.floor(Math.random()*2); n++)
        glitchBars.push({ y: Math.random()*H, dx: (Math.random()-0.5)*14*dpr, op: 0.6+Math.random()*0.35 });
    }
    glitchBars = glitchBars.filter(g => (g.op *= 0.85) > 0.02);

    ctx.textBaseline = 'top';

    // Clip text to canvas width so nothing escapes right edge
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, W - rightPad, H);
    ctx.clip();

    // Set font once per frame — only switch to bold inside loop if token requires it
    ctx.font = fontNormal;
    let lastFont = fontNormal;

    for (let li = 0; li < lines.length; li++) {
      const y = padY + li * lineH - scrollY;
      if (y < -lineH*2 || y > H + lineH) continue;
      if (!revealedChars[li]) continue;

      const revealed = lines[li].slice(0, Math.ceil(revealedChars[li]));
      const tokens   = _getTokens(li, revealed);

      let dx = 0;
      for (const g of glitchBars) { if (Math.abs(g.y - y) < lineH*2) dx += g.dx * g.op; }

      let x = padX + dx;
      for (const tok of tokens) {
        const f = tok.bold ? fontBold : fontNormal;
        if (f !== lastFont) { ctx.font = f; lastFont = f; }
        ctx.fillStyle = tok.color;
        ctx.fillText(tok.text, x, y);
        x += _measureWidth(tok.text, tok.bold);
      }
    }
    ctx.restore();

    // Gutter — only for code, not ascii art
    if (type !== 'ascii') {
      ctx.fillStyle = 'rgba(0,0,0,0.45)';
      ctx.fillRect(0, 0, padX * 0.88, H);
      ctx.font = fontGutter; lastFont = fontGutter;
      for (let li = 0; li < lines.length; li++) {
        if (!revealedChars[li]) continue;
        const y = padY + li * lineH - scrollY;
        if (y < -lineH || y > H + lineH) continue;
        ctx.fillStyle = _uiC(0.28);
        ctx.fillText(String(li+1).padStart(3), 2*dpr, y);
      }
    }

    // Cursor — blink while typing
    if (currentLine < lines.length && Math.floor(ts/500)%2 === 0) {
      const revealed = lines[currentLine].slice(0, Math.ceil(revealedChars[currentLine] || 0));
      if (ctx.font !== fontNormal) ctx.font = fontNormal;
      const tw = Math.min(_measureWidth(revealed, false), W - rightPad - padX - fontSize);
      const cy = padY + currentLine * lineH - scrollY;
      if (cy >= 0 && cy < H) {
        ctx.fillStyle = _avCodePalette.keyword;
        ctx.fillRect(padX + tw, cy, fontSize * 0.55, fontSize);
      }
    }

    // Vignette — use cached gradients
    ctx.fillStyle = gradTop; ctx.fillRect(0, 0, W, padY*4);
    ctx.fillStyle = gradBot; ctx.fillRect(0, H-padY*4, W, padY*4);

    // Fully revealed — hold then dismiss (ascii gets a linger, code dismisses sooner)
    if (done) {
      _avCodeActive = false;
      cancelAnimationFrame(_avCodeRafId); _avCodeRafId = null;
      const holdMs = type === 'ascii' ? _AV_ASCII_HOLD_MS : 1200;
      _avCodeDismissTimer = setTimeout(() => {
        if (_avCodeQueue.length > 0) {
          _avRunCode(...Object.values(_avCodeQueue.shift()));
        } else {
          _avDismissCode();
        }
      }, holdMs);
    }
  }
  _avCodeRafId = requestAnimationFrame(render);

  // Safety dismiss timeout — time to type all chars + hold + buffer, min 10s
  const totalChars = lines.reduce((s, l) => s + l.length, 0);
  const holdMs = type === 'ascii' ? _AV_ASCII_HOLD_MS : 1200;
  const dur = Math.max(10000, (totalChars / CHARS_PER_SEC) * 1000 + holdMs + 2000);
  _avCodeDismissTimer = setTimeout(() => {
    _avCodeActive = false;
    cancelAnimationFrame(_avCodeRafId); _avCodeRafId = null;
    if (_avCodeQueue.length > 0) _avRunCode(...Object.values(_avCodeQueue.shift()));
    else _avDismissCode();
  }, dur);
}

let _avCodeGeneration = 0;  // incremented on each _avRunCode call to invalidate stale RAFs
let _avCurrentCode = '';      // tracks content currently rendering — prevents SSE echo restarts

function _avDismissCode(soft=false) {
  _avCodeActive = false;
  if (!soft) _avCodeQueue = [];
  clearTimeout(_avCodeDismissTimer);
  cancelAnimationFrame(_avCodeRafId); _avCodeRafId = null;
  _avCurrentCode = '';
  if (soft) return;  // caller will immediately start a new run — leave canvas visible
  const img    = document.getElementById('avatar-img');
  const canvas = document.getElementById('avatar-code-canvas');
  if (img)    img.style.opacity    = '1';
  if (canvas) canvas.style.opacity = '0';
}

function _avIsAsciiArt(text) {
  const artChars = (text.match(/[⠀-⣿█▓▒░╔╗╚╝║═┌┐└┘│─┼╠╣╦╩╬▀▄▌▐]/g) || []).length;
  return artChars > 8;
}

function _avCheckBubbleForCode(text) {
  // Check for fenced ascii art first — pull just the content inside the backticks
  const fencedArt = text.match(/```\s*\n?([\s\S]*?)```/i);
  if (fencedArt && _avIsAsciiArt(fencedArt[1])) {
    _avShowCode(fencedArt[1].trim(), 'ascii'); return;
  }
  // Unfenced ascii art — whole message is art
  if (_avIsAsciiArt(text)) { _avShowCode(text, 'ascii'); return; }
  // Code detection
  const result = _avExtractCodeBlock(text);
  if (result) _avShowCode(result.code, result.type);
}

function _stripCodeForTTS(text) {
  // Remove closed fenced code blocks entirely
  let out = text.replace(/```[\s\S]*?```/g, ' ');
  // Remove any remaining unclosed fence (``` to end of string)
  out = out.replace(/```[\s\S]*$/g, ' ');
  // Remove inline code
  out = out.replace(/`[^`]+`/g, ' ');
  // Remove ANSI escape sequences
  out = out.replace(/\x1b\[[0-9;]*m/g, '');
  // Collapse excess whitespace/newlines left behind
  out = out.replace(/\n{3,}/g, '\n\n').trim();
  return out;
}

// ══════════════════════════════════════════════════════════════════════════════
// ── Ecko FX Engine ────────────────────────────────────────────────────────────
// Avatar-viewport-scoped canvas effects.  Lives inside #avatar-viewport so it
// is clipped to the avatar frame and never touches the chat/bubble UI.
// All colours derive from _uiHue so they follow the UI tint setting.
// ══════════════════════════════════════════════════════════════════════════════

(function(){

  // ── Canvas — injected into avatar-viewport, sits above code-canvas ────────
  const _fxCanvas = document.createElement('canvas');
  _fxCanvas.id = 'ecko-fx-canvas';
  _fxCanvas.style.cssText = [
    'position:absolute','inset:0','width:100%','height:100%',
    'pointer-events:none','z-index:10','opacity:0',  // z-index 10 = above glitch-bar(8) and sleep-overlay(9)
    'transition:opacity 0.35s ease',
  ].join(';');

  // Attach once avatar-viewport exists (it's in static HTML so it's always there)
  function _fxAttach() {
    const vp = document.getElementById('avatar-viewport');
    if (vp) { vp.appendChild(_fxCanvas); }
    else    { document.body.appendChild(_fxCanvas); }  // fallback
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', _fxAttach);
  else _fxAttach();

  // ── Tint helpers — read live _uiHue so colour always matches UI ──────────
  // h offset lets effects use analogous/split-complement hues from the base
  function _fxC(alpha, hOff=0, s=100, l=64) {
    return `hsla(${(_uiHue + hOff + 360) % 360},${s}%,${l}%,${alpha})`;
  }
  function _fxCDim(alpha, hOff=0)  { return _fxC(alpha, hOff, 60, 40);  }
  function _fxCGlow(alpha, hOff=0) { return _fxC(alpha, hOff, 100, 80); }

  // ── Canvas size sync — mirrors viewport size via ResizeObserver ───────────
  let _fxW = 0, _fxH = 0;
  function _fxSyncSize() {
    const vp = _fxCanvas.parentElement;
    if (!vp) return;
    const rect = vp.getBoundingClientRect();
    _fxW = Math.round(rect.width  * devicePixelRatio);
    _fxH = Math.round(rect.height * devicePixelRatio);
    if (_fxCanvas.width !== _fxW || _fxCanvas.height !== _fxH) {
      _fxCanvas.width  = _fxW;
      _fxCanvas.height = _fxH;
    }
  }
  const _fxRO = new ResizeObserver(_fxSyncSize);
  // observe once attached
  const _fxObserveWhenReady = () => {
    const vp = document.getElementById('avatar-viewport');
    if (vp) _fxRO.observe(vp);
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', _fxObserveWhenReady);
  else _fxObserveWhenReady();

  // ── Core state ────────────────────────────────────────────────────────────
  let _fxActive = false;
  let _fxRaf    = null;
  let _fxTimer  = null;
  let _fxGen    = 0;

  function _fxStop() {
    _fxActive = false;
    clearTimeout(_fxTimer);
    cancelAnimationFrame(_fxRaf);
    _fxGen++;
    _fxSyncSize();
    const ctx = _fxCanvas.getContext('2d');
    ctx.clearRect(0, 0, _fxCanvas.width, _fxCanvas.height);
    _fxCanvas.style.opacity = '0';
  }

  function _fxShow(durationMs, drawFn) {
    // Only render when avatar overlay is open — silently skip otherwise
    if (!_avOverlayOpen) return;
    _fxStop();
    _fxSyncSize();
    _fxActive = true;
    _fxCanvas.style.transition = 'opacity 0.35s ease';
    _fxCanvas.style.opacity = '1';
    const myGen = _fxGen;
    let start = null;
    function loop(ts) {
      if (!_fxActive || _fxGen !== myGen) return;
      if (!start) start = ts;
      const elapsed = ts - start;
      _fxSyncSize();
      drawFn(_fxCanvas.getContext('2d'), elapsed, _fxCanvas.width, _fxCanvas.height);
      if (elapsed < durationMs) {
        _fxRaf = requestAnimationFrame(loop);
      } else {
        _fxCanvas.style.transition = 'opacity 0.7s ease';
        _fxCanvas.style.opacity = '0';
        _fxTimer = setTimeout(_fxStop, 800);
      }
    }
    _fxRaf = requestAnimationFrame(loop);
  }

  // ── EFFECT: Matrix Rain ───────────────────────────────────────────────────
  // Uses tint hue for trail colour; bright white for the falling head
  function fxMatrixRain(durationMs = 4200) {
    const CHARS = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン0123456789ABCDEF<>{}[]|/\\░▒▓';
    let drops = [], fontSize = 0, lastW = 0;
    function init(W) {
      fontSize = Math.max(10, Math.floor(W / devicePixelRatio / 28)) * devicePixelRatio;
      const cols = Math.floor(W / fontSize);
      drops = Array.from({length: cols}, () => Math.random() * -60);
    }
    _fxShow(durationMs, (ctx, t, W, H) => {
      if (W !== lastW) { init(W); lastW = W; }
      const progress  = t / durationMs;
      const fadeAlpha = progress > 0.72 ? 1 - (progress - 0.72) / 0.28 : 1;
      // Smear background
      ctx.fillStyle = `rgba(0,0,0,${0.08 * fadeAlpha + (1-fadeAlpha)*0.2})`;
      ctx.fillRect(0, 0, W, H);
      ctx.font = `bold ${fontSize}px "Courier New",monospace`;
      drops.forEach((y, i) => {
        const x  = i * fontSize;
        const ch = CHARS[Math.floor(Math.random() * CHARS.length)];
        // Bright head — white-ish tint
        ctx.fillStyle = _fxCGlow(0.92 * fadeAlpha);
        ctx.fillText(ch, x, y * fontSize);
        // Trail — tint colour dimming
        ctx.fillStyle = _fxC(0.7 * fadeAlpha);
        if (y > 1) ctx.fillText(CHARS[Math.floor(Math.random() * CHARS.length)], x, (y-1)*fontSize);
        ctx.fillStyle = _fxCDim(0.5 * fadeAlpha);
        if (y > 3) ctx.fillText(CHARS[Math.floor(Math.random() * CHARS.length)], x, (y-3)*fontSize);
        if (Math.random() > 0.975 || drops[i] * fontSize > H) drops[i] = Math.random() * -25;
        drops[i] += 0.55;
      });
    });
  }

  // ── EFFECT: Glitch Storm ──────────────────────────────────────────────────
  function fxGlitchStorm(durationMs = 2600) {
    _fxShow(durationMs, (ctx, t, W, H) => {
      const progress  = t / durationMs;
      const intensity = Math.sin(progress * Math.PI);
      ctx.clearRect(0, 0, W, H);
      const numBars = Math.floor(4 + intensity * 20);
      for (let i = 0; i < numBars; i++) {
        const y  = Math.random() * H;
        const bh = (1 + Math.random() * 16) * devicePixelRatio;
        const dx = (Math.random() - 0.5) * 70 * devicePixelRatio * intensity;
        const bw = (20 + Math.random() * W * 0.7);
        const bx = Math.random() * (W - bw);
        const hOff = Math.floor(Math.random() * 60) - 30;
        ctx.fillStyle = _fxC(0.4 + Math.random()*0.5, hOff);
        ctx.fillRect(bx + dx, y, bw, bh);
        if (Math.random() > 0.55) {
          ctx.globalCompositeOperation = 'difference';
          ctx.fillStyle = `rgba(255,255,255,${0.12 * intensity})`;
          ctx.fillRect(bx + dx, y, bw, bh);
          ctx.globalCompositeOperation = 'source-over';
        }
      }
      // Full-width flash slice
      if (Math.random() < 0.09 * intensity) {
        ctx.fillStyle = _fxCGlow(0.5 * intensity);
        ctx.fillRect(0, Math.random() * H, W, (1 + Math.random()*3)*devicePixelRatio);
      }
      // RGB channel split ghost
      if (Math.random() < 0.04 * intensity) {
        const gy = Math.random() * H * 0.8;
        const gh = (8 + Math.random()*30)*devicePixelRatio;
        ctx.fillStyle = _fxC(0.18*intensity, -30);   // split left
        ctx.fillRect(-6*devicePixelRatio, gy, W, gh);
        ctx.fillStyle = _fxC(0.18*intensity,  30);   // split right
        ctx.fillRect(  6*devicePixelRatio, gy, W, gh);
      }
    });
  }

  // ── EFFECT: Signal Static ─────────────────────────────────────────────────
  function fxSignalStatic(durationMs = 1900) {
    _fxShow(durationMs, (ctx, t, W, H) => {
      const progress = t / durationMs;
      const alpha    = Math.sin(progress * Math.PI) * 0.6;
      // Tinted noise — bias toward the UI hue channel
      const hRad = (_uiHue / 360) * Math.PI * 2;
      const rBias = 0.5 + 0.5 * Math.cos(hRad);
      const gBias = 0.5 + 0.5 * Math.cos(hRad - 2.094);
      const bBias = 0.5 + 0.5 * Math.cos(hRad + 2.094);
      const imgData = ctx.createImageData(W, H);
      const d = imgData.data;
      for (let i = 0; i < d.length; i += 4) {
        const v = Math.random() > 0.45 ? Math.floor(Math.random() * 255) : 0;
        d[i]   = Math.floor(v * (0.4 + 0.6 * rBias));
        d[i+1] = Math.floor(v * (0.4 + 0.6 * gBias));
        d[i+2] = Math.floor(v * (0.4 + 0.6 * bBias));
        d[i+3] = Math.floor(alpha * 255);
      }
      ctx.putImageData(imgData, 0, 0);
      // Scanlines
      ctx.fillStyle = `rgba(0,0,0,0.28)`;
      for (let y = 0; y < H; y += 3*devicePixelRatio)
        ctx.fillRect(0, y, W, devicePixelRatio);
    });
  }

  // ── EFFECT: Particle Burst ────────────────────────────────────────────────
  function fxParticleBurst(durationMs = 2400) {
    // Init particles lazily from canvas centre (canvas size known after first frame)
    let particles = null;
    _fxShow(durationMs, (ctx, t, W, H) => {
      if (!particles) {
        const cx = W/2, cy = H/2;
        const NUM = 90;
        particles = Array.from({length: NUM}, (_, idx) => {
          const angle = Math.random() * Math.PI * 2;
          const speed = (1.5 + Math.random() * 7) * devicePixelRatio;
          const hOff  = (idx / NUM * 120) - 60;   // spread analogous hues
          return {
            x:cx, y:cy, vx:Math.cos(angle)*speed, vy:Math.sin(angle)*speed,
            r:(1.2 + Math.random()*2.8)*devicePixelRatio,
            hOff, trail:[],
          };
        });
      }
      const progress = t / durationMs;
      ctx.clearRect(0, 0, W, H);
      particles.forEach(p => {
        p.vy += 0.11 * devicePixelRatio;
        p.vx *= 0.992;
        p.x  += p.vx;
        p.y  += p.vy;
        p.trail.push({x:p.x, y:p.y});
        if (p.trail.length > 9) p.trail.shift();
        const a = Math.max(0, 1 - progress * 1.1);
        for (let i = 0; i < p.trail.length - 1; i++) {
          const ta = (i / p.trail.length) * a * 0.45;
          ctx.globalAlpha = ta;
          ctx.strokeStyle = _fxC(1, p.hOff);
          ctx.lineWidth   = p.r * 0.55;
          ctx.beginPath();
          ctx.moveTo(p.trail[i].x, p.trail[i].y);
          ctx.lineTo(p.trail[i+1].x, p.trail[i+1].y);
          ctx.stroke();
        }
        ctx.globalAlpha = a;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI*2);
        ctx.fillStyle = _fxCGlow(1, p.hOff);
        ctx.fill();
        ctx.globalAlpha = 1;
      });
      // Flash ring at start
      if (progress < 0.18) {
        const ring = progress / 0.18;
        ctx.beginPath();
        ctx.arc(W/2, H/2, ring * Math.min(W,H) * 0.22, 0, Math.PI*2);
        ctx.strokeStyle = _fxCGlow((1-ring)*0.9);
        ctx.lineWidth   = (1-ring) * 5 * devicePixelRatio;
        ctx.stroke();
      }
    });
  }

  // ── EFFECT: Scanline Warp ─────────────────────────────────────────────────
  function fxScanlineWarp(durationMs = 3200) {
    _fxShow(durationMs, (ctx, t, W, H) => {
      const progress = t / durationMs;
      const alpha    = Math.sin(progress * Math.PI) * 0.75;
      ctx.clearRect(0, 0, W, H);
      const spacing = 5 * devicePixelRatio;
      const waveAmp = 18 * devicePixelRatio * Math.sin(progress * Math.PI);
      for (let y = 0; y < H; y += spacing) {
        const warp = Math.sin(y * 0.012 + t * 0.0028) * waveAmp;
        ctx.fillStyle = _fxC(alpha * 0.35, 0, 80, 55);
        ctx.fillRect(warp, y, W, devicePixelRatio);
      }
      // Chromatic fringe lines — split-complement offset
      for (let i = 0; i < 4; i++) {
        const bx = (Math.sin(t * 0.0018 + i * 1.9) * 0.45 + 0.5) * W;
        ctx.fillStyle = _fxC(0.07*alpha, -60);
        ctx.fillRect(bx,               0, 2*devicePixelRatio, H);
        ctx.fillStyle = _fxC(0.07*alpha,  60);
        ctx.fillRect(bx+4*devicePixelRatio, 0, 2*devicePixelRatio, H);
      }
    });
  }

  // ── EFFECT: Data Corruption ───────────────────────────────────────────────
  function fxDataCorruption(durationMs = 3000) {
    const CC = '▓▒░█▄▀▌▐╔╗╚╝║═☠✦◈⬡⬢◉⊛⊕⌬⌭⌯#@%&?!~^*';
    _fxShow(durationMs, (ctx, t, W, H) => {
      const progress  = t / durationMs;
      const intensity = Math.sin(progress * Math.PI);
      ctx.clearRect(0, 0, W, H);
      const fs   = Math.max(10, 13) * devicePixelRatio;
      ctx.font   = `${fs}px monospace`;
      const cols = Math.floor(W / fs);
      const rows = Math.floor(H / fs);
      const num  = Math.floor(intensity * cols * rows * 0.18);
      for (let i = 0; i < num; i++) {
        const col  = Math.floor(Math.random() * cols);
        const row  = Math.floor(Math.random() * rows);
        const ch   = CC[Math.floor(Math.random() * CC.length)];
        const hOff = Math.floor(Math.random() * 80) - 40;
        ctx.fillStyle = _fxC(0.5 + Math.random()*0.5, hOff);
        ctx.fillText(ch, col*fs, row*fs + fs);
      }
      if (Math.random() < 0.08 * intensity) {
        ctx.fillStyle = _fxC(0.1 * intensity);
        ctx.fillRect(Math.random()*W*0.7, Math.random()*H*0.7,
          (20+Math.random()*160)*devicePixelRatio, (8+Math.random()*35)*devicePixelRatio);
      }
    });
  }

  // ── EFFECT: Heartbeat ─────────────────────────────────────────────────────
  function fxHeartbeat(durationMs = 3800) {
    _fxShow(durationMs, (ctx, t, W, H) => {
      const progress = t / durationMs;
      const alpha    = Math.sin(progress * Math.PI);
      ctx.clearRect(0, 0, W, H);
      // Grid
      ctx.strokeStyle = _fxCDim(0.22 * alpha);
      ctx.lineWidth   = devicePixelRatio;
      const gs = 38 * devicePixelRatio;
      for (let x = 0; x < W; x += gs) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }
      for (let y = 0; y < H; y += gs) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }
      // EKG sweep
      const midY    = H / 2;
      const headX   = (t * 0.0004 * W) % (W * 1.3);
      ctx.beginPath();
      ctx.strokeStyle = _fxCGlow(alpha);
      ctx.lineWidth   = 2.5 * devicePixelRatio;
      ctx.shadowColor = _fxC(0.8);
      ctx.shadowBlur  = 14;
      let started = false;
      for (let x = 0; x < W; x += devicePixelRatio) {
        const age  = headX / W - x / W;
        if (age < 0 || age > 0.55) continue;
        const fade = Math.max(0, 1 - age / 0.55);
        const phase = (x / W * 10 + t * 0.0015) % 1;
        let dy = 0;
        if      (phase < 0.05)  dy = -H*0.07*(phase/0.05);
        else if (phase < 0.10)  dy = -H*0.07 + H*0.28*((phase-0.05)/0.05);
        else if (phase < 0.15)  dy =  H*0.21 - H*0.25*((phase-0.10)/0.05);
        else if (phase < 0.20)  dy = -H*0.04 + H*0.04*((phase-0.15)/0.05);
        if (!started) { ctx.moveTo(x, midY + dy*fade); started = true; }
        else           ctx.lineTo(x, midY + dy*fade);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;
      // Scan head flash
      if (headX > 0 && headX < W) {
        const pa = Math.max(0, Math.sin((t*0.003)%1 * Math.PI*8)) * alpha * 0.35;
        ctx.fillStyle = _fxC(pa);
        ctx.fillRect(headX - 2*devicePixelRatio, 0, 4*devicePixelRatio, H);
      }
    });
  }

  // ── EFFECT: Hypno Spiral ──────────────────────────────────────────────────
  function fxHypnoSpiral(durationMs = 4800) {
    _fxShow(durationMs, (ctx, t, W, H) => {
      const progress = t / durationMs;
      const alpha    = Math.sin(progress * Math.PI) * 0.88;
      ctx.clearRect(0, 0, W, H);
      const cx   = W/2, cy = H/2;
      const maxR = Math.min(W, H) * 0.52;
      const rot  = t * 0.0014;
      const RINGS = 16;
      for (let ri = 0; ri < RINGS; ri++) {
        const frac   = ri / RINGS;
        const radius = frac * maxR;
        // Spin hue offset across rings — stays anchored to _uiHue
        const hOff = frac * 240 - 120;
        const pulse = 0.5 + 0.5 * Math.sin(frac * Math.PI * 5 + rot * 2.5);
        ctx.beginPath();
        ctx.arc(cx, cy, radius, rot + frac*0.6, rot + frac*0.6 + Math.PI*(1.55 + pulse*0.45));
        ctx.strokeStyle = _fxC(alpha * (0.28 + pulse*0.55), hOff, 100, 60 + pulse*20);
        ctx.lineWidth   = (2.5 + pulse*5) * devicePixelRatio;
        ctx.stroke();
      }
      // Centre glow
      const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, 55*devicePixelRatio);
      grd.addColorStop(0, _fxCGlow(alpha * 0.8));
      grd.addColorStop(1, _fxC(0));
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.arc(cx, cy, 55*devicePixelRatio, 0, Math.PI*2);
      ctx.fill();
    });
  }

  // ── EFFECT: Heart Pulse ───────────────────────────────────────────────────
  // One big heart centred in the avatar frame, beating in and out with a glow.
  function fxHeartPulse(durationMs = 3800) {
    // Draws a heart path centred at (cx, cy) with half-width r
    function _heartPath(ctx, cx, cy, r) {
      ctx.beginPath();
      ctx.moveTo(cx, cy + r * 0.35);
      ctx.bezierCurveTo(cx - r * 2,  cy - r * 0.6,  cx - r * 2,  cy - r * 1.6,  cx,      cy - r * 0.5);
      ctx.bezierCurveTo(cx + r * 2,  cy - r * 1.6,  cx + r * 2,  cy - r * 0.6,  cx,      cy + r * 0.35);
      ctx.closePath();
    }

    _fxShow(durationMs, (ctx, t, W, H) => {
      const progress = t / durationMs;
      // Overall envelope — fade in, hold, fade out
      const env = Math.sin(progress * Math.PI);
      ctx.clearRect(0, 0, W, H);

      const cx = W / 2, cy = H / 2;
      // Beat: two quick thumps per second with a rest between
      const beat = (t * 0.002) % 1;   // 0→1 over 500ms
      let scale;
      if      (beat < 0.12) scale = 1 + Math.sin(beat / 0.12 * Math.PI) * 0.22;   // thump 1
      else if (beat < 0.28) scale = 1 + Math.sin((beat - 0.17) / 0.11 * Math.PI) * 0.14; // thump 2
      else                  scale = 1;   // rest

      const baseR = Math.min(W, H) * 0.28 * scale;

      // Glow layers — several semi-transparent hearts expanding outward
      for (let layer = 3; layer >= 0; layer--) {
        const lr    = baseR * (1 + layer * 0.18);
        const la    = env * (0.18 - layer * 0.04);
        _heartPath(ctx, cx, cy, lr);
        ctx.fillStyle = _fxC(la, -10, 100, 70);
        ctx.fill();
      }

      // Solid fill heart
      _heartPath(ctx, cx, cy, baseR);
      const grad = ctx.createRadialGradient(cx, cy - baseR * 0.1, baseR * 0.05, cx, cy, baseR * 1.6);
      grad.addColorStop(0, _fxCGlow(env * 0.95, -5));
      grad.addColorStop(0.6, _fxC(env * 0.85, 0));
      grad.addColorStop(1,   _fxCDim(env * 0.5, 10));
      ctx.fillStyle = grad;
      ctx.fill();

      // Bright rim
      _heartPath(ctx, cx, cy, baseR);
      ctx.strokeStyle = _fxCGlow(env * 0.9);
      ctx.lineWidth   = 2.5 * devicePixelRatio;
      ctx.shadowColor = _fxC(0.7);
      ctx.shadowBlur  = 18 * scale;
      ctx.stroke();
      ctx.shadowBlur  = 0;
    });
  }

  // ── EFFECT: Heart Scatter ─────────────────────────────────────────────────
  // Several hearts of different sizes drifting / floating around the frame.
  function fxHeartScatter(durationMs = 4500) {
    // Each heart: position, size, phase offset, drift velocity, wobble
    const NUM = 7;
    let hearts = null;

    function _heartPath(ctx, cx, cy, r) {
      ctx.beginPath();
      ctx.moveTo(cx, cy + r * 0.35);
      ctx.bezierCurveTo(cx - r*2, cy - r*0.6,  cx - r*2, cy - r*1.6, cx,      cy - r*0.5);
      ctx.bezierCurveTo(cx + r*2, cy - r*1.6,  cx + r*2, cy - r*0.6, cx,      cy + r*0.35);
      ctx.closePath();
    }

    _fxShow(durationMs, (ctx, t, W, H) => {
      // Init hearts lazily so we have real canvas dimensions
      if (!hearts) {
        hearts = Array.from({length: NUM}, (_, i) => ({
          // Spread across the frame avoiding dead centre (that's for heart_pulse)
          x:    (0.12 + Math.random() * 0.76) * W,
          y:    (0.1  + Math.random() * 0.8)  * H,
          r:    (0.04 + Math.random() * 0.1)  * Math.min(W, H),
          phase: (i / NUM) * Math.PI * 2 + Math.random() * 0.8,  // stagger beats
          vx:   (Math.random() - 0.5) * 0.25 * devicePixelRatio,
          vy:   -(0.15 + Math.random() * 0.3) * devicePixelRatio, // float upward
          hOff: Math.floor(Math.random() * 40) - 20,
          wobble: 0.3 + Math.random() * 0.5,
        }));
      }

      const progress = t / durationMs;
      const env = Math.sin(progress * Math.PI);
      ctx.clearRect(0, 0, W, H);

      hearts.forEach(h => {
        // Drift
        h.x += h.vx;
        h.y += h.vy;

        // Beat pulse on individual phase
        const beat = (t * 0.0018 + h.phase) % (Math.PI * 2);
        const pulse = 1 + Math.sin(beat) * 0.15 * h.wobble;
        const r = h.r * pulse;

        // Per-heart fade tied to envelope, with slight individual variation
        const a = env * (0.7 + h.wobble * 0.3);

        // Soft glow
        _heartPath(ctx, h.x, h.y, r * 1.35);
        ctx.fillStyle = _fxC(a * 0.18, h.hOff, 100, 72);
        ctx.fill();

        // Filled heart
        _heartPath(ctx, h.x, h.y, r);
        const g = ctx.createRadialGradient(h.x, h.y - r*0.1, r*0.05, h.x, h.y, r*1.5);
        g.addColorStop(0,   _fxCGlow(a * 0.95, h.hOff - 5));
        g.addColorStop(0.5, _fxC(a * 0.85, h.hOff));
        g.addColorStop(1,   _fxCDim(a * 0.5, h.hOff + 10));
        ctx.fillStyle = g;
        ctx.fill();

        // Rim
        _heartPath(ctx, h.x, h.y, r);
        ctx.strokeStyle  = _fxCGlow(a * 0.8, h.hOff);
        ctx.lineWidth    = 1.5 * devicePixelRatio;
        ctx.shadowColor  = _fxC(0.5, h.hOff);
        ctx.shadowBlur   = 10;
        ctx.stroke();
        ctx.shadowBlur   = 0;
      });
    });
  }

  // ── Avatar layer distortion ───────────────────────────────────────────────
  // Applies temporary CSS filter/transform to the avatar image element,
  // synced to the currently running canvas effect.
  let _avFxDistortTimer = null;
  function _avDistort(mode, durationMs) {
    const img = document.getElementById('avatar-img');
    if (!img) return;
    clearTimeout(_avFxDistortTimer);
    // Filter only — never touch transform, avatar pan/zoom owns that
    img.style.transition = 'filter 0.2s ease';
    if (mode === 'rgb') {
      img.style.filter = 'hue-rotate(30deg) saturate(2.2) brightness(1.15)';
    } else if (mode === 'invert') {
      img.style.filter = 'invert(0.85) hue-rotate(180deg) brightness(1.2)';
    } else if (mode === 'mono') {
      img.style.filter = 'grayscale(1) brightness(0.7) contrast(1.4)';
    } else if (mode === 'bloom') {
      img.style.filter = 'brightness(1.4) saturate(1.8) blur(0.6px)';
    } else if (mode === 'glitch') {
      img.style.filter = 'hue-rotate(90deg) saturate(3) contrast(1.5)';
    } else if (mode === 'void') {
      img.style.filter = 'brightness(0.2) saturate(0.3) contrast(2)';
    } else if (mode === 'heat') {
      img.style.filter = 'sepia(0.6) saturate(2.5) hue-rotate(-20deg) brightness(1.1)';
    }
    // Restore filter only
    _avFxDistortTimer = setTimeout(() => {
      img.style.transition = 'filter 0.9s ease';
      img.style.filter = '';
    }, durationMs * 0.72);
  }

  // ── EFFECT: VHS Rewind ────────────────────────────────────────────────────
  function fxVhsRewind(durationMs = 2800) {
    _avDistort('rgb', durationMs);
    _fxShow(durationMs, (ctx, t, W, H) => {
      const progress = t / durationMs;
      const intensity = Math.sin(progress * Math.PI);
      ctx.clearRect(0, 0, W, H);
      // Horizontal scan tears
      const numTears = Math.floor(3 + intensity * 14);
      for (let i = 0; i < numTears; i++) {
        const y  = Math.random() * H;
        const bh = (1 + Math.random() * 8) * devicePixelRatio;
        const dx = (Math.random() < 0.5 ? -1 : 1) * (8 + Math.random() * 60) * devicePixelRatio * intensity;
        ctx.fillStyle = _fxC(0.25 + Math.random() * 0.4, Math.random() > 0.5 ? -40 : 40);
        ctx.fillRect(dx, y, W, bh);
      }
      // RGB separation lines
      const numRgb = Math.floor(intensity * 6);
      for (let i = 0; i < numRgb; i++) {
        const gy = Math.random() * H;
        const gh = (3 + Math.random() * 18) * devicePixelRatio;
        ctx.fillStyle = _fxC(0.22 * intensity, -50);
        ctx.fillRect(-8 * devicePixelRatio, gy, W, gh);
        ctx.fillStyle = _fxC(0.22 * intensity, 50);
        ctx.fillRect(8 * devicePixelRatio, gy, W, gh);
      }
      // Speed lines top-to-bottom
      if (intensity > 0.3) {
        ctx.fillStyle = `rgba(255,255,255,${0.04 * intensity})`;
        for (let y = 0; y < H; y += 2 * devicePixelRatio)
          if (Math.random() < 0.15) ctx.fillRect(0, y, W, devicePixelRatio);
      }
      // Tracking noise band sweeping downward
      const bandY = (t * 0.00045 * H) % (H * 1.2);
      const bH2 = 28 * devicePixelRatio;
      ctx.fillStyle = _fxCGlow(0.18 * intensity);
      ctx.fillRect(0, bandY, W, bH2);
    });
  }

  // ── EFFECT: Neural Fire ───────────────────────────────────────────────────
  function fxNeuralFire(durationMs = 4000) {
    let nodes = null;
    _fxShow(durationMs, (ctx, t, W, H) => {
      if (!nodes) {
        nodes = Array.from({length: 18}, () => ({
          x: Math.random() * W, y: Math.random() * H,
          vx: (Math.random()-0.5)*0.4*devicePixelRatio,
          vy: (Math.random()-0.5)*0.4*devicePixelRatio,
          r: (2+Math.random()*3)*devicePixelRatio,
          hOff: Math.floor(Math.random()*80)-40,
        }));
      }
      const progress = t / durationMs;
      const env = Math.sin(progress * Math.PI);
      ctx.clearRect(0, 0, W, H);
      // Move nodes
      nodes.forEach(n => {
        n.x = (n.x + n.vx + W) % W;
        n.y = (n.y + n.vy + H) % H;
      });
      // Draw arcs between nearby nodes
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i+1; j < nodes.length; j++) {
          const dx = nodes[j].x - nodes[i].x;
          const dy = nodes[j].y - nodes[i].y;
          const dist = Math.sqrt(dx*dx + dy*dy);
          const thresh = W * 0.38;
          if (dist > thresh) continue;
          const strength = (1 - dist/thresh) * env;
          if (Math.random() > 0.35) continue; // arc flickers
          const mx = (nodes[i].x + nodes[j].x)/2 + (Math.random()-0.5)*dist*0.6;
          const my = (nodes[i].y + nodes[j].y)/2 + (Math.random()-0.5)*dist*0.6;
          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.quadraticCurveTo(mx, my, nodes[j].x, nodes[j].y);
          ctx.strokeStyle = _fxCGlow(strength * 0.7, nodes[i].hOff);
          ctx.lineWidth = (0.5 + strength*2.5) * devicePixelRatio;
          ctx.shadowColor = _fxC(0.6, nodes[i].hOff);
          ctx.shadowBlur = 8;
          ctx.stroke();
          ctx.shadowBlur = 0;
        }
      }
      // Draw nodes
      nodes.forEach(n => {
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI*2);
        ctx.fillStyle = _fxCGlow(env * 0.9, n.hOff);
        ctx.fill();
      });
    });
  }

  // ── EFFECT: Pixel Melt ────────────────────────────────────────────────────
  function fxPixelMelt(durationMs = 3600) {
    _avDistort('glitch', durationMs);
    let cols = null;
    _fxShow(durationMs, (ctx, t, W, H) => {
      const progress = t / durationMs;
      const melt = Math.sin(progress * Math.PI);
      const colW = Math.max(6, Math.floor(W / 48));
      if (!cols) {
        const n = Math.ceil(W / colW);
        cols = Array.from({length: n}, () => ({
          offset: (0.1 + Math.random() * 0.6) * H,
          speed:  (0.5 + Math.random() * 1.5) * devicePixelRatio,
          hOff:   Math.floor(Math.random()*80)-40,
          len:    (0.1 + Math.random()*0.5) * H,
        }));
      }
      ctx.clearRect(0, 0, W, H);
      cols.forEach((col, i) => {
        col.offset = Math.min(col.offset + col.speed * melt * 2, H * 1.1);
        const a = melt * 0.6;
        const grad = ctx.createLinearGradient(0, col.offset - col.len, 0, col.offset);
        grad.addColorStop(0, _fxC(0, col.hOff));
        grad.addColorStop(0.3, _fxC(a * 0.5, col.hOff));
        grad.addColorStop(1, _fxCGlow(a, col.hOff));
        ctx.fillStyle = grad;
        ctx.fillRect(i * colW, col.offset - col.len, colW - 1, col.len);
      });
    });
  }

  // ── EFFECT: Void Pulse ────────────────────────────────────────────────────
  function fxVoidPulse(durationMs = 4200) {
    _avDistort('void', durationMs);
    _fxShow(durationMs, (ctx, t, W, H) => {
      const progress = t / durationMs;
      const env = Math.sin(progress * Math.PI);
      ctx.clearRect(0, 0, W, H);
      const cx = W/2, cy = H/2;
      const numRings = 5;
      for (let ri = 0; ri < numRings; ri++) {
        const phase = ((t * 0.0006 + ri/numRings) % 1);
        const r = phase * Math.max(W, H) * 0.75;
        const a = env * (1 - phase) * 0.55;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI*2);
        ctx.strokeStyle = _fxC(a, ri*15, 60, 35);
        ctx.lineWidth = (2 + (1-phase)*4) * devicePixelRatio;
        ctx.stroke();
      }
      // Brief black pulse between rings
      const blackAlpha = Math.pow(Math.sin(t * 0.008), 8) * env * 0.85;
      if (blackAlpha > 0.01) {
        ctx.fillStyle = `rgba(0,0,0,${blackAlpha})`;
        ctx.fillRect(0, 0, W, H);
      }
    });
  }

  // ── EFFECT: Static Burst ──────────────────────────────────────────────────
  function fxStaticBurst(durationMs = 900) {
    _avDistort('mono', durationMs);
    _fxShow(durationMs, (ctx, t, W, H) => {
      const progress = t / durationMs;
      // Sharp burst — strong at start, snap off
      const alpha = Math.pow(1 - progress, 1.4) * 0.9;
      const imgData = ctx.createImageData(W, H);
      const d = imgData.data;
      for (let i = 0; i < d.length; i += 4) {
        const v = Math.random() > 0.38 ? Math.floor(Math.random()*255) : 0;
        d[i] = d[i+1] = d[i+2] = v;
        d[i+3] = Math.floor(alpha * 255);
      }
      ctx.putImageData(imgData, 0, 0);
    });
  }

  // ── EFFECT: Cascade ───────────────────────────────────────────────────────
  function fxCascade(durationMs = 5000) {
    const CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789▓▒░█│┃┆┇┊┋╎╏║∥⫴⫿';
    let cols = null, fontSize = 0, lastW = 0;
    _fxShow(durationMs, (ctx, t, W, H) => {
      const progress = t / durationMs;
      const env = Math.sin(progress * Math.PI);
      if (W !== lastW) {
        fontSize = Math.max(12, Math.floor(W / devicePixelRatio / 22)) * devicePixelRatio;
        const n = Math.floor(W / fontSize);
        cols = Array.from({length: n}, () => ({
          y: Math.random() * -H,
          speed: (0.3 + Math.random() * 0.9) * devicePixelRatio,
          hOff: Math.floor(Math.random()*40)-20,
          lit: Math.floor(Math.random()*6)+2,
        }));
        lastW = W;
      }
      // Slow fade trail
      ctx.fillStyle = `rgba(0,0,0,${0.06 * env + (1-env)*0.25})`;
      ctx.fillRect(0, 0, W, H);
      ctx.font = `${fontSize}px monospace`;
      cols.forEach((col, i) => {
        col.y += col.speed * (0.4 + env*0.6);
        if (col.y > H + fontSize * col.lit) col.y = -fontSize * (2+Math.random()*4);
        for (let li = 0; li < col.lit; li++) {
          const cy2 = col.y - li * fontSize;
          if (cy2 < -fontSize || cy2 > H + fontSize) continue;
          const a = env * (li === 0 ? 0.95 : (1 - li/col.lit) * 0.6);
          ctx.fillStyle = li === 0 ? _fxCGlow(a, col.hOff) : _fxC(a, col.hOff, 80, 50);
          ctx.fillText(CHARS[Math.floor(Math.random()*CHARS.length)], i*fontSize, cy2);
        }
      });
    });
  }

  // ── EFFECT: Chromatic Bloom ───────────────────────────────────────────────
  function fxChromaticBloom(durationMs = 3000) {
    _avDistort('bloom', durationMs);
    _fxShow(durationMs, (ctx, t, W, H) => {
      const progress = t / durationMs;
      const env = Math.sin(progress * Math.PI);
      ctx.clearRect(0, 0, W, H);
      const cx = W/2, cy = H/2;
      // Expanding bloom rings radiating from centre
      const numRings = 7;
      for (let ri = 0; ri < numRings; ri++) {
        const phase = ((t * 0.0004 + ri * 0.14) % 1);
        const r = 20*devicePixelRatio + phase * Math.min(W,H) * 0.6;
        const a = env * (1 - phase) * 0.6;
        const hOff = (ri * 25) - 60;
        // Three colour rings offset slightly (chromatic aberration)
        [-6, 0, 6].forEach((off, ci) => {
          ctx.beginPath();
          ctx.arc(cx + off*devicePixelRatio, cy, r, 0, Math.PI*2);
          ctx.strokeStyle = _fxC(a * 0.7, hOff + ci*20);
          ctx.lineWidth = (1.5 + (1-phase)*3) * devicePixelRatio;
          ctx.stroke();
        });
      }
      // Central glow
      const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.min(W,H)*0.3);
      grd.addColorStop(0, _fxCGlow(env * 0.7));
      grd.addColorStop(1, _fxC(0));
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.arc(cx, cy, Math.min(W,H)*0.3, 0, Math.PI*2);
      ctx.fill();
    });
  }

  // ── EFFECT: Screen Crack ──────────────────────────────────────────────────
  function fxScreenCrack(durationMs = 3800) {
    let cracks = null;
    function buildCrack(sx, sy, angle, depth, maxDepth) {
      if (depth > maxDepth) return [];
      const len = (30 + Math.random() * 80) * devicePixelRatio;
      const ex = sx + Math.cos(angle) * len;
      const ey = sy + Math.sin(angle) * len;
      const segs = [{x1:sx, y1:sy, x2:ex, y2:ey, depth}];
      const branches = depth < 2 ? 2 : 1;
      for (let b = 0; b < branches; b++) {
        const da = (Math.random() - 0.5) * 1.1 + (b === 1 ? 0.5 : -0.5);
        segs.push(...buildCrack(ex, ey, angle + da, depth+1, maxDepth));
      }
      return segs;
    }
    _fxShow(durationMs, (ctx, t, W, H) => {
      if (!cracks) {
        const ox = (0.3 + Math.random()*0.4)*W;
        const oy = (0.2 + Math.random()*0.4)*H;
        const numArms = 5 + Math.floor(Math.random()*4);
        cracks = [];
        for (let a = 0; a < numArms; a++) {
          const angle = (a/numArms)*Math.PI*2 + (Math.random()-0.5)*0.5;
          cracks.push(...buildCrack(ox, oy, angle, 0, 3));
        }
      }
      const progress = t / durationMs;
      const reveal = Math.min(1, progress * 2.5);
      const fade   = progress > 0.65 ? 1 - (progress-0.65)/0.35 : 1;
      ctx.clearRect(0, 0, W, H);
      const visCount = Math.floor(reveal * cracks.length);
      cracks.slice(0, visCount).forEach(seg => {
        const a = fade * (0.9 - seg.depth * 0.18);
        ctx.beginPath();
        ctx.moveTo(seg.x1, seg.y1);
        ctx.lineTo(seg.x2, seg.y2);
        ctx.strokeStyle = _fxCGlow(a, seg.depth * 15);
        ctx.lineWidth = Math.max(0.5, (3 - seg.depth * 0.7)) * devicePixelRatio;
        ctx.shadowColor = _fxC(0.5);
        ctx.shadowBlur = 6;
        ctx.stroke();
        ctx.shadowBlur = 0;
      });
    });
  }

  // ── EFFECT: EKG Flatline ──────────────────────────────────────────────────
  function fxEkgFlatline(durationMs = 3500) {
    _avDistort('mono', durationMs * 0.5);
    _fxShow(durationMs, (ctx, t, W, H) => {
      const progress = t / durationMs;
      const env = Math.sin(progress * Math.PI);
      ctx.clearRect(0, 0, W, H);
      // Grid
      ctx.strokeStyle = _fxCDim(0.15 * env);
      ctx.lineWidth = devicePixelRatio;
      const gs = 32 * devicePixelRatio;
      for (let x = 0; x < W; x+=gs){ ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }
      for (let y = 0; y < H; y+=gs){ ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }
      const midY = H * 0.5;
      const headX = (t * 0.00038 * W) % (W * 1.2);
      // Flatline — just a straight line behind the head
      ctx.beginPath();
      ctx.strokeStyle = _fxC(env * 0.4);
      ctx.lineWidth = 1.5 * devicePixelRatio;
      ctx.moveTo(0, midY);
      ctx.lineTo(Math.max(0, headX - W * 0.08), midY);
      ctx.stroke();
      // Spike at ~30% of the effect
      const spikeAt = 0.3;
      const spikeW = 0.1;
      const inSpike = progress > spikeAt && progress < spikeAt + spikeW;
      if (inSpike) {
        const sp = (progress - spikeAt) / spikeW;
        const spikeAmp = Math.sin(sp * Math.PI) * H * 0.45;
        ctx.beginPath();
        ctx.strokeStyle = _fxCGlow(env);
        ctx.lineWidth = 3 * devicePixelRatio;
        ctx.shadowColor = _fxC(0.8);
        ctx.shadowBlur = 20;
        ctx.moveTo(headX - 20*devicePixelRatio, midY);
        ctx.lineTo(headX - 8*devicePixelRatio, midY - spikeAmp * 0.3);
        ctx.lineTo(headX, midY - spikeAmp);
        ctx.lineTo(headX + 8*devicePixelRatio, midY + spikeAmp * 0.5);
        ctx.lineTo(headX + 18*devicePixelRatio, midY);
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
      // Head flash
      if (headX > 0 && headX < W) {
        ctx.fillStyle = _fxC(env * (inSpike ? 0.5 : 0.2));
        ctx.fillRect(headX - 1.5*devicePixelRatio, 0, 3*devicePixelRatio, H);
      }
    });
  }

  // ── EFFECT: Binary Rain ───────────────────────────────────────────────────
  function fxBinaryRain(durationMs = 4500) {
    let drops = [], fontSize = 0, lastW = 0;
    _fxShow(durationMs, (ctx, t, W, H) => {
      const progress = t / durationMs;
      const fadeAlpha = progress > 0.75 ? 1-(progress-0.75)/0.25 : 1;
      if (W !== lastW) {
        fontSize = Math.max(11, Math.floor(W/devicePixelRatio/32)) * devicePixelRatio;
        drops = Array.from({length: Math.floor(W/fontSize)}, () => Math.random()*-50);
        lastW = W;
      }
      ctx.fillStyle = `rgba(0,0,0,${0.1*fadeAlpha + (1-fadeAlpha)*0.3})`;
      ctx.fillRect(0, 0, W, H);
      ctx.font = `bold ${fontSize}px monospace`;
      drops.forEach((y, i) => {
        const ch = Math.random() > 0.5 ? '1' : '0';
        ctx.fillStyle = _fxCGlow(0.95 * fadeAlpha);
        ctx.fillText(ch, i*fontSize, y*fontSize);
        ctx.fillStyle = _fxC(0.55 * fadeAlpha, 0, 70, 45);
        if (y>1) ctx.fillText(Math.random()>0.5?'1':'0', i*fontSize, (y-1)*fontSize);
        ctx.fillStyle = _fxCDim(0.3 * fadeAlpha);
        if (y>3) ctx.fillText(Math.random()>0.5?'1':'0', i*fontSize, (y-3)*fontSize);
        if (Math.random()>0.975 || drops[i]*fontSize>H) drops[i]=Math.random()*-30;
        drops[i] += 0.6;
      });
    });
  }

  // ── EFFECT: Warp Drive ────────────────────────────────────────────────────
  function fxWarpDrive(durationMs = 3200) {
    let stars = null;
    _fxShow(durationMs, (ctx, t, W, H) => {
      const progress = t / durationMs;
      const speed = Math.sin(progress * Math.PI);
      if (!stars) {
        stars = Array.from({length: 120}, () => ({
          angle: Math.random() * Math.PI * 2,
          dist: Math.random() * Math.min(W,H) * 0.45,
          hOff: Math.floor(Math.random()*60)-30,
          size: (0.5 + Math.random()*1.5)*devicePixelRatio,
        }));
      }
      ctx.clearRect(0, 0, W, H);
      const cx = W/2, cy = H/2;
      stars.forEach(s => {
        s.dist += speed * 4 * devicePixelRatio;
        if (s.dist > Math.max(W,H)*0.7) { s.dist = 2*devicePixelRatio; s.angle = Math.random()*Math.PI*2; }
        const x = cx + Math.cos(s.angle) * s.dist;
        const y = cy + Math.sin(s.angle) * s.dist;
        // Trail length proportional to speed and distance
        const trailLen = speed * s.dist * 0.18;
        const x2 = cx + Math.cos(s.angle) * (s.dist - trailLen);
        const y2 = cy + Math.sin(s.angle) * (s.dist - trailLen);
        const a = Math.min(1, s.dist / (Math.min(W,H)*0.2)) * speed;
        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(x, y);
        ctx.strokeStyle = _fxCGlow(a, s.hOff);
        ctx.lineWidth = s.size;
        ctx.stroke();
      });
    });
  }

  // ── EFFECT: Acid Wash ─────────────────────────────────────────────────────
  function fxAcidWash(durationMs = 4000) {
    _avDistort('heat', durationMs);
    _fxShow(durationMs, (ctx, t, W, H) => {
      const progress = t / durationMs;
      const env = Math.sin(progress * Math.PI);
      ctx.clearRect(0, 0, W, H);
      // Flowing noise bands using sine layering
      for (let y = 0; y < H; y += 2*devicePixelRatio) {
        const wave1 = Math.sin(y * 0.018 + t * 0.0022) * 0.5 + 0.5;
        const wave2 = Math.sin(y * 0.009 - t * 0.0018 + 1.2) * 0.5 + 0.5;
        const wave3 = Math.sin(y * 0.034 + t * 0.0031 + 2.4) * 0.5 + 0.5;
        const hOff = (wave1 * 60 + wave2 * 40 - 50);
        const sat  = 60 + wave3 * 40;
        const lig  = 30 + wave2 * 30;
        ctx.fillStyle = `hsla(${(_uiHue + hOff + 360)%360},${sat}%,${lig}%,${env * 0.45})`;
        ctx.fillRect(0, y, W, 2*devicePixelRatio);
      }
    });
  }

  // ── EFFECT: Ghost Signal ──────────────────────────────────────────────────
  function fxGhostSignal(durationMs = 5000) {
    _avDistort('invert', durationMs);
    _fxShow(durationMs, (ctx, t, W, H) => {
      const progress = t / durationMs;
      const env = Math.sin(progress * Math.PI);
      ctx.clearRect(0, 0, W, H);
      // Multiple ghost copies of a sine wave at different phases
      const midY = H * 0.5;
      for (let g = 0; g < 5; g++) {
        const phaseOff = g * 0.4 + t * 0.0009;
        const amp = H * 0.18 * (1 - g*0.15);
        const a = env * (0.6 - g*0.1);
        ctx.beginPath();
        ctx.strokeStyle = _fxC(a, g*18-36);
        ctx.lineWidth = (2 - g*0.3) * devicePixelRatio;
        for (let x = 0; x < W; x += devicePixelRatio) {
          const y = midY + Math.sin(x*0.012 + phaseOff) * amp
                        + Math.sin(x*0.025 - phaseOff*1.3) * amp * 0.4;
          x === 0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
        }
        ctx.stroke();
      }
      // Flickering vertical noise
      if (Math.random() < 0.3 * env) {
        ctx.fillStyle = _fxCGlow(Math.random()*0.15*env);
        ctx.fillRect(Math.random()*W, 0, (1+Math.random()*3)*devicePixelRatio, H);
      }
    });
  }

  // ── EFFECT: Memory Leak ───────────────────────────────────────────────────
  function fxMemoryLeak(durationMs = 4800) {
    const CHARS = '0123456789ABCDEFabcdef';
    let blocks = null;
    _fxShow(durationMs, (ctx, t, W, H) => {
      if (!blocks) {
        blocks = Array.from({length: 40}, () => ({
          x: Math.random()*W, y: Math.random()*H,
          w: (20+Math.random()*120)*devicePixelRatio,
          h: (8+Math.random()*24)*devicePixelRatio,
          hOff: Math.floor(Math.random()*80)-40,
          phase: Math.random()*Math.PI*2,
          speed: 0.002 + Math.random()*0.004,
        }));
      }
      const progress = t / durationMs;
      const env = Math.sin(progress * Math.PI);
      ctx.clearRect(0, 0, W, H);
      const fs = Math.max(8, 10)*devicePixelRatio;
      ctx.font = `${fs}px monospace`;
      blocks.forEach(b => {
        const flicker = 0.3 + 0.7*(0.5 + 0.5*Math.sin(t*b.speed + b.phase));
        const a = env * flicker;
        if (a < 0.05) return;
        // Block background
        ctx.fillStyle = _fxCDim(a * 0.25, b.hOff);
        ctx.fillRect(b.x, b.y, b.w, b.h);
        // Hex content
        ctx.fillStyle = _fxC(a * 0.85, b.hOff);
        const chars = Math.floor(b.w / fs);
        let str = '';
        for (let c = 0; c < chars; c++) str += CHARS[Math.floor(Math.random()*CHARS.length)];
        ctx.fillText(str, b.x + 2*devicePixelRatio, b.y + b.h - 3*devicePixelRatio);
      });
    });
  }

  // ── EFFECT: Hologram ──────────────────────────────────────────────────────
  function fxHologram(durationMs = 5500) {
    _avDistort('rgb', durationMs);
    _fxShow(durationMs, (ctx, t, W, H) => {
      const progress = t / durationMs;
      const env = Math.sin(progress * Math.PI);
      ctx.clearRect(0, 0, W, H);
      // Horizontal scan bars sweeping upward
      const barH = 3 * devicePixelRatio;
      const scrollY = (t * 0.00012 * H) % H;
      for (let y = 0; y < H + barH; y += barH * 2) {
        const ty = (y - scrollY + H) % H;
        ctx.fillStyle = _fxC(env * 0.08, 0, 60, 70);
        ctx.fillRect(0, ty, W, barH);
      }
      // Vertical interference fringe
      for (let x = 0; x < W; x += 4*devicePixelRatio) {
        const wave = Math.sin(x*0.008 + t*0.002) * 0.5 + 0.5;
        if (wave < 0.6) continue;
        ctx.fillStyle = _fxC(env * 0.06 * wave, 20);
        ctx.fillRect(x, 0, devicePixelRatio, H);
      }
      // Corner brackets — holographic frame
      const bLen = Math.min(W,H)*0.12;
      const bW = 2*devicePixelRatio;
      const pad = 12*devicePixelRatio;
      const corners = [[pad,pad,1,1],[W-pad,pad,-1,1],[pad,H-pad,1,-1],[W-pad,H-pad,-1,-1]];
      corners.forEach(([cx,cy,sx,sy]) => {
        ctx.strokeStyle = _fxCGlow(env*0.9);
        ctx.lineWidth = bW;
        ctx.shadowColor = _fxC(0.7);
        ctx.shadowBlur = 10;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + sx*bLen, cy); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx, cy + sy*bLen); ctx.stroke();
        ctx.shadowBlur = 0;
      });
      // Flicker
      if (Math.random() < 0.04) {
        ctx.fillStyle = `rgba(0,0,0,${Math.random()*0.6})`;
        ctx.fillRect(0, 0, W, H);
      }
    });
  }

  // ── EFFECT: Shockwave ─────────────────────────────────────────────────────
  function fxShockwave(durationMs = 2200) {
    _avDistort('bloom', durationMs);
    _fxShow(durationMs, (ctx, t, W, H) => {
      const progress = t / durationMs;
      ctx.clearRect(0, 0, W, H);
      const cx = W/2, cy = H/2;
      // 3 rings at staggered phases
      for (let ri = 0; ri < 3; ri++) {
        const p = Math.min(1, progress * 1.8 - ri * 0.18);
        if (p <= 0) continue;
        const r = p * Math.max(W,H) * 0.65;
        const a = (1-p) * (1 - ri*0.28) * 0.8;
        const thickness = (1-p) * 18 * devicePixelRatio + 1;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI*2);
        ctx.strokeStyle = _fxCGlow(a, ri*20);
        ctx.lineWidth = thickness;
        ctx.shadowColor = _fxC(0.6, ri*20);
        ctx.shadowBlur = 20;
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
      // Central flash
      if (progress < 0.15) {
        const fa = (1 - progress/0.15) * 0.7;
        const grd = ctx.createRadialGradient(cx,cy,0,cx,cy,Math.min(W,H)*0.3);
        grd.addColorStop(0, _fxCGlow(fa));
        grd.addColorStop(1, _fxC(0));
        ctx.fillStyle = grd;
        ctx.beginPath();
        ctx.arc(cx, cy, Math.min(W,H)*0.3, 0, Math.PI*2);
        ctx.fill();
      }
    });
  }

  // ── EFFECT: Morse Code ────────────────────────────────────────────────────
  function fxMorse(durationMs = 4000) {
    // Random sequence of dots and dashes as flashing light bars
    const pattern = Array.from({length: 24}, () => Math.random() > 0.4);
    let pulseIdx = 0, lastPulseT = 0;
    const pulseMs = durationMs / pattern.length;
    _fxShow(durationMs, (ctx, t, W, H) => {
      pulseIdx = Math.floor(t / pulseMs);
      const on = pattern[Math.min(pulseIdx, pattern.length-1)];
      const progress = t / durationMs;
      const env = Math.sin(progress * Math.PI);
      ctx.clearRect(0, 0, W, H);
      if (on) {
        // Flash bar across the top and bottom
        const barH2 = (6 + Math.random()*4) * devicePixelRatio;
        ctx.fillStyle = _fxCGlow(env * 0.9);
        ctx.shadowColor = _fxC(0.7);
        ctx.shadowBlur = 18;
        ctx.fillRect(0, 0, W, barH2);
        ctx.fillRect(0, H-barH2, W, barH2);
        ctx.shadowBlur = 0;
        // Centre crosshair flash
        ctx.fillStyle = _fxCGlow(env * 0.4);
        ctx.fillRect(W*0.1, H/2 - devicePixelRatio, W*0.8, 2*devicePixelRatio);
        ctx.fillRect(W/2 - devicePixelRatio, H*0.1, 2*devicePixelRatio, H*0.8);
      }
    });
  }

  // ── EFFECT: Thermal Vision ────────────────────────────────────────────────
  function fxThermalVision(durationMs = 4500) {
    _avDistort('heat', durationMs);
    _fxShow(durationMs, (ctx, t, W, H) => {
      const progress = t / durationMs;
      const env = Math.sin(progress * Math.PI);
      ctx.clearRect(0, 0, W, H);
      // Thermal colour gradient wash using per-pixel noise
      const imgData = ctx.createImageData(W, H);
      const d = imgData.data;
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          // Simulate heat concentration near centre
          const dx = (x/W - 0.5), dy = (y/H - 0.5);
          const dist = Math.sqrt(dx*dx + dy*dy);
          const heat = Math.max(0, 1 - dist*1.6) + Math.random()*0.15;
          // Map heat to thermal palette: cold=blue, warm=green, hot=red/white
          let r,g,b;
          if (heat < 0.33)      { r=0;         g=heat*3*200;   b=255; }
          else if (heat < 0.66) { r=heat*2*200; g=200;         b=200*(1-heat*2); }
          else                  { r=255;        g=255*(1-heat); b=0; }
          const i4 = (y*W + x)*4;
          d[i4]   = r;  d[i4+1] = g;  d[i4+2] = b;
          d[i4+3] = Math.floor(env * 0.55 * 255);
        }
      }
      ctx.putImageData(imgData, 0, 0);
      // Scan crosshair overlay
      ctx.strokeStyle = `rgba(255,255,255,${env*0.4})`;
      ctx.lineWidth = devicePixelRatio;
      ctx.beginPath(); ctx.moveTo(W/2,0); ctx.lineTo(W/2,H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0,H/2); ctx.lineTo(W,H/2); ctx.stroke();
      ctx.beginPath(); ctx.arc(W/2, H/2, Math.min(W,H)*0.15, 0, Math.PI*2); ctx.stroke();
    });
  }

  // ── EFFECT: Digital Rain (coloured) ───────────────────────────────────────
  function fxDigitalRainColor(durationMs = 5000) {
    const CHARS = '░▒▓█▄▀■□▪▫◆◇○●◎⊕⊗⊙⊚0123456789';
    let drops = [], fontSize = 0, lastW = 0;
    _fxShow(durationMs, (ctx, t, W, H) => {
      const progress = t / durationMs;
      const fadeAlpha = progress > 0.78 ? 1-(progress-0.78)/0.22 : 1;
      if (W !== lastW) {
        fontSize = Math.max(10, Math.floor(W/devicePixelRatio/26)) * devicePixelRatio;
        drops = Array.from({length: Math.floor(W/fontSize)}, (_, i) => ({
          y: Math.random()*-60, hOff: (i*13)%120-60, speed: 0.4+Math.random()*0.8,
        }));
        lastW = W;
      }
      ctx.fillStyle = `rgba(0,0,0,${0.07*fadeAlpha+(1-fadeAlpha)*0.2})`;
      ctx.fillRect(0,0,W,H);
      ctx.font = `bold ${fontSize}px monospace`;
      drops.forEach(d2 => {
        const ch = CHARS[Math.floor(Math.random()*CHARS.length)];
        ctx.fillStyle = _fxCGlow(0.9*fadeAlpha, d2.hOff);
        ctx.fillText(ch, drops.indexOf(d2)*fontSize, d2.y*fontSize);
        ctx.fillStyle = _fxC(0.6*fadeAlpha, d2.hOff, 80, 45);
        if(d2.y>2) ctx.fillText(CHARS[Math.floor(Math.random()*CHARS.length)], drops.indexOf(d2)*fontSize, (d2.y-2)*fontSize);
        if(Math.random()>0.97 || d2.y*fontSize>H) d2.y=Math.random()*-30;
        d2.y += d2.speed;
      });
    });
  }

  // ── Public dispatch ───────────────────────────────────────────────────────
  const _fxDispatch = {
    matrix_rain:        (ms) => fxMatrixRain(ms||undefined),
    glitch_storm:       (ms) => fxGlitchStorm(ms||undefined),
    signal_static:      (ms) => fxSignalStatic(ms||undefined),
    particle_burst:     (ms) => fxParticleBurst(ms||undefined),
    scanline_warp:      (ms) => fxScanlineWarp(ms||undefined),
    data_corruption:    (ms) => fxDataCorruption(ms||undefined),
    heartbeat:          (ms) => fxHeartbeat(ms||undefined),
    hypno_spiral:       (ms) => fxHypnoSpiral(ms||undefined),
    heart_pulse:        (ms) => fxHeartPulse(ms||undefined),
    heart_scatter:      (ms) => fxHeartScatter(ms||undefined),
    vhs_rewind:         (ms) => fxVhsRewind(ms||undefined),
    neural_fire:        (ms) => fxNeuralFire(ms||undefined),
    pixel_melt:         (ms) => fxPixelMelt(ms||undefined),
    void_pulse:         (ms) => fxVoidPulse(ms||undefined),
    static_burst:       (ms) => fxStaticBurst(ms||undefined),
    cascade:            (ms) => fxCascade(ms||undefined),
    chromatic_bloom:    (ms) => fxChromaticBloom(ms||undefined),
    screen_crack:       (ms) => fxScreenCrack(ms||undefined),
    ekg_flatline:       (ms) => fxEkgFlatline(ms||undefined),
    binary_rain:        (ms) => fxBinaryRain(ms||undefined),
    warp_drive:         (ms) => fxWarpDrive(ms||undefined),
    acid_wash:          (ms) => fxAcidWash(ms||undefined),
    ghost_signal:       (ms) => fxGhostSignal(ms||undefined),
    memory_leak:        (ms) => fxMemoryLeak(ms||undefined),
    hologram:           (ms) => fxHologram(ms||undefined),
    shockwave:          (ms) => fxShockwave(ms||undefined),
    morse:              (ms) => fxMorse(ms||undefined),
    thermal_vision:     (ms) => fxThermalVision(ms||undefined),
    digital_rain_color: (ms) => fxDigitalRainColor(ms||undefined),
  };

  window.triggerFX = function(effectName, durationMs) {
    if (!_avOverlayOpen) return;  // avatar must be open
    if (!_visualFxEnabled) return; // visual FX master switch
    const fn = _fxDispatch[effectName];
    if (!fn) { console.warn('[FX] Unknown effect:', effectName); return; }
    console.log('[FX] →', effectName);
    fn(durationMs || 0);
  };

  window.stopFX = _fxStop;
})();

// ══════════════════════════════════════════════════════════════════════════════
// ── Avatar Subtitle Engine ────────────────────────────────────────────────────
// Renders spoken text as typewriter-style subtitles inside the avatar viewport,
// timed to the actual audio playback position via audioCtx.currentTime.
//
// Timing model:
//   • Total audio duration estimated from char count at _SUB_CHARS_PER_SEC
//   • Text split into word tokens, each assigned a proportional start time
//   • A rAF loop checks audioCtx.currentTime against _ttsPlayStartTime and
//     reveals words whose cue time has passed — typewriter, not dump-all-at-once
//   • On stopAudio / barge-in the subtitle clears immediately
// ══════════════════════════════════════════════════════════════════════════════

let _subEnabled  = false;
let _subRafId    = null;
let _subCues     = [];       // [{word, startSec}]
let _subShownIdx = -1;
let _subGen      = 0;

// Subtitle speed — chars/sec. Higher = faster reveals. Saved to session.
// Range 4 (very slow) → 20 (very fast). Default 11 matches average TTS pace.
let _SUB_CHARS_PER_SEC = 11;
const _SUB_LINGER_SEC    = 2.5;
const _SUB_LINE_WORDS    = 5;   // words per subtitle line before rolling

function _setSubSpeed(val) {
  _SUB_CHARS_PER_SEC = Math.max(4, Math.min(30, Number(val)));
  // Sync all speed sliders
  for (const id of ['sub-speed-slider', 'av-sub-speed-slider']) {
    const el = document.getElementById(id);
    if (el) el.value = _SUB_CHARS_PER_SEC;
  }
  for (const id of ['sub-speed-val', 'av-sub-speed-val']) {
    const el = document.getElementById(id);
    if (el) el.textContent = _SUB_CHARS_PER_SEC;
  }
  fetch('/settings', { method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ sub_speed: _SUB_CHARS_PER_SEC }) }).catch(() => {});
}

function toggleAvatarCC() {
  _subEnabled = !_subEnabled;
  const btn = document.getElementById('avatar-cc-btn');
  if (btn) {
    btn.style.color       = _subEnabled ? 'var(--green)' : '';
    btn.style.borderColor = _subEnabled ? 'var(--green)' : '';
    btn.style.background  = _subEnabled ? 'var(--tint-dark)' : '';
  }
  if (!_subEnabled) _subClear(true);
}

function _subStart(text, gen) {
  if (!_subEnabled || !_avOverlayOpen) return;
  _subStop();

  let clean = text
    .replace(/```[\s\S]*?```/g, '')
    .replace(/`[^`]+`/g, '')
    .replace(/\x1b\[[0-9;]*m/g, '')
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/\*(.*?)\*/g, '$1')
    .trim();

  if (!clean) return;

  const words = clean.split(/\s+/).filter(Boolean);
  if (!words.length) return;

  const totalSec   = Math.max(1.5, clean.length / _SUB_CHARS_PER_SEC);
  const totalChars = words.reduce((s, w) => s + w.length, 0);
  let elapsed = 0;
  _subCues = words.map(word => {
    const frac = elapsed / Math.max(totalChars, 1);
    elapsed += word.length;
    return { word, startSec: frac * totalSec };
  });

  _subShownIdx = -1;
  _subGen      = gen;

  const sub = document.getElementById('avatar-subtitle');
  if (!sub) return;
  sub.innerHTML = '';
  sub.classList.add('visible');

  _subAdvanceLoop(gen);
}

function _subAdvanceLoop(gen) {
  if (_subGen !== gen) return;
  if (!audioCtx || !_subEnabled || !_avOverlayOpen) { _subClear(false); return; }

  const elapsed = audioCtx.currentTime - _ttsPlayStartTime;

  // Reveal words up to current time
  let newlyShown = false;
  for (let i = _subShownIdx + 1; i < _subCues.length; i++) {
    if (elapsed >= _subCues[i].startSec) {
      _subShownIdx = i;
      newlyShown = true;
    } else break;
  }

  if (newlyShown) _subRender();

  if (_subShownIdx >= _subCues.length - 1) {
    _subRafId = null;
    setTimeout(() => { if (_subGen === gen) _subClear(true); }, _SUB_LINGER_SEC * 1000);
    return;
  }

  _subRafId = requestAnimationFrame(() => _subAdvanceLoop(gen));
}

function _subRender() {
  const sub = document.getElementById('avatar-subtitle');
  if (!sub) return;

  const shown = _subCues.slice(0, _subShownIdx + 1).map(c => c.word);

  // Split shown words into lines of _SUB_LINE_WORDS each
  const lines = [];
  for (let i = 0; i < shown.length; i += _SUB_LINE_WORDS)
    lines.push(shown.slice(i, i + _SUB_LINE_WORDS));

  // Keep only the last 1 line
  const visible = lines.slice(-1);
  const isPartial = _subShownIdx < _subCues.length - 1;

  sub.innerHTML = '';

  visible.forEach((lineWords, li) => {
    const isLast = li === visible.length - 1;
    const div = document.createElement('div');

    if (isLast && isPartial) {
      // Current line — typewriter each word in
      div.className = 'sub-line current';
      // All words on previous lines are fully shown; only animate words
      // in this line that haven't been "committed" yet
      const lineIsComplete = lines.length > 1 && li < visible.length - 1;
      lineWords.forEach((w, wi) => {
        const span = document.createElement('span');
        span.className = 'sub-word shown'; // all shown — line rolled into view
        span.textContent = (wi === 0 ? '' : ' ') + w;
        div.appendChild(span);
      });
      // If this line is still being filled, the last word just appeared
    } else if (isLast) {
      // Final line, all words revealed
      div.className = 'sub-line current';
      lineWords.forEach((w, wi) => {
        const span = document.createElement('span');
        span.className = 'sub-word shown';
        span.textContent = (wi === 0 ? '' : ' ') + w;
        div.appendChild(span);
      });
    } else {
      // Previous line — dim, static
      div.className = 'sub-line prev';
      div.textContent = lineWords.join(' ');
    }

    sub.appendChild(div);
  });
}

function _subStop() {
  _subGen++;
  if (_subRafId) { cancelAnimationFrame(_subRafId); _subRafId = null; }
}

function _subClear(fade) {
  _subGen++;
  if (_subRafId) { cancelAnimationFrame(_subRafId); _subRafId = null; }
  const sub = document.getElementById('avatar-subtitle');
  if (!sub) return;
  if (fade) {
    sub.classList.remove('visible');
    setTimeout(() => { if (sub) sub.innerHTML = ''; }, 400);
  } else {
    sub.classList.remove('visible');
    sub.innerHTML = '';
  }
}

// ── Wave display toggles ───────────────────────────────────────────────────────
let _mainWaveVisible = true;
let _avatarWaveVisible = true;
let _visualFxEnabled = false;
let _moodFxEnabled   = false;

function toggleVisualFX() {
  _visualFxEnabled = !_visualFxEnabled;
  const btn = document.getElementById('s-vis-fx-btn');
  if (btn) { btn.textContent = _visualFxEnabled ? 'ON' : 'OFF'; btn.className = 'btn' + (_visualFxEnabled ? ' on' : ''); }
  fetch('/settings', { method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ visual_fx_enabled: _visualFxEnabled }) }).catch(() => {});
}

function toggleMoodFX() {
  _moodFxEnabled = !_moodFxEnabled;
  const btn = document.getElementById('s-mood-fx-btn');
  if (btn) { btn.textContent = _moodFxEnabled ? 'ON' : 'OFF'; btn.className = 'btn' + (_moodFxEnabled ? ' on' : ''); }
  fetch('/settings', { method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ mood_fx_enabled: _moodFxEnabled }) }).catch(() => {});
}

function toggleSpeakingIndicator() {
  const el = document.getElementById('playing-indicator');
  if (!el) return;
  el.classList.toggle('perm-hidden');
  _saveWaveState();
}

function toggleWaveDisplay() {
  _mainWaveVisible = !_mainWaveVisible;
  const ww  = document.getElementById('wave-wrap');
  const btn = document.getElementById('wave-toggle-btn');
  const mb  = document.getElementById('main-wave-en');
  if (ww) ww.classList.toggle('wave-hidden', !_mainWaveVisible);
  if (btn) btn.textContent = _mainWaveVisible ? '▼ WAVE' : '▶ WAVE';
  if (mb)  { mb.textContent = _mainWaveVisible ? 'ON' : 'OFF'; mb.className = 'btn' + (_mainWaveVisible ? ' on' : ''); }
  _saveWaveState();
}

function toggleAvatarWave() {
  _avatarWaveVisible = !_avatarWaveVisible;
  const ww  = document.getElementById('avatar-wave-wrap');
  const btn = document.getElementById('av-wave-en');
  if (ww) ww.classList.toggle('wave-hidden', !_avatarWaveVisible);
  if (btn) { btn.textContent = _avatarWaveVisible ? 'ON' : 'OFF'; btn.className = 'btn' + (_avatarWaveVisible ? ' on' : ''); }
  _saveWaveState();
}

function _saveWaveState() {
  fetch('/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      wave_mode:          waveModes[waveMode],
      main_wave_visible:  _mainWaveVisible,
      avatar_wave_visible: _avatarWaveVisible,
    })
  }).catch(() => {});
}

// ── Service worker + notifications ────────────────────────────────────────────
// ── Initiative toggle ─────────────────────────────────────────────────────────
let _initiativeEnabled = false;
let _initiativeMode = 'light';

function toggleInitiative() {
  _initiativeEnabled = !_initiativeEnabled;
  _applyInitiativeUI();
  fetch('/initiative/set', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled: _initiativeEnabled, mode: _initiativeMode }),
  });
}

function _applyInitiativeUI() {
  const btn = document.getElementById('s-init-btn');
  const ind = document.getElementById('init-indicator');
  const avInd = document.getElementById('av-init-indicator');
  if (btn) { btn.textContent = _initiativeEnabled ? 'ON' : 'OFF'; btn.className = 'btn' + (_initiativeEnabled ? ' on' : ''); }
  // Show indicator when active; give it .on glow; update text
  if (ind) {
    ind.style.display = '';
    ind.className = 'btn' + (_initiativeEnabled ? ' on' : '');
    ind.textContent = '◈ INIT';
  }
  if (avInd) {
    avInd.style.display = '';
    avInd.className = 'btn' + (_initiativeEnabled ? ' on' : '');
    avInd.textContent = '◈ INIT';
  }
  document.getElementById('s-init-mode').value = _initiativeMode;
}

document.getElementById('s-init-mode').addEventListener('change', function() {
  _initiativeMode = this.value;
  if (_initiativeEnabled) {
    fetch('/initiative/set', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: true, mode: _initiativeMode }),
    });
  }
});

// ── FX chance ─────────────────────────────────────────────────────────────────
function _saveFxChance() {
  const val = parseInt(document.getElementById('s-fx-chance').value) || 0;
  fetch('/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ initiative_fx_chance: val }) }).catch(() => {});
}

function _testFxNow() {
  fetch('/fx/trigger', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ effect: 'random' }) }).catch(() => {});
}

// ── Sleep timer ───────────────────────────────────────────────────────────────
let _sleepTimerEnabled = false;

function _toggleSleepTimer() {
  _sleepTimerEnabled = !_sleepTimerEnabled;
  const btn = document.getElementById('s-sleep-timer-btn');
  if (btn) { btn.textContent = _sleepTimerEnabled ? 'ON' : 'OFF'; btn.className = 'btn' + (_sleepTimerEnabled ? ' on' : ''); }
  _saveSleepTimer();
}

function _saveSleepTimer() {
  const start = parseInt(document.getElementById('s-sleep-start').value) || 0;
  const end   = parseInt(document.getElementById('s-sleep-end').value)   || 0;
  fetch('/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sleep_timer_enabled: _sleepTimerEnabled, sleep_start: start, sleep_end: end })
  }).catch(() => {});
}

function _applySleepTimerUI(enabled, inSleep) {
  _sleepTimerEnabled = !!enabled;
  const btn = document.getElementById('s-sleep-timer-btn');
  if (btn) { btn.textContent = _sleepTimerEnabled ? 'ON' : 'OFF'; btn.className = 'btn' + (_sleepTimerEnabled ? ' on' : ''); }
  const badge = document.getElementById('s-sleep-active-badge');
  if (badge) badge.style.display = (inSleep && _sleepTimerEnabled) ? '' : 'none';
}

// ── Init ──────────────────────────────────────────────────────────────────
// loadState must complete first so _loadedCharPath is set before the char
// dropdown is built — otherwise the session-restored character won't be selected.
(async function init(){
  await loadState();
  await loadCharacterList();
  loadRagFileList();
  pollSafetyStatus();
})();

// ── Guest / kiosk mode ────────────────────────────────────────────────────
(async function applyGuestMode(){
  try{
    const r=await fetch('/guest_config'); const g=await r.json();
    if(!g.guest_mode) return;
    // Hide settings gear button and settings panel
    document.querySelectorAll('[onclick="toggleSettings()"]').forEach(el=>el.style.display='none');
    // Hide safety indicators in wave area
    const safetyLight=document.getElementById('safety-light');
    const safetyScore=document.getElementById('safety-score-display');
    const safetyLevel=document.getElementById('safety-level-display');
    const safetyL1=document.getElementById('safety-l1-btn');
    const safetyL2=document.getElementById('safety-l2-btn');
    [safetyLight,safetyScore,safetyLevel,safetyL1,safetyL2].forEach(el=>{if(el)el.style.display='none';});
    // Also hide the safety section inside settings
    document.querySelectorAll('.setting-row').forEach(row=>{
      if(row.textContent.includes('SAFETY')||row.textContent.includes('L1 ON')||row.textContent.includes('L2 ON'))
        row.style.display='none';
    });
    // Keep settings panel permanently closed
    const sp=document.getElementById('settings-panel');
    if(sp){sp.style.display='none';sp.style.maxHeight='0';}
    // Update page title
    if(g.title){ document.title=g.title; const logo=document.querySelector('.logo'); if(logo)logo.textContent=g.title; }
    // Auto-load character
    if(g.character){
      const _gRes=await fetch('/characters/load',{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({path:g.character})});
      const _gData=await _gRes.json();
      if(_gData.ok){
        const _gChat=document.getElementById('chat');
        _gChat.innerHTML='';
        if(_gData.chat_history&&_gData.chat_history.length){
          _gData.chat_history.forEach(msg=>{
            const dispText=msg.user_image?msg.content.replace(/^\[image attached\]\s*/,''):msg.content;
            const b=addBubble(msg.role,dispText);
            if(msg.user_image){const ui=document.createElement('img');ui.src=msg.user_image;ui.className='bubble-img';ui.style.cursor='zoom-in';ui.onclick=()=>openLightbox(ui.src);b.insertBefore(ui,b.firstChild);}
            (msg.gen_images||[]).forEach(uri=>{
              const im=document.createElement('img');im.src=uri;
              im.className='bubble-img generated-img';
              im.style.cssText='max-width:100%;max-height:360px;margin-top:8px;cursor:zoom-in';
              im.onclick=()=>openLightbox(uri);b.appendChild(im);
            });
          });
        }
      }
      await loadState();
    }
  }catch(e){ console.warn('[guest]',e); }
})();

// ── Chat SSE sync — receive messages pushed from server ────────────────────
// Tracks the last message count we've seen so we skip replaying our own sends.
let _sseOwnCount=0; // incremented when we send, so we skip that push
let _sseSynced=false; // true once we've received the history burst
let _sseBootId=null; // tracks server boot ID — changes on server restart
let _sseSource=null; // current EventSource
let _sseCharPath=''; // char path this stream was opened for

function _startChatStream(charPath){
  if(_sseSource){ try{_sseSource.close();}catch(e){} _sseSource=null; }
  _sseSynced=false;
  _sseCharPath=charPath||'';
  const url='/chat/stream'+(charPath?'?char='+encodeURIComponent(charPath):'');
  const src=new EventSource(url);
  _sseSource=src;
  src.onmessage=async function(e){
    if(!e.data||e.data.startsWith(':')) return;
    try{
      const msg=JSON.parse(e.data);

      // ── Visual FX effect triggered by agent or initiative ──
      if(msg.type==='fx'){
        if(typeof window.triggerFX === 'function') window.triggerFX(msg.effect, msg.duration_ms||0);
        return;
      }

      // Boot ID — detect server restarts and reload state cleanly
      if(msg.type==='boot'){
        if(_sseBootId && _sseBootId !== msg.id){
          // Server restarted — reload state and re-sync chat
          console.log('[SSE] server restart detected, reloading state');
          _sseSynced=false;
          await loadState();
          await loadCharacterList();
        }
        _sseBootId=msg.id;
        return;
      }

      // History burst on connect — populate chat from server state
      if(msg.type==='history'){
        if(!_sseSynced){
          _sseSynced=true;
          const chat=document.getElementById('chat');
          const alreadyShown=chat.querySelectorAll('.bubble').length;
          // Only add messages we don't already have showing
          const toAdd=msg.messages.slice(alreadyShown);
          toAdd.forEach(m=>{
            const dispText=m.user_image?m.content.replace(/^\[image attached\]\s*/,''):m.content;
            const bubble=addBubble(m.role,dispText);
            if(m.user_image){const ui=document.createElement('img');ui.src=m.user_image;ui.className='bubble-img';ui.style.cursor='zoom-in';ui.onclick=()=>openLightbox(ui.src);bubble.insertBefore(ui,bubble.firstChild);}
            (m.gen_images||[]).forEach(uri=>{const im=document.createElement('img');im.src=uri;im.className='bubble-img generated-img';im.style.cssText='max-width:100%;max-height:360px;margin-top:8px;cursor:zoom-in';im.onclick=()=>openLightbox(uri);bubble.appendChild(im);});
          });
        }
        return;
      }

      const isInitiative = msg.type === 'initiative';

      if(!isInitiative){
        // Skip if this is the echo of a message we just sent from this tab
        if(_sseOwnCount>0){_sseOwnCount--;return;}
        // Dedup: compare against last bubble text without forcing layout on canvas frames
        // Use textContent (no layout) and compare trimmed to handle whitespace differences
        const chat=document.getElementById('chat');
        const bubbles=chat.querySelectorAll('.bubble');
        const last=bubbles.length?bubbles[bubbles.length-1]:null;
        const lastText=last?last.textContent.trim():'';
        if(lastText===msg.text.trim()) return; // already shown — don't retrigger canvas
      }

      if(msg.role==='user') addBubble('user',msg.text);
      else addBubble('assistant',msg.text);

      // Initiative message — reschedule if user is typing or busy, otherwise play
      if(isInitiative){
        if(_userIsTyping()||isBusy||isPlaying){
          fetch('/initiative/reschedule',{method:'POST'}).catch(()=>{});
          return;
        }
        isBusy=true;
        const gen=++_ttsGeneration;
        _currentReplyText=msg.text;
        _ttsPlayStartTime=audioCtx?audioCtx.currentTime+0.06:0;
        if(openMicState!=='off') setOpenMicState('playing');
        playTTS(_stripCodeForTTS(msg.text), gen).then(()=>{
          if(openMicState==='playing') setOpenMicState('listening');
        }).finally(()=>{ isBusy=false; });
      }
    }catch(err){}
  };
  src.addEventListener('heartbeat',function(){
    // keep-alive received — connection healthy
  });
  let _sseRetryDelay=3000;
  src.onerror=function(){
    _sseSynced=false; // reset so we re-sync on reconnect
    src.close();
    _sseSource=null;
    const delay=_sseRetryDelay;
    _sseRetryDelay=Math.min(_sseRetryDelay*1.5,30000); // backoff up to 30s
    console.warn('[SSE] disconnected, retrying in',delay,'ms');
    setTimeout(function(){ _sseRetryDelay=3000; _startChatStream(_sseCharPath); },delay);
  };
  src.onopen=function(){ _sseRetryDelay=3000; }; // reset backoff on successful open
}
// Initial connection — char path not yet known; loadState will reconnect with correct char
_startChatStream('');
</script>
</body>
</html>
"""
