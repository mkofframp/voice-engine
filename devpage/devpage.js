// Local developer test page. Holds no API key: sessions are created through /dev/api/session.
const $ = (id) => document.getElementById(id);
const STAGES = ["user_last_word", "end_of_turn", "final_transcript", "llm_first_token", "tts_first_audio", "audio_out", "total_ms"];

const [copy, defaults] = await Promise.all([
  fetch("/dev/copy.json").then((r) => r.json()),
  fetch("/dev/config.json").then((r) => r.json()),
]);
$("h-config").textContent = copy.config_editor;
$("h-log").textContent = copy.event_log;
$("h-metrics").textContent = copy.metrics;
$("connect").textContent = copy.connect;
$("disconnect").textContent = copy.disconnect;
$("config").value = JSON.stringify(defaults, null, 2);

let client = null;
const totals = [];

function log(label, payload) {
  const row = document.createElement("div");
  row.textContent = payload === undefined ? label : `${label} ${JSON.stringify(payload)}`;
  const box = $("log");
  box.appendChild(row);
  box.scrollTop = box.scrollHeight;
}

function quantile(sorted, q) {
  return sorted[Math.min(sorted.length - 1, Math.ceil(q * sorted.length) - 1)];
}

function renderMetrics(e) {
  if (typeof e.total_ms === "number") totals.push(e.total_ms);
  const lines = STAGES.map((s) => `${s}: ${e[s] === null || e[s] === undefined ? "-" : Math.round(e[s]) + " ms"}`);
  if (totals.length) {
    const sorted = [...totals].sort((a, b) => a - b);
    const mid = sorted.length >> 1;
    const median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    lines.push(`turns: ${sorted.length}`, `median_ms: ${Math.round(median)}`, `p90_ms: ${Math.round(quantile(sorted, 0.9))}`);
  }
  $("metrics").textContent = lines.join("\n");
}

function setConnected(on) {
  $("connect").disabled = on;
  $("disconnect").disabled = !on;
}

$("connect").onclick = async () => {
  if (client) return;
  $("connect").disabled = true;
  try {
    await connectSession();
  } finally {
    if (!client) setConnected(false);
  }
};

async function connectSession() {
  let config;
  try {
    config = JSON.parse($("config").value);
  } catch (err) {
    log("config", String(err));
    return;
  }
  const res = await fetch("/dev/api/session", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(config),
  });
  const body = await res.json();
  if (!res.ok) {
    log(String(res.status), body);
    return;
  }
  let mod;
  try {
    mod = await import("/dev/client.js");
  } catch (err) {
    log("client.js", String(err));
    return;
  }
  totals.length = 0;
  const c = new mod.VoiceEngineClient({
    engineUrl: location.origin,
    clientToken: body.client_token,
    iceServers: body.ice_servers,
  });
  client = c;
  c.onAny((e) => {
    log(e.type, e);
    if (e.type === "turn_metrics") renderMetrics(e);
  });
  try {
    await c.connect();
    $("status").textContent = body.session_id;
    setConnected(true);
  } catch (err) {
    log("connect", String(err));
    try { c.disconnect(); } catch {}
    client = null;
  }
}

$("disconnect").onclick = () => {
  if (client) client.disconnect();
  client = null;
  $("status").textContent = "";
  setConnected(false);
};

// ---- Text to read aloud: POST /dev/api/speak, stream raw PCM into Web Audio ----
$("h-speak").textContent = copy.speak_heading;
$("speak-text").placeholder = copy.speak_placeholder;
$("speak-rewrite-label").textContent = copy.speak_rewrite;
$("speak-go").textContent = copy.speak_button;
$("speak-stop").textContent = copy.speak_stop;

let speakAbort = null;
let speakCtx = null;

function stopSpeaking() {
  if (speakAbort) speakAbort.abort();
  speakAbort = null;
  if (speakCtx) speakCtx.close();
  speakCtx = null;
  $("speak-go").disabled = false;
  $("speak-stop").disabled = true;
}

$("speak-stop").onclick = stopSpeaking;

$("speak-go").onclick = async () => {
  const text = $("speak-text").value;
  if (!text.trim()) return;
  stopSpeaking();
  $("speak-go").disabled = true;
  $("speak-stop").disabled = false;
  $("speak-status").textContent = "";
  let cfg = {};
  try { cfg = JSON.parse($("config").value); } catch { /* use engine defaults */ }
  const abort = (speakAbort = new AbortController());
  const t0 = performance.now();
  try {
    const res = await fetch("/dev/api/speak", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text, rewrite: $("speak-rewrite").checked, config: cfg }),
      signal: abort.signal,
    });
    if (!res.ok) {
      log("speak error", await res.json().catch(() => ({ status: res.status })));
      return stopSpeaking();
    }
    const rate = Number(res.headers.get("x-sample-rate")) || 24000;
    const ctx = (speakCtx = new AudioContext({ sampleRate: rate }));
    let at = ctx.currentTime + 0.05;
    let carry = new Uint8Array(0);
    let first = true;
    const reader = res.body.getReader();
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      const bytes = new Uint8Array(carry.length + value.length);
      bytes.set(carry); bytes.set(value, carry.length);
      const usable = bytes.length - (bytes.length % 2);
      carry = bytes.slice(usable);
      if (!usable) continue;
      const pcm = new Int16Array(bytes.buffer, 0, usable / 2);
      const buf = ctx.createBuffer(1, pcm.length, rate);
      const ch = buf.getChannelData(0);
      for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
      const src = ctx.createBufferSource();
      src.buffer = buf; src.connect(ctx.destination);
      at = Math.max(at, ctx.currentTime);
      src.start(at); at += buf.duration;
      if (first) {
        first = false;
        $("speak-status").textContent = `${Math.round(performance.now() - t0)} ms`;
      }
    }
    // Re-enable the button once playback has finished.
    const left = Math.max(0, at - ctx.currentTime) * 1000;
    setTimeout(() => { if (speakCtx === ctx) stopSpeaking(); }, left);
  } catch (e) {
    if (e.name !== "AbortError") { log("speak error", { message: String(e) }); stopSpeaking(); }
  }
};
