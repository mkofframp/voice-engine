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
  const c = new mod.VoiceEngineClient({ engineUrl: location.origin, clientToken: body.client_token });
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
