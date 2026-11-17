# Voice Engine — Design

Date: 2026-11-17 · Status: approved by owner 2026-11-17 (including §7a defaults)
(Supersedes the earlier "Kid Voice Companion" draft; scope narrowed to the engine only.)

## 1. Intent and the why

The owner wants to reproduce what makes ChatGPT voice mode feel human — natural emotion, near
real-time turn-taking, being interruptible, and audibly listening — as a **standalone voice engine**
that a separate web app (with its own server) will plug in later. The first use is a companion for
children aged 5–8, but the engine itself is persona-agnostic; persona and safety rules are supplied
by the host app as configuration.

Reverse-engineering OpenAI's service directly is out of scope (unauthorized, and the human feel
lives in a server-side model). Instead we rebuild the experience from separate, controllable parts.

## 2. Goals

1. **Real-time conversation:** median time from the person's last word to first reply audio
   ≤ 1.0 s at default settings on a normal connection; p90 ≤ 1.5 s.
2. **Emotion:** every reply sentence carries an emotion tag that changes voice delivery, and the tag
   is emitted as an event for the host app.
3. **Interruption mid-sentence (barge-in):** see §6.4.
4. **Active listening (backchannels):** see §6.5.
5. **Embeddable:** a host web app with its own server can create sessions, configure them, and
   connect a browser, with no engine code in the host beyond a small JS client.
6. **Observable:** per-turn timings and events available to the host and to a dev test page.

## 3. Non-goals

- Any end-user UI: face/avatar, parent gate, tuning panel, session timer UI, accounts. These belong
  to the host app. The engine emits the events the host needs to build them.
- Detecting the speaker's emotion from tone of voice (possible later).
- Languages other than English. Mobile-native SDKs.
- Public hosting, auth hardening, scaling. v1 runs locally for the owner only.

## 4. Anti-goals

- Dead air, talking over the person, or not stopping when interrupted.
- Stopping its own speech because the person said "uh-huh".
- Backchannels that are too frequent, mistimed (landing mid-word), or trigger the bot to interrupt
  itself.
- Uniform, theatrical emotion (every sentence `[excited]`).
- Persisting audio or transcripts.

## 5. Constraints

- Owner's global rule: any fixed human-readable or human-heard text shipped *by the engine*
  (default redirect/goodbye lines, backchannel phrase list, test-page labels, error messages meant
  for people) is written by ChatGPT via the `chatgpt-copy` skill on `gpt-6-astra`; `TODO(copy)`
  placeholders until then. LLM replies generated at runtime and prompt configuration are not copy.
- API keys only on the engine server (`.env`), never in a browser.
- Must work with a host server in any language (host server language unknown) — integration is
  over HTTP + WebRTC, not by importing code.

## 6. The design

### 6.1 Approaches considered

| | Approach | Trade-off |
|---|---|---|
| A | OpenAI Realtime API (single speech-to-speech model) | Fastest, closest to ChatGPT; emotion only via prompt, opaque; little control over backchannels |
| **B (chosen)** | **Pipeline: streaming STT → fast LLM with emotion tags → expressive TTS, on Pipecat** | Explicit, inspectable control of emotion, interruptions and backchannels; ~0.8–1.2 s, several vendors |
| C | Self-hosted open speech-to-speech (Moshi, Sesame CSM) | Cheap per minute; GPU, lower quality, weak safety controls — dropped |

Integration shape (designer's call): **separate engine service + JS client**, because it works
with any host server language. Embedding the engine as a library inside the host server was
rejected since it only works if the host is Python.

### 6.2 Architecture

```
Host web app (browser)                Engine service (Python, Pipecat)
┌──────────────────────┐  WebRTC   ┌───────────────────────────────────────────────┐
│ engine JS client     │◀─audio──▶│ transport → VAD + end-of-turn model             │
│  (mic, playback,     │◀─events─▶│   → Deepgram streaming STT                      │
│   events, controls)  │          │   → turn manager (barge-in, backchannels)       │
└──────────────────────┘          │   → Claude Haiku 5.5 (streamed, emotion tags)   │
                                  │   → tag parser + output filter                  │
Host server ──HTTP (API key)────▶│   → Cartesia TTS (emotion-mapped)               │
  POST /sessions {config}         │ Session store (in memory)                       │
  ◀── {session_id, client_token}  └───────────────────────────────────────────────┘
```

- **Session flow:** host server calls `POST /sessions` with an engine API key and a session config;
  gets back `session_id` + short-lived `client_token`; passes the token to its browser; the JS
  client connects over WebRTC with it. Host server can `PATCH /sessions/{id}` (config changes apply
  from the next turn) and `DELETE /sessions/{id}`.
- **Transport:** Pipecat SmallWebRTC; browser echo cancellation on (required so backchannels and
  replies aren't heard as the person speaking).
- **STT:** Deepgram streaming with interim results.
- **LLM:** Claude Haiku 5.5 (`claude-haiku-5-5`), streamed; builder loads the `claude-api` skill
  before writing this integration. Context in memory per session.
- **TTS:** Cartesia lowest-latency model, streamed sentence by sentence; ElevenLabs adapter behind
  the same interface is optional.

### 6.3 Session config (supplied by host)

- `persona_prompt` (text), `safety_prompt` (text; locked block placed before persona; engine ships
  a kid-safe 5–8 default), `output_blocklist` (list; kid default shipped).
- `voice_id`, `speaking_speed`, `energy`, `emotion_intensity` (0–100), `response_length`.
- `turn_eagerness` (patient ↔ quick), `backchannel_frequency` (off / low / normal / high).
- Fixed lines: `redirect_line`, `goodbye_line` (defaults are `TODO(copy)`).
- `max_session_seconds` (engine enforces; host decides the value).

### 6.4 Interruption (barge-in)

- While the bot is speaking, VAD + interim STT watch for the person speaking.
- **Real interruption** = speech that lasts ≥ a threshold (~300 ms, tunable) *or* whose interim
  transcript is not a backchannel word (“mm”, “uh-huh”, “yeah”, “ok”, “right”, laugh). On real
  interruption: stop playback within ~200 ms of detection, cancel in-flight LLM and TTS, emit
  `interrupted`.
- **Context truncation:** the assistant message in history is cut to the words actually played
  (using TTS word timestamps / played-audio position), with a marker that it was cut off, so the
  next reply can acknowledge it naturally.
- **User backchannel** (short, backchannel word): bot keeps talking; nothing added to history as a
  turn (optionally noted in context).

### 6.5 Active listening (bot backchannels)

- While the person holds the turn, at pauses where VAD reports silence but the end-of-turn model
  says “not finished” (pause ~250–700 ms), the turn manager may play a short backchannel.
- **Bank:** per voice, pre-rendered at session start (or cached per `voice_id`): ~15–25 clips across
  moods — neutral acknowledgments (“mm-hm”, “yeah”), interested (“oh?”, “really?”), amused (soft
  laugh), sympathetic (“aww”). Phrase list is copy → `TODO(copy)` until written.
- **Choice:** mood picked from interim transcript via a cheap heuristic (keywords/punctuation of
  interim text); no LLM call in this path. Never the same clip twice in a row.
- **Rate limits:** at most one per ~4–6 s of person speech (scaled by `backchannel_frequency`),
  none in the first ~2 s of a turn, none if a reply is about to start.
- Backchannel audio is **not** treated as bot speech for barge-in and **never** enters history.
- Emits `backchannel {mood}` events so the host can animate (nod, smile).

### 6.6 Emotion system

- Tag set (config file): `neutral, happy, excited, giggle, curious, surprised, gentle, sad,
  whisper, proud`.
- LLM prefixes each sentence with one tag; prompt asks for natural variation, defaulting to
  `neutral`/`gentle`.
- Streaming tag parser yields `(emotion, sentence)`; each sentence goes to TTS when complete, so
  speech starts before the LLM finishes.
- Emotion mapper converts tag → Cartesia controls (emotion, speed, volume, laughter/nonverbal
  markup where supported), scaled by `emotion_intensity`; unsupported tags map to nearest and the
  fallback is logged. Builder verifies Cartesia's current control surface and documents the table.
- Output filter: each sentence checked against `output_blocklist`; a hit is replaced with
  `redirect_line`. Fixed lines are pre-rendered at session start.

### 6.7 Events to the host (over the WebRTC data channel, mirrored in JS client callbacks)

`state` (listening | thinking | speaking), `user_transcript` (interim/final), `bot_sentence
{emotion, text}` timed to audio start, `backchannel {mood}`, `interrupted {spoken_text}`,
`turn_metrics {timings}`, `safety_flag {category, excerpt}`, `session_ending`, `error`.
Host commands and webhook: see §7a.

### 6.8 Turn metrics

Per turn: last word → end-of-turn decision → final transcript → LLM first token → first TTS audio →
audio out. Kept in memory only; emitted as `turn_metrics`. Goal 1 is measured at default
`turn_eagerness`; more patient settings may exceed it. The LLM may start speculatively on the
interim transcript and be discarded if the person keeps talking.

### 6.9 Dev test page

A bare page served by the engine at `localhost` only: connect, talk, see the event stream and
metrics, edit config JSON. For the owner's testing; not a product UI.

### 6.10 Layout and running

- `/Users/michaelhegai/Documents/ReverseEng/kid-voice/` (rename to `voice-engine` at builder's
  discretion), new git repo. `engine/` (Python service), `client/` (JS client package, ESM,
  TypeScript types), `devpage/`.
- One command to run locally; `.env.example`: `DEEPGRAM_API_KEY`, `ANTHROPIC_API_KEY`,
  `CARTESIA_API_KEY`, `ENGINE_API_KEY`.
- Tests: unit tests for tag parser, emotion mapper, output filter, interruption classifier
  (backchannel vs real), backchannel scheduler (timing/rate limits), history truncation; an
  end-to-end script feeding recorded/synthetic audio that asserts reply, interruption stop time,
  and backchannel timing. No real child recordings collected.

## 7. Decided

- Owner: approach B; ages 5–8 for first use; local only, owner only; engine only, no product UI;
  interruption mid-sentence and active listening are required; host is a web app with its own
  server.
- Designer's calls (owner may override): separate service + JS client; Claude Haiku 5.5, Deepgram,
  Cartesia; safety as locked config block + output filter with kid defaults; latency target at
  default eagerness; fixed lines and backchannel phrases via `chatgpt-copy`.

## 7a. Calls made after second builder review (designer's calls, owner may override)

- **Worrying things the person says** (harm, self-harm, personal details like address/school):
  each final user transcript gets an async check (fast LLM classification, run in parallel with
  the reply so it adds no latency). A hit emits `safety_flag {category, excerpt}` to the browser
  and to the host webhook (below). The bot's reply is governed by the safety prompt (gentle,
  suggest talking to a grown-up); the engine does nothing else — what to do with a flag is the
  host's decision.
- **Opening and host commands:** config gets optional `greeting` (if set, the bot speaks first,
  otherwise it waits). Host can send, via JS client or `POST /sessions/{id}/commands`:
  `say {text}` (spoken as a bot turn, interruptible like any other), `add_context {text}`
  (appended to LLM context, not spoken), `end {graceful}` (graceful = say `goodbye_line`, then
  close), `mute_mic` / `unmute_mic`.
- **Safety prompt ownership:** the host may replace `safety_prompt` (the engine is
  persona-agnostic), but it can't be empty: if omitted, the kid-safe 5–8 default is used. It always
  comes before `persona_prompt`. `PATCH` may change it.
- **Backchannel audio:** generated with the session's TTS voice, no recorded clips in v1. Moods
  the TTS can't render convincingly (e.g. laugh) are dropped from the bank for that voice, not
  faked; the bank is cached per `voice_id`.
- **Host server events:** optional `webhook_url` in config; engine POSTs `session_ended {reason,
  metrics_summary}`, `error`, and `safety_flag`. `GET /sessions/{id}` returns status.
- **Definition of done:** automated thresholds (latency goal 1; interruption stop ≤ 300 ms from
  speech onset in the e2e test; zero backchannels in the first 2 s or closer than the rate limit;
  zero history entries from backchannels) **plus** an owner listening session against a short
  checklist: emotion varies and fits, backchannels land in pauses and feel natural, interruptions
  feel immediate and the bot acknowledges being cut off, "uh-huh" doesn't stop it.

## 8. Left to the builder

- Exact VAD/end-of-turn parameters and thresholds in §6.4–6.5.
- Deepgram and Cartesia model/voice choices; ElevenLabs adapter or not.
- Backchannel mood heuristic details; blocklist contents; default persona wording.
- HTTP API details beyond §6.2 (error shapes, token lifetime).
