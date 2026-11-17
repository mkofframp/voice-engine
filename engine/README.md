# voice-engine

Real-time voice conversation engine for children aged 5-8 (Python 3.12, Pipecat 1.16).
Speech in over WebRTC → Deepgram STT → Claude Haiku → Cartesia TTS (emotion-tagged) → speech
out, with turn-taking, interruption, backchannels and a safety check.

## Run

    cd engine
    cp .env.example .env      # then fill in the keys (below)
    uv run voice-engine       # serves the API on HOST:PORT (default 127.0.0.1:7860)

Dev test page (loopback only): build the browser bundle once with
`cd client && npm run build:browser`, then open `http://127.0.0.1:7860/dev`.

Tests: `uv run pytest -q` (no network, no keys needed).

## `.env`

Secrets are read only from `engine/.env` (never from the request or the browser).

| variable | used for |
|---|---|
| `DEEPGRAM_API_KEY` | speech-to-text |
| `ANTHROPIC_API_KEY` | replies and the safety classifier |
| `CARTESIA_API_KEY` | text-to-speech, backchannel/fixed-line audio, e2e fixtures |
| `ENGINE_API_KEY` | the bearer key your host server sends to this API |

Optional: `HOST` (default `127.0.0.1`), `PORT` (default `7860`), `CLIENT_TOKEN_TTL_S`
(default `120`), `ALLOWED_ORIGINS` (JSON list, default `["http://localhost:5173"]`),
`LOG_LEVEL` (`INFO`, `WARNING` or `ERROR`; default `INFO`).
`HOST`, `PORT`, `CLIENT_TOKEN_TTL_S`, `ALLOWED_ORIGINS` and `LOG_LEVEL` can go in `.env`. `DEBUG` is read
from the shell environment only, not from `.env`: run `DEBUG=1 uv run voice-engine` to add
`history_roles` to `GET /sessions/{id}` (for the e2e script only).

Nothing is persisted. Sessions live in memory and transcripts are not logged: `voice-engine`
replaces loguru's default DEBUG sink with one at `LOG_LEVEL` (never below `INFO`, since
Pipecat's DEBUG lines carry transcripts and replies), sets `PIPECAT_LOG_LLM_CONTEXT=off` and
drops Pipecat's few text-bearing lines at any level.

## HTTP API

Every host call sends `Authorization: Bearer $ENGINE_API_KEY`. Errors are JSON
`{"error": code, "message": text}` (plus `details` for `invalid_request`):
`unauthorized` (401), `token_invalid` (401), `session_not_found` (404),
`session_busy` (409), `session_not_connected` (409), `invalid_request` (422).

### `POST /sessions` → 201

Body: a session config (all fields optional). Returns a single-use `client_token` for the browser.

```sh
curl -s -X POST http://127.0.0.1:7860/sessions \
  -H "Authorization: Bearer $ENGINE_API_KEY" -H "Content-Type: application/json" \
  -d '{"persona_prompt": "You are Pip, a friendly robot.", "greeting": "Hi! I am Pip."}'
# {"session_id": "…", "client_token": "…", "expires_in": 120}
```

| field | type / range | default |
|---|---|---|
| `persona_prompt` | string | `""` |
| `safety_prompt` | string; blank → default | kid-safe 5-8 prompt (always placed before the persona) |
| `output_blocklist` | list of strings | built-in list |
| `voice_id` | Cartesia voice id | built-in voice |
| `speaking_speed` | 0.6-1.5 | 1.0 |
| `energy` | 0-1 | 0.5 |
| `emotion_intensity` | 0-100 | 70 |
| `response_length` | `short` \| `medium` \| `long` | `short` |
| `turn_eagerness` | `patient` \| `normal` \| `quick` | `normal` |
| `backchannel_frequency` | `off` \| `low` \| `normal` \| `high` | `normal` |
| `redirect_line` | string \| null | null |
| `goodbye_line` | string \| null | null |
| `greeting` | string \| null (set → the bot speaks first) | null |
| `max_session_seconds` | 30-3600 | 600 |
| `webhook_url` | URL \| null | null |

Unknown fields are rejected (422).

### `GET /sessions/{id}` → 200

```sh
curl -s http://127.0.0.1:7860/sessions/$SID -H "Authorization: Bearer $ENGINE_API_KEY"
# {"status": "created|connected|ended", "created_at": …, "config": {…}}
```

With `DEBUG=1` the response also has `history_roles`, a count of LLM-context messages per
role plus `backchannel_like`, the number of non-system messages whose whole text is a
backchannel phrase or word (for example `{"user": 3, "assistant": 3, "backchannel_like": 0}`).
The system prompt is passed as the LLM's `system_instruction`, so there is no `"system"`
entry; an `add_context` note counts as a `"user"` message. No message text is returned.

### `PATCH /sessions/{id}` → 200

Partial config; returns the new config. It is validated as a whole (422 on failure, nothing
changed) and applies from the next turn. `turn_eagerness` applies from the next session.

```sh
curl -s -X PATCH http://127.0.0.1:7860/sessions/$SID \
  -H "Authorization: Bearer $ENGINE_API_KEY" -H "Content-Type: application/json" \
  -d '{"emotion_intensity": 40}'
```

### `DELETE /sessions/{id}` → 204

Stops the conversation (reason `deleted`) and removes the session.

### `POST /sessions/{id}/commands` → 202

Needs a connected session (409 `session_not_connected` otherwise).

| command | effect |
|---|---|
| `{"type": "say", "text": "…"}` | spoken as a bot turn, interruptible |
| `{"type": "add_context", "text": "…"}` | appended to the LLM context, not spoken (as a user message marked "Note from the app, not the child", so it reaches the model even before the child's first turn) |
| `{"type": "end", "graceful": true}` | graceful: say `goodbye_line`, then close |
| `{"type": "mute_mic"}` / `{"type": "unmute_mic"}` | ignore / resume the child's audio |

```sh
curl -s -X POST http://127.0.0.1:7860/sessions/$SID/commands \
  -H "Authorization: Bearer $ENGINE_API_KEY" -H "Content-Type: application/json" \
  -d '{"type": "say", "text": "Time for one more question!"}'
```

The browser can send the same commands over the data channel (`client.sendCommand`).

### `POST /webrtc/offer` (browser)

Called by the JS client, not by the host. Body `{"sdp", "type": "offer", "client_token"}`;
no API key, the single-use token authorises it. A second connection is refused (409
`session_busy`). `ALLOWED_ORIGINS` is the CORS allow-list: browsers on other origins cannot
read the response, but the server does not reject a request that has no `Origin` header.

## Events (data channel → browser)

| type | payload |
|---|---|
| `state` | `state`: `listening` \| `thinking` \| `speaking` |
| `user_transcript` | `text`, `final` (false = interim) |
| `bot_sentence` | `emotion`, `text` (sent as the sentence starts playing) |
| `backchannel` | `mood` |
| `interrupted` | `spoken_text`: what the child actually heard |
| `turn_metrics` | ms from the child's last word: `end_of_turn`, `final_transcript`, `llm_first_token`, `tts_first_audio`, `audio_out`, `total_ms` (null if a stage did not happen) |
| `safety_flag` | `category`, `excerpt` |
| `session_ending` | `reason` |
| `error` | `code`, `message` |

### Webhook (engine → host server)

If `webhook_url` is set the engine POSTs JSON (2 retries, never blocks the conversation):
`{"type": "session_ended", "session_id", "reason", "metrics_summary": {"turns", "median_ms", "p90_ms"}}`,
`{"type": "safety_flag", "session_id", "category", "excerpt"}`,
`{"type": "error", "session_id", "code", "message"}`.
What to do with a safety flag is the host's decision.

## Turn-taking: `turn_eagerness`

End of turn is decided by Silero VAD plus Smart Turn v3. `turn_eagerness` sets two
silences (seconds), fixed when the session's pipeline is built:

| turn_eagerness | VAD stop_secs | Smart Turn stop_secs |
|---|---|---|
| patient | 0.5 | 4.0 |
| normal  | 0.3 | 3.0 |
| quick   | 0.2 | 2.0 |

VAD stop_secs is the pause before Smart Turn is asked whether the child has finished;
Smart Turn stop_secs is the pause after which the turn ends regardless. Children pause
mid-thought often, so `patient` waits longest. Source: `EAGERNESS` in
`src/voice_engine/pipeline/bot.py`. A patched `turn_eagerness` applies from the next session.

These values are not yet tuned against a live run (see "Status" below).

## Emotion map

The LLM tags each sentence; the tag picks Cartesia controls (live TTS uses `sonic-3`; `sonic-3.6` is used only for
pre-rendered audio). Values are at full
intensity; `emotion_intensity` scales them toward neutral (0 = neutral delivery, speed factor
and volume 1.0). `speed_factor` multiplies `speaking_speed`; the result is clamped to
0.6-1.5, volume to 0.5-2.0. The prefix is added only at intensity ≥ 50. Unknown tags fall back
to neutral. Source: `src/voice_engine/data/emotion_map.toml`.

| tag | Cartesia emotion | speed factor | volume | prefix |
|---|---|---|---|---|
| neutral | neutral | 1.0 | 1.0 | |
| happy | happy | 1.05 | 1.1 | |
| excited | excited | 1.15 | 1.25 | |
| giggle | happy | 1.1 | 1.1 | `[laughter]` |
| curious | curious | 1.0 | 1.0 | |
| surprised | excited | 1.1 | 1.2 | |
| gentle | calm | 0.9 | 0.9 | |
| sad | sad | 0.85 | 0.85 | |
| whisper | calm | 0.85 | 0.5 | |
| proud | content | 0.95 | 1.1 | |

## Integrating with your host server

1. **Server:** `POST /sessions` with `ENGINE_API_KEY` and your config. Keep the key on the server.
2. **Server → browser:** hand over only `client_token` (single use, expires after
   `expires_in` seconds).
3. **Browser:** connect with the client package (`client/`):

   ```ts
   import { VoiceEngineClient } from "@voice-engine/client";

   const client = new VoiceEngineClient({ engineUrl: "https://engine.example", clientToken });
   client.on("bot_sentence", (e) => console.log(e.emotion, e.text));
   client.on("state", (e) => setAvatar(e.state));
   await client.connect();            // asks for the mic, plays the bot's audio
   client.sendCommand({ type: "mute_mic" });
   client.disconnect();
   ```

4. **Server, optional:** `PATCH` config mid-session, send commands, and receive webhooks.
   `DELETE` when you are done.

## End-to-end check: `scripts/e2e.py`

Runs the spec §7a automated Definition-of-done checks against a running engine:

| check | threshold |
|---|---|
| latency | median `turn_metrics.total_ms` ≤ 1000, p90 ≤ 1500, over ≥ 10 turns from the plain-turn phase only, at default config |
| interruption | speech onset → `interrupted` ≤ 300 ms |
| uh-huh | "uh-huh" during bot speech → no `interrupted` |
| backchannel timing | none in the first 2 s of a user turn, none closer than `MIN_GAP_S["normal"]` (5 s) |
| history | `history_roles.backchannel_like` = 0 and assistant messages ≤ replies |

    # once: generate the synthetic child-like fixtures (needs CARTESIA_API_KEY)
    uv run python scripts/e2e.py --make-fixtures --voice-id <child-like Cartesia voice id>
    # shell 1
    DEBUG=1 uv run voice-engine
    # shell 2
    uv run python scripts/e2e.py

It connects as a headless WebRTC client (aiortc), plays the fixtures as microphone audio,
prints a PASS/FAIL table and exits 1 on any failure (2 on setup errors, such as a missing key
or fixture). See `scripts/fixtures/README.md`. The check logic is unit-tested in
`tests/test_e2e_checks.py`.

## Status

**The latency and interruption targets are not yet verified live.** The e2e run needs real
Deepgram, Anthropic and Cartesia keys, which were not available when this was built. Run the
e2e script once keys exist; if latency fails, tune the `EAGERNESS` mapping and the Deepgram
endpointing and record the final values here. Then do the owner listening session in
`docs/listening-checklist.md`.
