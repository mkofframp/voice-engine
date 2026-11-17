# Voice Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone voice engine that holds natural, emotional, near-real-time spoken conversations. It must be interruptible mid-sentence and make listening sounds while the person talks. A host web app with its own server can embed it through HTTP and a small JS client.

**Architecture:** A Python service built on Pipecat. Audio comes in from the browser over SmallWebRTC, then goes through VAD and an end-of-turn model, Deepgram streaming STT, a custom turn manager (barge-in and backchannels), and Claude Haiku 5.5 producing emotion-tagged sentences. Each sentence passes a tag parser and an output filter, then goes to Cartesia TTS with emotion controls. Events go back to the browser on the WebRTC data channel and to the host server through an optional webhook. The host server creates and controls sessions over an HTTP API protected by an API key. The browser connects with a single-use client token.

**Tech Stack:** Python 3.12 (pinned with `uv`; Pipecat's native deps lag 3.14), `pipecat-ai` (extras: `webrtc,silero,deepgram,anthropic,cartesia,local-smart-turn-v3`), FastAPI + uvicorn, pydantic v2 + pydantic-settings, httpx, pytest + pytest-asyncio. JS client: TypeScript, ESM, `@pipecat-ai/small-webrtc-transport` + `@pipecat-ai/client-js`, built with `tsc`, tested with vitest.

**Spec:** `docs/superpowers/specs/2026-11-17-kid-voice-companion-design.md` (read §6 and §7a before any task).

## Global Constraints

- Latency: median last-word → first-reply-audio ≤ 1.0 s at default settings; p90 ≤ 1.5 s (spec Goal 1).
- LLM model id `claude-haiku-5-5`. Load the `claude-api` skill before writing any Anthropic code.
- API keys (`DEEPGRAM_API_KEY`, `ANTHROPIC_API_KEY`, `CARTESIA_API_KEY`, `ENGINE_API_KEY`) live only in `engine/.env`. They are never sent to the browser or the dev page.
- Copy rule: all fixed human-readable or human-heard text shipped by the engine comes from `engine/src/voice_engine/copy.json` (Task 2). That covers the default redirect and goodbye lines, backchannel phrases, dev page labels, and error messages meant for people. It is written by ChatGPT through the `chatgpt-copy` skill on `gpt-6-astra`; no task writes this text itself. If the skill is unavailable, the value is the literal `TODO(copy)`. Prompts are not copy.
- Persist nothing: no audio or transcripts on disk. Transcript text is logged only at DEBUG level.
- The server binds to `127.0.0.1` by default. English only.
- Host integration is HTTP + WebRTC only, so any server language works.
- The plan uses Pipecat class names as of late 2025 (`InterruptionFrame`, `OutputTransportMessageUrgentFrame`, `SmallWebRTCConnection`, `LocalSmartTurnAnalyzerV3`). Check them against the installed version. If one was renamed, use the new name and note it in the commit message.

## Review Focus

1. **The person interrupts the greeting or a host `say` line in its first 300 ms.** Expected: it is cut off like any other bot speech. Test in Task 11: `test_interrupts_host_say_line`.
2. **Interruption after the LLM has finished but while TTS audio is still playing, or before the first word played.** Expected: the history holds only the words actually heard plus the cut-off marker, and never the full unplayed text. Tests in Task 5: `test_truncate_before_first_word`, `test_truncate_after_llm_done`.
3. **`PATCH` with invalid values or during a turn.** Expected: invalid → 422 and the config is unchanged; a valid patch takes effect from the next turn, not mid-sentence. Tests in Task 8: `test_patch_invalid_leaves_config`. Task 10: `test_patch_applies_next_turn`.
4. **A client token used twice, after it expires, or by a second browser on a live session.** Expected: 401/409, and the first connection is unaffected. Tests in Task 8: `test_token_single_use`, `test_token_expired`, `test_second_connection_rejected`.
5. **Deepgram, Cartesia or Anthropic drops or returns 429 mid-session.** Expected: an `error` event plus a webhook, then the bot goes back to `listening`. No hang, no silent dead session. Test in Task 10: `test_llm_failure_emits_error_and_recovers`.

---

## File structure

```
kid-voice/
  engine/
    pyproject.toml  .env.example  README.md
    src/voice_engine/
      settings.py        # env settings
      config.py          # SessionConfig + patch
      copy.json  copy.py # shipped copy (Task 2)
      emotions.py        # tag set + streaming tag parser
      emotion_map.py     # tag -> Cartesia controls
      output_filter.py   # blocklist check
      interruption.py    # backchannel-vs-interruption classifier
      history.py         # assistant-message truncation
      backchannel.py     # scheduler, mood heuristic, clip choice
      metrics.py         # per-turn timings
      safety.py          # user-speech safety classifier
      sessions.py        # in-memory store + client tokens
      webhook.py         # host-server webhook sender
      api.py             # FastAPI app
      pipeline/
        events.py        # data-channel event helper
        processors.py    # TagParser/OutputFilter/EmotionTTS glue processors
        turn_manager.py  # barge-in + backchannel processor
        prerender.py     # fixed lines + backchannel bank audio cache
        bot.py           # builds and runs one session's pipeline
      data/default_safety_prompt.md  default_blocklist.txt  emotion_map.toml
      __main__.py        # `uv run voice-engine`
    tests/ ...           # one test file per module
    scripts/e2e.py  scripts/fixtures/
  client/                # TS package @local/voice-engine-client
  devpage/               # static index.html + devpage.js
```

---

### Task 1: Scaffold, settings, session config

**Files:**
- Create: `engine/pyproject.toml`, `engine/.env.example`, `.gitignore`, `engine/src/voice_engine/__init__.py`, `settings.py`, `config.py`, `data/default_safety_prompt.md`, `data/default_blocklist.txt`
- Test: `engine/tests/test_config.py`

**Interfaces:**
- Produces:
  - `Settings(BaseSettings)`, which reads `engine/.env`. Fields: `deepgram_api_key: str`, `anthropic_api_key: str`, `cartesia_api_key: str`, `engine_api_key: str`, `host: str = "127.0.0.1"`, `port: int = 7860`, `client_token_ttl_s: int = 120`, `allowed_origins: list[str] = ["http://localhost:5173"]`.
  - `SessionConfig(BaseModel, extra="forbid")`. Fields:
    - `persona_prompt: str = ""`
    - `safety_prompt: str` (default: the contents of `default_safety_prompt.md`; a blank value is replaced by the default)
    - `output_blocklist: list[str]` (default: lines of `default_blocklist.txt`)
    - `voice_id: str` (builder picks the default Cartesia voice)
    - `speaking_speed: float = Field(1.0, ge=0.6, le=1.5)`
    - `energy: float = Field(0.5, ge=0, le=1)`
    - `emotion_intensity: int = Field(70, ge=0, le=100)`
    - `response_length: Literal["short","medium","long"] = "short"`
    - `turn_eagerness: Literal["patient","normal","quick"] = "normal"`
    - `backchannel_frequency: Literal["off","low","normal","high"] = "normal"`
    - `redirect_line: str | None = None`, `goodbye_line: str | None = None` (`None` means the copy default)
    - `greeting: str | None = None`
    - `max_session_seconds: int = Field(600, ge=30, le=3600)`
    - `webhook_url: AnyHttpUrl | None = None`
  - `SessionConfig.patched(self, patch: dict) -> SessionConfig`: returns a new validated copy and raises `ValidationError` on bad input.

- [ ] **Step 1:** Run `git init` in `kid-voice/`. Create the uv project at `engine/` with `requires-python = "==3.12.*"` and the deps in Tech Stack. Add a `voice-engine` script entry. Add pytest config with `asyncio_mode = "auto"`. The `.gitignore` covers `.env`, `.venv`, `node_modules`, `dist`. `.env.example` lists the four keys.
- [ ] **Step 2: Write failing tests** in `test_config.py`:
  - `test_defaults`: `SessionConfig()` has `emotion_intensity == 70`, `turn_eagerness == "normal"`, and a non-empty `safety_prompt`.
  - `test_blank_safety_prompt_uses_default`: `SessionConfig(safety_prompt="  ").safety_prompt == SessionConfig().safety_prompt`.
  - `test_rejects_out_of_range`: `emotion_intensity=101` and `backchannel_frequency="always"` both raise `ValidationError`.
  - `test_rejects_unknown_field`: `SessionConfig(foo=1)` raises.
  - `test_patched_returns_new`: `c.patched({"emotion_intensity": 10})` gives 10 and `c` is unchanged.
- [ ] **Step 3:** Run `cd engine && uv run pytest tests/test_config.py -v`. Expected: FAIL (import error).
- [ ] **Step 4:** Implement `settings.py` and `config.py`. Write `default_safety_prompt.md`, a kid-safe prompt for ages 5–8 covering the topics in spec §7a. Seed `default_blocklist.txt` with the builder's list.
- [ ] **Step 5:** Run the tests again. Expected: 5 passed.
- [ ] **Step 6:** Commit: `feat(engine): scaffold, settings, session config`.

### Task 2: Shipped copy via ChatGPT

**Files:**
- Create: `engine/src/voice_engine/copy.json`, `engine/src/voice_engine/copy.py`
- Test: `engine/tests/test_copy.py`

**Interfaces:**
- Produces: `copy.get(key: str) -> str` and `copy.backchannel_phrases() -> dict[Mood, list[str]]`, where `Mood = Literal["neutral","interested","amused","sympathetic"]`.
- Keys: `redirect_line`, `goodbye_line`, `devpage.*` (labels for connect, disconnect, config editor, event log, metrics), `errors.*` (`token_invalid`, `session_not_found`, `session_busy`, `upstream_failed`). `backchannel` holds 4–7 phrases per mood, 15–25 in total, matching spec §6.5 (e.g. a neutral acknowledgement, interested, a soft laugh, a sympathetic sound).

- [ ] **Step 1:** Invoke the `chatgpt-copy` skill on model `gpt-6-astra`. The brief should cover: the audience (children aged 5–8 hearing a friendly voice, plus the owner reading the dev page); every key above with its purpose; the length limits (fixed lines under 15 words, backchannel phrases of 1–3 words that work when spoken alone). Check the result against the brief. Write it into `copy.json` unchanged. If the skill fails, put `"TODO(copy)"` as every value and tell the owner.
- [ ] **Step 2: Write tests:**
  - `test_all_keys_present`: every key listed above exists.
  - `test_backchannel_bank_size`: there are 15–25 phrases and every mood is non-empty.
- [ ] **Step 3:** Implement `copy.py`, which loads the JSON once. Run `uv run pytest tests/test_copy.py -v`. Expected: PASS.
- [ ] **Step 4:** Commit: `feat(engine): shipped copy from chatgpt-copy`.

### Task 3: Emotion tags and streaming tag parser

**Files:**
- Create: `engine/src/voice_engine/emotions.py`
- Test: `engine/tests/test_emotions.py`

**Interfaces:**
- Produces:
  - `EMOTION_TAGS = ("neutral","happy","excited","giggle","curious","surprised","gentle","sad","whisper","proud")`
  - `@dataclass(frozen=True) Sentence(emotion: str, text: str)`
  - `class TagParser` with `feed(chunk: str) -> list[Sentence]` (returns the sentences completed so far), `flush() -> list[Sentence]` and `reset() -> None`
  - `EMOTION_PROMPT: str`, the instruction appended to the system prompt
- Format: the LLM writes `[tag] Sentence text.` A sentence ends at `.`, `!` or `?` followed by whitespace or end of stream, or at a newline. A sentence with no tag inherits the previous tag, or `neutral` at the start. An unknown tag is kept as-is; the mapper handles it in Task 4. Tags are stripped from the text.

- [ ] **Step 1: Write failing tests:**
  - `test_single_sentence_in_chunks`: feeding `"[hap"`, `"py] Hi the"`, `"re! "` returns `[Sentence("happy","Hi there!")]` on the third feed.
  - `test_two_sentences_two_tags`: `"[curious] What is it? [excited] Wow!"` then `flush()` returns two sentences with the right emotions.
  - `test_untagged_inherits`: `"[sad] Oh no. That hurts."` returns two sentences, both `sad`.
  - `test_decimal_not_sentence_end`: `"[neutral] It is 3.5 meters."` returns one sentence.
  - `test_flush_partial`: `"[gentle] okay"` then `flush()` returns `Sentence("gentle","okay")`.
- [ ] **Step 2:** Run `uv run pytest tests/test_emotions.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement. `EMOTION_PROMPT` lists the tags and asks for one tag per sentence and natural variation, mostly `neutral`/`gentle` and rarely `excited` (spec §6.6).
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(engine): emotion tag parser`.

### Task 4: Emotion mapper and output filter

**Files:**
- Create: `engine/src/voice_engine/emotion_map.py`, `engine/src/voice_engine/data/emotion_map.toml`, `engine/src/voice_engine/output_filter.py`
- Test: `engine/tests/test_emotion_map.py`, `engine/tests/test_output_filter.py`

**Interfaces:**
- Consumes: `EMOTION_TAGS` (Task 3).
- Produces:
  - `@dataclass TTSControls(emotion: list[str], speed: float, volume: float, prefix_markup: str = "")`
  - `map_emotion(tag: str, intensity: int, base_speed: float) -> TTSControls`
  - `filter_sentence(text: str, blocklist: list[str], redirect_line: str) -> tuple[str, bool]`, which returns `(text_to_speak, was_blocked)`

- [ ] **Step 1:** Read Cartesia's current docs for the lowest-latency Sonic model. Note its controls for emotion, speed, volume and laughter/nonverbal sounds. Fill `emotion_map.toml` with one entry per tag holding the full-intensity controls. Every tag needs an entry; unsupported ones map to the nearest supported delivery. Add a comment at the top naming the model and doc date.
- [ ] **Step 2: Write failing tests:**
  - `test_every_tag_mapped`: each tag in `EMOTION_TAGS` maps without warning.
  - `test_zero_intensity_is_neutral`: for every tag, `map_emotion(tag, 0, 1.0)` equals `map_emotion("neutral", 0, 1.0)`.
  - `test_intensity_scales_speed`: for `excited`, the absolute difference between speed and base is larger at 100 than at 50.
  - `test_unknown_tag_falls_back`: `map_emotion("angry", 70, 1.0)` equals neutral, and `caplog` records one warning.
  - `test_blocked_word_replaced`: `filter_sentence("That is stupid.", ["stupid"], "R")` gives `("R", True)`.
  - `test_word_boundary`: the blocklist `["ass"]` does not block `"class"`.
  - `test_case_insensitive`.
- [ ] **Step 3:** Run `uv run pytest tests/test_emotion_map.py tests/test_output_filter.py -v`. Expected: FAIL.
- [ ] **Step 4:** Implement. Intensity scales linearly from the neutral controls to the tag's full controls.
- [ ] **Step 5:** Run the tests. Expected: PASS.
- [ ] **Step 6:** Commit: `feat(engine): emotion mapper and output filter`.

### Task 5: Interruption classifier and history truncation

**Files:**
- Create: `engine/src/voice_engine/interruption.py`, `engine/src/voice_engine/history.py`
- Test: `engine/tests/test_interruption.py`, `engine/tests/test_history.py`

**Interfaces:**
- Produces:
  - `BACKCHANNEL_WORDS: frozenset[str]` (mm, mhm, mm-hm, uh-huh, yeah, yep, ok, okay, right, sure, wow, oh, aha, ha, haha)
  - `classify_user_speech(transcript: str, speech_ms: int, *, min_speech_ms: int = 200, no_text_ms: int = 600) -> Literal["undecided","backchannel","interruption"]`
  - `@dataclass PlayedWord(word: str, start_ms: int)`
  - `@dataclass SpokenSentence(text: str, start_ms: int, words: list[PlayedWord])`, where times are relative to the start of the bot turn
  - `CUT_MARKER = " [interrupted by the user here]"`
  - `truncate_reply(sentences: list[SpokenSentence], played_ms: int) -> str`
- Classifier rule:
  - `speech_ms < min_speech_ms` → `undecided`.
  - Empty transcript: `speech_ms >= no_text_ms` → `interruption`, otherwise `undecided`.
  - At most 3 tokens, all in `BACKCHANNEL_WORDS` (lowercased, punctuation stripped) → `backchannel`.
  - Anything else → `interruption`.
- Truncation: keep the words with `start_ms < played_ms`, join them, append `CUT_MARKER`. If no word has played, return `CUT_MARKER.strip()`. If a sentence has no word timestamps, it counts as played only if `played_ms` ≥ the next sentence's `start_ms`.

- [ ] **Step 1: Write failing tests:**
  - `test_short_speech_undecided`: `("hey", 150)` → `undecided`.
  - `test_uh_huh_is_backchannel`: `("Uh-huh.", 400)` → `backchannel`.
  - `test_yeah_okay_backchannel`: `("yeah okay", 500)` → `backchannel`.
  - `test_real_words_interrupt`: `("wait I want", 300)` → `interruption`.
  - `test_long_speech_no_text_interrupts`: `("", 700)` → `interruption`.
  - `test_truncate_mid_sentence`: words at 0/300/600 ms with `played_ms=450` → `"Once upon" + CUT_MARKER`.
  - `test_truncate_before_first_word`: `played_ms=0` → `CUT_MARKER.strip()`.
  - `test_truncate_after_llm_done`: three sentences fully generated, `played_ms` inside the 2nd → no words from the 3rd.
- [ ] **Step 2:** Run `uv run pytest tests/test_interruption.py tests/test_history.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement both modules.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(engine): barge-in classifier and history truncation`.

### Task 6: Backchannel scheduler, mood heuristic, clip choice

**Files:**
- Create: `engine/src/voice_engine/backchannel.py`
- Test: `engine/tests/test_backchannel.py`

**Interfaces:**
- Consumes: `Mood` (Task 2).
- Produces:
  - `MIN_GAP_S = {"low": 8.0, "normal": 5.0, "high": 3.5}`, `WARMUP_S = 2.0`, `PAUSE_WINDOW_MS = (250, 700)`
  - `class BackchannelScheduler(frequency: str)` with:
    - `on_user_turn_start(t: float) -> None`
    - `should_play(t: float, pause_ms: int, turn_complete: bool, bot_speaking: bool) -> bool`
    - `on_played(t: float) -> None`
    - `set_frequency(f: str) -> None`
  - `pick_mood(interim_text: str) -> Mood`
  - `class ClipBank(clips: dict[Mood, list[bytes]])` with `pick(mood: Mood) -> tuple[int, bytes] | None`, which never returns the index it returned last time if another clip exists, and falls back to `neutral` when the mood is empty
- `should_play` is true only if all of these hold:
  - frequency is not `off`
  - not `bot_speaking`
  - not `turn_complete`
  - the pause is within `PAUSE_WINDOW_MS`
  - `t - turn_start >= WARMUP_S`
  - `t - last_played >= MIN_GAP_S[f]`
- Mood heuristic: no LLM call. Keyword lists: sad/hurt/scared words → `sympathetic`; funny/silly/laugh words → `amused`; a question mark or "guess what"/"you know what" → `interested`; otherwise `neutral`.

- [ ] **Step 1: Write failing tests:**
  - `test_no_backchannel_in_warmup`: turn starts at 0, pause at t=1.5 → False.
  - `test_plays_in_pause_window`: t=3, pause 400 → True.
  - `test_pause_too_short_or_long`: 100 ms and 900 ms → False.
  - `test_rate_limit`: played at 3, t=6 (normal) → False; t=8.1 → True.
  - `test_never_during_bot_speech`
  - `test_never_when_turn_complete`
  - `test_off_never_plays`
  - `test_mood_heuristics`: "my dog got hurt" → `sympathetic`; "that's so silly" → `amused`.
  - `test_clip_no_repeat`: 50 picks from a 3-clip mood, no two consecutive indices equal.
  - `test_empty_mood_falls_back`
- [ ] **Step 2:** Run `uv run pytest tests/test_backchannel.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement. Pass `t` in explicitly; there is no wall clock inside the module.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(engine): backchannel scheduler`.

### Task 7: Turn metrics

**Files:**
- Create: `engine/src/voice_engine/metrics.py`
- Test: `engine/tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `STAGES = ("user_last_word","end_of_turn","final_transcript","llm_first_token","tts_first_audio","audio_out")`
  - `class TurnMetrics` with `mark(stage: str, t: float) -> None` (the first mark per stage wins) and `as_event() -> dict` (ms deltas from `user_last_word` per stage, plus `total_ms` = `audio_out - user_last_word`; missing stages are `None`)
  - `class SessionMetrics` with `add(turn: TurnMetrics) -> None` and `summary() -> dict` (`{"turns": n, "median_ms": ..., "p90_ms": ...}` over `total_ms`)

- [ ] **Step 1: Write failing tests:**
  - `test_deltas`
  - `test_first_mark_wins`
  - `test_missing_stage_none`
  - `test_summary_median_p90`: totals 100..1000 step 100 → median 550, p90 910 (linear interpolation, `statistics.quantiles(method="inclusive")`).
- [ ] **Step 2:** Run `uv run pytest tests/test_metrics.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(engine): turn metrics`.

### Task 8: Session store, HTTP API, webhook

**Files:**
- Create: `engine/src/voice_engine/sessions.py`, `engine/src/voice_engine/webhook.py`, `engine/src/voice_engine/api.py`
- Test: `engine/tests/test_api.py`, `engine/tests/test_webhook.py`

**Interfaces:**
- Consumes: `Settings`, `SessionConfig` (Task 1), `copy.get` (Task 2).
- Produces:
  - `@dataclass Session`. Fields:
    - `id: str`
    - `config: SessionConfig`
    - `pending_config: SessionConfig | None`
    - `status: Literal["created","connected","ended"]`
    - `created_at: float`
    - `runner: SessionRunnerProtocol | None`
  - `class SessionStore(token_ttl_s: int, clock=time.monotonic)` with:
    - `create(config) -> tuple[Session, str]` (the token)
    - `redeem(token) -> Session` (raises `TokenInvalid` if unknown, already used or expired)
    - `get(id) -> Session` (raises `SessionNotFound`)
    - `patch(id, patch: dict) -> SessionConfig`, which stores the result as `pending_config`; the runner picks it up at the next turn start
    - `end(id, reason: str) -> None`
  - `Protocol SessionRunnerProtocol` with `async handle_offer(sdp: str, type: str) -> dict`, `async command(cmd: dict) -> None` and `async stop(reason: str) -> None`. It is implemented in Task 10.
  - `RunnerFactory = Callable[[Session], SessionRunnerProtocol]`
  - `class WebhookSender` with `async send(url: str | None, event: dict) -> None`. It uses httpx with a 3 s timeout and 2 retries on 5xx or network error, logs failures, never raises, and does nothing when `url` is `None`.
  - `create_app(settings: Settings, store: SessionStore, runner_factory: RunnerFactory) -> FastAPI`
- HTTP API (JSON error shape: `{"error": code, "message": copy.get("errors."+code)}`):
  - `POST /sessions`: requires `Authorization: Bearer <ENGINE_API_KEY>`, body = partial `SessionConfig`. Returns 201 `{session_id, client_token, expires_in}`.
  - `GET /sessions/{id}` (auth): returns `{status, created_at, config}`.
  - `PATCH /sessions/{id}` (auth): returns 200 with the pending config, 422 on validation error, 404 for an unknown id.
  - `DELETE /sessions/{id}` (auth): stops the runner, returns 204.
  - `POST /sessions/{id}/commands` (auth): body is one of `{type:"say",text}`, `{type:"add_context",text}`, `{type:"end",graceful:bool}`, `{type:"mute_mic"}`, `{type:"unmute_mic"}`. Returns 202, or 409 if not connected.
  - `POST /webrtc/offer`: no API key. Body `{client_token, sdp, type}`. Redeems the token and creates the runner. Returns 200 with the answer, 401 for an invalid token, 409 if the session is already connected.
  - CORS allows `settings.allowed_origins`, for `/webrtc/offer` only.

- [ ] **Step 1: Write failing tests** with FastAPI `TestClient`, a fake runner factory and a fake clock:
  - `test_create_requires_key`: a missing or bad key gives 401.
  - `test_create_returns_token`
  - `test_patch_invalid_leaves_config`: `{"emotion_intensity": 500}` gives 422, and `GET` shows the old config with no pending config.
  - `test_patch_valid_is_pending`
  - `test_token_single_use`: a second offer with the same token gives 401.
  - `test_token_expired`: advance the clock past the TTL → 401.
  - `test_second_connection_rejected`: a session already connected gives 409.
  - `test_command_forwarded`: `say` reaches `fake_runner.command`.
  - `test_delete_stops_runner`
  - `test_webhook_retries_then_gives_up` (`test_webhook.py`, with `httpx.MockTransport`): three 500s, no exception, 3 calls.
  - `test_webhook_none_url_noop`
- [ ] **Step 2:** Run `uv run pytest tests/test_api.py tests/test_webhook.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement. Create the token with `secrets.token_urlsafe(32)` and compare keys with `secrets.compare_digest`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(engine): session store, HTTP API, webhook`.

### Task 9: Safety classifier on user speech

**Files:**
- Create: `engine/src/voice_engine/safety.py`
- Test: `engine/tests/test_safety.py`

**Interfaces:**
- Produces:
  - `@dataclass SafetyFlag(category: Literal["harm","self_harm","personal_info"], excerpt: str)`
  - `class SafetyClassifier(client: anthropic.AsyncAnthropic, model: str = "claude-haiku-5-5")` with `async classify(text: str) -> SafetyFlag | None`. It never raises: on any API error it logs and returns `None`. `excerpt` is at most 120 characters of the input.

- [ ] **Step 1:** Load the `claude-api` skill. Use forced tool use (one tool, `report`, with an enum `category` that includes `"none"`) so the output is structured.
- [ ] **Step 2: Write failing tests** with a fake client returning canned tool-use responses:
  - `test_flag_returned`
  - `test_none_category_returns_none`
  - `test_api_error_returns_none`
  - `test_excerpt_truncated`
- [ ] **Step 3:** Run `uv run pytest tests/test_safety.py -v`. Expected: FAIL.
- [ ] **Step 4:** Implement. The classification prompt defines the three categories for a child speaker aged 5–8; a child saying their own first name is not `personal_info`. Add a test that runs only with `RUN_LIVE=1`: `test_live_examples`, with 3 positive and 3 negative sentences.
- [ ] **Step 5:** Run the tests. Expected: PASS (the live test is skipped).
- [ ] **Step 6:** Commit: `feat(engine): user-speech safety classifier`.

### Task 10: Speech pipeline: one session end to end (no barge-in or backchannels yet)

**Files:**
- Create: `engine/src/voice_engine/pipeline/events.py`, `processors.py`, `prerender.py`, `bot.py`, `engine/src/voice_engine/__main__.py`
- Test: `engine/tests/pipeline/test_processors.py`, `engine/tests/pipeline/test_bot.py`

**Interfaces:**
- Consumes: Tasks 1–9.
- Produces:
  - `events.event_frame(type: str, **payload) -> OutputTransportMessageUrgentFrame`, with message `{"type": type, **payload}`. The types are the spec §6.7 list: `state`, `user_transcript`, `bot_sentence`, `backchannel`, `interrupted`, `turn_metrics`, `safety_flag`, `session_ending`, `error`.
  - `class EmotionSentenceFrame(DataFrame)` with `emotion: str`, `text: str`, `blocked: bool = False`.
  - `class TagParserProcessor(FrameProcessor)`: `LLMTextFrame` → `EmotionSentenceFrame`. It resets on `LLMFullResponseStartFrame` and on `InterruptionFrame`, and flushes on `LLMFullResponseEndFrame`.
  - `class OutputFilterProcessor(FrameProcessor)` (`config_ref: Callable[[], SessionConfig]`): applies `filter_sentence`.
  - `class EmotionTTSAdapter(FrameProcessor)`: for each `EmotionSentenceFrame`, pushes the controls for `map_emotion(...)` to Cartesia (via `TTSUpdateSettingsFrame`, or a `CartesiaTTSService` subclass if per-sentence settings don't apply to the next sentence in the installed version), then a `TextFrame` of the sentence. It also emits `bot_sentence {emotion, text}` when `TTSStartedFrame` arrives for that sentence.
  - `prerender.PrerenderCache`, keyed by `(voice_id, speed)`. It has `async fixed_line(key: Literal["redirect","goodbye"], cfg) -> bytes` and `async backchannel_bank(cfg) -> ClipBank`. It renders through Cartesia's HTTP API once, keeps the result in memory only, and leaves out moods whose phrase render fails.
  - `class SessionRunner(SessionRunnerProtocol)` (`session`, `settings`, `store`, `webhook`, `cache`). `handle_offer` builds the pipeline:

    ```
    transport.input()
      → STT (Deepgram, interim results)
      → [TurnManager slot]
      → user context aggregator
      → LLM (claude-haiku-5-5, system = safety_prompt + persona_prompt + EMOTION_PROMPT + response-length line)
      → TagParserProcessor → OutputFilterProcessor → EmotionTTSAdapter
      → Cartesia TTS
      → transport.output()
      → assistant context aggregator
    ```

    VAD is `SileroVADAnalyzer`. Turn analysis is `LocalSmartTurnAnalyzerV3`, with the stop-secs/eagerness mapping from `turn_eagerness` chosen by the builder and documented in the README.
  - Responsibilities:
    - Apply `session.pending_config` at each user turn start.
    - Run `SafetyClassifier.classify` as a background task on every final transcript, then emit `safety_flag` and the webhook.
    - Feed `TurnMetrics` from frame timestamps.
    - Emit `state` changes.
    - Greeting: if `cfg.greeting` is set, queue it as `TTSSpeakFrame` on client connect.
    - Commands: `say` → `TTSSpeakFrame` and append to the assistant context; `add_context` → system message; `end(graceful)` → play goodbye clip then stop; `mute_mic`/`unmute_mic` → drop input audio.
    - Session timer: at `max_session_seconds`, after the current sentence, emit `session_ending`, play the goodbye clip and stop.
    - On stop: emit `session_ending`, webhook `session_ended {reason, metrics_summary}`, `store.end`.
    - Upstream `ErrorFrame`: emit `error` + webhook and go back to `listening`. Only fatal errors (transport closed) end the session.
  - `__main__`: `uv run voice-engine` starts uvicorn on `settings.host:port` with `create_app(..., runner_factory=SessionRunner factory)`.

- [ ] **Step 1: Write failing processor tests** using Pipecat's test helper (`pipecat.tests.utils.run_test`, or the installed equivalent):
  - `test_tag_parser_emits_sentences`: `LLMTextFrame`s `"[happy] Hi! [sad] Oh."` → two `EmotionSentenceFrame`s.
  - `test_tag_parser_resets_on_interruption`: a half-sentence, then `InterruptionFrame`, then a new response → no leftover text.
  - `test_filter_replaces_blocked`
  - `test_tts_adapter_sets_controls_per_sentence`: two sentences with different emotions → two different settings payloads, each before its `TextFrame`.
- [ ] **Step 2:** Run `uv run pytest tests/pipeline/test_processors.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `events.py` and `processors.py`. Run again. Expected: PASS.
- [ ] **Step 4: Write failing runner tests** in `test_bot.py`. Swap the Deepgram, Anthropic and Cartesia services for Pipecat-compatible fakes through an injectable `services` factory, so no network is used:
  - `test_patch_applies_next_turn`: a patch arrives while the bot is speaking; the current turn's sentences still use the old intensity and the next turn uses the new one.
  - `test_llm_failure_emits_error_and_recovers`: the fake LLM pushes `ErrorFrame`; the events contain `error` then `state: listening`, the webhook fake got `error`, and the session is still `connected`.
  - `test_greeting_spoken_first`
  - `test_session_timer_goodbye`: `max_session_seconds` with a fake clock → `session_ending` then stop, with `session_ended` posted to the webhook.
  - `test_safety_flag_event`: the fake classifier flags → `safety_flag` event and webhook.
- [ ] **Step 5:** Run `uv run pytest tests/pipeline/test_bot.py -v`. Expected: FAIL.
- [ ] **Step 6:** Implement `prerender.py`, `bot.py` and `__main__.py`. Run the full suite with `uv run pytest -v`. Expected: all pass.
- [ ] **Step 7: Manual smoke test** (needs real keys in `engine/.env`). Run `uv run voice-engine`, then `curl -X POST localhost:7860/sessions -H "Authorization: Bearer $ENGINE_API_KEY" -d '{}'`. Expected: 201 with a token. The full voice check waits for the dev page (Task 13).
- [ ] **Step 8:** Commit: `feat(engine): session pipeline with emotion TTS, events, commands`.

### Task 11: Turn manager: barge-in, truncation, backchannels

**Files:**
- Create: `engine/src/voice_engine/pipeline/turn_manager.py`
- Modify: `engine/src/voice_engine/pipeline/bot.py` (insert into the TurnManager slot; set the Pipecat transport/pipeline so VAD alone never triggers an interruption)
- Test: `engine/tests/pipeline/test_turn_manager.py`

**Interfaces:**
- Consumes: `classify_user_speech`, `truncate_reply`, `SpokenSentence` (Task 5), `BackchannelScheduler`, `pick_mood`, `ClipBank` (Task 6), `event_frame` (Task 10).
- Produces: `class TurnManager(FrameProcessor)` (`config_ref`, `clip_bank: ClipBank`, `context: LLMContext`, `clock: Callable[[], float] = time.monotonic`).
- Behaviour:
  - While the bot is speaking (between `BotStartedSpeakingFrame` and `BotStoppedSpeakingFrame`, including greeting and `say` lines), track user speech duration from VAD and the latest interim transcript, and call `classify_user_speech` on every update.
    - `interruption`: push `InterruptionFrame` (Pipecat's interruption task API), emit `interrupted {spoken_text}`, and replace the last assistant message in `context` with `truncate_reply(...)`. Build `SpokenSentence`s from `TTSTextFrame` word timestamps; get `played_ms` from the time of the first `BotStartedSpeakingFrame` of this reply.
    - `backchannel`: do nothing to the bot and keep the transcript out of the context. Pipecat's user aggregator must not form a turn from it: drop that `TranscriptionFrame`.
  - While the user holds the turn: on each VAD pause where the end-of-turn analyzer reports incomplete, call `scheduler.should_play`. If true, push the `ClipBank.pick(pick_mood(interim))` audio as `OutputAudioRawFrame` downstream, flagged so it does not raise `BotStartedSpeakingFrame` (verify; otherwise suppress the bot-speaking state for clips). Emit `backchannel {mood}` and call `on_played`. Clips never touch the context.
  - Target: playback stops within ~200 ms of the interruption being detected (spec §6.4).

- [ ] **Step 1: Write failing tests** with frame-script scenarios and a fake clock:
  - `test_uh_huh_does_not_interrupt`: bot speaking; user VAD 400 ms and interim "uh-huh" → no `InterruptionFrame`, context unchanged, no user turn.
  - `test_real_speech_interrupts`: interim "wait stop" at 300 ms → exactly one `InterruptionFrame`, an `interrupted` event, last assistant message ends with `CUT_MARKER`.
  - `test_interrupts_host_say_line`: speech during a `say` line → interrupted.
  - `test_backchannel_in_pause`: user talking 3 s, then a 400 ms incomplete pause → one `OutputAudioRawFrame` + a `backchannel` event, and no `BotStartedSpeakingFrame` downstream.
  - `test_backchannel_not_repeated_too_soon`
  - `test_backchannel_never_in_context`
  - `test_no_backchannel_while_bot_speaking`
- [ ] **Step 2:** Run `uv run pytest tests/pipeline/test_turn_manager.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement and wire into `bot.py`.
- [ ] **Step 4:** Run `uv run pytest -v`. Expected: all pass, including Task 10's.
- [ ] **Step 5:** Commit: `feat(engine): barge-in and active listening`.

### Task 12: JS client package

**Files:**
- Create: `client/package.json`, `client/tsconfig.json`, `client/src/index.ts`, `client/src/events.ts`
- Test: `client/test/client.test.ts`

**Interfaces:**
- Consumes: the `/webrtc/offer` contract and event types (Tasks 8 and 10).
- Produces (ESM, with types):
  - `type EngineEvent`: a discriminated union over the 9 §6.7 types, plus payloads.
  - `class VoiceEngineClient(opts: { engineUrl: string; clientToken: string })` with:
    - `connect(): Promise<void>` (asks for the mic, sends the WebRTC offer with `client_token`, attaches remote audio to an internal `<audio>`)
    - `disconnect(): void`
    - `on<T extends EngineEvent["type"]>(type: T, cb: (e: Extract<EngineEvent,{type:T}>) => void): () => void` (the return value unsubscribes)
    - `onAny(cb)`
    - `sendCommand(cmd: Command): void` over the data channel, with the same command shapes as the HTTP endpoint. The engine accepts commands from both paths, so add the data-channel path to `SessionRunner` in this task.

- [ ] **Step 1: Write failing vitest tests** with a fake transport:
  - `test_routes_events_by_type`
  - `test_unsubscribe`
  - `test_offer_includes_token`
  - `test_unknown_event_type_ignored_not_thrown`
  - Python side: `test_datachannel_command_forwarded` in `tests/pipeline/test_bot.py`.
- [ ] **Step 2:** Run `cd client && npm test` and `cd engine && uv run pytest tests/pipeline/test_bot.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement on top of `@pipecat-ai/small-webrtc-transport`, with no API key anywhere in the client.
- [ ] **Step 4:** Run `npm test && npm run build`, and the Python test. Expected: PASS, with `dist/` emitted.
- [ ] **Step 5:** Commit: `feat(client): JS client`.

### Task 13: Dev test page

**Files:**
- Create: `devpage/index.html`, `devpage/devpage.js`
- Modify: `engine/src/voice_engine/api.py` (serve `/dev` and `/dev/api/session` only when the request comes from `127.0.0.1`/`::1`)
- Test: `engine/tests/test_devpage.py`

**Interfaces:**
- Consumes: `VoiceEngineClient` (Task 12, imported from `client/dist`), the API (Task 8).
- Produces: `GET /dev` returns the page. `POST /dev/api/session` takes a config JSON body and creates a session server-side using `ENGINE_API_KEY` internally, returning only `{session_id, client_token}`, because the page must never hold the key.
- The page has: a config JSON textarea (prefilled from `SessionConfig().model_dump()`), Connect/Disconnect, a live event log, and a metrics panel showing the last turn's stages and the session median/p90. Labels come from `copy.get("devpage.*")`, exposed through `GET /dev/copy.json`.

- [ ] **Step 1: Write failing tests:**
  - `test_dev_localhost_only`: a request whose client host is not loopback → 404.
  - `test_dev_session_hides_key`: the response JSON contains no `ENGINE_API_KEY` value.
- [ ] **Step 2:** Run `uv run pytest tests/test_devpage.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement. Run the tests. Expected: PASS.
- [ ] **Step 4: Manual check** with real keys:
  - Run `uv run voice-engine`, open `http://localhost:7860/dev`, Connect, and talk.
  - Expected: a reply with varied `bot_sentence` emotions; "uh-huh" during a reply does not stop it; "wait" stops it within a beat; "mm-hm" sounds appear in your pauses; the metrics panel fills in.
- [ ] **Step 5:** Commit: `feat(devpage): local test page`.

### Task 14: End-to-end script, README, listening checklist

**Files:**
- Create: `engine/scripts/e2e.py`, `engine/scripts/fixtures/README.md`, `engine/README.md`, `docs/listening-checklist.md`

**Interfaces:**
- Consumes: the running engine plus the API.
- `e2e.py` connects as a headless WebRTC client (aiortc via Pipecat's SmallWebRTC client utilities), creates a session through the API, and plays synthetic child-like TTS fixtures. Fixtures are generated once by the script from Cartesia with a child-like voice; no real child recordings. It reads the event stream and asserts these Definition-of-done thresholds (spec §7a):
  - median `turn_metrics.total_ms` ≤ 1000 and p90 ≤ 1500 over ≥10 turns at default config
  - interruption: user speech onset (the fixture's known offset) → `interrupted` event ≤ 300 ms
  - "uh-huh" fixture played during bot speech → no `interrupted`
  - no `backchannel` event within the first 2 s of a user turn or closer than `MIN_GAP_S["normal"]`
  - `GET /sessions/{id}` after the run does not show backchannel text in the transcript (expose a debug-only `history_roles` count when `DEBUG=1` for this)

- [ ] **Step 1:** Write `e2e.py` with those assertions; it exits non-zero on failure and prints a table of results.
- [ ] **Step 2:** Run `uv run voice-engine` in one shell and `uv run python scripts/e2e.py` in another. Expected: all checks PASS. If latency fails, tune the Task 10 eagerness mapping and the Deepgram endpointing, and record the final values in the README.
- [ ] **Step 3:** Write `engine/README.md` covering: one-command run, `.env` setup, the HTTP API reference with examples, the event reference, the `turn_eagerness` mapping, the emotion map table (from `emotion_map.toml`), and how the host server integrates (create session → hand token to browser → `VoiceEngineClient`). Write `docs/listening-checklist.md` from the spec §7a owner checklist; it is an internal doc for the owner, not shipped copy.
- [ ] **Step 4:** Commit: `test: e2e thresholds, docs`.
