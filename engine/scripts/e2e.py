"""End-to-end check of a running engine against the Definition of done (spec §7a).

    DEBUG=1 uv run voice-engine                                   # shell 1
    uv run python scripts/e2e.py --make-fixtures --voice-id <id>  # once: needs CARTESIA_API_KEY
    uv run python scripts/e2e.py                                  # shell 2

The script connects as a headless WebRTC client (aiortc), creates a session through the
API, plays the synthetic child-like fixtures in `scripts/fixtures/` and reads the event
stream. It prints a table of results and exits 0 only if every check passes (1 if any
check fails, 2 if it could not run). It never prints transcript or reply text.

The check functions below are pure (events in, result out) and are unit-tested in
`tests/test_e2e_checks.py`. Event times are the client's `time.monotonic()` at receipt.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import struct
import sys
import time
import wave
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from voice_engine.backchannel import MIN_GAP_S
from voice_engine.backchannel import WARMUP_S as BACKCHANNEL_WARMUP_S  # no backchannel in the first 2 s of a turn

ENGINE_DIR = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# Thresholds from the spec (goal 1 and §7a).
MIN_TURNS = 10
MEDIAN_MAX_MS = 1000
P90_MAX_MS = 1500
INTERRUPT_MAX_MS = 300
GAP_JITTER_S = 0.05  # allowed spread in data-channel delivery between two events

RUN_KEYS = ["ENGINE_API_KEY", "DEEPGRAM_API_KEY", "ANTHROPIC_API_KEY", "CARTESIA_API_KEY"]
FIXTURE_KEYS = ["CARTESIA_API_KEY"]

Events = list[tuple[float, dict]]


# ---------------------------------------------------------------- checks (pure)


@dataclass
class CheckResult:
    name: str
    passed: bool
    measured: str
    threshold: str


def _of_type(events: Iterable[tuple[float, dict]], type: str) -> list[tuple[float, dict]]:
    return [(t, e) for t, e in events if e.get("type") == type]


def _state_at(events: Events, t: float) -> str | None:
    """The bot state last reported at or before `t`."""
    current = None
    for at, e in events:
        if at > t:
            break
        if e.get("type") == "state":
            current = e.get("state")
    return current


def latency_stats(totals: list[int]) -> tuple[int, int]:
    """(median, p90) in ms, computed the same way as `metrics.SessionMetrics`."""
    if len(totals) == 1:
        return totals[0], totals[0]
    median = round(statistics.median(totals))
    p90 = round(statistics.quantiles(totals, n=10, method="inclusive")[8])
    return median, p90


def check_latency(
    events: Events,
    min_turns: int = MIN_TURNS,
    median_max_ms: int = MEDIAN_MAX_MS,
    p90_max_ms: int = P90_MAX_MS,
    window: tuple[float, float] | None = None,
) -> CheckResult:
    """`window` (start, end): count only turn_metrics received in the plain-turn phase."""
    threshold = f"median<={median_max_ms}ms, p90<={p90_max_ms}ms, n>={min_turns}"
    totals = [
        e["total_ms"]
        for t, e in _of_type(events, "turn_metrics")
        if e.get("total_ms") is not None and (window is None or window[0] <= t <= window[1])
    ]
    if not totals:
        return CheckResult("latency", False, "n=0 turns with total_ms", threshold)
    median, p90 = latency_stats(totals)
    ok = len(totals) >= min_turns and median <= median_max_ms and p90 <= p90_max_ms
    return CheckResult("latency", ok, f"median={median}ms p90={p90}ms n={len(totals)}", threshold)


def check_interruption(events: Events, onset_t: float, max_ms: int = INTERRUPT_MAX_MS) -> CheckResult:
    threshold = f"speech onset -> interrupted <= {max_ms}ms"
    if _state_at(events, onset_t) != "speaking":
        return CheckResult("interruption", False, "bot was not speaking at speech onset", threshold)
    after = [t for t, _ in _of_type(events, "interrupted") if t >= onset_t]
    if not after:
        return CheckResult("interruption", False, "no interrupted event", threshold)
    ms = round((after[0] - onset_t) * 1000)
    return CheckResult("interruption", ms <= max_ms, f"{ms}ms", threshold)


def check_uhhuh(events: Events, start_t: float, end_t: float) -> CheckResult:
    threshold = "no interrupted while 'uh-huh' plays over the bot"
    if _state_at(events, start_t) != "speaking":
        return CheckResult("uh-huh", False, "bot was not speaking when 'uh-huh' started", threshold)
    hits = [t for t, _ in _of_type(events, "interrupted") if start_t <= t <= end_t]
    if hits:
        return CheckResult("uh-huh", False, f"interrupted {hits[0] - start_t:.2f}s after start", threshold)
    return CheckResult("uh-huh", True, "no interrupted", threshold)


def check_backchannel_timing(
    events: Events,
    user_turn_starts: list[float],
    warmup_s: float = BACKCHANNEL_WARMUP_S,
    min_gap_s: float = MIN_GAP_S["normal"],
    jitter_s: float = GAP_JITTER_S,
) -> CheckResult:
    threshold = f"none in first {warmup_s:g}s of a turn; gap >= {min_gap_s:g}s"
    times = [t for t, _ in _of_type(events, "backchannel")]
    starts = sorted(user_turn_starts)
    problems: list[str] = []
    for t in times:
        begun = [s for s in starts if s <= t]
        if not begun:
            problems.append("one before any user turn")
        elif t - begun[-1] < warmup_s:
            problems.append(f"one {t - begun[-1]:.2f}s into a turn")
    for a, b in zip(times, times[1:]):
        if b - a < min_gap_s - jitter_s:
            problems.append(f"two {b - a:.2f}s apart")
    measured = f"{len(times)} backchannels"
    if problems:
        return CheckResult("backchannel timing", False, f"{measured}; " + "; ".join(problems), threshold)
    return CheckResult("backchannel timing", True, measured, threshold)


def check_history(
    history_roles: Mapping[str, int] | None, events: Events, extra_replies: int = 0
) -> CheckResult:
    """Backchannels must never enter the LLM context.

    The engine counts context messages whose whole text is a backchannel phrase or word
    (`backchannel_like`, DEBUG only); it must be 0. As a sanity check, assistant messages
    must not outnumber the replies: each reply follows a "thinking" state, plus any the
    script started itself (greeting, `say`).
    """
    threshold = "backchannel_like == 0, assistant messages <= replies"
    if history_roles is None:
        return CheckResult(
            "history", False, "no history_roles: start the engine with DEBUG=1", threshold
        )
    leaked = history_roles.get("backchannel_like")
    if leaked is None:
        return CheckResult("history", False, "no backchannel_like count in history_roles", threshold)
    replies = sum(1 for _, e in _of_type(events, "state") if e.get("state") == "thinking")
    replies += extra_replies
    assistant = history_roles.get("assistant", 0)
    bcs = len(_of_type(events, "backchannel"))
    measured = f"backchannel_like={leaked} assistant={assistant} replies={replies} backchannels={bcs}"
    return CheckResult("history", leaked == 0 and assistant <= replies, measured, threshold)


def format_table(results: list[CheckResult]) -> str:
    rows = [("check", "result", "measured", "threshold")]
    rows += [(r.name, "PASS" if r.passed else "FAIL", r.measured, r.threshold) for r in results]
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def exit_code(results: list[CheckResult]) -> int:
    return 0 if results and all(r.passed for r in results) else 1


# ---------------------------------------------------------------- env and keys


def load_env() -> dict[str, str]:
    """engine/.env, overridden by the process environment (secrets only from there)."""
    from dotenv import dotenv_values

    env = {k: v for k, v in dotenv_values(ENGINE_DIR / ".env").items() if v is not None}
    env.update(os.environ)
    return env


def missing_keys(env: Mapping[str, str], required: list[str]) -> list[str]:
    return [k for k in required if not env.get(k, "").strip()]


def require_keys(env: Mapping[str, str], required: list[str]) -> None:
    missing = missing_keys(env, required)
    if missing:
        print(
            f"e2e: missing {', '.join(missing)}. Put them in {ENGINE_DIR / '.env'} "
            "(see engine/README.md) and try again.",
            file=sys.stderr,
        )
        raise SystemExit(2)


# ---------------------------------------------------------------- fixtures

FIXTURE_RATE = 16000
FIXTURE_ONSET_S = 0.5  # every fixture's speech starts exactly this far into the file
SILENCE_THRESHOLD = 500  # |sample| above this counts as speech
RAMBLE_PAUSE_S = 0.6  # pauses inside the ramble, inside the backchannel window

# Test inputs only (spoken to the engine), not shipped copy.
TURN_FIXTURES = {
    "turn_01": "Hi! My name is Sam and I am six.",
    "turn_02": "Do you like dinosaurs?",
    "turn_03": "My favorite color is green.",
    "turn_04": "I have a cat called Pickle.",
    "turn_05": "What do bunnies eat?",
    "turn_06": "I went to the park today.",
    "turn_07": "Can you count to five?",
    "turn_08": "Why is the sky blue?",
    "turn_09": "I like pizza with lots of cheese.",
    "turn_10": "My tooth is wobbly!",
    "turn_11": "What is your favorite animal?",
    "turn_12": "I built a big tower with blocks.",
}
OTHER_FIXTURES = {
    "story_request": "Can you tell me a long story about a friendly dragon?",
    "interrupt": "Wait, wait! I want to tell you something!",
    "uhhuh": "Uh-huh.",
}
RAMBLE_SEGMENTS = [
    "So yesterday at school, um,",
    "we had a really big race, and",
    "I ran super super fast, and then",
    "my friend fell down, but",
    "she was okay and we got stickers.",
]


def first_loud_sample(pcm: bytes, threshold: int = SILENCE_THRESHOLD) -> int | None:
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm[: len(pcm) // 2 * 2])
    for i, s in enumerate(samples):
        if abs(s) > threshold:
            return i
    return None


def _last_loud_sample(pcm: bytes, threshold: int = SILENCE_THRESHOLD) -> int:
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm[: len(pcm) // 2 * 2])
    for i in range(len(samples) - 1, -1, -1):
        if abs(samples[i]) > threshold:
            return i
    return -1


def place_onset(pcm: bytes, onset_s: float, rate: int) -> bytes:
    """Trim leading silence and pad so speech starts exactly `onset_s` in."""
    first = first_loud_sample(pcm)
    if first is None:
        raise ValueError("fixture audio has no speech")
    return bytes(2 * round(onset_s * rate)) + pcm[2 * first :]


def _trim(pcm: bytes) -> bytes:
    first = first_loud_sample(pcm)
    if first is None:
        raise ValueError("fixture audio has no speech")
    return pcm[2 * first : 2 * (_last_loud_sample(pcm) + 1)]


def write_wav(path: Path, pcm: bytes, rate: int = FIXTURE_RATE) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def read_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, FIXTURE_RATE):
            raise ValueError(f"{path.name}: expected 16-bit mono {FIXTURE_RATE} Hz")
        return w.readframes(w.getnframes())


async def make_fixtures(api_key: str, voice_id: str, out_dir: Path = FIXTURES_DIR) -> None:
    import httpx

    from voice_engine.pipeline.prerender import CARTESIA_MODEL, CARTESIA_URL, CARTESIA_VERSION

    headers = {"Cartesia-Version": CARTESIA_VERSION, "X-API-Key": api_key}

    async def render(client: httpx.AsyncClient, text: str) -> bytes:
        payload = {
            "model_id": CARTESIA_MODEL,
            "transcript": text,
            "voice": {"mode": "id", "id": voice_id},
            "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": FIXTURE_RATE},
            "language": "en",
        }
        resp = await client.post(CARTESIA_URL, json=payload, headers=headers)
        if resp.status_code != 200 or not resp.content:
            raise SystemExit(f"e2e: Cartesia render failed with status {resp.status_code}")
        return resp.content

    out_dir.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=30) as client:
        for name, text in {**TURN_FIXTURES, **OTHER_FIXTURES}.items():
            write_wav(out_dir / f"{name}.wav", place_onset(await render(client, text), FIXTURE_ONSET_S, FIXTURE_RATE))
            print(f"wrote {name}.wav")
        pause = bytes(2 * round(RAMBLE_PAUSE_S * FIXTURE_RATE))
        parts = [_trim(await render(client, seg)) for seg in RAMBLE_SEGMENTS]
        ramble = place_onset(pause.join(parts), FIXTURE_ONSET_S, FIXTURE_RATE)
        write_wav(out_dir / "ramble.wav", ramble)
        print("wrote ramble.wav")


def load_fixtures(fixtures_dir: Path = FIXTURES_DIR) -> dict[str, bytes]:
    names = [*TURN_FIXTURES, *OTHER_FIXTURES, "ramble"]
    missing = [n for n in names if not (fixtures_dir / f"{n}.wav").exists()]
    if missing:
        print(
            f"e2e: fixtures missing ({', '.join(missing)}). Generate them once with\n"
            "  uv run python scripts/e2e.py --make-fixtures --voice-id <child-like Cartesia voice id>",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return {n: read_wav(fixtures_dir / f"{n}.wav") for n in names}


# ---------------------------------------------------------------- headless client


class ScenarioError(Exception):
    pass


def _fixture_track_class():
    import fractions

    from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
    from av import AudioFrame

    frame_samples = FIXTURE_RATE // 50  # 20 ms

    class FixtureTrack(MediaStreamTrack):
        """Microphone stand-in: silence, or a queued fixture, paced in real time."""

        kind = "audio"

        def __init__(self) -> None:
            super().__init__()
            self._clips: list[tuple[bytes, asyncio.Future, asyncio.Future]] = []
            self._pos = 0
            self._start: float | None = None
            self._sent = 0

        def play(self, pcm: bytes) -> tuple[asyncio.Future, asyncio.Future]:
            """Queue a clip; the futures resolve to the monotonic times its first
            and last frames go out."""
            loop = asyncio.get_running_loop()
            started, ended = loop.create_future(), loop.create_future()
            self._clips.append((pcm, started, ended))
            return started, ended

        async def recv(self) -> AudioFrame:
            if self.readyState != "live":
                raise MediaStreamError
            if self._start is None:
                self._start = time.monotonic()
            else:
                wait = self._start + self._sent / FIXTURE_RATE - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
            need = frame_samples * 2
            chunk = b""
            if self._clips:
                pcm, started, ended = self._clips[0]
                if self._pos == 0 and not started.done():
                    started.set_result(time.monotonic())
                chunk = pcm[self._pos : self._pos + need]
                self._pos += need
                if self._pos >= len(pcm):
                    ended.set_result(time.monotonic())
                    self._clips.pop(0)
                    self._pos = 0
            chunk = chunk.ljust(need, b"\x00")
            frame = AudioFrame(format="s16", layout="mono", samples=frame_samples)
            frame.planes[0].update(chunk)
            frame.pts = self._sent
            frame.sample_rate = FIXTURE_RATE
            frame.time_base = fractions.Fraction(1, FIXTURE_RATE)
            self._sent += frame_samples
            return frame

    return FixtureTrack


class Scenario:
    def __init__(self, fixtures: dict[str, bytes]) -> None:
        self.fixtures = fixtures
        self.events: Events = []
        self.user_turn_starts: list[float] = []
        self.interrupt_onset: float | None = None
        self.uhhuh_window: tuple[float, float] | None = None
        self.latency_window: tuple[float, float] | None = None
        self.track = None

    def on_message(self, message) -> None:
        if not isinstance(message, str):
            return
        try:
            msg = json.loads(message)
        except ValueError:
            return
        if isinstance(msg, dict) and "type" in msg:
            self.events.append((time.monotonic(), msg))

    async def wait(self, pred, timeout: float, what: str) -> None:
        deadline = time.monotonic() + timeout
        while not pred():
            if time.monotonic() > deadline:
                raise ScenarioError(f"timed out waiting for {what}")
            await asyncio.sleep(0.02)

    def _state(self) -> str | None:
        return _state_at(self.events, float("inf"))

    def _state_since(self, t: float, state: str) -> bool:
        return any(at >= t and e.get("type") == "state" and e.get("state") == state for at, e in self.events)

    async def play(self, name: str, turn: bool = True) -> tuple[float, float]:
        started, ended = self.track.play(self.fixtures[name])
        t0 = await started
        if turn:
            self.user_turn_starts.append(t0 + FIXTURE_ONSET_S)
        return t0, await ended

    async def bot_done(self, after: float, timeout: float = 30.0, quiet_s: float = 1.5) -> None:
        """Wait for the reply to a turn to start and then finish (quiet for `quiet_s`)."""
        await self.wait(lambda: self._state_since(after, "speaking"), timeout, "the bot to reply")
        await self.wait(self._quiet(quiet_s), timeout * 2, "the bot to finish")

    def _quiet(self, quiet_s: float):
        def pred() -> bool:
            states = [(at, e["state"]) for at, e in self.events if e.get("type") == "state"]
            return bool(states) and states[-1][1] == "listening" and time.monotonic() - states[-1][0] >= quiet_s

        return pred

    async def run(self) -> None:
        # 1. Latency: plain turns at the default config.
        phase_start = time.monotonic()
        for name in TURN_FIXTURES:
            t0, _ = await self.play(name)
            try:
                await self.bot_done(after=t0)
            except ScenarioError as e:
                print(f"e2e: {name}: {e}; continuing", file=sys.stderr)
                await self.wait(self._quiet(1.5), 60, "quiet")
        self.latency_window = (phase_start, time.monotonic())
        # 2. "uh-huh" over the bot: it must keep talking.
        await self._over_long_reply("uhhuh")
        # 3. A real interruption over the bot.
        await self._over_long_reply("interrupt")
        # 4. A long turn with pauses, for backchannels.
        t0, _ = await self.play("ramble")
        try:
            await self.bot_done(after=t0)
        except ScenarioError as e:
            print(f"e2e: ramble: {e}", file=sys.stderr)

    async def _over_long_reply(self, name: str) -> None:
        t0, _ = await self.play("story_request")
        try:
            await self.wait(lambda: self._state_since(t0, "speaking"), 30, "the story to start")
        except ScenarioError as e:
            print(f"e2e: {name}: {e}", file=sys.stderr)
            return
        await asyncio.sleep(1.5)  # well into the reply
        start, end = await self.play(name, turn=(name == "interrupt"))
        if name == "uhhuh":
            self.uhhuh_window = (start, end + 1.5)
        else:
            self.interrupt_onset = start + FIXTURE_ONSET_S
        await asyncio.sleep(1.5)
        try:
            await self.wait(self._quiet(1.5), 90, "the bot to finish")
        except ScenarioError as e:
            print(f"e2e: {name}: {e}", file=sys.stderr)


def results_for(sc: Scenario, history_roles: Mapping[str, int] | None) -> list[CheckResult]:
    results = [check_latency(sc.events, window=sc.latency_window)]
    if sc.interrupt_onset is None:
        results.append(CheckResult("interruption", False, "scenario did not run", f"<= {INTERRUPT_MAX_MS}ms"))
    else:
        results.append(check_interruption(sc.events, sc.interrupt_onset))
    if sc.uhhuh_window is None:
        results.append(CheckResult("uh-huh", False, "scenario did not run", "no interrupted"))
    else:
        results.append(check_uhhuh(sc.events, *sc.uhhuh_window))
    results.append(check_backchannel_timing(sc.events, sc.user_turn_starts))
    results.append(check_history(history_roles, sc.events))
    return results


async def run_e2e(base_url: str, api_key: str, fixtures: dict[str, bytes]) -> list[CheckResult]:
    import httpx
    from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import MediaStreamError

    auth = {"Authorization": f"Bearer {api_key}"}
    sc = Scenario(fixtures)
    async with httpx.AsyncClient(base_url=base_url, timeout=15) as http:
        try:
            r = await http.post("/sessions", json={}, headers=auth)  # default config
        except httpx.ConnectError:
            print(f"e2e: no engine at {base_url}; start it with `DEBUG=1 uv run voice-engine`", file=sys.stderr)
            raise SystemExit(2) from None
        if r.status_code != 201:
            print(f"e2e: POST /sessions returned {r.status_code}", file=sys.stderr)
            raise SystemExit(2)
        sid, token = r.json()["session_id"], r.json()["client_token"]

        # No STUN: the engine runs on this machine and gathers host candidates only.
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        sc.track = _fixture_track_class()()
        pc.addTrack(sc.track)
        channel = pc.createDataChannel("events")
        channel.on("message", sc.on_message)
        sinks: list[asyncio.Task] = []

        @pc.on("track")
        def _on_track(track) -> None:
            async def drain() -> None:  # the bot's audio; read so it does not pile up
                try:
                    while True:
                        await track.recv()
                except MediaStreamError:
                    pass

            sinks.append(asyncio.create_task(drain()))

        async def ping() -> None:  # SmallWebRTC treats the peer as gone without pings
            while True:
                if channel.readyState == "open":
                    channel.send(f"ping: {int(time.time() * 1000)}")
                await asyncio.sleep(1.0)

        pinger = asyncio.create_task(ping())
        history_roles = None
        try:
            await pc.setLocalDescription(await pc.createOffer())
            offer = {"client_token": token, "sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            r = await http.post("/webrtc/offer", json=offer)
            if r.status_code != 200:
                print(f"e2e: POST /webrtc/offer returned {r.status_code}", file=sys.stderr)
                raise SystemExit(2)
            await pc.setRemoteDescription(RTCSessionDescription(sdp=r.json()["sdp"], type=r.json()["type"]))
            await sc.wait(lambda: channel.readyState == "open", 20, "the data channel")
            await sc.wait(lambda: sc._state() == "listening", 20, "the engine to listen")
            await sc.run()
            got = (await http.get(f"/sessions/{sid}", headers=auth)).json()
            history_roles = got.get("history_roles")
        except ScenarioError as e:
            print(f"e2e: {e}", file=sys.stderr)
        finally:
            pinger.cancel()
            await http.delete(f"/sessions/{sid}", headers=auth)
            await pc.close()
            for t in sinks:
                t.cancel()
    return results_for(sc, history_roles)


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    env = load_env()
    default_url = f"http://{env.get('HOST', '127.0.0.1')}:{env.get('PORT', '7860')}"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=default_url, help=f"engine base URL (default {default_url})")
    p.add_argument("--make-fixtures", action="store_true", help="render the fixtures with Cartesia and exit")
    p.add_argument("--voice-id", help="child-like Cartesia voice id for --make-fixtures")
    args = p.parse_args(argv)

    if args.make_fixtures:
        require_keys(env, FIXTURE_KEYS)
        if not args.voice_id:
            p.error("--make-fixtures needs --voice-id (a child-like Cartesia voice)")
        asyncio.run(make_fixtures(env["CARTESIA_API_KEY"], args.voice_id))
        return 0

    require_keys(env, RUN_KEYS)
    fixtures = load_fixtures()
    results = asyncio.run(run_e2e(args.url, env["ENGINE_API_KEY"], fixtures))
    print(format_table(results))
    return exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
