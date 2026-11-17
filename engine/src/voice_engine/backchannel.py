"""Active-listening backchannel: when to play, which mood, which clip."""

import random
import re

from voice_engine.copy import Mood

MIN_GAP_S = {"low": 8.0, "normal": 5.0, "high": 3.5}
WARMUP_S = 2.0
PAUSE_WINDOW_MS = (250, 700)
_FREQUENCIES = ("off", "low", "normal", "high")


def _check(f: str) -> str:
    if f not in _FREQUENCIES:
        raise ValueError(f"unknown frequency: {f!r}")
    return f


class BackchannelScheduler:
    def __init__(self, frequency: str) -> None:
        self._freq = _check(frequency)
        self._turn_start: float | None = None
        self._last_played: float | None = None

    def set_frequency(self, f: str) -> None:
        self._freq = _check(f)

    def on_user_turn_start(self, t: float) -> None:
        self._turn_start = t

    def on_played(self, t: float) -> None:
        self._last_played = t

    def should_play(
        self, t: float, pause_ms: int, turn_complete: bool, bot_speaking: bool
    ) -> bool:
        if self._freq == "off" or bot_speaking or turn_complete:
            return False
        if self._turn_start is None:
            return False
        lo, hi = PAUSE_WINDOW_MS
        if not (lo <= pause_ms <= hi):
            return False
        if t - self._turn_start < WARMUP_S:
            return False
        if self._last_played is not None and t - self._last_played < MIN_GAP_S[self._freq]:
            return False
        return True


def _words(words: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE)


# Word forms, not stems: "hurtful" is not a hurt, but "hurts" and "hurting" are.
_SYMPATHETIC = _words(
    (
        r"hurt(?:s|ing)?", r"sad(?:der|dest)?", r"cr(?:y|ies|ied|ying)",
        r"scar(?:ed|y|es|ier|iest)", r"afraid", r"sick", r"lost", r"broke(?:n)?",
    )
)
_AMUSED = _words(
    (
        r"funn(?:y|ier|iest)", r"sill(?:y|ier|iest)", r"laugh(?:s|ed|ing)?", r"(?:ha){2,}h?",
        r"jok(?:e|es|ed|ing)", r"giggl(?:e|es|ed|ing)",
    )
)
_INTERESTED = _words((r"guess\s+what", r"you\s+know\s+what"))


def pick_mood(interim_text: str) -> Mood:
    if _SYMPATHETIC.search(interim_text):
        return "sympathetic"
    if _AMUSED.search(interim_text):
        return "amused"
    if "?" in interim_text or _INTERESTED.search(interim_text):
        return "interested"
    return "neutral"


class ClipBank:
    def __init__(self, clips: dict[Mood, list[bytes]], rng: random.Random | None = None) -> None:
        self._clips = clips
        self._rng = rng or random.Random()
        self._last: dict[Mood, int] = {}

    def pick(self, mood: Mood) -> tuple[int, bytes] | None:
        key: Mood = mood
        pool = self._clips.get(key) or []
        if not pool:
            key = "neutral"
            pool = self._clips.get(key) or []
        if not pool:
            return None
        choices = [i for i in range(len(pool)) if i != self._last.get(key)]
        if not choices:
            choices = [0]
        i = self._rng.choice(choices)
        self._last[key] = i
        return i, pool[i]
