"""Emotion tags and a streaming parser for `[tag] Sentence.` LLM output."""

import re
from dataclasses import dataclass

EMOTION_TAGS = (
    "neutral",
    "happy",
    "excited",
    "giggle",
    "curious",
    "surprised",
    "gentle",
    "sad",
    "whisper",
    "proud",
)

EMOTION_PROMPT = (
    "Start every sentence of your reply with exactly one emotion tag in square "
    "brackets, chosen from: "
    + ", ".join(f"[{t}]" for t in EMOTION_TAGS)
    + ". Example: [gentle] That sounds hard. [curious] What happened next? "
    "Vary the tags naturally to match what you say. Use [neutral] and [gentle] "
    "most of the time, and use [excited] only rarely. Never mention the tags."
)

_TAG_RE = re.compile(r"^\s*\[([^\]\[]*)\]\s*")


@dataclass(frozen=True)
class Sentence:
    emotion: str
    text: str


class TagParser:
    def __init__(self) -> None:
        self._buf = ""
        self._emotion = "neutral"

    def reset(self) -> None:
        self._buf = ""
        self._emotion = "neutral"

    def feed(self, chunk: str) -> list[Sentence]:
        self._buf += chunk
        out: list[Sentence] = []
        while (end := self._find_end()) is not None:
            raw, self._buf = self._buf[:end], self._buf[end:]
            s = self._make(raw)
            if s:
                out.append(s)
        return out

    def flush(self) -> list[Sentence]:
        raw, self._buf = self._buf, ""
        s = self._make(raw)
        return [s] if s else []

    def _find_end(self) -> int | None:
        buf = self._buf
        for i, ch in enumerate(buf):
            if ch == "\n":
                return i + 1
            if ch in ".!?" and i + 1 < len(buf) and buf[i + 1].isspace():
                return i + 1
        return None

    def _make(self, raw: str) -> Sentence | None:
        m = _TAG_RE.match(raw)
        if m:
            self._emotion = m.group(1).strip().lower() or self._emotion
            raw = raw[m.end():]
        text = raw.strip()
        return Sentence(self._emotion, text) if text else None
