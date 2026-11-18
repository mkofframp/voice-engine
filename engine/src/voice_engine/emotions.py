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
    + ". Example: [curious] Ooh, wait, a real dragon? [giggle] That's so silly! "
    "Match the tag to the feeling, like a warm grown-up who is really enjoying "
    "the chat: lean on [happy], [curious], [gentle] and [giggle], use [excited] "
    "and [surprised] when something is genuinely fun, and [neutral] only for "
    "plain facts. Never mention the tags.\n"
    "Your words are spoken aloud by a voice, never read, so write the way people "
    "actually talk, not the way they write: use contractions, start with a quick "
    "reaction (oh, ooh, hmm, wow, aw, okay so), allow a natural filler or a "
    "trailing ... now and then, react to what the child just said before adding "
    "anything new, and keep most sentences under about 12 words. When something "
    "is truly funny, you may put [laughter] inside a sentence for a real laugh. "
    "Avoid formal, textbook phrasing and avoid starting with \"Great question\"."
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
