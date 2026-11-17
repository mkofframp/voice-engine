"""Trim an interrupted bot reply to what the child actually heard."""

from __future__ import annotations

from dataclasses import dataclass, field

CUT_MARKER = " [interrupted by the user here]"


@dataclass
class PlayedWord:
    word: str
    start_ms: int


@dataclass
class SpokenSentence:
    text: str
    start_ms: int
    words: list[PlayedWord] = field(default_factory=list)


def truncate_reply(sentences: list[SpokenSentence], played_ms: int) -> str:
    parts: list[str] = []
    for i, sent in enumerate(sentences):
        if sent.words:
            parts.extend(w.word for w in sent.words if w.start_ms < played_ms)
        elif i + 1 < len(sentences) and played_ms >= sentences[i + 1].start_ms:
            parts.append(sent.text)
    if not parts:
        return CUT_MARKER.strip()
    return " ".join(parts) + CUT_MARKER
