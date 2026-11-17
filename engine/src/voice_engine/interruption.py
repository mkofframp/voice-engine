"""Classify user speech heard while the bot is talking."""

from __future__ import annotations

from typing import Literal

BACKCHANNEL_WORDS: frozenset[str] = frozenset(
    {"mm", "mhm", "mm-hm", "uh-huh", "yeah", "yep", "ok", "okay", "right",
     "sure", "wow", "oh", "aha", "ha", "haha"}
)

_MAX_BACKCHANNEL_TOKENS = 3


def _tokens(transcript: str) -> list[str]:
    out = []
    for raw in transcript.lower().split():
        tok = raw.strip(".,!?;:\"'()[]…-–—")
        if tok:
            out.append(tok)
    return out


def classify_user_speech(
    transcript: str,
    speech_ms: int,
    *,
    min_speech_ms: int = 200,
    no_text_ms: int = 600,
) -> Literal["undecided", "backchannel", "interruption"]:
    if speech_ms < min_speech_ms:
        return "undecided"
    tokens = _tokens(transcript)
    if not tokens:
        return "interruption" if speech_ms >= no_text_ms else "undecided"
    if len(tokens) <= _MAX_BACKCHANNEL_TOKENS and all(t in BACKCHANNEL_WORDS for t in tokens):
        return "backchannel"
    return "interruption"
