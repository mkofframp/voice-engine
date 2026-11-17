"""Classify user speech heard while the bot is talking."""

from __future__ import annotations

import re
from typing import Literal

# Backchannel words as people write them; a transcript is compared after
# `_canon`, so the many spellings STT produces ("mm-hmm", "mmhmm", "Mhm.", "uh huh")
# all match.
BACKCHANNEL_WORDS: frozenset[str] = frozenset(
    {"mm", "mhm", "mm-hm", "mm-hmm", "hmm", "uh-huh", "yeah", "yep", "yup", "ok", "okay",
     "right", "sure", "wow", "oh", "ooh", "ah", "aha", "ha", "haha"}
)

_MAX_BACKCHANNEL_TOKENS = 3

# Speech over the bot with no transcript yet becomes an interruption once it has
# lasted this long (spec §6.4: "speech that lasts >= a threshold (~300 ms, tunable)").
# Set below 300 so the stop can land within 300 ms of speech onset (spec §7a); the
# STT's first interim usually comes later than this, so a backchannel longer than
# this ("uh-huuuh") stops the bot unless its interim has already arrived. Tune here.
INTERRUPT_SPEECH_MS = 250
_LAUGH = re.compile(r"^(?:ha|he){2,}h?$")  # "haha", "hehehe"; not "he"


def _canon(word: str) -> str:
    """One spelling per backchannel: lower case, no hyphens or spaces, runs of a
    letter other than h collapsed ("mmhmm" -> "mhm", "hmmm" -> "hm", "ooh" -> "oh"),
    trailing h's collapsed. A doubled h inside a word is kept, so "uh-huh" ("uhhuh")
    and "uh-uh" ("uhuh", a no) stay apart."""
    w = re.sub(r"[\s\-–—]+", "", word.lower())
    w = re.sub(r"([a-gi-z])\1+", r"\1", w)
    return re.sub(r"h+$", "h", w)


_CANON_WORDS: frozenset[str] = frozenset(_canon(w) for w in BACKCHANNEL_WORDS)


def is_backchannel_word(word: str) -> bool:
    c = _canon(word)
    return c in _CANON_WORDS or bool(_LAUGH.match(c))


def _tokens(transcript: str) -> list[str]:
    out = []
    for raw in transcript.lower().split():
        tok = raw.strip(".,!?;:\"'()[]…-–—")
        if tok:
            out.append(tok)
    return out


def backchannel_units(transcript: str) -> list[str]:
    """The transcript's words, with a pair that spells one backchannel when written
    together ("uh huh", "mm hmm") joined into one."""
    tokens = _tokens(transcript)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and _canon(tokens[i] + tokens[i + 1]) in _CANON_WORDS:
            out.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def classify_user_speech(
    transcript: str,
    speech_ms: int,
    *,
    min_speech_ms: int = 200,
    no_text_ms: int = INTERRUPT_SPEECH_MS,
) -> Literal["undecided", "backchannel", "interruption"]:
    if speech_ms < min_speech_ms:
        return "undecided"
    tokens = backchannel_units(transcript)
    if not tokens:
        return "interruption" if speech_ms >= no_text_ms else "undecided"
    if len(tokens) <= _MAX_BACKCHANNEL_TOKENS and all(is_backchannel_word(t) for t in tokens):
        return "backchannel"
    return "interruption"
