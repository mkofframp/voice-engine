"""Debug-only counts over the LLM context (for DEBUG=1 reads and scripts/e2e.py).

Returns counts only, never message text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from voice_engine import copy
from voice_engine.interruption import BACKCHANNEL_WORDS

_TRIM = " \t\r\n.,!?;:\"'()[]…-–—"
_TAG = re.compile(r"^\s*\[[a-z_]+\]\s*")  # a leading emotion tag such as "[happy]"


def _norm(text: str) -> str:
    return _TAG.sub("", text.strip().lower()).strip(_TRIM)


_BACKCHANNEL_LIKE: frozenset[str] = frozenset(
    {_norm(p) for phrases in copy.backchannel_phrases().values() for p in phrases}
    | {_norm(w) for w in BACKCHANNEL_WORDS}
)


def is_backchannel_like(text: str) -> bool:
    """True if the whole text, normalised, is a backchannel phrase or word."""
    return _norm(text) in _BACKCHANNEL_LIKE


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p["text"] for p in content
            if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str)
        )
    return ""


def count_history(messages: Iterable[Any]) -> dict[str, int]:
    """Messages per role, plus `backchannel_like`: non-system messages whose text is
    exactly a backchannel phrase (copy.json, all moods) or word (BACKCHANNEL_WORDS)."""
    counts: dict[str, int] = {}
    leaked = 0
    for msg in messages:
        inner = getattr(msg, "message", msg)  # LLMSpecificMessage wraps a raw message
        role = inner.get("role") if isinstance(inner, dict) else None
        key = role if isinstance(role, str) else "other"
        counts[key] = counts.get(key, 0) + 1
        if key != "system" and isinstance(inner, dict) and is_backchannel_like(_text(inner.get("content"))):
            leaked += 1
    counts["backchannel_like"] = leaked
    return counts
