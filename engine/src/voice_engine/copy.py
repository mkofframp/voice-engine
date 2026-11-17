"""Shipped fixed text (written by ChatGPT), loaded once from copy.json."""
import json
from pathlib import Path
from typing import Literal

Mood = Literal["neutral", "interested", "amused", "sympathetic"]

_COPY: dict = json.loads((Path(__file__).parent / "copy.json").read_text(encoding="utf-8"))


def get(key: str) -> str:
    node = _COPY
    for part in key.split("."):
        node = node[part]
    if not isinstance(node, str):
        raise KeyError(key)
    return node


def backchannel_phrases() -> dict[Mood, list[str]]:
    return {m: list(p) for m, p in _COPY["backchannel"].items()}  # type: ignore[misc]
