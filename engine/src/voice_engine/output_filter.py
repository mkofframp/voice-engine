"""Blocklist filter for bot output sentences."""

import re


def filter_sentence(
    text: str, blocklist: list[str], redirect_line: str
) -> tuple[str, bool]:
    """Return (text_to_speak, was_blocked)."""
    for term in blocklist:
        term = term.strip()
        if not term:
            continue
        if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE):
            return redirect_line, True
    return text, False
