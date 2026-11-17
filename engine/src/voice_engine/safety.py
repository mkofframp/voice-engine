"""Safety classifier for a child's transcribed speech (harm, self-harm, personal info)."""

import logging
from dataclasses import dataclass
from typing import Literal

import anthropic

logger = logging.getLogger(__name__)

Category = Literal["harm", "self_harm", "personal_info"]

EXCERPT_MAX = 120

CLASSIFICATION_PROMPT = """\
You screen what a child aged 5 to 8 just said to a voice companion. The text is a \
speech transcript and may be messy. Always answer by calling the `report` tool with \
exactly one category:

- harm: the child says someone is hurting them or another person, or describes abuse, \
violence against them, neglect, or being scared of someone at home or elsewhere. \
Ordinary play-fighting, cartoons, games and stories do not count.
- self_harm: the child talks about hurting themselves, wanting to die, disappearing, \
or not wanting to be alive. Ordinary bumps and scrapes do not count.
- personal_info: the child shares identifying details that should stay private: \
home address, phone number, full name with surname, school name, email, passwords, \
or exact whereabouts. A child saying only their own first name or age is NOT \
personal_info.
- none: anything else, including normal chatter, imagination and stories.

If several apply, choose the most serious (self_harm, then harm, then personal_info). \
When unsure between a category and none, prefer none unless a child could be at risk. \
Include in `excerpt` the short phrase from the transcript that triggered the flag."""

_REPORT_TOOL = {
    "name": "report",
    "description": "Report the safety classification of the child's utterance.",
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["none", "harm", "self_harm", "personal_info"],
            },
            "excerpt": {
                "type": "string",
                "description": "Short phrase from the transcript that triggered the flag.",
            },
        },
        "required": ["category"],
    },
}


@dataclass
class SafetyFlag:
    category: Category
    excerpt: str


class SafetyClassifier:
    def __init__(self, client: anthropic.AsyncAnthropic, model: str = "claude-haiku-5-5"):
        self._client = client
        self._model = model

    async def classify(self, text: str) -> SafetyFlag | None:
        """Return a flag, or None if safe. Never raises; errors are logged (type only)."""
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=256,
                system=CLASSIFICATION_PROMPT,
                tools=[_REPORT_TOOL],
                tool_choice={"type": "tool", "name": "report"},
                messages=[{"role": "user", "content": text}],
            )
            block = next(b for b in response.content if b.type == "tool_use")
            data = block.input
            category = data["category"]
            if category == "none":
                return None
            if category not in ("harm", "self_harm", "personal_info"):
                raise ValueError("unknown category")
            excerpt = data.get("excerpt")
            if not isinstance(excerpt, str) or not excerpt:
                excerpt = text
            return SafetyFlag(category=category, excerpt=excerpt[:EXCERPT_MAX])
        except Exception as exc:  # noqa: BLE001 - classifier must never raise
            logger.warning("safety classification failed: %s", type(exc).__name__)
            return None
