from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator

DATA_DIR = Path(__file__).parent / "data"

DEFAULT_SAFETY_PROMPT = (DATA_DIR / "default_safety_prompt.md").read_text(encoding="utf-8").strip()
DEFAULT_BLOCKLIST = [
    line.strip()
    for line in (DATA_DIR / "default_blocklist.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
# Default Cartesia voice; host may override per session.
DEFAULT_VOICE_ID = "9779d1bc-9070-4a31-a3e0-7cc034da01ca"  # owner's cloned Cartesia voice "Motie" (private)


class SessionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persona_prompt: str = ""
    safety_prompt: str = DEFAULT_SAFETY_PROMPT
    output_blocklist: list[str] = Field(default_factory=lambda: list(DEFAULT_BLOCKLIST))
    voice_id: str = DEFAULT_VOICE_ID
    speaking_speed: float = Field(1.0, ge=0.6, le=1.5)
    energy: float = Field(0.5, ge=0, le=1)
    emotion_intensity: int = Field(70, ge=0, le=100)
    response_length: Literal["short", "medium", "long"] = "short"
    turn_eagerness: Literal["patient", "normal", "quick"] = "normal"
    backchannel_frequency: Literal["off", "low", "normal", "high"] = "normal"
    redirect_line: str | None = None
    goodbye_line: str | None = None
    greeting: str | None = None
    max_session_seconds: int = Field(600, ge=30, le=3600)
    webhook_url: AnyHttpUrl | None = None

    @field_validator("safety_prompt")
    @classmethod
    def _blank_safety_prompt_uses_default(cls, v: str) -> str:
        return v if v.strip() else DEFAULT_SAFETY_PROMPT

    def patched(self, patch: dict) -> "SessionConfig":
        """Return a new validated copy with `patch` applied; raises ValidationError."""
        return type(self).model_validate({**self.model_dump(), **patch})
