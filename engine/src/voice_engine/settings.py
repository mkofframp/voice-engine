from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ENGINE_DIR = Path(__file__).resolve().parents[2]

# Nothing below INFO: Pipecat's DEBUG lines carry transcripts and replies.
LOG_LEVELS = ("INFO", "WARNING", "ERROR")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENGINE_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    deepgram_api_key: str
    anthropic_api_key: str
    cartesia_api_key: str
    engine_api_key: str
    host: str = "127.0.0.1"
    port: int = 7860
    client_token_ttl_s: int = 120
    allowed_origins: list[str] = ["http://localhost:5173"]
    log_level: Literal["INFO", "WARNING", "ERROR"] = "INFO"
