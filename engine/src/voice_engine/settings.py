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
    # Hosted demo: STUN/TURN URLs for the server's WebRTC side (JSON list in env).
    ice_servers: list[str] = []
    # Hosted demo: Cloudflare TURN relay key (both set = relay on), for hosts without direct UDP.
    turn_key_id: str | None = None
    turn_api_token: str | None = None
    # Hosted demo: at most this many live conversations at once (None = no cap).
    max_connected_sessions: int | None = None
