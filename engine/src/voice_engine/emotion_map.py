"""Map emotion tags and intensity to Cartesia TTS controls."""

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from voice_engine.emotions import EMOTION_TAGS

log = logging.getLogger(__name__)

_SPEED_RANGE = (0.6, 1.5)
_VOLUME_RANGE = (0.5, 2.0)
_PREFIX_MIN_INTENSITY = 50


@dataclass
class TTSControls:
    emotion: list[str] = field(default_factory=list)
    speed: float = 1.0
    volume: float = 1.0
    prefix_markup: str = ""


def _load() -> dict[str, dict]:
    path = Path(__file__).parent / "data" / "emotion_map.toml"
    with path.open("rb") as f:
        return tomllib.load(f)


_MAP = _load()
_missing = [t for t in EMOTION_TAGS if t not in _MAP]
if _missing:
    raise RuntimeError(f"emotion_map.toml lacks tags: {_missing}")


def _clamp(v: float, lo_hi: tuple[float, float]) -> float:
    return max(lo_hi[0], min(lo_hi[1], v))


def map_emotion(tag: str, intensity: int, base_speed: float) -> TTSControls:
    entry = _MAP.get(tag)
    if entry is None:
        log.warning("unknown emotion tag, using neutral")  # the tag is model text: not logged
        entry, intensity = _MAP["neutral"], 0
    k = max(0, min(100, intensity)) / 100
    speed = base_speed * (1 + (entry["speed_factor"] - 1) * k)
    volume = 1.0 + (entry["volume"] - 1.0) * k
    emotion = [entry["emotion"]] if k > 0 else [_MAP["neutral"]["emotion"]]
    prefix = entry["prefix"] if intensity >= _PREFIX_MIN_INTENSITY else ""
    return TTSControls(
        emotion=emotion,
        speed=_clamp(speed, _SPEED_RANGE),
        volume=_clamp(volume, _VOLUME_RANGE),
        prefix_markup=prefix,
    )
