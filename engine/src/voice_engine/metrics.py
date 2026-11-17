"""Per-turn latency metrics and per-session summary. Times are seconds, passed in."""

import statistics

STAGES = (
    "user_last_word",
    "end_of_turn",
    "final_transcript",
    "llm_first_token",
    "tts_first_audio",
    "audio_out",
)


class TurnMetrics:
    def __init__(self) -> None:
        self._marks: dict[str, float] = {}

    def mark(self, stage: str, t: float) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage: {stage!r}")
        self._marks.setdefault(stage, t)

    def as_event(self) -> dict:
        base = self._marks.get("user_last_word")
        event: dict = {}
        for stage in STAGES:
            at = self._marks.get(stage)
            if base is None or at is None:
                event[stage] = None
            else:
                event[stage] = round((at - base) * 1000)
        if base is None or "audio_out" not in self._marks:
            event["total_ms"] = None
        else:
            event["total_ms"] = round((self._marks["audio_out"] - base) * 1000)
        return event


class SessionMetrics:
    def __init__(self) -> None:
        self._totals: list[int] = []

    def add(self, turn: TurnMetrics) -> None:
        total = turn.as_event()["total_ms"]
        if total is not None:
            self._totals.append(total)

    def summary(self) -> dict:
        totals = self._totals
        if not totals:
            return {"turns": 0, "median_ms": None, "p90_ms": None}
        if len(totals) == 1:
            only = totals[0]
            return {"turns": 1, "median_ms": only, "p90_ms": only}
        return {
            "turns": len(totals),
            "median_ms": round(statistics.median(totals)),
            "p90_ms": round(
                statistics.quantiles(totals, n=10, method="inclusive")[8]
            ),
        }
