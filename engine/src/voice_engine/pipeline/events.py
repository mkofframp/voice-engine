"""Events sent to the browser over the WebRTC data channel (spec §6.7)."""

from pipecat.frames.frames import OutputTransportMessageUrgentFrame

EVENT_TYPES = (
    "state",
    "user_transcript",
    "bot_sentence",
    "backchannel",
    "interrupted",
    "turn_metrics",
    "safety_flag",
    "session_ending",
    "error",
)


def event_frame(type: str, **payload) -> OutputTransportMessageUrgentFrame:
    if type not in EVENT_TYPES:
        raise ValueError(f"unknown event type: {type!r}")
    return OutputTransportMessageUrgentFrame(message={"type": type, **payload})
