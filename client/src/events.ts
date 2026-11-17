export type TurnStage =
  | "user_last_word"
  | "end_of_turn"
  | "final_transcript"
  | "llm_first_token"
  | "tts_first_audio"
  | "audio_out";

/** Events the engine sends over the WebRTC data channel (spec section 6.7). */
export type EngineEvent =
  | { type: "state"; state: "listening" | "thinking" | "speaking" }
  | { type: "user_transcript"; text: string; final: boolean }
  | { type: "bot_sentence"; emotion: string; text: string }
  | { type: "backchannel"; mood: string }
  | { type: "interrupted"; spoken_text: string }
  | ({
      type: "turn_metrics";
      /** Milliseconds after the child's last word; null when the stage did not happen. */
    } & Record<TurnStage | "total_ms", number | null>)
  | { type: "safety_flag"; category: string; excerpt: string }
  | { type: "session_ending"; reason?: string }
  | { type: "error"; code: string; message: string };

export type EngineEventType = EngineEvent["type"];

export const EVENT_TYPES: readonly EngineEventType[] = [
  "state",
  "user_transcript",
  "bot_sentence",
  "backchannel",
  "interrupted",
  "turn_metrics",
  "safety_flag",
  "session_ending",
  "error",
];

/** Host commands; the same shapes as POST /sessions/{id}/commands. */
export type Command =
  | { type: "say"; text: string }
  | { type: "add_context"; text: string }
  | { type: "end"; graceful?: boolean }
  | { type: "mute_mic" }
  | { type: "unmute_mic" };
