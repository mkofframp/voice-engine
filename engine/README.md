# voice-engine

Real-time voice conversation engine (Python, Pipecat). Run with `uv run voice-engine`;
it serves the session API on `HOST:PORT` from `engine/.env`.

## Turn-taking: `turn_eagerness`

End of turn is decided by Silero VAD plus Smart Turn v3. `turn_eagerness` sets two
silences (seconds), fixed when the session's pipeline is built:

| turn_eagerness | VAD stop_secs | Smart Turn stop_secs |
|---|---|---|
| patient | 0.5 | 4.0 |
| normal  | 0.3 | 3.0 |
| quick   | 0.2 | 2.0 |

VAD stop_secs is the pause before Smart Turn is asked whether the child has finished;
Smart Turn stop_secs is the pause after which the turn ends regardless. Children pause
mid-thought often, so `patient` waits longest. Source: `EAGERNESS` in
`src/voice_engine/pipeline/bot.py`. A patched `turn_eagerness` applies from the next session.
