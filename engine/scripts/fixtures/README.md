# e2e fixtures

Synthetic child-like speech for `scripts/e2e.py`. No real child recordings. The WAV files
are generated, not committed (this README is the only tracked file here):

    uv run python scripts/e2e.py --make-fixtures --voice-id <child-like Cartesia voice id>

This needs `CARTESIA_API_KEY` in `engine/.env`. Pick a child-like voice from the Cartesia
voice library and pass its id; the script does not choose one for you.

Format: 16-bit mono PCM WAV, 16 kHz. Rendered with the same Cartesia model and API version
as the engine (`voice_engine.pipeline.prerender`).

## Speech onset

Every fixture is trimmed of leading silence (first sample with |amplitude| > 500) and then
padded so that speech starts exactly **0.50 s** into the file (`FIXTURE_ONSET_S` in
`e2e.py`). The e2e script uses this offset as the user's speech onset: the interruption
check measures from playback start + 0.50 s to the `interrupted` event, and user turns
start at that point for the backchannel check.

| file | text | onset |
|---|---|---|
| turn_01.wav | Hi! My name is Sam and I am six. | 0.50 s |
| turn_02.wav | Do you like dinosaurs? | 0.50 s |
| turn_03.wav | My favorite color is green. | 0.50 s |
| turn_04.wav | I have a cat called Pickle. | 0.50 s |
| turn_05.wav | What do bunnies eat? | 0.50 s |
| turn_06.wav | I went to the park today. | 0.50 s |
| turn_07.wav | Can you count to five? | 0.50 s |
| turn_08.wav | Why is the sky blue? | 0.50 s |
| turn_09.wav | I like pizza with lots of cheese. | 0.50 s |
| turn_10.wav | My tooth is wobbly! | 0.50 s |
| turn_11.wav | What is your favorite animal? | 0.50 s |
| turn_12.wav | I built a big tower with blocks. | 0.50 s |
| story_request.wav | Can you tell me a long story about a friendly dragon? | 0.50 s |
| interrupt.wav | Wait, wait! I want to tell you something! | 0.50 s |
| uhhuh.wav | Uh-huh. | 0.50 s |
| ramble.wav | five clauses joined by 0.6 s pauses (see `RAMBLE_SEGMENTS`) | 0.50 s |

`ramble.wav` gives the engine mid-turn pauses inside the backchannel window
(250-700 ms), so backchannels have a chance to play. Each clause is trimmed of silence
before joining, so the pauses are exactly 0.6 s.

If you change a text in `e2e.py`, regenerate the fixtures and update this table.
