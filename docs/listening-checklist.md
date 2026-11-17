# Owner listening checklist

Internal doc for the owner, not shipped copy. This is the human half of the Definition of
done (spec §7a). Run it after `scripts/e2e.py` passes.

## Setup

1. Fill in `engine/.env` (see `engine/README.md`).
2. `cd client && npm run build:browser` (once), then `cd engine && uv run voice-engine`.
3. Open `http://127.0.0.1:7860/dev`, keep the default config, and talk for about 5 minutes
   the way a 5-8 year old would: short answers, long rambles with pauses, questions, and
   cutting in.

## Checklist

Tick each item, or note what felt wrong and when.

- [ ] **Emotion varies and fits.** The voice changes with the content (happy, excited,
  gentle, sad, whisper and so on) and the change suits what is being said. It is not flat,
  and it is not over the top. Try `emotion_intensity` at 0, 70 and 100.
- [ ] **Backchannels land in pauses and feel natural.** Short "mm-hm" / "yeah" / "oh!"
  sounds come only when you pause mid-thought, never over your words, not in the first
  couple of seconds, and not too often. Try a long story with pauses. With
  `backchannel_frequency: off` there are none.
- [ ] **Interruptions feel immediate, and the bot acknowledges being cut off.** Start talking
  while the bot speaks: it stops at once, and its next reply picks up from what it actually
  got to say, not from the unheard rest.
- [ ] **"Uh-huh" doesn't stop it.** Say "uh-huh", "yeah" or "okay" while the bot speaks: it
  keeps talking.

## Also worth a listen

- Turn-taking: the bot waits through mid-sentence pauses and does not jump in early. If it
  does, try `turn_eagerness: patient`; if replies feel slow, try `quick`.
- Latency: replies start about a second after you stop. The dev page shows `turn_metrics`.

## Result

Date, config used, pass or fail per item, and notes. If something fails, say what you heard
and roughly when, so it can be reproduced.
