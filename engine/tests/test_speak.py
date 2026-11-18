import asyncio
import json
from types import SimpleNamespace

import pytest

from voice_engine.config import SessionConfig
from voice_engine.emotions import Sentence
from voice_engine.pipeline.prerender import RenderFailed
from voice_engine.speak import (
    PAUSE_BYTES,
    SentenceTagger,
    Speaker,
    split_sentences,
    wav_header,
)


# ---------- split ----------

def test_split_sentences_keeps_words():
    assert split_sentences("The sun is a star. It is very hot! Why? Because") == [
        "The sun is a star.", "It is very hot!", "Why?", "Because",
    ]


def test_split_sentences_blank():
    assert split_sentences("   \n ") == []


# ---------- tagger (fake Anthropic client) ----------

class FakeMessages:
    def __init__(self, tool_input=None, exc=None, delay=0.0):
        self.tool_input, self.exc, self.delay = tool_input, exc, delay
        self.calls = []

    async def create(self, **kw):
        self.calls.append(kw)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        block = SimpleNamespace(type="tool_use", name="tag_sentences", input=self.tool_input)
        return SimpleNamespace(content=[block])


def tagger(tool_input=None, exc=None, delay=0.0, timeout_s=5.0):
    msgs = FakeMessages(tool_input, exc, delay)
    return SentenceTagger(SimpleNamespace(messages=msgs), timeout_s=timeout_s), msgs


SENTS = ["Dinosaurs were huge.", "Some were as long as three buses!", "Isn't that wild?"]


async def test_exact_mode_keeps_original_words():
    t, msgs = tagger({"items": [
        {"tag": "curious", "text": "Dinos were big."},
        {"tag": "excited"},
        {"tag": "surprised"},
    ]})
    out = await t.tag(SENTS, rewrite=False, cfg=SessionConfig())
    assert out == [
        Sentence("curious", SENTS[0]), Sentence("excited", SENTS[1]), Sentence("surprised", SENTS[2]),
    ]
    call = msgs.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "tag_sentences"}
    assert call["model"] == "claude-haiku-5-5"
    # the sentences are numbered in the request so tags line up by index
    assert '1. Dinosaurs were huge.' in call["messages"][0]["content"]


async def test_exact_mode_unknown_or_missing_tags_fall_back_to_neutral():
    t, _ = tagger({"items": [{"tag": "angry"}, {"tag": "HAPPY"}]})
    out = await t.tag(SENTS, rewrite=False, cfg=SessionConfig())
    assert [s.emotion for s in out] == ["neutral", "happy", "neutral"]
    assert [s.text for s in out] == SENTS


@pytest.mark.parametrize("exc,delay", [(RuntimeError("boom"), 0.0), (None, 0.5)])
async def test_tagger_failure_or_timeout_speaks_plainly(exc, delay):
    t, _ = tagger({"items": []}, exc=exc, delay=delay, timeout_s=0.05)
    out = await t.tag(SENTS, rewrite=True, cfg=SessionConfig())
    assert out == [Sentence("neutral", s) for s in SENTS]


async def test_rewrite_mode_uses_new_text_and_safety_prompt():
    t, msgs = tagger({"items": [
        {"tag": "curious", "text": "Ooh, guess what?"},
        {"tag": "excited", "text": "Some dinosaurs were as long as three buses!"},
        {"tag": "happy", "text": "  "},
    ]})
    cfg = SessionConfig()
    out = await t.tag(SENTS, rewrite=True, cfg=cfg)
    assert out == [
        Sentence("curious", "Ooh, guess what?"),
        Sentence("excited", "Some dinosaurs were as long as three buses!"),
    ]
    assert msgs.calls[0]["system"].startswith(cfg.safety_prompt)


async def test_rewrite_mode_empty_result_falls_back_to_original():
    t, _ = tagger({"items": []})
    out = await t.tag(SENTS, rewrite=True, cfg=SessionConfig())
    assert [s.text for s in out] == SENTS


# ---------- speaker ----------

class FakeTagger:
    def __init__(self, tags):
        self.tags = tags

    async def tag(self, sentences, rewrite, cfg):
        return [Sentence(t, s) for t, s in zip(self.tags, sentences)]


class FakeRender:
    def __init__(self, fail_on=()):
        self.calls, self.fail_on = [], set(fail_on)

    async def __call__(self, text, voice_id, generation_config):
        self.calls.append((text, voice_id, generation_config))
        if len(self.calls) - 1 in self.fail_on:
            raise RenderFailed("status 500")
        return f"<{text}>".encode()


async def collect(agen):
    return [c async for c in agen]


async def test_speaker_maps_emotion_and_streams_in_order():
    render = FakeRender()
    sp = Speaker(FakeTagger(["excited", "gentle"]), render)
    cfg = SessionConfig(voice_id="v1", speaking_speed=1.0, emotion_intensity=100)
    chunks = await collect(sp.stream("Wow, look! It is sleeping.", cfg, rewrite=False))
    assert chunks == [b"<Wow, look!>", PAUSE_BYTES, b"<It is sleeping.>"]
    (_, v1, g1), (_, _, g2) = render.calls
    assert v1 == "v1"
    assert g1["emotion"] == "excited" and g1["speed"] > 1.0 and g1["volume"] > 1.0
    assert g2["emotion"] == "calm" and g2["speed"] < 1.0


async def test_speaker_giggle_gets_laughter_markup():
    render = FakeRender()
    sp = Speaker(FakeTagger(["giggle"]), render)
    await collect(sp.stream("That is so silly.", SessionConfig(emotion_intensity=80), rewrite=False))
    assert render.calls[0][0].startswith("[laughter]")


async def test_speaker_blocklist_replaces_sentence():
    render = FakeRender()
    sp = Speaker(FakeTagger(["happy", "happy"]), render)
    cfg = SessionConfig(output_blocklist=["zebra"], redirect_line="Let's talk about stars.")
    await collect(sp.stream("I like cats. I like a zebra.", cfg, rewrite=False))
    assert render.calls[1][0] == "Let's talk about stars."


async def test_speaker_first_render_failure_raises():
    sp = Speaker(FakeTagger(["happy", "happy"]), FakeRender(fail_on={0}))
    with pytest.raises(RenderFailed):
        await collect(sp.stream("One. Two.", SessionConfig(), rewrite=False))


async def test_speaker_later_failure_skips_sentence():
    sp = Speaker(FakeTagger(["happy"] * 3), FakeRender(fail_on={1}))
    chunks = await collect(sp.stream("One. Two. Three.", SessionConfig(), rewrite=False))
    assert chunks == [b"<One.>", PAUSE_BYTES, b"<Three.>"]


def test_wav_header():
    h = wav_header(48000)
    assert h[:4] == b"RIFF" and h[8:16] == b"WAVEfmt " and len(h) == 44
    assert int.from_bytes(h[24:28], "little") == 24000
    assert int.from_bytes(h[40:44], "little") == 48000


# ---------- HTTP ----------

from fastapi.testclient import TestClient  # noqa: E402

from voice_engine.api import create_app  # noqa: E402
from voice_engine.sessions import SessionStore  # noqa: E402
from voice_engine.settings import Settings  # noqa: E402

KEY = "speak-test-key"


def app_client(render=None, local=False):
    settings = Settings(
        deepgram_api_key="d", anthropic_api_key="a", cartesia_api_key="c", engine_api_key=KEY,
    )
    sp = Speaker(FakeTagger(["happy"] * 10), render or FakeRender())
    app = create_app(settings, SessionStore(token_ttl_s=120), lambda s: None, speaker=sp)
    if local:
        return TestClient(app, base_url="http://localhost:7860", client=("127.0.0.1", 5000))
    return TestClient(app)


def test_speak_requires_key():
    assert app_client().post("/speak", json={"text": "Hi."}).status_code == 401


def test_speak_streams_pcm():
    r = app_client().post(
        "/speak", json={"text": "Hi there. Bye now."},
        headers={"Authorization": f"Bearer {KEY}"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/pcm")
    assert r.headers["x-sample-rate"] == "24000"
    assert r.content == b"<Hi there.>" + PAUSE_BYTES + b"<Bye now.>"


@pytest.mark.parametrize("body", [
    {}, {"text": "   "}, {"text": "x" * 4001}, {"text": "Hi.", "rewrite": "yes"},
    {"text": "Hi.", "config": {"emotion_intensity": 500}},
])
def test_speak_invalid_422(body):
    r = app_client().post("/speak", json=body, headers={"Authorization": f"Bearer {KEY}"})
    assert r.status_code == 422 and r.json()["error"] == "invalid_request"


def test_speak_first_render_failure_502():
    r = app_client(FakeRender(fail_on={0})).post(
        "/speak", json={"text": "Hi."}, headers={"Authorization": f"Bearer {KEY}"},
    )
    assert r.status_code == 502 and r.json()["error"] == "upstream_failed"


def test_dev_speak_loopback_only():
    assert app_client().post("/dev/api/speak", json={"text": "Hi."}).status_code == 404
    r = app_client(local=True).post("/dev/api/speak", json={"text": "Hi."})
    assert r.status_code == 200 and r.content == b"<Hi.>"
