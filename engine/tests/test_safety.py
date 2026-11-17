import os
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from voice_engine.safety import SafetyClassifier, SafetyFlag


def _response(**tool_input):
    block = SimpleNamespace(type="tool_use", name="report", input=tool_input)
    return SimpleNamespace(content=[block])


class FakeClient:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return self._response


async def test_flag_returned():
    client = FakeClient(_response(category="harm", excerpt="he hits me"))
    flag = await SafetyClassifier(client).classify("he hits me at home")
    assert flag == SafetyFlag(category="harm", excerpt="he hits me")
    call = client.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "report"}
    assert call["model"] == "claude-haiku-5-5"


async def test_none_category_returns_none():
    client = FakeClient(_response(category="none"))
    assert await SafetyClassifier(client).classify("I like dinosaurs") is None


async def test_api_error_returns_none():
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    client = FakeClient(error=anthropic.APIConnectionError(request=req))
    assert await SafetyClassifier(client).classify("hello") is None


async def test_malformed_response_returns_none():
    client = FakeClient(SimpleNamespace(content=[]))
    assert await SafetyClassifier(client).classify("hello") is None


async def test_excerpt_truncated():
    long_text = "x" * 500
    client = FakeClient(_response(category="self_harm", excerpt="y" * 300))
    flag = await SafetyClassifier(client).classify(long_text)
    assert flag is not None and len(flag.excerpt) == 120
    client = FakeClient(_response(category="self_harm"))
    flag = await SafetyClassifier(client).classify(long_text)
    assert flag is not None and flag.excerpt == "x" * 120


@pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="live API test")
async def test_live_examples():
    classifier = SafetyClassifier(anthropic.AsyncAnthropic())
    positives = [
        "My dad hits me when he is angry.",
        "I want to hurt myself and not be alive.",
        "I live at 12 Oak Street and my phone number is 555 0134.",
    ]
    negatives = [
        "My name is Mia.",
        "I like drawing dragons.",
        "We went to the zoo and saw a lion.",
    ]
    for text in positives:
        assert await classifier.classify(text) is not None, text
    for text in negatives:
        assert await classifier.classify(text) is None, text


async def test_aclose_closes_the_client():
    client = anthropic.AsyncAnthropic(api_key="test-key")
    await SafetyClassifier(client).aclose()
    assert client.is_closed()
