import httpx

from voice_engine.webhook import WebhookSender


async def test_webhook_retries_then_gives_up():
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(500)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await WebhookSender(client=client, backoff_s=0).send("http://x.test/hook", {"a": 1})
    assert len(calls) == 3


async def test_webhook_network_error_no_raise():
    def handler(req):
        raise httpx.ConnectError("boom")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await WebhookSender(client=client, backoff_s=0).send("http://x.test/hook", {})


async def test_webhook_success_single_call():
    calls = []
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: calls.append(r) or httpx.Response(200))
    )
    await WebhookSender(client=client, backoff_s=0).send("http://x.test/hook", {})
    assert len(calls) == 1


async def test_webhook_none_url_noop():
    def handler(req):
        raise AssertionError("must not be called")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await WebhookSender(client=client).send(None, {"a": 1})
