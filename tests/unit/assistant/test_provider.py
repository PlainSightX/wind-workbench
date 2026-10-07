"""通信协议的注入测试；不冒充真实 GPU 输出质量或吞吐量。"""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from power_forecast_service.assistant.contracts import AssistantError
from power_forecast_service.assistant.provider import VLLMProvider, make_assistant
from power_forecast_service.assistant.workflow import Assistant


def stream(*, content='{"status":"answered"}', finish="stop", done=True, model="candidate", usage=True):
    chunks = [{"model": model, "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish}]}]
    if usage:
        chunks.append({"model": model, "choices": [], "usage": {
            "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
            "prompt_tokens_details": {"cached_tokens": 64}}})
    return "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + ("data: [DONE]\n\n" if done else "")


def execute(handler, schema=None):
    async def check():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = VLLMProvider("http://127.0.0.1:18112", "candidate", client=client)
        try:
            return await provider.ainvoke_structured([("system", "规则"), ("user", "完整证据")],
                schema=schema or {"type": "object"}, call_id="request-1")
        finally:
            await provider.aclose()
    return asyncio.run(check())


def test_real_request_schema_and_fixed_sampling_forwarded():
    schema = {"type": "object", "properties": {"body": {"enum": ["context-a"]}}}
    def handler(request):
        payload = json.loads(request.content)
        assert payload["response_format"]["json_schema"]["schema"] == schema
        assert payload["messages"][1]["content"] == "完整证据"
        assert payload["temperature"] == 0.7 and payload["seed"] == 42
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert request.headers["x-request-id"] == "request-1"
        return httpx.Response(200, text=stream())
    reply = execute(handler, schema)
    assert reply.response_metadata["cached_tokens"] == 64
    assert reply.usage_metadata["input_tokens"] == 100


@pytest.mark.parametrize("changes", [{"finish": "length"}, {"done": False}, {"usage": False}, {"content": ""}])
def test_incomplete_stream_is_unknown_not_retried(changes):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, text=stream(**changes))
    with pytest.raises(AssistantError, match="provider_result_unknown"):
        execute(handler)
    assert len(calls) == 1


@pytest.mark.parametrize("code,expected", [(429, "provider_rate_limited"), (400, "provider_rejected"), (503, "provider_result_unknown")])
def test_status_does_not_trigger_transport_retry(code, expected):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(code, text="private error")
    with pytest.raises(AssistantError, match=expected):
        execute(handler)
    assert len(calls) == 1


def test_network_failure_and_model_identity_are_not_repaired():
    def broken(request):
        raise httpx.ReadTimeout("private url and secret", request=request)
    with pytest.raises(AssistantError, match="provider_result_unknown"):
        execute(broken)
    with pytest.raises(AssistantError, match="provider_identity_mismatch"):
        execute(lambda request: httpx.Response(200, text=stream(model="other")))


@pytest.mark.parametrize("url", ["https://remote/v1", "http://remote/v1", "http://user:secret@127.0.0.1", "http://127.0.0.1/path"])
def test_only_explicit_local_service_address(url):
    with pytest.raises(ValueError):
        VLLMProvider(url, "candidate")


def test_default_factory_preserves_legacy():
    assistant = make_assistant(None, SimpleNamespace(assistant_backend="default"))
    try:
        assert assistant.provider is None and assistant.response_mode == "legacy"
        assert assistant.prompt_layout == "original" and assistant.journal is None
    finally:
        assistant.close()


@pytest.mark.parametrize("capacity", [0, 5, True])
def test_capacity_is_bounded(capacity):
    with pytest.raises(ValueError):
        Assistant(None, capacity=capacity)
