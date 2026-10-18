"""通信协议的注入测试；不冒充真实 GPU 输出质量或吞吐量。"""

import asyncio
from copy import deepcopy
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


def test_explicit_greedy_profile_is_consumed_by_http_payload():
    seen = []
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, text=stream())
    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = VLLMProvider("http://127.0.0.1:18112", "candidate", client=client,
            generation_profile="offline-greedy-v1", timeout_seconds=60)
        try:
            await provider.ainvoke_structured([("user", "完整证据")], schema={"type": "object"}, call_id="greedy")
            assert provider.identity["generation_profile"] == "offline-greedy-v1"
            assert provider.identity["timeout_seconds"] == 60
            assert provider.identity["timeout_semantics"] == "total_and_io_v1"
        finally:
            await provider.aclose()
    asyncio.run(run())
    assert len(seen) == 1
    assert {key: seen[0][key] for key in ("temperature", "top_p", "top_k", "max_tokens")} == {
        "temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 1024}


def test_total_budget_closes_a_continuously_active_stream_without_retry():
    class ActiveStream(httpx.AsyncByteStream):
        def __init__(self):
            self.chunks, self.closed = 0, False

        async def __aiter__(self):
            for _ in range(100):
                await asyncio.sleep(0.005)
                self.chunks += 1
                yield b'data: {"model":"candidate","choices":[{"index":0,"delta":{"content":" "}}]}\n\n'

        async def aclose(self):
            self.closed = True

    async def scenario():
        active, requests = ActiveStream(), []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, stream=active)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = VLLMProvider("http://127.0.0.1:18112", "candidate", client=client, timeout_seconds=0.04)
        try:
            with pytest.raises(AssistantError, match="provider_result_unknown"):
                await provider.ainvoke_structured([("user", "完整证据")], schema={"type": "object"}, call_id="deadline")
            assert 1 <= active.chunks < 100
            assert active.closed and len(requests) == 1
        finally:
            await provider.aclose()
    asyncio.run(scenario())


def test_total_budget_includes_waiting_for_response_headers():
    async def handler(request):
        await asyncio.sleep(1)
        return httpx.Response(200, text=stream())

    async def scenario():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = VLLMProvider("http://127.0.0.1:18112", "candidate", client=client, timeout_seconds=0.02)
        try:
            with pytest.raises(AssistantError, match="provider_result_unknown"):
                await provider.ainvoke_structured([("user", "完整证据")], schema={"type": "object"}, call_id="headers")
        finally:
            await provider.aclose()
    asyncio.run(scenario())


@pytest.mark.parametrize("value", [0, -1, 76, float("nan"), True])
def test_explicit_provider_budget_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="workflow budget"):
        VLLMProvider("http://127.0.0.1:18112", "candidate", timeout_seconds=value)


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


def test_requested_factory_consumes_versioned_contract_without_enabling_auxiliary_policy():
    settings = SimpleNamespace(assistant_backend="vllm", assistant_response_mode="program_requested",
        assistant_capacity=2, assistant_vllm_url="http://127.0.0.1:18112", assistant_vllm_model="named-adapter")
    assistant = make_assistant(None, settings)
    try:
        assert assistant.response_mode == "program_requested"
        assert assistant.prompt_version == "wind-results-program-23-requested-presentation"
        assert assistant.provider.model == "named-adapter"
        assert assistant.journal is not None
    finally:
        asyncio.run(assistant.provider.aclose())
        assistant.close()


@pytest.mark.parametrize("capacity", [0, 5, True])
def test_capacity_is_bounded(capacity):
    with pytest.raises(ValueError):
        Assistant(None, capacity=capacity)


def test_unique_schema_is_transported_equivalently_with_original_response_validation():
    schema = {"type": "object", "properties": {"ids": {"type": "array", "uniqueItems": True,
        "items": {"type": "string", "enum": ["a", "b"]}}}, "required": ["ids"], "additionalProperties": False}
    original, calls = deepcopy(schema), []
    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        assert body["response_format"]["json_schema"]["schema"]["properties"]["ids"]["enum"] == [[], ["a"], ["b"], ["a", "b"], ["b", "a"]]
        return httpx.Response(200, text=stream(content='{"ids":["b","a"]}'))
    result = execute(handler, schema)
    assert len(calls) == 1 and schema == original
    assert result.response_metadata["original_schema_sha256"] != result.response_metadata["wire_schema_sha256"]
    # 即使上游声称结构化输出已成功，也要按原约束拒绝重复元素。
    with pytest.raises(AssistantError, match="answer_schema_invalid"):
        execute(lambda request: httpx.Response(200, text=stream(content='{"ids":["a","a"]}')), schema)


def test_incompatible_schema_fails_before_transport():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, text=stream())
    with pytest.raises(AssistantError, match="provider_rejected"):
        execute(handler, {"type": "array", "uniqueItems": True, "items": {"type": "string"}})
    assert calls == []


@pytest.mark.parametrize("content", ['{"wrong":1}', 'not json'])
def test_complete_bad_answer_is_known_schema_failure_and_not_transport_retry(content):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, text=stream(content=content))
    with pytest.raises(AssistantError, match="answer_schema_invalid"):
        execute(handler, {"type": "object", "required": ["status"]})
    assert len(calls) == 1


def test_schema_property_order_is_default_off_and_explicitly_recorded():
    schema = {"type": "object", "properties": {"requests": {"type": "string"}, "presentation": {"type": "string"}}}
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, text=stream(content='{"requests":"a"}'))

    async def run(**options):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = VLLMProvider("http://127.0.0.1:18112", "candidate", client=client, **options)
        try:
            reply = await provider.ainvoke_structured([("user", "完整证据")], schema=schema, call_id="order")
            return provider.identity, reply
        finally:
            await provider.aclose()

    default_identity, default_reply = asyncio.run(run())
    sorted_identity, sorted_reply = asyncio.run(run(sort_schema_properties=True))
    assert "schema_property_order" not in default_identity
    assert "schema_property_order" not in default_reply.response_metadata
    assert sorted_identity["schema_property_order"] == "sorted-properties-v1"
    assert sorted_reply.response_metadata["schema_property_order"] == "sorted-properties-v1"
    assert list(seen[0]["response_format"]["json_schema"]["schema"]["properties"]) == ["requests", "presentation"]
    assert list(seen[1]["response_format"]["json_schema"]["schema"]["properties"]) == ["presentation", "requests"]
    # 业务侧原schema摘要不随传输顺序改变；线上schema摘要按排序后JSON计算也相同。
    assert default_reply.response_metadata["original_schema_sha256"] == sorted_reply.response_metadata["original_schema_sha256"]
    with pytest.raises(ValueError):
        VLLMProvider("http://127.0.0.1:18112", "candidate", sort_schema_properties="yes")
