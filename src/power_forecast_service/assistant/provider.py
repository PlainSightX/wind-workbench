"""可选 vLLM 通信边界；消费本次 schema，不承担业务验收或自动重试。"""

import asyncio
import json
import math
from hashlib import sha256
from urllib.parse import urlsplit

import httpx
from langchain_core.messages import AIMessage
from jsonschema import Draft202012Validator, SchemaError, ValidationError

from .contracts import AssistantError
from .structured_output import (SCHEMA_TRANSPORT_VERSION, SORTED_PROPERTIES_VERSION,
    SchemaCompatibilityError, vllm_schema)


def vllm_request(messages, *, model, schema, generation, sort_properties=False):
    """服务与冻结导出共用请求构造；只转换约束表示，不修改消息或采样。"""
    return {"model": model, "messages": [{"role": role, "content": text} for role, text in messages],
        **{key: generation[key] for key in ("temperature", "seed", "top_p", "top_k", "min_p", "max_tokens")},
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True, "stream_options": {"include_usage": True},
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "wind_answer", "schema": vllm_schema(schema, sort_properties=sort_properties), "strict": True}}}


def schema_digest(schema):
    return sha256(json.dumps(schema, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


class VLLMProvider:
    def __init__(self, base_url, model, *, client=None, generation_profile="ordinary", timeout_seconds=35,
                 sort_schema_properties=False):
        url = urlsplit(base_url)
        # 运维显式配置，不能由问题/正文指定地址；本阶段仅开放受控的本机隧道。
        if (url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1"}
                or url.username or url.password or url.query or url.fragment
                or url.path not in {"", "/", "/v1", "/v1/"} or not model.strip()):
            raise ValueError("vLLM requires an explicit loopback URL and model")
        profiles = {
            "ordinary": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": 1700},
            "offline-greedy-v1": {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 1024},
        }
        if generation_profile not in profiles:
            raise ValueError("Unknown vLLM generation profile")
        if (type(timeout_seconds) not in {int, float} or not math.isfinite(timeout_seconds)
                or not 0 < timeout_seconds <= 75):
            raise ValueError("vLLM timeout must be positive and within the workflow budget")
        if type(sort_schema_properties) is not bool:
            raise ValueError("vLLM schema property order must be an explicit boolean")
        self.sort_schema_properties = sort_schema_properties
        self.url = base_url.rstrip("/")
        if not self.url.endswith("/v1"):
            self.url += "/v1"
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds, connect=5),
            transport=httpx.AsyncHTTPTransport(retries=0), trust_env=False)
        self.identity = {"backend": "vllm", "url": self.url, "model": model,
            **profiles[generation_profile], "seed": 42, "min_p": 0.0,
            "enable_thinking": False, "generation_profile": generation_profile,
            "timeout_seconds": timeout_seconds, "timeout_semantics": "total_and_io_v1",
            "schema_transport": SCHEMA_TRANSPORT_VERSION}
        if sort_schema_properties:
            # 只在显式开启时记录，默认身份与既有请求登记保持逐字相同。
            self.identity["schema_property_order"] = SORTED_PROPERTIES_VERSION

    async def aclose(self):
        await self.client.aclose()

    async def ainvoke_structured(self, messages, *, schema, call_id):
        # HTTPX的read timeout只约束两次读取之间的等待；持续输出也必须遵守总预算。
        try:
            async with asyncio.timeout(self.timeout_seconds):
                return await self._ainvoke_structured(messages, schema=schema, call_id=call_id)
        except TimeoutError:
            # 关闭流不等于确认远端未执行；保持未知结果及不自动重发的合同。
            raise AssistantError("provider_result_unknown") from None

    async def _ainvoke_structured(self, messages, *, schema, call_id):
        try:
            payload = vllm_request(messages, model=self.model, schema=schema, generation=self.identity,
                sort_properties=self.sort_schema_properties)
        except (SchemaCompatibilityError, SchemaError) as exc:
            # 本地明确拒绝，不向引擎提交、更不能放宽约束或触发回答修复。
            raise AssistantError("provider_rejected") from exc
        content, usage, finish, done, size = [], None, None, False, 0
        try:
            async with self.client.stream("POST", self.url + "/chat/completions",
                    json=payload, headers={"X-Request-ID": call_id}) as response:
                if response.status_code != 200:
                    code = "provider_rate_limited" if response.status_code == 429 else (
                        "provider_result_unknown" if response.status_code >= 500 else "provider_rejected")
                    raise AssistantError(code)
                async for line in response.aiter_lines():
                    size += len(line)
                    if size > 2_000_000:
                        raise AssistantError("provider_result_unknown")
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    chunk = json.loads(data)
                    if chunk.get("model") != self.model:
                        raise AssistantError("provider_identity_mismatch")
                    choices = chunk.get("choices", [])
                    if len(choices) > 1:
                        raise AssistantError("provider_result_unknown")
                    if choices:
                        choice = choices[0]
                        if choice.get("index") != 0:
                            raise AssistantError("provider_result_unknown")
                        delta = choice.get("delta", {}).get("content")
                        if delta is not None:
                            if not isinstance(delta, str):
                                raise AssistantError("provider_result_unknown")
                            content.append(delta)
                        if choice.get("finish_reason") is not None:
                            finish = choice["finish_reason"]
                    if chunk.get("usage") is not None:
                        usage = chunk["usage"]
            if not done or finish != "stop" or not "".join(content).strip():
                raise AssistantError("provider_result_unknown")
            if (not isinstance(usage, dict) or any(type(usage.get(key)) is not int
                    or usage[key] < 0 for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
                    or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
                raise AssistantError("provider_result_unknown")
            raw_content = "".join(content)
            try:
                Draft202012Validator(schema).validate(json.loads(raw_content))
            except (ValidationError, ValueError) as exc:
                # 已完整返回的非法答案与传输未知分开，由原工作流控制至多一次回答修复。
                raise AssistantError("answer_schema_invalid") from exc
            return AIMessage(content=raw_content, usage_metadata={
                "input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"]}, response_metadata={
                "model_name": self.model, "finish_reason": finish, "call_id": call_id,
                "schema_transport": SCHEMA_TRANSPORT_VERSION,
                **({"schema_property_order": SORTED_PROPERTIES_VERSION} if self.sort_schema_properties else {}),
                "original_schema_sha256": schema_digest(schema),
                "wire_schema_sha256": schema_digest(payload["response_format"]["json_schema"]["schema"]),
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")})
        except asyncio.CancelledError:
            raise
        except AssistantError:
            raise
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            # HTTP timeout/EOF 不证明远端没有执行；不回显异常或重新请求。
            raise AssistantError("provider_result_unknown") from exc


def make_assistant(sessions, settings, *, repair_mode="full", response_mode=None):
    """普通入口不变；只有运维选择 vllm 后才创建可选客户端与请求登记。"""
    from .workflow import Assistant, INFERENCE_RESPONSE_MODES
    if settings.assistant_backend == "default":
        return Assistant(sessions)
    if settings.assistant_backend != "vllm":
        raise ValueError("Unknown assistant backend")
    response_mode = response_mode or settings.assistant_response_mode
    if response_mode not in INFERENCE_RESPONSE_MODES:
        raise ValueError("Unknown inference response mode")
    if type(settings.assistant_capacity) is not int or not 1 <= settings.assistant_capacity <= 4:
        raise ValueError("Assistant capacity must be between one and four")
    provider = VLLMProvider(settings.assistant_vllm_url, settings.assistant_vllm_model,
        generation_profile=getattr(settings, "inference_generation_profile", "ordinary"),
        timeout_seconds=getattr(settings, "inference_provider_timeout", 35),
        sort_schema_properties=getattr(settings, "inference_sort_schema_properties", False))
    return Assistant(sessions, provider=provider, response_mode=response_mode,
        prompt_layout="original", durable_requests=True,
        capacity=settings.assistant_capacity, repair_mode=repair_mode)
