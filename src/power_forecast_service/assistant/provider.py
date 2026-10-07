"""可选 vLLM 通信边界；消费本次 schema，不承担业务验收或自动重试。"""

import asyncio
import json
from urllib.parse import urlsplit

import httpx
from langchain_core.messages import AIMessage

from .contracts import AssistantError


class VLLMProvider:
    def __init__(self, base_url, model, *, client=None):
        url = urlsplit(base_url)
        # 运维显式配置，不能由问题/正文指定地址；本阶段仅开放受控的本机隧道。
        if (url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1"}
                or url.username or url.password or url.query or url.fragment
                or url.path not in {"", "/", "/v1", "/v1/"} or not model.strip()):
            raise ValueError("vLLM requires an explicit loopback URL and model")
        self.url = base_url.rstrip("/")
        if not self.url.endswith("/v1"):
            self.url += "/v1"
        self.model = model
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(35, connect=5),
            transport=httpx.AsyncHTTPTransport(retries=0), trust_env=False)
        self.identity = {"backend": "vllm", "url": self.url, "model": model,
            "temperature": 0.7, "seed": 42, "top_p": 0.8, "top_k": 20,
            "min_p": 0.0, "max_tokens": 1700, "enable_thinking": False}

    async def aclose(self):
        await self.client.aclose()

    async def ainvoke_structured(self, messages, *, schema, call_id):
        payload = {"model": self.model, "messages": [
            {"role": role, "content": text} for role, text in messages],
            "temperature": 0.7, "seed": 42, "top_p": 0.8, "top_k": 20,
            "min_p": 0.0, "max_tokens": 1700,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": True, "stream_options": {"include_usage": True},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "wind_answer", "schema": schema, "strict": True}}}
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
            return AIMessage(content="".join(content), usage_metadata={
                "input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"]}, response_metadata={
                "model_name": self.model, "finish_reason": finish, "call_id": call_id,
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")})
        except asyncio.CancelledError:
            raise
        except AssistantError:
            raise
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            # HTTP timeout/EOF 不证明远端没有执行；不回显异常或重新请求。
            raise AssistantError("provider_result_unknown") from exc


def make_assistant(sessions, settings):
    """普通入口不变；只有运维选择 vllm 后才创建可选客户端与请求登记。"""
    from .workflow import Assistant
    if settings.assistant_backend == "default":
        return Assistant(sessions)
    if settings.assistant_backend != "vllm":
        raise ValueError("Unknown assistant backend")
    if type(settings.assistant_capacity) is not int or not 1 <= settings.assistant_capacity <= 4:
        raise ValueError("Assistant capacity must be between one and four")
    provider = VLLMProvider(settings.assistant_vllm_url, settings.assistant_vllm_model)
    return Assistant(sessions, provider=provider, response_mode="references",
        prompt_layout="original", durable_requests=True,
        capacity=settings.assistant_capacity)
