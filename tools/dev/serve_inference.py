"""显式启动宿主侧可选推理 API；检查不迁移数据库、不发送生成请求。"""

import argparse
import asyncio
from dataclasses import dataclass, fields
import json
import os
import socket
from urllib.parse import urlsplit

from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import text

from power_forecast_service.assistant.provider import VLLMProvider
from power_forecast_service.assistant.workflow import INFERENCE_RESPONSE_MODES
from power_forecast_service.serve import make_event_loop
from power_forecast_service.settings import ROOT, Settings
from power_forecast_service.storage.database import make_sync_engine


class EntryError(RuntimeError):
    """启动失败只公开稳定错误码，不回显连接串、密码或远端正文。"""


@dataclass(frozen=True)
class InferenceSettings(Settings):
    """研究入口显式配置采样；普通API的Settings和默认值保持不变。"""

    inference_generation_profile: str = "ordinary"
    inference_provider_timeout: float = 35
    inference_sort_schema_properties: bool = False


def check_port(port, provider_url):
    """CLI 端口独立于 run-local 注入的普通 API 端口；不接管已有监听。"""
    if not 1024 <= port <= 65535:
        raise EntryError("inference_port_invalid")
    ordinary = os.getenv("WIND_API_PORT")
    upstream = urlsplit(provider_url).port or 80
    if (ordinary and port == int(ordinary)) or port == upstream:
        raise EntryError("inference_port_conflict")
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    except OSError as exc:
        raise EntryError("inference_port_in_use") from exc


def database_revision(settings):
    """检查实际连接的数据库，不用文件存在或新建 ORM 表代替迁移验收。"""
    expected = set(ScriptDirectory.from_config(Config(str(ROOT / "alembic.ini"))).get_heads())
    engine = make_sync_engine(settings)
    try:
        with engine.connect() as connection:
            connection.execute(text("SET TRANSACTION READ ONLY"))
            actual = set(MigrationContext.configure(connection).get_current_heads())
            table = connection.scalar(text("SELECT to_regclass('assistant_requests')"))
            if actual != expected or not table:
                raise EntryError("inference_database_migration_required")
            return sorted(actual)
    except EntryError:
        raise
    except Exception as exc:
        raise EntryError("inference_database_unavailable") from exc
    finally:
        engine.dispose()


async def check_configuration(settings, *, client=None):
    if (settings.assistant_backend != "vllm" or type(settings.assistant_capacity) is not int
            or not 1 <= settings.assistant_capacity <= 4
            or settings.assistant_response_mode not in INFERENCE_RESPONSE_MODES):
        raise EntryError("inference_configuration_invalid")
    try:
        provider = VLLMProvider(settings.assistant_vllm_url, settings.assistant_vllm_model,
            client=client, generation_profile=getattr(settings, "inference_generation_profile", "ordinary"),
            timeout_seconds=getattr(settings, "inference_provider_timeout", 35),
            sort_schema_properties=getattr(settings, "inference_sort_schema_properties", False))
    except ValueError as exc:
        raise EntryError("inference_configuration_invalid") from exc
    try:
        revisions = database_revision(settings)
        # 只读模型目录；身份检查不消费 chat/completions、不以生成探针证明可用。
        async with provider.client.stream("GET", provider.url + "/models") as response:
            if response.status_code != 200:
                raise EntryError("inference_metadata_unavailable")
            chunks, size = [], 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > 65536:
                    raise EntryError("inference_metadata_invalid")
                chunks.append(chunk)
        metadata = json.loads(b"".join(chunks))
        models = metadata.get("data") if isinstance(metadata, dict) else None
        if not isinstance(models, list) or any(not isinstance(row, dict) for row in models):
            raise EntryError("inference_metadata_invalid")
        selected = [row for row in models if row.get("id") == provider.model]
        if len(selected) != 1:
            raise EntryError("inference_model_identity_mismatch")
        context = selected[0].get("max_model_len")
        context_model = provider.model
        # vLLM 0.11.2 的 LoRA 卡没有容量字段；只能继承唯一父底座，不能猜测可用长度。
        if context is None and selected[0].get("parent") is not None:
            parent = selected[0]["parent"]
            if not isinstance(parent, str) or not parent or parent == provider.model:
                raise EntryError("inference_model_identity_mismatch")
            parents = [row for row in models if row.get("id") == parent]
            if len(parents) != 1 or parents[0].get("parent") is not None:
                raise EntryError("inference_model_identity_mismatch")
            context, context_model = parents[0].get("max_model_len"), parent
        if type(context) is not int or context < 32768:
            raise EntryError("inference_context_insufficient")
        return {"backend": "vllm", "model": provider.model, "max_model_len": context,
            "context_model": context_model,
            "database_revisions": revisions, "capacity": settings.assistant_capacity,
            "response_mode": settings.assistant_response_mode,
            "generation": provider.identity,
            "generation_sent": False}
    except EntryError:
        raise
    except Exception as exc:
        raise EntryError("inference_metadata_unavailable") from exc
    finally:
        await provider.aclose()


def serve(settings, port):
    import uvicorn
    from power_forecast_service.api.app import create_app
    server = uvicorn.Server(uvicorn.Config(create_app(settings), host="127.0.0.1", port=port))
    asyncio.run(server.serve(), loop_factory=make_event_loop)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-url", required=True)
    parser.add_argument("--model", required=True, help="vLLM served-model-name, not a local weight path")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--capacity", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--response-mode", choices=INFERENCE_RESPONSE_MODES, default="references")
    parser.add_argument("--generation-profile", choices=["ordinary", "offline-greedy-v1"], default="ordinary")
    parser.add_argument("--provider-timeout", type=float, default=35)
    parser.add_argument("--sort-schema-properties", action="store_true",
        help="Send transport schema properties in key order (for adapters trained on sort_keys targets)")
    parser.add_argument("--check", action="store_true", help="Read-only metadata/migration/port checks")
    args = parser.parse_args(argv)
    try:
        original = Settings.from_environment()
        base = {field.name: getattr(original, field.name) for field in fields(Settings)}
        base.update(assistant_backend="vllm",
            assistant_vllm_url=args.provider_url, assistant_vllm_model=args.model,
            assistant_capacity=args.capacity, assistant_response_mode=args.response_mode)
        settings = InferenceSettings(**base, inference_generation_profile=args.generation_profile,
            inference_provider_timeout=args.provider_timeout,
            inference_sort_schema_properties=args.sort_schema_properties)
        check_port(args.port, args.provider_url)
        report = asyncio.run(check_configuration(settings), loop_factory=make_event_loop)
        print(json.dumps(report | {"port": args.port, "check_only": args.check}), flush=True)
        if not args.check:
            serve(settings, args.port)
    except EntryError as exc:
        parser.exit(2, str(exc) + "\n")
    except (ValueError, OSError):
        parser.exit(2, "inference_environment_invalid\n")


if __name__ == "__main__":
    main()
