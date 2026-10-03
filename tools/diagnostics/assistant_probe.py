"""分离本地配置、固定 embedding 准备与显式付费探针；不输出凭据。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from power_forecast_service.assistant.configuration import provider_key
from power_forecast_service.assistant.retrieval import runtime_root, embed
from power_forecast_service.assistant.retrieval import EMBEDDING_MODEL as MODEL, EMBEDDING_REVISION as REVISION


def save_key():
    target = runtime_root()
    target.mkdir(parents=True, exist_ok=True)
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    path = target / "provider-key.txt"
    if key:
        if path.exists():
            if path.read_text(encoding="utf-8").strip() != key:
                raise ValueError("provider_key_conflict: remove the owned stale key explicitly")
        else:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target, suffix=".partial", delete=False) as stream:
                    temporary = Path(stream.name)
                    # 在临时文件上设权限并完整落盘，再独占发布；失败不留下空的正式凭据。
                    os.chmod(temporary, 0o600)
                    stream.write(key)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    if path.read_text(encoding="utf-8").strip() != key:
                        raise ValueError("provider_key_conflict: remove the owned stale key explicitly")
            finally:
                if temporary is not None:
                    temporary.unlink()
    return {"provider_configured": bool(provider_key()), "network_requests": 0}


def prepare_embedding():
    from huggingface_hub import snapshot_download

    target = runtime_root()
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / "embedding"
    snapshot_download(MODEL, revision=REVISION, local_dir=model_path,
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "README.md"])
    manifest = {"model": MODEL, "revision": REVISION, "license": "MIT", "files": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in model_path.iterdir() if p.is_file()}}
    (target / "embedding-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def probe_vector():
    from sqlalchemy import create_engine, text
    from power_forecast_service.settings import Settings

    engine = create_engine(Settings.from_environment().database_url)
    try:
        with engine.connect() as conn:
            version = conn.execute(text("SELECT extversion FROM pg_extension WHERE extname='vector'")).scalar_one()
            distance = conn.execute(text("SELECT '[1,0,0]'::vector <=> '[1,0,0]'::vector")).scalar_one()
    finally:
        engine.dispose()
    vectors = embed(["为什么不能使用未来天气？", "只能使用起报时刻已知的历史观测。"])
    return {"pgvector": version, "self_cosine_distance": distance,
        "embedding_model": MODEL, "revision": REVISION, "dimension": len(vectors[0])}


def probe_provider():
    from langchain_deepseek import ChatDeepSeek

    reply = ChatDeepSeek(model="deepseek-chat", api_key=provider_key(), timeout=30,
        max_retries=0, temperature=0, max_tokens=80).bind(response_format={"type": "json_object"}).invoke(
            [("system", 'Return only JSON with status="ok".'), ("user", "Probe.")])
    if json.loads(reply.content).get("status") != "ok":
        raise ValueError("provider_probe_invalid")
    return {"provider_requested_model": "deepseek-chat",
        "provider_reported_model": reply.response_metadata.get("model_name"), "usage": reply.usage_metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["configure", "prepare-embedding", "vector", "provider"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"configure": save_key, "prepare-embedding": prepare_embedding,
        "vector": probe_vector, "provider": probe_provider}[args.action]()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
