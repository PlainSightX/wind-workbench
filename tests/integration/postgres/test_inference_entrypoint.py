"""真实 PG 迁移与只读入口；provider 仅返回模型目录，不生成回答。"""

import asyncio
from dataclasses import replace
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from uuid import uuid4

from alembic.config import Config
from alembic.script import ScriptDirectory
import httpx
import pytest
from sqlalchemy import text

from power_forecast_service.serve import make_event_loop
from power_forecast_service.storage.database import make_sync_engine

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("serve_inference_pg", ROOT / "tools/dev/serve_inference.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def test_real_head_and_metadata_check_do_not_write(config):
    settings = replace(config, assistant_backend="vllm", assistant_vllm_url="http://127.0.0.1:18110",
        assistant_vllm_model="pg-test-model")
    seen = []
    def handle(request):
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json={"data": [{"id": "pg-test-model", "max_model_len": 32768}]})
    engine = make_sync_engine(config)
    try:
        with engine.connect() as connection:
            before = connection.scalar(text("SELECT count(*) FROM assistant_requests"))
        result = asyncio.run(entry.check_configuration(settings,
            client=httpx.AsyncClient(transport=httpx.MockTransport(handle))), loop_factory=make_event_loop)
        with engine.connect() as connection:
            after = connection.scalar(text("SELECT count(*) FROM assistant_requests"))
        assert before == after
        assert result["database_revisions"] == ScriptDirectory.from_config(Config(str(ROOT / "alembic.ini"))).get_heads()
        assert seen == [("GET", "/v1/models")]
    finally:
        engine.dispose()


def test_stale_migration_is_rejected_without_running_upgrade(config):
    engine = make_sync_engine(config)
    try:
        with engine.begin() as connection:
            current = connection.scalar(text("SELECT version_num FROM alembic_version"))
            connection.execute(text("UPDATE alembic_version SET version_num='0007_engie_monitor'"))
        try:
            with pytest.raises(entry.EntryError, match="inference_database_migration_required"):
                entry.database_revision(config)
            with engine.connect() as connection:
                assert connection.scalar(text("SELECT version_num FROM alembic_version")) == "0007_engie_monitor"
        finally:
            with engine.begin() as connection:
                connection.execute(text("UPDATE alembic_version SET version_num=:version"), {"version": current})
    finally:
        engine.dispose()


def test_head_label_without_request_table_is_rejected(config):
    engine = make_sync_engine(config)
    try:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE assistant_requests RENAME TO i4_owned_requests_hidden"))
        try:
            with pytest.raises(entry.EntryError, match="inference_database_migration_required"):
                entry.database_revision(config)
        finally:
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE i4_owned_requests_hidden RENAME TO assistant_requests"))
    finally:
        engine.dispose()


def test_public_cli_check_and_real_host_api_lifecycle(config, tmp_path):
    """真实 TCP/CLI/PG 消费，不把模型目录替身算作真实 GPU 或回答质量验证。"""
    seen = []
    class Metadata(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            seen.append(("GET", self.path))
            payload = json.dumps({"data": [{"id": "cli-test-model", "max_model_len": 32768}]}).encode()
            self.send_response(200 if self.path == "/v1/models" else 404)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def do_POST(self):
            seen.append(("POST", self.path))
            self.send_error(500, "generation is not part of this test")
    metadata = ThreadingHTTPServer(("127.0.0.1", 0), Metadata)
    thread = threading.Thread(target=metadata.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    environment = os.environ | {"WIND_DB_NAME": config.database_url.database,
        "WIND_DB_HOST": config.database_url.host, "WIND_DB_PORT": str(config.database_url.port),
        "WIND_ARTIFACT_ROOT": str(config.artifact_root), "PYTHONDONTWRITEBYTECODE": "1"}
    argv = [sys.executable, "-X", "utf8", "-B", str(ROOT / "tools/dev/serve_inference.py"),
        "--provider-url", f"http://127.0.0.1:{metadata.server_port}/v1", "--model", "cli-test-model",
        "--port", str(port)]
    process = None
    try:
        checked = subprocess.run([*argv, "--check"], cwd=ROOT, env=environment,
            capture_output=True, text=True, encoding="utf-8", timeout=30)
        assert checked.returncode == 0, checked.stderr
        assert json.loads(checked.stdout)["check_only"] is True
        with socket.socket() as probe:
            assert probe.connect_ex(("127.0.0.1", port)) != 0
        with (tmp_path / "owned-api.log").open("wb") as output:
            process = subprocess.Popen(argv, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            with httpx.Client(trust_env=False, timeout=2) as client:
                for _ in range(100):
                    assert process.poll() is None, (tmp_path / "owned-api.log").read_text("utf-8")
                    try:
                        healthy = client.get(f"http://127.0.0.1:{port}/health")
                        if healthy.status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.1)
                else:
                    pytest.fail("owned host API did not become ready")
                assert healthy.json()["training_on_read"] is False
                unknown = client.get(f"http://127.0.0.1:{port}/assistant/requests/{uuid4()}")
                assert unknown.status_code == 404
                assert unknown.json()["detail"] == "assistant_request_not_found"
                assert seen == [("GET", "/v1/models"), ("GET", "/v1/models")]
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=15)
        metadata.shutdown()
        metadata.server_close()
        thread.join(5)
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", port)) != 0
