"""可选服务真实 HTTP/PG 验证；故障代理只截断交付，不伪造模型回答。"""

import argparse
import asyncio
from dataclasses import replace
from datetime import UTC, datetime
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

import httpx
import psycopg
from psycopg import sql
from alembic import command
from alembic.config import Config
from sqlalchemy import select
from sqlalchemy.orm import Session

from power_forecast_service.assistant.evidence import digest
from power_forecast_service.assistant.retrieval import corpus
from power_forecast_service.settings import ROOT, Settings
from power_forecast_service.storage.database import make_sync_engine
from power_forecast_service.storage.models import ImportedRun, ImportedArtifact
from power_forecast_service.serve import make_event_loop


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def assert_free(port):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))


def verify_citations(citations, documents):
    """HTTP 交付用 revision 表示来源 hash；不能套用内部文档字段名。"""
    for citation in citations:
        source = documents.get(citation["id"])
        if (source is None or citation.get("revision") != source["source_sha256"]
                or not citation.get("quote") or citation["quote"] not in source["text"]):
            raise ValueError("Citation version or original quote changed")


def service_port(base, proxy, index):
    """独立进程使用不同已登记端口，避免 Windows TIME_WAIT 被误判成占用。"""
    candidate = base + index
    return candidate + (1 if base <= proxy <= candidate else 0)


def source_rows(settings):
    engine = make_sync_engine(settings)
    try:
        with Session(engine) as session:
            session.execute(__import__("sqlalchemy").text("SET TRANSACTION READ ONLY"))
            rows = {}
            for model in (ImportedRun, ImportedArtifact):
                rows[model.__tablename__] = [{column.name: getattr(row, column.name)
                    for column in model.__table__.columns}
                    for row in session.scalars(select(model).order_by(model.id))]
            return rows
    finally:
        engine.dispose()


class FaultProxy:
    """远端完整返回后才注入 EOF/暂扣交付；不替换原问题、schema 或答案。"""
    def __init__(self, port, upstream):
        self.upstream, self.mode = upstream.rstrip("/"), "eof"
        self.finished, self.release = threading.Event(), threading.Event()
        self.calls, self.errors = [], []
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                try:
                    payload = self.rfile.read(int(self.headers["Content-Length"]))
                    with httpx.Client(timeout=75, trust_env=False) as client:
                        response = client.post(owner.upstream + "/chat/completions", content=payload,
                            headers={"Content-Type": "application/json", "X-Request-ID": self.headers["X-Request-ID"]})
                    response.raise_for_status()
                    lines = response.text.splitlines()
                    chunks = [json.loads(line[5:].strip()) for line in lines if line.startswith("data:") and line[5:].strip() != "[DONE]"]
                    if not any(line.strip() == "data: [DONE]" for line in lines) or not any(
                            choice.get("finish_reason") == "stop" for chunk in chunks for choice in chunk.get("choices", [])):
                        raise ValueError("Remote did not complete before fault injection")
                    usage = next(chunk["usage"] for chunk in reversed(chunks) if chunk.get("usage"))
                    owner.calls.append({"id": self.headers["X-Request-ID"], "remote_finish": "stop", "usage": usage,
                                        "injection": owner.mode, "at": datetime.now(UTC).isoformat()})
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    if owner.mode == "hold":
                        self.wfile.write((lines[0] + "\n\n").encode())
                        self.wfile.flush()
                        owner.finished.set()
                        if not owner.release.wait(90):
                            raise TimeoutError("Owned fault hold deadline elapsed")
                    else:
                        # 原始远端输出保持，仅去掉协议终止标志模拟不完整交付。
                        self.wfile.write(response.text.replace("data: [DONE]", "").encode())
                        self.wfile.flush()
                        owner.finished.set()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as error:
                    owner.errors.append(type(error).__name__)
                    owner.finished.set()
                    self.close_connection = True
        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)


class ServiceProcess:
    def __init__(self, port, database, provider_url, model, output, *, capacity=1):
        assert_free(port)
        environment = os.environ | {"WIND_API_PORT": str(port), "WIND_DB_NAME": database,
            "WIND_ASSISTANT_BACKEND": "vllm", "WIND_ASSISTANT_VLLM_URL": provider_url,
            "WIND_ASSISTANT_VLLM_MODEL": model, "WIND_ASSISTANT_CAPACITY": str(capacity)}
        self.log = output.open("xb")
        self.process = subprocess.Popen([sys.executable, "-X", "utf8", "-B", "-m", "power_forecast_service.serve"],
            cwd=ROOT, env=environment, stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.url = f"http://127.0.0.1:{port}"
        self.forced = False

    async def ready(self, client):
        for _ in range(100):
            if self.process.poll() is not None:
                raise RuntimeError("Owned API exited during startup; inspect private log")
            try:
                if (await client.get(self.url + "/openapi.json", timeout=1)).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.1)
        raise TimeoutError("Owned API did not become ready")

    def stop(self, *, force=False):
        if self.process.poll() is None:
            self.forced = force
            self.process.kill() if force else self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.log.close()


async def consume(args, database, rows, output):
    development = sorted([row for row in rows["imported_runs"] if row["manifest"].get("scope") != "final_2015"], key=lambda row: row["quarter"])
    final = next(row for row in rows["imported_runs"] if row["manifest"].get("scope") == "final_2015")
    if len(development) < 2:
        raise ValueError("Two real development objects and one final object are required")
    cases = [{"id": "development-a", "question": "lightgbm 相对持续性的 MAE 改善是多少？请同时给出两者 MAE。", "contexts": [{"kind": "engie_import", "id": str(development[0]["id"]), "model": "lightgbm"}]},
             {"id": "development-b", "question": "lightgbm 相对持续性的 MAE 改善是多少？请同时给出两者 MAE。", "contexts": [{"kind": "engie_import", "id": str(development[1]["id"]), "model": "lightgbm"}]},
             {"id": "final", "question": "开发采用门与最终留出结果应该如何区分？请说明默认策略及最终 MAE 改善。", "contexts": [{"kind": "engie_import", "id": str(final["id"]), "model": "lightgbm_l1_shrink"}]}]
    artifact_map = {(str(row["import_id"]), row["family"]): row for row in rows["imported_artifacts"]}
    save(output / "frozen-cases.json", {"cases": cases, "source_sha256": digest(rows),
        "corpus_sha256": corpus()["sha256"], "acceptance": "Current-object exact fact/source bindings plus independent prose review; observed development contexts, not unseen generalization"})
    results, checks, services = [], {}, []
    proxy = None
    async with httpx.AsyncClient(timeout=85, trust_env=False) as client:
        def start(url, name, capacity=1):
            port = service_port(args.api_port, args.proxy_port, len(services))
            service = ServiceProcess(port, database, url, args.model, output / (name + ".log"), capacity=capacity)
            services.append(service)
            return service
        async def ask(service, case, *, key=None, timeout=85):
            key = key or str(uuid4())
            payload = {k: case[k] for k in ("question", "contexts")}
            started = time.monotonic()
            response = await client.post(service.url + "/assistant/answers", json=payload,
                headers={"X-Request-ID": key}, timeout=timeout)
            item = {"case": case["id"], "request_id": key, "http_status": response.status_code,
                "seconds": time.monotonic() - started, "result": response.json()}
            results.append(item)
            save(output / "results.json", results)
            return item
        def verify(item, case):
            result = item["result"]
            if item["http_status"] != 200 or result["status"] != "answered":
                raise ValueError("Real consumer did not deliver: " + case["id"] + "/" + str(result.get("error")))
            chosen = case["contexts"][0]["id"]
            required = {"c0." + case["contexts"][0]["model"] + ".mae", "c0.persistence.mae",
                        "c0." + case["contexts"][0]["model"] + ".mae_gain"}
            observed = {fact["id"] for fact in result["facts"]}
            if case["id"] == "final":
                required = {"c0.lightgbm_l1_shrink.mae_gain", "c0.adopted"}
            if not required <= observed:
                raise ValueError("Independent metric coverage failed: " + case["id"])
            for fact in result["facts"]:
                if fact["object_id"] != chosen:
                    raise ValueError("Fact from previous or different object")
                parts = fact["id"].split(".")
                if len(parts) == 3 and parts[-1] in {"mae", "rmse", "bias"}:
                    artifact = artifact_map[(chosen, parts[1])]
                    if fact["artifact_id"] != str(artifact["id"]) or fact["value"] != artifact["manifest"]["metrics"][parts[2]]:
                        raise ValueError("Metric binding differs from original PG")
            documents = {chunk["id"]: chunk for chunk in corpus()["chunks"]}
            verify_citations(result["citations"], documents)
        try:
            service = start(args.provider_url, "normal")
            await service.ready(client)
            for case in cases:
                item = await ask(service, case)
                verify(item, case)
                state = (await client.get(service.url + "/assistant/requests/" + item["request_id"])).json()
                if state["status"] != "completed":
                    raise ValueError("Successful answer was not durably finalized")
            checks["sequential_current_object_bindings"] = True
            # 跨对象负对照必须在当前模型之前被工具层拒绝，不能借用缓存回答。
            negative = {**cases[0], "id": "outside-object", "question": "请解释对象 " + cases[1]["contexts"][0]["id"]}
            item = await ask(service, negative)
            checks["outside_object_no_model_call"] = (item["result"].get("error") == "context_out_of_scope"
                and not item["result"]["trace"]["model_calls"])
            # 真实 TCP 客户端超时；模型进程仍在执行，容量不能提前返还。
            key = str(uuid4())
            task = asyncio.create_task(ask(service, cases[0], key=key, timeout=0.3))
            try:
                await task
                raise ValueError("Disconnect drill did not interrupt the HTTP client")
            except httpx.TimeoutException:
                pass
            state = None
            for _ in range(50):
                response = await client.get(service.url + "/assistant/requests/" + key)
                if response.status_code == 200:
                    state = response.json()
                    break
                await asyncio.sleep(0.1)
            if state is None:
                raise ValueError("Disconnect occurred before request admission")
            busy = await ask(service, cases[1])
            checks["disconnect_keeps_slot_and_429"] = busy["http_status"] == 429
            for _ in range(800):
                state = (await client.get(service.url + "/assistant/requests/" + key)).json()
                if state["status"] != "pending":
                    break
                await asyncio.sleep(0.1)
            checks["disconnected_request_completed"] = state["status"] == "completed"
            duplicate = await ask(service, cases[0], key=key)
            checks["completed_duplicate_no_resend"] = duplicate["http_status"] == 409
            service.stop()
            mixed = start(args.provider_url, "mixed", capacity=2)
            await mixed.ready(client)
            delivered = await asyncio.gather(*(ask(mixed, case) for case in cases[:2]))
            for item, case in zip(delivered, cases[:2]):
                verify(item, case)
            checks["concurrent_object_bindings"] = True
            mixed.stop()
            assert_free(args.proxy_port)
            proxy = FaultProxy(args.proxy_port, args.provider_url)
            failed = start(f"http://127.0.0.1:{args.proxy_port}/v1", "faults")
            await failed.ready(client)
            key = str(uuid4())
            item = await ask(failed, cases[0], key=key)
            checks["real_remote_return_then_eof_unknown"] = (item["result"].get("error") == "provider_result_unknown"
                and len(item["result"]["trace"]["model_calls"]) == 1 and len(proxy.calls) == 1)
            duplicate = await ask(failed, cases[0], key=key)
            checks["unknown_duplicate_no_resend"] = duplicate["http_status"] == 409 and len(proxy.calls) == 1
            proxy.mode, proxy.finished = "hold", threading.Event()
            key = str(uuid4())
            interrupted = asyncio.create_task(ask(failed, cases[1], key=key))
            if not await asyncio.to_thread(proxy.finished.wait, 65) or proxy.errors or len(proxy.calls) != 2:
                raise ValueError("Remote completion barrier failed")
            before = (await client.get(failed.url + "/assistant/requests/" + key)).json()
            if before["status"] != "pending" or before["calls"][-1]["status"] != "dispatching":
                raise ValueError("Kill window was not after durable dispatch and before local return")
            failed.stop(force=True)
            try:
                await interrupted
                raise ValueError("Forced process exit unexpectedly delivered an answer")
            except httpx.HTTPError:
                pass
            proxy.release.set()
            recovered = start(f"http://127.0.0.1:{args.proxy_port}/v1", "recovered")
            await recovered.ready(client)
            duplicate = await ask(recovered, cases[1], key=key)
            checks["restart_does_not_resend"] = duplicate["http_status"] == 409 and len(proxy.calls) == 2
            deadline = datetime.fromisoformat(before["deadline_at"])
            await asyncio.sleep(max(0, (deadline - datetime.now(UTC)).total_seconds()) + 0.1)
            after = (await client.get(recovered.url + "/assistant/requests/" + key)).json()
            checks["killed_request_visible_as_unknown"] = after["status"] == "unknown"
            save(output / "kill-window.json", {"remote": proxy.calls[-1], "before": before, "after": after,
                "killed_pid": failed.process.pid, "exit_code": failed.process.returncode,
                "claims": "Remote stop observed, local pre-dispatch persisted, no local complete result, no automatic retransmission"})
            recovered.stop()
            save(output / "fault-proxy.json", {"calls": proxy.calls, "errors": proxy.errors})
            if not all(checks.values()):
                raise ValueError("A frozen service check failed: " + str([key for key, value in checks.items() if not value]))
            return checks
        finally:
            for service in reversed(services):
                service.stop()
            if proxy:
                proxy.close()
            save(output / "checks.json", checks)
            save(output / "processes.json", [{"pid": service.process.pid,
                "exit_code": service.process.poll(), "forced_drill": service.forced} for service in services])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-port", type=int, default=18113)
    parser.add_argument("--proxy-port", type=int, default=18114)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert_free(args.api_port)
    assert_free(args.proxy_port)
    settings = Settings.from_environment()
    rows = source_rows(settings)
    save(args.output / "source-snapshot.json", rows)
    database = "wind_i3_" + uuid4().hex
    password = Path(os.environ["WIND_DB_ADMIN_PASSWORD_FILE"]).read_text().strip()
    outcome = {"status": "running", "database": database, "source_before": digest(rows)}
    save(args.output / "outcome.json", outcome)
    with psycopg.connect(host=settings.database_url.host, port=settings.database_url.port,
            dbname=settings.database_url.database, user="wind_admin", password=password,
            autocommit=True, connect_timeout=5) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {} OWNER wind_app").format(sql.Identifier(database)))
        try:
            with psycopg.connect(host=settings.database_url.host, port=settings.database_url.port,
                    dbname=database, user="wind_admin", password=password, autocommit=True) as setup:
                setup.execute("CREATE EXTENSION vector")
            original_name = os.environ.get("WIND_DB_NAME")
            os.environ["WIND_DB_NAME"] = database
            try:
                command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
                command.check(Config(str(ROOT / "alembic.ini")))
            finally:
                if original_name is None:
                    os.environ.pop("WIND_DB_NAME", None)
                else:
                    os.environ["WIND_DB_NAME"] = original_name
            engine = make_sync_engine(replace(settings, database_url=settings.database_url.set(database=database)))
            try:
                with engine.begin() as connection:
                    for model in (ImportedRun, ImportedArtifact):
                        connection.execute(model.__table__.insert(), rows[model.__tablename__])
                outcome["checks"] = asyncio.run(consume(args, database, rows, args.output), loop_factory=make_event_loop)
                with engine.connect() as connection:
                    for table in ("assistant_requests", "answer_audits"):
                        values = connection.execute(__import__("sqlalchemy").text("SELECT * FROM " + table + " ORDER BY created_at")).mappings().all()
                        save(args.output / (table + ".json"), [dict(row) for row in values])
                outcome["status"] = "technical_checks_passed_semantic_review_pending"
            finally:
                engine.dispose()
        except BaseException as error:
            outcome.update(status="failed", exception=type(error).__name__)
            raise
        finally:
            admin.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s AND pid<>pg_backend_pid()", (database,))
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database)))
            outcome["owned_database_removed"] = True
            outcome["source_after"] = digest(source_rows(settings))
            outcome["source_unchanged"] = outcome["source_before"] == outcome["source_after"]
            save(args.output / "outcome.json", outcome)
    if not outcome["source_unchanged"]:
        raise ValueError("Ordinary source rows changed; inspect before continuing")
    print(json.dumps(outcome, ensure_ascii=False))


if __name__ == "__main__":
    main()
