"""W1 有限负载：先冻结历史参照，再执行一次；中断后只读分析，不重复提交。"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import httpx
import numpy as np
import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
Q1_SOURCE = "docs/results/wind-sequence-round4/final-delivery/verification-a39c567f234a44ffac9abe6dc71b2ee7.json"
ENGIE_SOURCE = "docs/results/wind-engie-a3-20260923/http-delivery.json"
TABLES = ("experiment_tasks", "experiment_attempts", "experiment_runs", "model_artifacts",
          "imported_runs", "imported_artifacts", "engie_deliveries", "answer_audits")
TERMINAL = {"succeeded", "failed"}
SPEC = {"dataset_id": "wind-2019-q1", "purpose": "development",
        "training_policy": "fixed_iterations", "candidate_key": "none",
        "sequence_key": "transformer_direct"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value, *, exclusive=False):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as stream:
        stream.write(encode(value) + "\n")


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(encode(value) + "\n")
        stream.flush()


def command(argv, timeout=15):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", timeout=timeout)
    if result.returncode:
        # 服务错误只报告操作/退出码，不把可能包含连接参数的 stderr 扩散到结果。
        raise RuntimeError(f"command_failed:{argv[0]}:{result.returncode}")
    return result.stdout


def preflight(record, output, checker):
    package = read(record)
    if package["status"] != "in_progress" or Path(package["worktree"]).resolve() != ROOT:
        raise ValueError("active_owned_package_required")
    relative = output.resolve().relative_to(ROOT).as_posix()
    if not any(relative == p.rstrip("/") or relative.startswith(p.rstrip("/") + "/")
               for p in package["allowed_paths"]):
        raise ValueError("output_outside_package")
    # 历史负载执行属于维护者工作包，不猜测兄弟仓库；纯分析函数仍可独立复用。
    if checker is None or not checker.is_file():
        raise ValueError("maintainer_checker_required: pass --checker explicitly")
    command([sys.executable, str(checker), "--record", str(record), "--check-environment"], 30)


async def connect():
    from power_forecast_service.settings import Settings
    url = Settings.from_environment().database_url
    return await psycopg.AsyncConnection.connect(host=url.host, port=url.port,
        user=url.username, password=url.password, dbname=url.database,
        connect_timeout=5, autocommit=True, row_factory=dict_row)


async def inventory(conn):
    # 行身份与完整行 hash 验证旧对象不被修改；不复制整份历史评分数组。
    async with conn.transaction():
        await conn.execute("SET TRANSACTION READ ONLY")
        await conn.execute("SET LOCAL statement_timeout = '3000ms'")
        result = {}
        for table in TABLES:
            cursor = await conn.execute(f"SELECT id::text, md5(row_to_json(t)::text) AS hash FROM {table} t ORDER BY id")
            result[table] = {row["id"]: row["hash"] for row in await cursor.fetchall()}
        return result


async def observation(conn, keys):
    begin = utc()
    async with conn.transaction():
        await conn.execute("SET TRANSACTION READ ONLY")
        await conn.execute("SET LOCAL statement_timeout = '3000ms'")
        cursor = await conn.execute("SELECT clock_timestamp() AS db_now")
        db_time = (await cursor.fetchone())["db_now"]
        cursor = await conn.execute("""
            SELECT t.id::text AS task_id, t.idempotency_key, t.status, t.attempt_count,
                   t.active_attempt_id::text, t.created_at, t.lease_until, t.error_code,
                   a.id::text AS attempt_id, a.number, a.status AS attempt_status,
                   a.started_at, a.finished_at, o.id::text AS outbox_id,
                   o.sent_at, o.publish_count, o.last_error,
                   r.id::text AS run_id, r.attempt_id::text AS run_attempt_id
            FROM experiment_tasks t
            LEFT JOIN experiment_attempts a ON a.task_id=t.id
            LEFT JOIN experiment_outbox o ON o.task_id=t.id
            LEFT JOIN experiment_runs r ON r.task_id=t.id
            WHERE t.idempotency_key = ANY(%s) ORDER BY t.idempotency_key, a.number
        """, (keys,))
        rows = await cursor.fetchall()
    return {"host_begin": begin, "host_end": utc(), "db_now": db_time.isoformat(), "tasks": rows}


def stamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def overlap(row, observations):
    """请求整个不确定时间区间都落在已闭合 attempt 内，才声称任务执行重叠。"""
    if not observations:
        return "unobserved"
    offsets = [(stamp(o["db_now"]) - stamp(o["host_end"]),
                stamp(o["db_now"]) - stamp(o["host_begin"])) for o in observations]
    low, high = min(o[0] for o in offsets), max(o[1] for o in offsets)
    start, end = stamp(row["started_at"]), stamp(row["finished_at"])
    attempts = {t["attempt_id"]: t for o in observations for t in o["tasks"]
                if t.get("started_at") and t.get("finished_at")}
    for attempt in attempts.values():
        a, b = stamp(str(attempt["started_at"])), stamp(str(attempt["finished_at"]))
        if start + low >= a and end + high <= b:
            return "task_execution"
    if any(start + low < stamp(str(t["finished_at"])) and end + high > stamp(str(t["started_at"])) for t in attempts.values()):
        return "boundary_or_clock_uncertain"
    return "outside_task_execution"


def same(expected, actual, *, atol=1e-8):
    """数值沿用原验收容差；带时区的时间比较瞬间，无时区的源时钟不擅自加时区。"""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and same(v, actual[k], atol=atol)
                                               for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(
            same(a, b, atol=atol) for a, b in zip(expected, actual))
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and math.isfinite(actual) and math.isclose(
            expected, actual, rel_tol=1e-8, abs_tol=atol)
    if isinstance(expected, str) and "T" in expected and isinstance(actual, str):
        try:
            a, b = datetime.fromisoformat(expected.replace("Z", "+00:00")), datetime.fromisoformat(actual.replace("Z", "+00:00"))
            return a == b
        except ValueError:
            pass
    return expected == actual


def container_identity():
    prefix = ["docker", "compose", "--profile", "app"]
    ids = command(prefix + ["ps", "-q", "api", "worker", "dispatcher", "postgres", "redis"]).split()
    items = json.loads(command(["docker", "inspect", *ids]))
    selected = []
    for item in items:
        labels = item["Config"]["Labels"]
        if Path(labels["com.docker.compose.project.working_dir"]).resolve() != ROOT:
            raise ValueError("wrong_runtime_checkout")
        if labels["com.docker.compose.project"] != "wind-stage-20260922" or not item["State"]["Running"] or item["State"]["OOMKilled"]:
            raise ValueError("wrong_or_stopped_runtime")
        selected.append({"service": labels["com.docker.compose.service"], "id": item["Id"],
            "image": item["Image"], "pid": item["State"]["Pid"], "started_at": item["State"]["StartedAt"],
            "oom_killed": item["State"]["OOMKilled"], "nano_cpus": item["HostConfig"]["NanoCpus"],
            "cpuset": item["HostConfig"]["CpusetCpus"], "memory_limit": item["HostConfig"]["Memory"],
            "command": item["Config"]["Cmd"],
            "mounts": [{"source": m["Source"], "destination": m["Destination"]} for m in item["Mounts"]]})
    if {s["service"] for s in selected} != {"api", "worker", "dispatcher", "postgres", "redis"}:
        raise ValueError("incomplete_runtime")
    return sorted(selected, key=lambda item: item["service"])


def runtime_identity():
    selected = container_identity()
    prefix = ["docker", "compose", "--profile", "app"]
    probe = """import hashlib,json,os,importlib.metadata as m
from pathlib import Path
root=Path('/app/src')
print(json.dumps({'source':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*.py'))},'corpus_file_sha256':hashlib.sha256((root/'power_forecast_service/assistant/corpus.json').read_bytes()).hexdigest(),'data_sha256':hashlib.sha256(Path('/app/data/wind_2019_q1.csv').read_bytes()).hexdigest(),'versions':{k:m.version(k) for k in ['fastapi','starlette','celery','httpx','torch']},'cpu_count':os.cpu_count(),'threads':{k:os.getenv(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}}))"""
    hosts = {str(p.relative_to(ROOT / "src")).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((ROOT / "src").rglob("*.py"))}
    probes = {}
    for service in ("api", "worker", "dispatcher"):
        data = json.loads(command(prefix + ["exec", "-T", service, "python", "-c", probe]))
        if data["source"] != hosts:
            raise ValueError(f"source_identity_mismatch:{service}")
        if data["corpus_file_sha256"] != hashlib.sha256((ROOT / "src/power_forecast_service/assistant/corpus.json").read_bytes()).hexdigest():
            raise ValueError(f"corpus_identity_mismatch:{service}")
        data["source_digest"] = digest(data.pop("source"))
        probes[service] = data
    return {"containers": selected, "probes": probes, "host_cpu_count": os.cpu_count(),
            "host_versions": {k: importlib.metadata.version(k) for k in ("httpx", "numpy", "psycopg")}}


async def assistant_oracles(questions):
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from power_forecast_service.storage.database import make_async_engine
    from power_forecast_service.settings import Settings
    from power_forecast_service.assistant.contracts import Question
    from power_forecast_service.assistant.evidence import get_results, compare_results
    from power_forecast_service.assistant.retrieval import corpus, keyword_rank, expand_sections
    from power_forecast_service.assistant.workflow import PROMPT_VERSION
    engine = make_async_engine(Settings.from_environment())
    try:
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        pack = corpus()
        result = []
        for payload in questions:
            question = Question.model_validate(payload)
            evidence = compare_results(await get_results(sessions, question.contexts))
            allowed = [c for c in pack["chunks"] if set(c["scopes"]) & set(evidence["scopes"])]
            docs = expand_sections(keyword_rank(question.question, allowed), allowed)
            result.append({"facts": {f["id"]: f for f in evidence["facts"]},
                "citations": {d["id"]: {"id": d["id"], "title": d["title"],
                    "revision": d["source_sha256"], "quote": d["text"], "url": f"/assistant/documents/{d['id']}"} for d in docs},
                "trace": {"contexts": question.model_dump(mode="json")["contexts"], "strategy": "keyword",
                    "mode": "workflow", "prompt_version": PROMPT_VERSION, "corpus_sha256": pack["sha256"],
                    "tools": ["get_result", "compare_results", "retrieve_documents"], "documents": [d["id"] for d in docs]}})
        return result
    finally:
        await engine.dispose()


def verify_assistant(endpoint, body):
    oracle = endpoint["oracle"]
    facts, citations, trace = body.get("facts", []), body.get("citations", []), body.get("trace", {})
    calls = trace.get("model_calls")
    if body.get("status") != "answered" or not isinstance(calls, list) or not 1 <= len(calls) <= 2:
        return False
    return (same(oracle["trace"], trace) and bool(facts or citations)
        and trace.get("fact_ids") == [f["id"] for f in facts]
        and all(f == oracle["facts"].get(f["id"]) for f in facts)
        and all(c == oracle["citations"].get(c["id"]) for c in citations)
        and all(call.get("status") == "returned" for call in calls))


async def prepare(output, base_url):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("protocol_already_frozen")
    runtime = await asyncio.to_thread(runtime_identity)
    q1 = read(ROOT / Q1_SOURCE)
    replay = next(r for r in q1["replays"] if r["forecast"]["model_key"] == "ridge_0_1")
    engie = next(c for c in read(ROOT / ENGIE_SOURCE)["cases"] if c["kind"] == "normal")["service"]["record"]["result"]
    run_id, task_id = q1["run"]["run_id"], q1["task"]["task_id"]
    async with await connect() as conn:
        before = await inventory(conn)
        cursor = await conn.execute("SELECT id::text,status FROM experiment_tasks WHERE status NOT IN ('succeeded','failed')")
        if await cursor.fetchall():
            raise ValueError("unrelated_active_experiment")
    async with httpx.AsyncClient(base_url=base_url, trust_env=False, timeout=10) as client:
        health = await client.get("/health")
        health.raise_for_status()
        task = (await client.get(f"/tasks/{task_id}")).json()
        run = (await client.get(f"/runs/{run_id}")).json()
        if run["task_id"] != task_id or task["status"] != "succeeded":
            raise ValueError("old_q1_identity_invalid")
        if not same(q1["run"], run) or not same(q1["task"], task):
            raise ValueError("historical_q1_response_changed")
        imports = (await client.get("/engie/imports")).json()
        final = next(r for r in imports if r["import_id"] == engie["import_id"] and r["scope"] == "final_2015")
        if not any(m["artifact_id"] == engie["artifact_id"] for m in final["models"]):
            raise ValueError("old_engie_identity_invalid")
    from power_forecast_service.experiments.comparison import compare_results
    expected = json.loads(compare_results(run_id, q1["run"]["result"], "persistence", run_id,
                                        q1["run"]["result"], "ridge_0_1").model_dump_json())
    if expected["status"] != "comparable":
        raise ValueError("old_q1_comparison_invalid")
    compare_params = {"left_run_id": run_id, "right_run_id": run_id,
                      "left_model": "persistence", "right_model": "ridge_0_1"}
    rows = q1["run"]["result"]["scoring"]["rows"]
    prediction_file = ROOT / ".local/work-packages/wind-engie-a3-20260923/artifacts/2015-predictions.npz"
    if hashlib.sha256(prediction_file.read_bytes()).hexdigest() != "b92d62b7c8f0e33d2fb403f8ae9a7d91a10d2ffda299c2cb51efed195eb98679":
        raise ValueError("engie_saved_prediction_changed")
    # 原HTTP参照另与原始逐点数组核对，不能以首次在线响应自证正确。
    with np.load(prediction_file, allow_pickle=False) as saved:
        idx = np.flatnonzero(saved["issue_ns"] == int(datetime.fromisoformat(engie["issue_time"]).timestamp() * 1e9))
        if len(idx) != 1 or not saved["input_valid"][idx[0]]:
            raise ValueError("engie_issue_identity_invalid")
        frozen_values = saved["prediction_lightgbm_l1_shrink"][idx[0]]
        np.testing.assert_allclose(frozen_values, engie["predictions"], atol=1e-6, rtol=1e-8)
        # HTTP的点顺序由下面预执行核对检查；完整预测oracle来自已冻结历史HTTP。
        save(output / "engie-array-oracle.json", {"index": int(idx[0]), "predictions": frozen_values.tolist(), "source_sha256": hashlib.sha256(prediction_file.read_bytes()).hexdigest()})
    endpoints = [
        {"name": "task", "method": "GET", "path": f"/tasks/{task_id}", "expected_hash": digest(task)},
        {"name": "run", "method": "GET", "path": f"/runs/{run_id}", "expected_hash": digest(run)},
        {"name": "q1_forecast", "method": "POST", "path": "/forecasts",
         "json": {"artifact_id": replay["forecast"]["artifact_id"], "observations": replay["history"]}, "expected": replay["forecast"]},
        {"name": "engie_replay", "method": "POST", "path": "/engie/replays",
         "json": {"artifact_id": engie["artifact_id"], "issue_time": datetime.fromisoformat(engie["issue_time"]).astimezone(timezone.utc).isoformat()},
         "expected": {k: v for k, v in engie.items() if k != "input_sha256"}, "atol": 1e-6},
        {"name": "compare", "method": "GET", "path": "/runs/compare", "params": compare_params, "expected": expected},
        {"name": "compare_series", "method": "GET", "path": "/runs/compare-series", "params": {**compare_params, "limit": 24},
         "expected_comparison": expected, "expected_total": len(rows),
         "expected_rows": [{"cutoff": r["cutoff"], "target_time": r["target_time"], "actual": r["actual"],
                            "left_prediction": r["predictions"]["persistence"], "right_prediction": r["predictions"]["ridge_0_1"]} for r in rows[:24]]},
    ]
    questions = [{"question": "Q1 推荐模型为什么没有采用 Transformer？", "contexts": [{"kind": "q1_run", "id": run_id}]},
        {"question": "ENGIE 开发与最终评价改善分别是多少，为什么默认仍是持久性？",
         "contexts": [{"kind": "engie_import", "id": engie["import_id"]}]}]
    from power_forecast_service.settings import Settings
    from power_forecast_service.experiments.contracts import ExperimentRequest, freeze_spec
    frozen_spec = freeze_spec(ExperimentRequest.model_validate(SPEC), Settings.from_environment())
    protocol = {"created_at": utc(), "base_url": base_url, "endpoints": endpoints, "questions": questions,
        "assistant_oracles": await assistant_oracles(questions), "frozen_spec": frozen_spec,
        "spec": SPEC, "keys": ["w1-20260930-development-1", "w1-20260930-development-2"],
        "concurrency": [1, 2, 4], "attempts_per_cell": 30, "percentile_method": "linear",
        "connection_limits": {"max_connections": 8, "max_keepalive_connections": 8, "keepalive_expiry": 30},
        "warm_wall_seconds": 10, "first_wall_seconds": 60, "assistant_wall_seconds": 90,
        "max_training": 2, "max_provider_calls": 4, "max_task_wait_seconds": 600,
        "resource_stop": "OOM, expired lease, failed/retry/unknown task, wrong output, unknown remote outcome",
        "oracles": [{"path": Q1_SOURCE, "sha256": hashlib.sha256((ROOT / Q1_SOURCE).read_bytes()).hexdigest()},
                    {"path": ENGIE_SOURCE, "sha256": hashlib.sha256((ROOT / ENGIE_SOURCE).read_bytes()).hexdigest()}]}
    save(output / "protocol.json", protocol, exclusive=True)
    save(output / "runtime.json", runtime)
    save(output / "inventory-before.json", before)
    save(output / "old-run.json", run)
    save(output / "freeze.json", {name: digest(read(output / name)) for name in
        ("protocol.json", "runtime.json", "inventory-before.json", "old-run.json", "engie-array-oracle.json")}, exclusive=True)
    print(encode({"prepared": True, "output": str(output), "runtime": [s["service"] for s in runtime["containers"]],
                  "old_counts": {k: len(v) for k, v in before.items()}, "protocol_sha256": digest(protocol)}), flush=True)


def verify_response(endpoint, body):
    if "expected_hash" in endpoint:
        return digest(body) == endpoint["expected_hash"]
    if "expected" in endpoint:
        if endpoint["name"] == "engie_replay":
            # HTTP回放包有history/actual外壳；本轮oracle验证的是其中的固定预测。
            body = body.get("forecast")
        return same(endpoint["expected"], body, atol=endpoint.get("atol", 1e-8))
    return same(endpoint["expected_comparison"], body.get("comparison")) and body.get("total") == endpoint["expected_total"] and body.get("offset") == 0 and same(endpoint["expected_rows"], body.get("rows"))


class Measurement:
    def __init__(self, output, protocol):
        self.output, self.protocol = output, protocol
        self.stop = None
        self.latest = None
        self.latest_monotonic = 0.0
        self.sequence = 0
        self.inflight = 0
        self.max_inflight = 0
        self.monitor_finished = asyncio.Event()
        self.observer_ready = asyncio.Event()
        self.resource_ready = asyncio.Event()

    def halt(self, reason):
        # 保留第一个停线原因；后续收尾错误另记，不能覆盖最初现场。
        if self.stop is None:
            self.stop = reason

    def running(self):
        return self.latest and time.monotonic() - self.latest_monotonic < 2 and any(
            t["status"] == "running" for t in self.latest["tasks"])

    async def request(self, client, endpoint, purpose, concurrency=1, wall=10, headers=None):
        self.sequence += 1
        row = {"request_id": self.sequence, "purpose": purpose, "endpoint": endpoint["name"],
               "target_concurrency": concurrency, "request_sha256": digest(endpoint), "started_at": utc()}
        begin = time.perf_counter()
        body = None
        append(self.output / "request-intents.jsonl", row)
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        row["observed_inflight_at_start"] = self.inflight
        try:
            async with asyncio.timeout(wall):
                response = await client.request(endpoint["method"], endpoint["path"],
                    params=endpoint.get("params"), json=endpoint.get("json"), headers=headers,
                    timeout=httpx.Timeout(read=wall, connect=3, write=3, pool=3))
            row.update(finished_at=utc(), elapsed_ms=(time.perf_counter() - begin) * 1000,
                       status=response.status_code, bytes=len(response.content), body_sha256=hashlib.sha256(response.content).hexdigest())
            decode_start = time.perf_counter()
            body = response.json()
            row["reason"] = body.get("detail", body.get("error")) if isinstance(body, dict) else None
            if response.status_code == 200:
                row["correct"] = verify_response(endpoint, body) if purpose != "assistant" else verify_assistant(endpoint, body)
                row["category"] = "accepted_correct" if row["correct"] else "wrong_output"
                if purpose == "assistant" and row["correct"]:
                    row.update(category="accepted_evidence_checked", semantic_review="pending")
            elif response.status_code == 503 and row["reason"] == "forecast_capacity_busy":
                row["category"] = "capacity_rejected"
            elif response.status_code == 429 and row["reason"] == "assistant_busy":
                row["category"] = "assistant_rejected"
            elif response.status_code == 202 and purpose == "submission":
                row["category"] = "submitted"
                if not isinstance(body, dict) or not body.get("task_id"):
                    row["category"] = "wrong_output"
            elif response.status_code == 409 and purpose == "conflict":
                row["category"] = "expected_conflict"
            else:
                row["category"] = "http_error"
            row["decode_validation_ms"] = (time.perf_counter() - decode_start) * 1000
            if row["category"] in {"wrong_output", "http_error"}:
                self.halt(row["category"])
                save(self.output / f"error-response-{row['request_id']}.json", body)
            if purpose in {"assistant", "submission", "conflict"}:
                append(self.output / "action-responses.jsonl", {"request": row, "body": body})
        except (httpx.HTTPError, TimeoutError) as exc:
            body = None
            row.update(finished_at=utc(), elapsed_ms=(time.perf_counter() - begin) * 1000,
                       category="timeout_or_transport", error_type=type(exc).__name__)
            self.halt("unknown_inflight_after_transport")
        except Exception as exc:
            row.update(category="measurement_error", error_type=type(exc).__name__)
            self.halt("measurement_error")
            if body is not None:
                save(self.output / f"error-response-{row['request_id']}.json", body)
        finally:
            row.setdefault("finished_at", utc())
            row.setdefault("elapsed_ms", (time.perf_counter() - begin) * 1000)
            self.inflight -= 1
            append(self.output / "requests.jsonl", row)
        return row, body

    async def observe(self):
        try:
            async with await connect() as conn:
                while not self.monitor_finished.is_set():
                    current = await observation(conn, self.protocol["keys"])
                    append(self.output / "observations.jsonl", current)
                    self.latest, self.latest_monotonic = current, time.monotonic()
                    self.observer_ready.set()
                    db_now = stamp(current["db_now"])
                    for task in current["tasks"]:
                        if task["status"] in {"failed", "retry_wait"} or task["attempt_status"] in {"failed", "expired"} or task["attempt_count"] > 1:
                            self.halt("task_failed_or_retry")
                        if task["status"] == "running" and db_now > stamp(str(task["lease_until"])):
                            self.halt("expired_task_lease")
                    await asyncio.sleep(0.5)
        except Exception as exc:
            self.halt("observer_unavailable")
            append(self.output / "observations.jsonl", {"error_type": type(exc).__name__, "at": utc()})
        finally:
            self.observer_ready.set()

    async def resources(self):
        frozen = read(self.output / "runtime.json")["containers"]
        names = [s["id"] for s in frozen if s["service"] in {"api", "worker", "postgres"}]
        while not self.monitor_finished.is_set():
            begin = utc()
            try:
                current = await asyncio.to_thread(container_identity)
                if current != frozen:
                    self.halt("runtime_identity_changed")
                    append(self.output / "resources.jsonl", {"at": utc(), "identity_changed": True, "containers": current})
                    return
                self.resource_ready.set()
                raw = await asyncio.to_thread(command, ["docker", "stats", "--no-stream", "--format", "{{json .}}", *names], 8)
                append(self.output / "resources.jsonl", {"started_at": begin, "finished_at": utc(),
                    "containers": [json.loads(line) for line in raw.splitlines() if line]})
            except Exception as exc:
                self.halt("resource_observer_unavailable")
                self.resource_ready.set()
                append(self.output / "resources.jsonl", {"started_at": begin, "error_type": type(exc).__name__})
                return
            await asyncio.sleep(1)

    async def matrix(self, client, purpose):
        # 各格轮转小批次，避免长请求让后面的端点完全错过有限任务窗口。
        counts = {(e["name"], c): 0 for e in self.protocol["endpoints"] for c in self.protocol["concurrency"]}
        while not self.stop and any(n < 30 for n in counts.values()):
            for endpoint in self.protocol["endpoints"]:
                for c in self.protocol["concurrency"]:
                    key = endpoint["name"], c
                    n = min(c, 30 - counts[key])
                    if not n:
                        continue
                    if self.stop or (purpose == "task_background" and not self.running()):
                        save(self.output / f"{purpose}-cells.json", {f"{a}:{b}": v for (a, b), v in counts.items()})
                        return
                    barrier = asyncio.Event()
                    async def one():
                        await barrier.wait()
                        return await self.request(client, endpoint, purpose, c)
                    pending = [asyncio.create_task(one()) for _ in range(n)]
                    barrier.set()
                    await asyncio.gather(*pending)
                    counts[key] += n
            print(encode({"phase": purpose, "attempts": sum(counts.values()), "stop": self.stop}), flush=True)
        save(self.output / f"{purpose}-cells.json", {f"{a}:{b}": v for (a, b), v in counts.items()})

    async def assistant(self, client):
        endpoints = [{"name": f"assistant_{i+1}", "method": "POST", "path": "/assistant/answers", "json": q,
                      "oracle": self.protocol["assistant_oracles"][i]}
                     for i, q in enumerate(self.protocol["questions"])]
        first = asyncio.create_task(self.request(client, endpoints[0], "assistant", wall=90))
        await asyncio.sleep(0.1)
        if self.stop:
            result = await first
            save(self.output / "assistant-summary.json", {"calls": None, "accepted_questions": None,
                "responses": [result[0]], "unknown_requests": 1, "stop": self.stop})
            return
        second_task = asyncio.create_task(self.request(client, endpoints[1], "assistant", wall=90))
        companion = asyncio.create_task(self.request(client, self.protocol["endpoints"][2], "assistant_companion"))
        first_result, second, forecast = await asyncio.gather(first, second_task, companion)
        if second[0]["category"] == "assistant_rejected" and first_result[0]["category"] == "accepted_evidence_checked" and not self.stop:
            second = await self.request(client, endpoints[1], "assistant", wall=90)
        answers = [r for r in (first_result, second) if r[0]["category"] != "assistant_rejected"]
        known = all(r[1] is not None and isinstance(r[1].get("trace", {}).get("model_calls"), list) for r in answers)
        calls = sum(len(r[1].get("trace", {}).get("model_calls", [])) for r in answers) if known else None
        if calls is not None and calls > 4:
            self.halt("provider_budget_exceeded")
        save(self.output / "assistant-summary.json", {"calls": calls, "accepted_questions": len(answers) if known else None,
            "unknown_requests": sum(r[1] is None for r in answers),
            "responses": [r[0] for r in (first_result, second)], "forecast": forecast[0]})


def validate_frozen(output):
    frozen = read(output / "freeze.json")
    if any(digest(read(output / name)) != sha for name, sha in frozen.items()):
        raise ValueError("frozen_inputs_changed")
    protocol = read(output / "protocol.json")
    if (protocol["spec"] != SPEC or len(protocol["keys"]) != 2 or len(set(protocol["keys"])) != 2
        or protocol["max_training"] != 2 or protocol["max_provider_calls"] != 4
        or len(protocol["questions"]) != 2 or protocol["concurrency"] != [1, 2, 4]
        or protocol["attempts_per_cell"] != 30 or protocol["base_url"] != "http://127.0.0.1:18000"
        or [e["name"] for e in protocol["endpoints"]] != ["task", "run", "q1_forecast", "engie_replay", "compare", "compare_series"]):
        raise ValueError("frozen_scope_invalid")
    for source in protocol["oracles"]:
        if hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("historical_oracle_changed")
    return protocol


async def verify_new_tasks(conn, protocol, output):
    evidence = []
    frozen = protocol["frozen_spec"]
    for key in protocol["keys"]:
        cursor = await conn.execute("SELECT * FROM experiment_tasks WHERE idempotency_key=%s", (key,))
        tasks = await cursor.fetchall()
        assert len(tasks) == 1, "unique_task_required"
        task = tasks[0]
        assert task["status"] == "succeeded" and task["spec"] == frozen
        assert task["error_code"] is None and task["lease_until"] is None and task["source_task_id"] is None
        cursor = await conn.execute("SELECT * FROM experiment_attempts WHERE task_id=%s", (task["id"],))
        attempts = await cursor.fetchall()
        assert len(attempts) == task["attempt_count"] == 1
        assert attempts[0]["status"] == "succeeded" and attempts[0]["id"] == task["active_attempt_id"]
        cursor = await conn.execute("SELECT * FROM experiment_runs WHERE task_id=%s", (task["id"],))
        runs = await cursor.fetchall()
        assert len(runs) == 1 and runs[0]["attempt_id"] == task["active_attempt_id"]
        run = runs[0]
        result = run["result"]
        assert result["frozen_spec"] == frozen and result["model_set"] == frozen["model_set"]
        assert result["purpose"] == "development" and result["evaluation_split"] == "validation"
        assert result["split"]["test_scored"] is False
        assert len(result["scoring"]["rows"]) == 3872
        assert result["scoring"]["samples_sha256"] == "b5626097acd5a2b1fcb20c905bb0e9414f90734e642870ba7589f014e4ebb9a0"
        training = result["candidate_training"]["transformer_direct"]
        assert training["config"] == frozen["sequence_recipe"] and training["seed"] == 42 and training["device"] == "cpu"
        assert training["epochs"] == 20 and len(training["train_loss"]) == 20 and all(math.isfinite(v) for v in training["train_loss"])
        cursor = await conn.execute("SELECT * FROM model_artifacts WHERE run_id=%s", (run["id"],))
        artifacts = await cursor.fetchall()
        assert len(artifacts) == 4 and {a["model_key"] for a in artifacts} == set(frozen["model_set"])
        registered = [{"model_key": a["model_key"], "artifact_id": str(a["id"]), "manifest_sha256": a["manifest_sha256"]} for a in artifacts]
        assert sorted(registered, key=lambda a: a["model_key"]) == sorted(result["model_artifacts"], key=lambda a: a["model_key"])
        assert len(result["model_verification"]) == 4
        for verified in result["model_verification"]:
            assert {k: verified[k] for k in ("model_key", "artifact_id", "manifest_sha256")} in registered
            assert verified["samples"] == 3
        for artifact in artifacts:
            manifest = artifact["manifest"]
            assert artifact["status"] == "ready"
            for field in ("task_id", "attempt_id", "run_id", "artifact_id"):
                expected = {"task_id": task["id"], "attempt_id": task["active_attempt_id"], "run_id": run["id"], "artifact_id": artifact["id"]}[field]
                assert manifest[field] == str(expected)
            assert manifest["frozen_spec"] == frozen
            assert manifest["provenance"] == result["execution"]
        evidence.append({"key": key, "task_id": task["id"], "run_id": run["id"], "attempt_id": task["active_attempt_id"],
                         "spec_sha256": digest(frozen), "models": [{"id": a["id"], "key": a["model_key"], "status": a["status"], "manifest_sha256": a["manifest_sha256"]} for a in artifacts]})
    assert len({r["task_id"] for r in evidence}) == 2
    save(output / "new-task-verification.json", {"verified": True, "tasks": evidence})


async def run(output):
    protocol = validate_frozen(output)
    if (output / "execution-started.json").exists():
        raise ValueError("execution_already_started_read_only_analysis_required")
    if await asyncio.to_thread(runtime_identity) != read(output / "runtime.json"):
        raise ValueError("runtime_changed_after_freeze")
    async with await connect() as conn:
        existing = await observation(conn, protocol["keys"])
        if existing["tasks"]:
            raise ValueError("frozen_keys_already_exist_no_resubmit")
        if await inventory(conn) != read(output / "inventory-before.json"):
            raise ValueError("inventory_changed_after_freeze")
    save(output / "execution-started.json", {"started_at": utc(), "protocol_sha256": digest(protocol)}, exclusive=True)
    measurement = Measurement(output, protocol)
    observers = [asyncio.create_task(measurement.observe()), asyncio.create_task(measurement.resources())]
    error_type = None
    try:
        await asyncio.wait_for(asyncio.gather(measurement.observer_ready.wait(), measurement.resource_ready.wait()), timeout=15)
        async with httpx.AsyncClient(base_url=protocol["base_url"], trust_env=False,
                limits=httpx.Limits(**protocol["connection_limits"])) as client:
            for endpoint in protocol["endpoints"]:
                if measurement.stop:
                    break
                await measurement.request(client, endpoint, "first_observed", wall=60)
                if measurement.stop:
                    break
                await measurement.request(client, endpoint, "warmup")
            if not measurement.stop:
                await measurement.matrix(client, "idle_warm")
            receipts = []
            for index, key in enumerate(protocol["keys"]):
                if measurement.stop:
                    break
                endpoint = {"name": f"submit_{index+1}", "method": "POST", "path": "/experiments", "json": protocol["spec"]}
                # 先持久化发送意图。任何中断都只读原 key，不自动重发可能已受理的请求。
                append(output / "submit-intents.jsonl", {"key": key, "spec": protocol["spec"], "started_at": utc()})
                row, body = await measurement.request(client, endpoint, "submission", headers={"Idempotency-Key": key})
                if row["category"] == "submitted":
                    receipts.append(body)
            if not measurement.stop:
                key = protocol["keys"][0]
                row, body = await measurement.request(client, {"name": "same_key", "method": "POST", "path": "/experiments", "json": protocol["spec"]},
                    "submission", headers={"Idempotency-Key": key})
                if body is None or body.get("task_id") != receipts[0]["task_id"]:
                    measurement.halt("same_key_task_mismatch")
                if not measurement.stop:
                    await measurement.request(client, {"name": "different_parameters", "method": "POST", "path": "/experiments", "json": {**protocol["spec"], "sequence_key": "transformer_delta"}},
                        "conflict", headers={"Idempotency-Key": key})
                deadline = time.monotonic() + 30
                while not measurement.running() and not measurement.stop and time.monotonic() < deadline:
                    await asyncio.sleep(0.2)
                if measurement.running() and not measurement.stop:
                    try:
                        info = {}
                        for action in ("active", "reserved"):
                            raw = await asyncio.to_thread(command, ["docker", "compose", "--profile", "app", "exec", "-T", "worker", "celery", "-A", "power_forecast_service.jobs.worker:app", "inspect", action, "--json", "--timeout=1"], 8)
                            info[action] = json.loads(raw)
                        save(output / "celery-observation.json", {"at": utc(), "observations": info})
                    except Exception as exc:
                        save(output / "celery-observation.json", {"at": utc(), "error_type": type(exc).__name__})
                    await measurement.matrix(client, "task_background")
                print(encode({"phase": "waiting_for_tasks", "stop": measurement.stop}), flush=True)
            deadline = time.monotonic() + protocol["max_task_wait_seconds"]
            while time.monotonic() < deadline:
                if measurement.latest and len({t["task_id"] for t in measurement.latest["tasks"]}) == 2 and all(t["status"] in TERMINAL for t in measurement.latest["tasks"]):
                    break
                if not (output / "submit-intents.jsonl").exists() or measurement.stop == "observer_unavailable":
                    break
                await asyncio.sleep(1)
            if not measurement.stop:
                tasks = measurement.latest["tasks"] if measurement.latest else []
                if len({t["task_id"] for t in tasks}) != 2 or not all(t["status"] == "succeeded" for t in tasks):
                    measurement.halt("tasks_not_successful")
            if not measurement.stop:
                async with await connect() as conn:
                    await verify_new_tasks(conn, protocol, output)
            if not measurement.stop:
                await measurement.assistant(client)
    except Exception as exc:
        error_type = type(exc).__name__
        measurement.halt("execution_exception")
        append(output / "execution-errors.jsonl", {"at": utc(), "error_type": error_type, "message": str(exc) if isinstance(exc, AssertionError) else "details_retained_in_local_exception"})
    finally:
        measurement.monitor_finished.set()
        await asyncio.gather(*observers, return_exceptions=True)
        try:
            async with await connect() as conn:
                save(output / "inventory-after.json", await inventory(conn))
                save(output / "final-observation.json", await observation(conn, protocol["keys"]))
                ids = [r["body"]["id"] for r in lines(output / "action-responses.jsonl")
                       if r["request"]["purpose"] == "assistant" and r.get("body") and r["body"].get("id")]
                cursor = await conn.execute("SELECT id::text,status,question_sha256,answer_sha256,trace FROM answer_audits WHERE id=ANY(%s::uuid[])", (ids,))
                save(output / "assistant-audits.json", await cursor.fetchall())
        except Exception as exc:
            measurement.halt("final_database_unconfirmed")
            append(output / "execution-errors.jsonl", {"at": utc(), "error_type": type(exc).__name__, "phase": "final_database"})
        save(output / "execution-finished.json", {"finished_at": utc(), "stop": measurement.stop,
             "error_type": error_type, "max_measured_inflight": measurement.max_inflight, "requests": measurement.sequence})
    print(encode({"execution_finished": True, "stop": measurement.stop, "requests": measurement.sequence}), flush=True)
    if measurement.stop:
        raise RuntimeError("execution_stopped_inspect_saved_evidence")


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line] if path.exists() else []


def analyze(output):
    observations = [o for o in lines(output / "observations.jsonl") if "db_now" in o]
    final = read(output / "final-observation.json")
    observations.append(final)
    grouped = defaultdict(list)
    for row in lines(output / "requests.jsonl"):
        row["overlap"] = overlap(row, observations)
        grouped[(row["purpose"], row["endpoint"], row["target_concurrency"], row["overlap"], row["category"])].append(row["elapsed_ms"])
    summaries = []
    for key, values in sorted(grouped.items()):
        summaries.append(dict(zip(("purpose", "endpoint", "concurrency", "overlap", "category"), key),
            n=len(values), p50_ms=float(np.quantile(values, .5, method="linear")),
            p95_ms=float(np.quantile(values, .95, method="linear")), max_ms=max(values)))
    before, after = read(output / "inventory-before.json"), read(output / "inventory-after.json")
    changes = {table: [key for key, value in rows.items() if after[table].get(key) != value]
               for table, rows in before.items()}
    additions = {table: sorted(set(after[table]) - set(before[table])) for table in TABLES}
    expected_additions = {"experiment_tasks": 2, "experiment_attempts": 2, "experiment_runs": 2,
        "model_artifacts": 8, "imported_runs": 0, "imported_artifacts": 0, "engie_deliveries": 0, "answer_audits": 2}
    audit_verified = []
    audits = {a["id"]: a for a in read(output / "assistant-audits.json")} if (output / "assistant-audits.json").exists() else {}
    for action in lines(output / "action-responses.jsonl"):
        row, body = action["request"], action["body"]
        if row["purpose"] != "assistant" or row["category"] != "accepted_evidence_checked":
            continue
        audit = audits.get(body["id"])
        from power_forecast_service.assistant.evidence import digest as assistant_digest
        verified = bool(audit) and audit["status"] == body["status"] and audit["answer_sha256"] == assistant_digest({k: v for k, v in body.items() if k != "trace"})
        audit_verified.append({"answer_id": body["id"], "verified": verified,
            "call_records": len(audit["trace"]["model_calls"]) if audit else None})
    queued_overlap = any(any(t["status"] == "running" for t in o["tasks"]) and
                         any(t["status"] == "queued" for t in o["tasks"]) for o in observations)
    save(output / "summary.json", {"groups": summaries, "old_row_changes": changes,
        "new_ids": additions, "running_and_queued_observed": queued_overlap,
        "addition_counts_expected": expected_additions,
        "addition_counts_match": all(len(additions[t]) == n for t, n in expected_additions.items()),
        "assistant_audit_verification": audit_verified,
        "tasks": final["tasks"], "execution": read(output / "execution-finished.json"),
        "categories": dict(Counter(r["category"] for r in lines(output / "requests.jsonl")))})
    print(encode({"analyzed": True, "old_changed": sum(map(len, changes.values())),
                  "running_and_queued": queued_overlap, "group_count": len(summaries)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checker", type=Path, help="维护者工作包检查器；历史负载入口不属于公开启动流程")
    parser.add_argument("--base-url", default=os.getenv("WIND_TEST_BASE_URL", "http://127.0.0.1:18000"))
    args = parser.parse_args()
    preflight(args.record.resolve(), args.output.resolve(), args.checker)
    if args.base_url != "http://127.0.0.1:18000":
        raise ValueError("explicit_current_18000_stack_required")
    if args.action == "analyze":
        analyze(args.output)
    else:
        # Windows默认Proactor不能被psycopg异步连接使用；不更改项目环境或服务。
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(prepare(args.output, args.base_url) if args.action == "prepare" else run(args.output))


if __name__ == "__main__":
    main()
