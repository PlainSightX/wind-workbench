"""公开原模型回放检查；需要已启动独立服务，不训练、不调用模型API。"""

import argparse
import asyncio
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4
from uuid import UUID
from urllib.request import Request, ProxyHandler, build_opener

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from power_forecast_service.settings import Settings
from power_forecast_service.storage.database import make_sync_engine, make_async_engine
from power_forecast_service.storage.engie_packages import checked_bytes
from power_forecast_service.storage.models import ImportedArtifact, ImportedRun, Run


def client(base):
    opener = build_opener(ProxyHandler({}))

    def call(method, path, body=None):
        payload = json.dumps(body).encode() if body is not None else None
        request = Request(base + path, method=method, data=payload,
                          headers={"Content-Type": "application/json"})
        with opener.open(request, timeout=90) as response:
            return json.load(response)
    return call


async def check_bindings(settings, new_run, final_id):
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from power_forecast_service.assistant.contracts import ContextRef
    from power_forecast_service.assistant.evidence import get_results

    engine = make_async_engine(settings)
    try:
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        fresh = await get_results(sessions, [ContextRef(kind="q1_run", id=new_run, model="persistence")])
        assert fresh["records"][0]["scope"] == "q1_unbound"
        assert "stage_boundary" not in fresh["records"][0]
        original = await get_results(sessions, [ContextRef(kind="engie_import", id=final_id, model="persistence")])
        assert original["records"][0]["scope"] == "engie_final"
        assert original["records"][0]["stage_boundary"]
        assert any(f["stage"] == "development_summary" for f in original["facts"])
        return {"fresh_scope": "q1_unbound", "original_scope": "engie_final",
                "new_experiment_does_not_inherit_historical_selection": True}
    finally:
        await engine.dispose()


def verify(base, output, readback=False, q1_run_id=None, *, request_key=None, delivery_prepared=None):
    settings = Settings.from_environment()
    call = client(base)
    if readback:
        previous = json.loads(output.read_text("utf-8"))
        for record in previous["replays"]:
            assert call("POST", "/engie/replays", record["request"]) == record["response"]
        delivery = previous["delivery"]
        assert call("GET", f"/engie/deliveries/{delivery['id']}") == delivery
        return {"status": "passed", "replays": len(previous["replays"]), "persisted_delivery_identical": True,
                "training_performed": False}

    engine = make_sync_engine(settings)
    records = []
    try:
        with Session(engine) as session:
            imports = session.scalars(select(ImportedRun).where(ImportedRun.quarter == "2015-final")).all()
            assert len(imports) == 1 and imports[0].quarter == "2015-final"
            final_id = imports[0].id
            query = select(Run).where(Run.id == q1_run_id) if q1_run_id else select(Run)
            runs = session.scalars(query).all()
            assert len(runs) == 1
            new_run = runs[0].id
            artifacts = session.scalars(select(ImportedArtifact).where(ImportedArtifact.import_id == final_id)).all()
            assert len(artifacts) == 5
            for item in artifacts:
                manifest = imports[0].manifest
                arrays = checked_bytes(settings.artifact_root, manifest["predictions_path"], manifest["predictions_sha256"])
                with np.load(BytesIO(arrays), allow_pickle=False) as saved:
                    legal = np.flatnonzero(saved["scoreable"])
                    for index in (legal[0], legal[len(legal) // 2], legal[-1]):
                        issue = datetime.fromtimestamp(int(saved["issue_ns"][index]) / 1e9,
                                                       tz=timezone.utc).isoformat()
                        body = {"artifact_id": str(item.id), "issue_time": issue}
                        response = call("POST", "/engie/replays", body)
                        predicted = np.asarray(response["forecast"]["predictions"])
                        expected = saved[f"prediction_{item.family}"][index]
                        np.testing.assert_allclose(predicted, expected, atol=1e-6, rtol=1e-8)
                        assert response["forecast"]["unit"] == "kW"
                        assert response["forecast"]["import_id"] == str(final_id)
                        records.append({"family": item.family, "request": body, "response": response,
                                        "max_abs_difference_kw": float(np.abs(predicted - expected).max())})
    finally:
        engine.dispose()
    record = records[-1]
    body = {**record["request"], "history": record["response"]["history"],
            "request_key": request_key or "public-replay-" + uuid4().hex, "budget_ms": 60000}
    if delivery_prepared is not None:
        delivery_prepared(body)
    published = call("POST", "/engie/deliveries", body)
    assert published["status"] == "published"
    assert call("POST", "/engie/deliveries", body) == published
    assert call("GET", f"/engie/deliveries/{published['id']}") == published
    bindings = asyncio.run(check_bindings(settings, new_run, final_id))
    return {"status": "passed", "replays": records, "delivery": published, "bindings": bindings,
            "training_performed": False, "replay_scope": "15 selected original scored windows, not new holdout"}


def write_json_atomic(path, value, *, replace=False):
    """同目录完整落盘后发布；exclusive模式不会覆盖并发创建的已有回执。"""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def publish_pending(destination):
    pending = destination.with_name(destination.name + ".pending.json")
    record = json.loads(pending.read_text("utf-8"))
    if record.get("status") != "complete" or not isinstance(record.get("result"), dict):
        raise ValueError("receipt_not_complete: reconcile the recorded request_key; do not start another publication")
    write_json_atomic(destination, record["result"])
    pending.unlink()
    return record["result"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readback", action="store_true")
    parser.add_argument("--recover-receipt", action="store_true", help="只发布已完整保存的pending回执，不发送HTTP")
    parser.add_argument("--q1-run-id", type=UUID, help="多实验实例必须显式选择新开发run，不猜测最新对象")
    args = parser.parse_args(argv)
    if args.readback and args.recover_receipt:
        parser.error("--readback and --recover-receipt cannot be combined")
    destination = args.output.with_name("restart-readback.json") if args.readback else args.output
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if args.recover_receipt:
        result = publish_pending(destination)
    else:
        pending = destination.with_name(destination.name + ".pending.json")
        if pending.exists() or pending.is_symlink():
            raise FileExistsError(pending)
        if args.readback and not args.output.is_file():
            raise FileNotFoundError(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # request_key必须在HTTP前持久化；远端结果未知时保留它，拒绝用新key盲目重发。
        request_key = "public-replay-" + uuid4().hex
        journal = {"status": "started", "url": args.url, "request_key": request_key, "readback": args.readback}
        write_json_atomic(pending, journal)
        def delivery_prepared(body):
            journal.update(status="delivery_prepared", delivery_request=body)
            write_json_atomic(pending, journal, replace=True)
        result = verify(args.url, args.output, args.readback, args.q1_run_id,
                        request_key=request_key, delivery_prepared=delivery_prepared)
        journal.update(status="complete", result=result)
        write_json_atomic(pending, journal, replace=True)
        publish_pending(destination)
    replay_count = result["replays"] if isinstance(result["replays"], int) else len(result["replays"])
    print(json.dumps({"status": result["status"], "receipt": destination.name, "replays": replay_count}))


if __name__ == "__main__":
    main()
