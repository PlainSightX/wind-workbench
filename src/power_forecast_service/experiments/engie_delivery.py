"""短时预测发布：PG时钟、不可续期的请求、确定结果的幂等终结。"""

from datetime import datetime, timedelta
from hashlib import sha256
import json
import math
from uuid import uuid4

from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..storage.models import EngieDelivery, ImportedArtifact
from ..storage.model_packages import PackageError
from ..forecasting.engie_service_contract import ROSTER, HORIZONS


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def public(row):
    return {name: getattr(row, name) for name in (
        "id", "request_key", "artifact_id", "issue_time", "input_sha256", "budget_ms", "status",
        "created_at", "deadline_at", "completed_at", "finalized_at", "result_sha256", "result", "reason")}


def clock(session):
    # now()固定在事务开始时；等锁之后必须读取真正的数据库当前时刻。
    return session.scalar(select(func.clock_timestamp()))


def expire(row, now):
    if row.status == "pending" and now > row.deadline_at:
        row.status, row.reason, row.finalized_at = "expired", "delivery_deadline_exceeded", now


def reserve(engine, body):
    payload = body.model_dump(mode="json", exclude={"request_key"})
    fingerprint = digest(payload)
    with Session(engine) as session, session.begin():
        now = clock(session)
        identity = uuid4()
        inserted = session.execute(insert(EngieDelivery).values(
            id=identity, request_key=body.request_key, fingerprint=fingerprint,
            artifact_id=body.artifact_id, issue_time=body.issue_time, input_sha256=digest(payload["history"]),
            budget_ms=body.budget_ms, status="pending", created_at=now,
            deadline_at=now + timedelta(milliseconds=body.budget_ms),
        ).on_conflict_do_nothing(index_elements=["request_key"]).returning(EngieDelivery.id)).scalar_one_or_none()
        row = session.scalar(select(EngieDelivery).where(EngieDelivery.request_key == body.request_key).with_for_update())
        if row.fingerprint != fingerprint:
            raise PackageError("engie_delivery_request_conflict")
        if inserted is None:
            expire(row, clock(session))
        return public(row), inserted is not None


def validate_result(row, result, artifact):
    if (result["artifact_id"] != str(row.artifact_id)
            or result["input_sha256"] != row.input_sha256
            or result["issue_time"] != row.issue_time.isoformat()):
        raise PackageError("engie_delivery_result_identity_conflict")
    if (result["roster"] != list(ROSTER) or result["unit"] != "kW" or result["clock"] != "UTC"
            or result["import_id"] != str(artifact.import_id) or result["family"] != artifact.family
            or result["model_version"] != artifact.manifest["model_version"]
            or datetime.fromisoformat(result["source_cutoff"]) != row.issue_time - timedelta(minutes=20)
            or [datetime.fromisoformat(t) for t in result["target_times"]] != [row.issue_time + timedelta(minutes=h) for h in HORIZONS]):
        raise PackageError("engie_delivery_result_contract_conflict")
    values = result["predictions"]
    if (len(values) != 4 or any(len(v) != 6 for v in values)
            or any(not math.isfinite(x) for v in values for x in v)
            or len(result["farm_predictions"]) != 6
            or any(not math.isclose(sum(v[h] for v in values), x, abs_tol=1e-8, rel_tol=1e-10)
                   for h, x in enumerate(result["farm_predictions"]))):
        raise PackageError("engie_delivery_result_invalid")


def finalize(engine, delivery_id, result=None, *, reason=None):
    """结果重投不写第二条；超时仍保存完成时刻/摘要，但不发布失效曲线。"""
    with Session(engine) as session, session.begin():
        row = session.scalar(select(EngieDelivery).where(EngieDelivery.id == delivery_id).with_for_update())
        if row is None:
            raise PackageError("engie_delivery_not_found")
        now = clock(session)
        if result is not None:
            try:
                validate_result(row, result, session.get(ImportedArtifact, row.artifact_id))
                result_hash = digest(result)
            except PackageError:
                raise
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise PackageError("engie_delivery_result_invalid") from exc
            if row.result_sha256 is not None:
                if row.result_sha256 != result_hash:
                    raise PackageError("engie_delivery_result_conflict")
                return public(row)
            if row.status == "failed":
                raise PackageError("engie_delivery_already_failed")
            row.result_sha256, row.completed_at = result_hash, now
            expire(row, now)
            if row.status == "pending":
                row.status, row.result, row.finalized_at = "published", result, now
        elif row.status == "pending":
            expire(row, now)
            if row.status == "pending":
                row.status, row.reason, row.finalized_at = "failed", reason or "engie_delivery_inference_failed", now
        return public(row)


def read_delivery(engine, delivery_id):
    with Session(engine) as session, session.begin():
        row = session.scalar(select(EngieDelivery).where(EngieDelivery.id == delivery_id).with_for_update())
        if row is None:
            return None
        expire(row, clock(session))
        return public(row)


def deliver(engine, root, registration, body):
    from ..forecasting.engie_predictor import forecast

    record, created = reserve(engine, body)
    if not created:
        return record
    try:
        # 不因故障预算极短而跳过真实推理，迟到结果由数据库发布边界拒绝。
        result = forecast(root, registration, body)
    except (PackageError, OSError) as exc:
        return finalize(engine, record["id"], reason=str(exc) if isinstance(exc, PackageError) else "artifact_storage_unavailable")
    return finalize(engine, record["id"], result)
