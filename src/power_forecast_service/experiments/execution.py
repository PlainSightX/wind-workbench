"""数据库控制执行权；重复投递允许发生，只有当前有效 attempt 能登记结果。"""

import logging
from datetime import timedelta
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..settings import Settings
from ..storage.models import Attempt, ModelArtifact, Outbox, Run, Task
from ..storage.model_packages import PackageError, PackageRegistration, checked_manifest

log = logging.getLogger(__name__)


def db_now(session: Session):
    return session.scalar(select(func.clock_timestamp()))


def claim(engine, task_id: UUID, settings: Settings) -> tuple[UUID, dict] | None:
    with Session(engine) as session, session.begin():
        task = session.scalar(select(Task).where(Task.id == task_id).with_for_update())
        now = db_now(session)
        if task is None or task.status not in {"pending_dispatch", "queued", "retry_wait"}:
            return None
        if task.next_attempt_at and task.next_attempt_at > now:
            return None
        if task.attempt_count >= settings.max_attempts:
            return None
        attempt_id = uuid4()
        session.add(
            Attempt(id=attempt_id, task_id=task_id, number=task.attempt_count + 1, status="running")
        )
        session.flush()
        task.status = "running"
        task.attempt_count += 1
        task.active_attempt_id = attempt_id
        task.lease_until = now + timedelta(seconds=settings.lease_seconds)
        task.next_attempt_at = None
        task.error_code = None
        task.updated_at = now
        return attempt_id, task.spec


def owned(task: Task | None, attempt_id: UUID, now) -> bool:
    return bool(
        task
        and task.status == "running"
        and task.active_attempt_id == attempt_id
        and task.lease_until
        and task.lease_until > now
    )


def heartbeat(engine, task_id: UUID, attempt_id: UUID, settings: Settings) -> bool:
    with Session(engine) as session, session.begin():
        task = session.scalar(select(Task).where(Task.id == task_id).with_for_update())
        now = db_now(session)
        if not owned(task, attempt_id, now):
            return False
        task.lease_until = now + timedelta(seconds=settings.lease_seconds)
        task.updated_at = now
        return True


def publish_result(
    engine, task_id: UUID, attempt_id: UUID, result: dict, path: str, sha256: str,
    *, packages: list[PackageRegistration], artifact_root: Path,
) -> bool:
    # 文件校验在短事务外；登记前仍会再次裁决租约，打包耗时不会延长旧执行权。
    required = set(result["frozen_spec"]["model_set"])
    if (set(result.get("model_set", [])) != required or len(result["model_set"]) != len(required)
            or {item.model_key for item in packages} != required or len(packages) != len(required)
            or set(result.get("metrics", {})) != required
            or not result.get("scoring", {}).get("rows")
            or any(set(row["predictions"]) != required for row in result["scoring"]["rows"])
            or len({item.artifact_id for item in packages}) != len(packages)):
        raise PackageError("artifact_model_set_incomplete")
    for item in packages:
        manifest = checked_manifest(artifact_root, item)
        if (manifest.task_id != task_id or manifest.attempt_id != attempt_id
                or str(manifest.run_id) != result["run_id"]
                or manifest.frozen_spec != result["frozen_spec"]):
            raise PackageError("artifact_identity_mismatch")
    expected = {str(item.artifact_id): item.manifest_sha256 for item in packages}
    observed = {item["artifact_id"]: item["manifest_sha256"]
                for item in result.get("model_verification", [])}
    if observed != expected or len(result.get("model_verification", [])) != len(packages):
        raise PackageError("artifact_verification_missing")
    with Session(engine) as session, session.begin():
        task = session.scalar(select(Task).where(Task.id == task_id).with_for_update())
        now = db_now(session)
        if not owned(task, attempt_id, now):
            return False
        if result["frozen_spec"] != task.spec:
            raise PackageError("artifact_frozen_spec_mismatch")
        # task.spec是提交时已验证合同。发布再核模型身份，不能用一组自洽但缺项的结果替代。
        from ..forecasting.development_protocol import candidate_model_set
        from ..forecasting.sequence_protocol import sequence_model_set
        expected_models = (sequence_model_set(task.spec["sequence_key"]) if task.spec.get("sequence_key")
                           else candidate_model_set(task.spec.get("candidate_key", "none")))
        if task.spec["model_set"] != expected_models:
            raise PackageError("artifact_model_set_incomplete")
        attempt = session.get(Attempt, attempt_id)
        session.add(
            Run(
                id=UUID(result["run_id"]),
                task_id=task_id,
                attempt_id=attempt_id,
                result=result,
                artifact_path=path,
                artifact_sha256=sha256,
            )
        )
        # 显式先flush父对象；任一工件插入失败时，整笔事务回滚而非只丢第二个模型。
        session.flush()
        for item in packages:
            session.add(ModelArtifact(
                id=item.artifact_id, run_id=UUID(result["run_id"]), model_key=item.model_key,
                path=item.path, manifest_sha256=item.manifest_sha256,
                manifest=item.manifest, status="ready",
            ))
        session.flush()
        task.status = "succeeded"
        task.lease_until = None
        task.updated_at = now
        attempt.status = "succeeded"
        attempt.finished_at = now
        # 同一 commit 登记数值、文件校验和与成功状态；文件存在本身不是成功。
        return True


def fail_attempt(
    engine, task_id: UUID, attempt_id: UUID, code: str, settings: Settings, *, retryable: bool
) -> bool:
    with Session(engine) as session, session.begin():
        task = session.scalar(select(Task).where(Task.id == task_id).with_for_update())
        now = db_now(session)
        if not owned(task, attempt_id, now):
            return False
        attempt = session.get(Attempt, attempt_id)
        attempt.status = "failed"
        attempt.error_code = code
        attempt.finished_at = now
        schedule_retry(session, task, now, code, settings, retryable=retryable)
        return True


def schedule_retry(session, task, now, code, settings, *, retryable):
    task.lease_until = None
    task.error_code = code
    task.updated_at = now
    if retryable and task.attempt_count < settings.max_attempts:
        task.status = "retry_wait"
        task.next_attempt_at = now + timedelta(seconds=settings.retry_delay_seconds)
        notification = session.scalar(select(Outbox).where(Outbox.task_id == task.id))
        notification.available_at = task.next_attempt_at
        notification.sent_at = None
    else:
        task.status = "failed"
        task.next_attempt_at = None


def recover_expired(engine, settings: Settings) -> int:
    recovered = 0
    with Session(engine) as session, session.begin():
        rows = session.scalars(
            select(Task)
            .where(Task.status == "running", Task.lease_until <= func.clock_timestamp())
            .order_by(Task.lease_until)
            .limit(20)
            .with_for_update(skip_locked=True)
        ).all()
        for task in rows:
            now = db_now(session)
            attempt = session.get(Attempt, task.active_attempt_id)
            attempt.status = "expired"
            attempt.error_code = "worker_lease_expired"
            attempt.finished_at = now
            schedule_retry(session, task, now, "worker_lease_expired", settings, retryable=True)
            log.warning(
                "expired task=%s attempt=%s next_state=%s", task.id, attempt.id, task.status
            )
            recovered += 1
    return recovered
