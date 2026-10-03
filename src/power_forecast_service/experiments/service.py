"""提交的事务边界；这里不调用 Redis，也不执行训练。重跑编排留给结对。"""

import asyncio
from uuid import UUID, uuid4

from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..settings import Settings
from ..storage.models import Outbox, Task
from .contracts import (
    ExperimentRequest, TaskConflict, TaskReceipt, freeze_spec, submission_fingerprint,
)


def receipt(task: Task) -> TaskReceipt:
    return TaskReceipt(
        task_id=task.id,
        status=task.status,
        status_url=f"/tasks/{task.id}",
        source_task_id=task.source_task_id,
    )


async def create_task_with_outbox(
    session: AsyncSession,
    *,
    idempotency_key: str,
    request_fingerprint: str,
    spec: dict,
    source_task_id: UUID | None = None,
) -> Task:
    """必须在调用者事务内运行；任务与通知一同提交，唯一键竞争由 PG 裁决。"""
    if not session.in_transaction():
        raise RuntimeError("Caller must own an active transaction")
    task_id = uuid4()
    inserted = await session.scalar(
        insert(Task)
        .values(
            id=task_id,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            spec=spec,
            source_task_id=source_task_id,
            status="pending_dispatch",
            attempt_count=0,
        )
        .on_conflict_do_nothing(index_elements=[Task.idempotency_key])
        .returning(Task.id)
    )
    if inserted is None:
        task = await session.scalar(select(Task).where(Task.idempotency_key == idempotency_key))
        if task is None or task.request_fingerprint != request_fingerprint:
            raise TaskConflict("idempotency_key_conflict")
        return task
    session.add(Outbox(id=uuid4(), task_id=task_id, publish_count=0))
    await session.flush()
    return await session.get(Task, task_id)


async def submit_experiment(
    session: AsyncSession,
    *,
    request: ExperimentRequest,
    idempotency_key: str,
    settings: Settings,
) -> TaskReceipt:
    request_hash = submission_fingerprint(request)
    async with session.begin():
        # 重放先返回原任务；数据后来变化不能把同一次提交变成另一份实验。
        existing = await session.scalar(select(Task).where(Task.idempotency_key == idempotency_key))
        if existing is not None:
            if existing.request_fingerprint != request_hash:
                raise TaskConflict("idempotency_key_conflict")
            return receipt(existing)
        spec = await asyncio.to_thread(freeze_spec, request, settings)
        if request.purpose == "final_evaluation":
            # 请求键防网络重发，协议锁防不同请求键重复查看同一正式留出。
            # advisory锁随事务释放；数据库唯一索引仍是绕过本入口后的最后裁决。
            lock = int(spec["final_protocol_id"][:15], 16)
            await session.execute(select(func.pg_advisory_xact_lock(lock)))
            previous = await session.scalar(select(Task).where(
                Task.spec["purpose"].astext == "final_evaluation",
                Task.spec["final_protocol_id"].astext == spec["final_protocol_id"]))
            if previous is not None:
                return receipt(previous)
        task = await create_task_with_outbox(
            session,
            idempotency_key=idempotency_key,
            request_fingerprint=request_hash,
            spec=spec,
        )
        result = receipt(task)
    # 只有上下文成功 commit，才向 API 返回回执。
    return result
