"""同键竞争、任务与 outbox 原子提交、重放原始请求。"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import Session

from power_forecast_service.experiments.contracts import ExperimentRequest, TaskConflict, fingerprint
from power_forecast_service.experiments.service import create_task_with_outbox
from power_forecast_service.serve import make_event_loop
from power_forecast_service.storage.database import make_async_engine
from power_forecast_service.storage.models import Outbox, Task


def test_concurrent_duplicate_submission_has_one_task(config, engine, submit):
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(lambda _: submit(config), range(4)))
    assert len({item.task_id for item in receipts}) == 1
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(Outbox)) == 1


def test_key_conflict_and_task_outbox_rollback(config, engine, submit):
    submit(config)

    async def call():
        async_engine = make_async_engine(config)
        try:
            async with async_sessionmaker(async_engine)() as session:
                with pytest.raises(TaskConflict):
                    async with session.begin():
                        await create_task_with_outbox(
                            session,
                            idempotency_key="request-1",
                            request_fingerprint="different",
                            spec={},
                        )
                with pytest.raises(RuntimeError, match="rollback"):
                    async with session.begin():
                        await create_task_with_outbox(
                            session,
                            idempotency_key="rolled-back",
                            request_fingerprint="a" * 64,
                            spec={},
                        )
                        raise RuntimeError("rollback")
        finally:
            await async_engine.dispose()

    asyncio.run(call(), loop_factory=make_event_loop)
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(Task)) == 1
        assert session.scalar(select(func.count()).select_from(Outbox)) == 1


def test_replay_returns_original_after_dataset_disappears(config, engine, submit):
    first = submit(config)
    missing = replace(config, data_path=config.data_path.parent / "not-present.csv")
    assert submit(missing).task_id == first.task_id


def test_literal_legacy_request_replays_without_mutation(config, engine, submit, legacy_spec):
    task_id, outbox_id = uuid4(), uuid4()
    old_hash = fingerprint({"action": "submit", "request": {
        "dataset_id": "wind-2019-q1", "purpose": "development",
    }})
    with Session(engine) as session, session.begin():
        session.add(Task(id=task_id, idempotency_key="legacy", request_fingerprint=old_hash,
                         spec=legacy_spec, status="pending_dispatch"))
        session.flush()
        session.add(Outbox(id=outbox_id, task_id=task_id))
    missing = replace(config, data_path=config.data_path.parent / "not-present.csv")
    for request in (ExperimentRequest(), ExperimentRequest(training_policy="auto_early_stopping")):
        assert submit(missing, key="legacy", request=request).task_id == task_id
    with pytest.raises(TaskConflict):
        submit(missing, key="legacy", request=ExperimentRequest(training_policy="fixed_iterations"))
    with Session(engine) as session:
        task = session.get(Task, task_id)
        assert task.spec == legacy_spec and task.request_fingerprint == old_hash
        assert task.status == "pending_dispatch" and task.attempt_count == 0
        assert session.get(Outbox, outbox_id).publish_count == 0
        assert session.scalar(select(func.count()).select_from(Task)) == 1
        assert session.scalar(select(func.count()).select_from(Outbox)) == 1


def test_different_strategies_cannot_share_a_concurrent_key(config, engine, submit):
    barrier = Barrier(2)

    def call(policy):
        barrier.wait(timeout=10)
        try:
            return policy, submit(config, request=ExperimentRequest(training_policy=policy))
        except TaskConflict:
            return policy, None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(call, ["auto_early_stopping", "fixed_iterations"]))
    winners = [(policy, receipt) for policy, receipt in outcomes if receipt is not None]
    assert len(winners) == 1
    with Session(engine) as session:
        assert session.get(Task, winners[0][1].task_id).spec["training_policy"] == winners[0][0]
        assert session.scalar(select(func.count()).select_from(Task)) == 1
        assert session.scalar(select(func.count()).select_from(Outbox)) == 1
