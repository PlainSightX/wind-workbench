"""真实行锁与租约裁决；旧 attempt 不能发布结果，耗尽后进入终态。"""

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.orm import Session

from power_forecast_service.experiments.execution import (
    claim,
    heartbeat,
    publish_result,
    recover_expired,
)
from power_forecast_service.storage.models import Attempt, ModelArtifact, Run, Task
from power_forecast_service.storage.model_packages import PackageError


def test_duplicate_delivery_and_expired_attempt_cannot_publish(config, engine, submit, expire,
                                                              publication):
    task_id = submit(config).task_id
    old, spec = claim(engine, task_id, config)
    assert claim(engine, task_id, config) is None
    expire(engine, task_id)
    assert heartbeat(engine, task_id, old, config) is False
    assert recover_expired(engine, config) == 1
    current, _ = claim(engine, task_id, config)
    assert current != old
    payload, packages = publication(task_id, old, spec)
    assert publish_result(engine, task_id, old, payload, "old/result.json", "a" * 64,
                          packages=packages, artifact_root=config.artifact_root) is False
    payload, packages = publication(task_id, current, spec)
    args = dict(packages=packages, artifact_root=config.artifact_root)
    assert publish_result(engine, task_id, current, payload, "new/result.json", "b" * 64, **args)
    assert publish_result(engine, task_id, current, payload, "new/result.json", "b" * 64,
                          **args) is False
    with Session(engine) as session:
        assert session.get(Task, task_id).status == "succeeded"
        assert session.get(Attempt, old).status == "expired"
        assert session.scalar(select(func.count()).select_from(Run)) == 1
        assert session.scalar(select(func.count()).select_from(ModelArtifact)) == 2


@pytest.mark.parametrize("after_models", [False, True])
def test_publication_real_flush_rolls_back_atomically(config, engine, submit, publication,
                                                     after_models):
    task_id = submit(config).task_id
    attempt_id, spec = claim(engine, task_id, config)
    payload, packages = publication(task_id, attempt_id, spec)
    observed = []

    def fail_after_insert(session, context):
        connection = session.connection()
        runs = connection.scalar(select(func.count()).select_from(Run))
        models = connection.scalar(select(func.count()).select_from(ModelArtifact))
        if runs == 1 and models == (2 if after_models else 0):
            observed.append((runs, models))
            raise RuntimeError("injected after confirmed database INSERT")

    event.listen(Session, "after_flush_postexec", fail_after_insert)
    try:
        with pytest.raises(RuntimeError, match="confirmed database INSERT"):
            publish_result(engine, task_id, attempt_id, payload, "result.json", "a" * 64,
                           packages=packages, artifact_root=config.artifact_root)
    finally:
        event.remove(Session, "after_flush_postexec", fail_after_insert)
    assert observed == [(1, 2 if after_models else 0)]
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(Run)) == 0
        assert session.scalar(select(func.count()).select_from(ModelArtifact)) == 0
        task = session.get(Task, task_id)
        assert task.status == "running" and task.active_attempt_id == attempt_id
        assert task.lease_until is not None
        assert session.get(Attempt, attempt_id).status == "running"
    assert all((config.artifact_root / item.path / "manifest.json").exists() for item in packages)


@pytest.mark.parametrize("count", [0, 1])
def test_incomplete_model_set_cannot_publish(config, engine, submit, publication, count):
    task_id = submit(config).task_id
    attempt_id, spec = claim(engine, task_id, config)
    payload, packages = publication(task_id, attempt_id, spec)
    payload["model_set"] = payload["model_set"][:count]
    with pytest.raises(PackageError, match="artifact_model_set_incomplete"):
        publish_result(engine, task_id, attempt_id, payload, "result.json", "a" * 64,
                       packages=packages[:count], artifact_root=config.artifact_root)
    with Session(engine) as session:
        assert session.get(Task, task_id).status == "running"


def test_frozen_task_not_caller_result_is_authority(config, engine, submit, publication):
    task_id = submit(config).task_id
    attempt_id, spec = claim(engine, task_id, config)
    payload, packages = publication(task_id, attempt_id, {**spec, "random_seed": 999})
    with pytest.raises(PackageError, match="artifact_frozen_spec_mismatch"):
        publish_result(engine, task_id, attempt_id, payload, "result.json", "a" * 64,
                       packages=packages, artifact_root=config.artifact_root)


def test_three_expired_attempts_become_terminal(config, engine, submit, expire):
    task_id = submit(config).task_id
    for _ in range(3):
        assert claim(engine, task_id, config)
        expire(engine, task_id)
        assert recover_expired(engine, config) == 1
    assert claim(engine, task_id, config) is None
    with Session(engine) as session:
        task = session.get(Task, task_id)
        assert task.status == "failed" and task.attempt_count == 3
