"""真实训练、文件与 PG 结果登记；不把故障注入结果当作队列集成验收。"""

import json
import math

import pytest

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from power_forecast_service.experiments.execution import recover_expired
from power_forecast_service.experiments.contracts import ExperimentRequest
from power_forecast_service.storage.models import ModelArtifact, Run, Task


@pytest.mark.parametrize("policy", ["auto_early_stopping", "fixed_iterations", "legacy"])
def test_worker_real_training_and_result_file(config, engine, submit, legacy_spec, policy, monkeypatch):
    from power_forecast_service.jobs.worker import execute_experiment
    from power_forecast_service.forecasting import pipeline

    original_fit = pipeline.fit_models
    fits = []

    def count_fit(*args, **kwargs):
        fits.append(1)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(pipeline, "fit_models", count_fit)

    actual_policy = "auto_early_stopping" if policy == "legacy" else policy
    task_id = submit(config, request=ExperimentRequest(training_policy=actual_policy)).task_id
    if policy == "legacy":
        with Session(engine) as session, session.begin():
            session.get(Task, task_id).spec = legacy_spec
    assert execute_experiment(task_id, config) == "succeeded"
    assert execute_experiment(task_id, config) == "duplicate_or_ineligible"
    assert fits == [1]
    with Session(engine) as session:
        run = session.scalar(select(Run).where(Run.task_id == task_id))
        output = config.artifact_root / run.artifact_path
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload == run.result
        models = session.scalars(select(ModelArtifact).where(ModelArtifact.run_id == run.id)).all()
        assert len(models) == 2 and {m.status for m in models} == {"ready"}
        assert {m.model_key for m in models} == set(payload["model_set"])
        assert all(m.manifest["attempt_id"] == str(run.attempt_id) for m in models)
        assert len(payload["model_verification"]) == 2
        assert all(item["samples"] == 3 and item["max_absolute_difference"] <= 1e-8
                   for item in payload["model_verification"])
        assert payload["split"]["test_scored"] is False
        assert len(payload["execution"]["source_tree_sha256"]) == 64
        assert payload["execution"]["packages"]["scikit-learn"]
        assert set(payload["metrics"]) == {"persistence", "hist_gradient_boosting"}
        assert payload["execution"]["source_fingerprint_version"] == "package-recursive-v2"
        assert payload["frozen_spec"] == session.get(Task, task_id).spec
        details = payload["training"]
        assert details["training_policy"] == actual_policy
        effective = payload["training"]["effective_parameters"]
        assert effective["random_state"] == payload["frozen_spec"]["random_seed"]
        assert all(effective[name] == value for name, value in payload["frozen_spec"]["hgb"].items())
        assert 0 < details["n_iter"] <= details["max_iter"]
        assert details["input_samples"] == payload["split"]["train"]
        assert details["feature_names"] == payload["feature_columns"]
        assert math.isfinite(details["fit_elapsed_seconds"]) and details["fit_elapsed_seconds"] >= 0
        for name in ("train_objective_scores", "internal_validation_objective_scores"):
            curve = details[name]
            assert all(math.isfinite(value) for value in curve)
            assert len(curve) == (details["n_iter"] + 1 if details["early_stopping_enabled"] else 0)
        json.dumps(payload, allow_nan=False)
        if policy == "legacy":
            assert payload["frozen_spec"] == legacy_spec
            assert "spec_version" not in session.get(Task, task_id).spec
        if policy == "fixed_iterations":
            assert payload["training"]["early_stopping_enabled"] is False
            assert payload["training"]["n_iter"] == 180
            assert payload["training"]["train_objective_scores"] == []


def test_result_file_without_db_commit_is_not_success(config, engine, submit, expire, monkeypatch):
    from power_forecast_service.jobs import worker

    task_id = submit(config).task_id

    def unavailable(*args, **kwargs):
        raise SQLAlchemyError("simulated commit failure after real result file")

    # 仅注入故障时点；训练、文件和数据库都是真实对象。
    monkeypatch.setattr(worker, "publish_result", unavailable)
    assert worker.execute_experiment(task_id, config) == "database_outcome_unconfirmed"
    assert list((config.artifact_root / str(task_id)).glob("*/result.json"))
    with Session(engine) as session:
        assert session.get(Task, task_id).status == "running"
        assert session.scalar(select(func.count()).select_from(Run)) == 0
        assert session.scalar(select(func.count()).select_from(ModelArtifact)) == 0
    assert len(list((config.artifact_root / str(task_id)).glob("*/models/*/manifest.json"))) == 2
    expire(engine, task_id)
    assert recover_expired(engine, config) == 1


@pytest.mark.parametrize("mutation", ["input_sha256", "policy", "parameter"])
def test_changed_contract_is_terminal_failure(config, engine, submit, monkeypatch, mutation):
    from power_forecast_service.jobs import worker

    def forbidden(*args, **kwargs):
        pytest.fail("invalid frozen contract must fail before training")

    monkeypatch.setattr(worker, "train_experiment", forbidden)

    task_id = submit(config).task_id
    with Session(engine) as session, session.begin():
        task = session.get(Task, task_id)
        if mutation == "input_sha256":
            task.spec = {**task.spec, "input_sha256": "0" * 64}
        elif mutation == "policy":
            task.spec = {**task.spec, "training_policy": "unknown"}
        else:
            task.spec = {**task.spec, "hgb": {**task.spec["hgb"], "max_iter": 1}}
    assert worker.execute_experiment(task_id, config) == "failed"
    with Session(engine) as session:
        task = session.get(Task, task_id)
        assert task.status == "failed" and task.error_code == "input_or_contract_invalid"
        assert session.scalar(select(func.count()).select_from(Run)) == 0
    assert not list((config.artifact_root / str(task_id)).rglob("result.json"))


def test_model_verification_error_is_not_input_failure(config, engine, submit, monkeypatch):
    from power_forecast_service.jobs import worker
    from power_forecast_service.storage.model_packages import PackageError

    def fail(*args):
        raise PackageError("artifact_fresh_process_verification_failed")

    monkeypatch.setattr(worker, "verify_fresh_process", fail)
    task_id = submit(config).task_id
    assert worker.execute_experiment(task_id, config) == "failed"
    with Session(engine) as session:
        task = session.get(Task, task_id)
        assert task.error_code == "artifact_fresh_process_verification_failed"
        assert session.scalar(select(func.count()).select_from(ModelArtifact)) == 0


def test_candidate_missing_package_rejected_before_success(config, engine, submit, monkeypatch):
    from power_forecast_service.jobs import worker

    save = worker.save_packages

    def incomplete(*args, **kwargs):
        return save(*args, **kwargs)[:2]

    monkeypatch.setattr(worker, "save_packages", incomplete)
    task_id = submit(config, request=ExperimentRequest(
        training_policy="fixed_iterations", candidate_key="ridge_0_1")).task_id
    assert worker.execute_experiment(task_id, config) == "failed"
    with Session(engine) as session:
        assert session.get(Task, task_id).error_code == "artifact_model_set_incomplete"
        assert session.scalar(select(func.count()).select_from(Run)) == 0
        assert session.scalar(select(func.count()).select_from(ModelArtifact)) == 0
