"""真实 PG 上的当前 HTTP 合同；不启动队列，不在请求中训练。"""

import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from power_forecast_service.api.app import create_app
from power_forecast_service.experiments.execution import claim, publish_result
from power_forecast_service.serve import make_event_loop
from power_forecast_service.storage.models import Attempt, ModelArtifact, Outbox, Run, Task


def test_api_no_training_and_idempotent_post(config, engine, monkeypatch):
    from power_forecast_service.forecasting import pipeline

    def forbidden(*args, **kwargs):
        pytest.fail("API reads or submission must not train")

    monkeypatch.setattr(pipeline, "train_service", forbidden)
    monkeypatch.setattr(pipeline, "run_experiment", forbidden)
    monkeypatch.setattr(pipeline, "train_experiment", forbidden)
    with TestClient(
        create_app(config),
        base_url="http://localhost",
        backend_options={"loop_factory": make_event_loop},
    ) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/datasets").status_code == 200
        first = client.post("/experiments", json={}, headers={"Idempotency-Key": "api-key"})
        second = client.post("/experiments", json={"training_policy": "auto_early_stopping"},
                             headers={"Idempotency-Key": "api-key"})
        assert first.status_code == second.status_code == 202
        assert first.json() == second.json()
        assert client.post("/experiments", json={"training_policy": "fixed_iterations"},
                           headers={"Idempotency-Key": "api-key"}).status_code == 409
        for payload in ({"training_policy": "unknown"}, {"hgb": {"max_iter": 1}}, {"extra": 1}):
            assert client.post("/experiments", json=payload,
                               headers={"Idempotency-Key": "invalid"}).status_code == 422
        detail = client.get(first.json()["status_url"]).json()
        assert detail["status"] == "pending_dispatch"
        assert detail["attempt_count"] == 0
        assert detail["result_url"] is None
        assert client.get("/tasks").status_code == 200
        assert client.get(f"/tasks/{uuid4()}").status_code == 404
        assert (
            client.post(
                "/experiments",
                json={},
                headers={"Origin": "https://untrusted.invalid", "Idempotency-Key": "cross"},
            ).status_code
            == 403
        )
        assert client.get("/health", headers={"Host": "untrusted.invalid"}).status_code == 400
        assert (
            client.post(
                "/experiments",
                json={"purpose": "final_evaluation"},
                headers={"Idempotency-Key": "final"},
            ).status_code
            == 422
        )
        assert client.get("/evaluate").status_code == 404
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(Task)) == 1
        assert session.scalar(select(func.count()).select_from(Outbox)) == 1


def test_compare_keeps_legacy_results_readable_without_inventing_evidence(config, engine, submit):
    task = submit(config)
    attempt_id, _ = claim(engine, task.task_id, config)
    run_id = uuid4()
    legacy = {"run_id": str(run_id), "metrics": {"persistence": {"mae": 1, "samples": 2}}}
    # 历史fixture不通过要求完整模型的现行发布函数，不补造历史ready能力。
    with Session(engine) as session, session.begin():
        item = session.get(Task, task.task_id)
        item.status, item.lease_until = "succeeded", None
        session.get(Attempt, attempt_id).status = "succeeded"
        session.add(Run(id=run_id, task_id=task.task_id, attempt_id=attempt_id, result=legacy,
                        artifact_path="legacy.json", artifact_sha256="a" * 64))
    with TestClient(
        create_app(config), base_url="http://localhost",
        backend_options={"loop_factory": make_event_loop},
    ) as client:
        response = client.get("/runs/compare", params={
            "left_run_id": str(run_id), "right_run_id": str(run_id),
        })
        assert response.status_code == 200
        assert response.json()["status"] == "not_comparable"
        assert response.json()["delta"] is None
        assert "left:scoring_evidence_missing" in response.json()["reasons"]
        assert client.get(f"/runs/{run_id}").json()["result"] == legacy
        assert client.get("/runs/compare", params={
            "left_run_id": str(run_id), "right_run_id": str(uuid4()),
        }).status_code == 404
        assert len(client.get("/tasks").json()) == 1
        assert client.get("/artifacts").json() == []


def test_explicit_artifact_prediction_reload_and_errors(config, engine, submit, publication,
                                                       monkeypatch):
    from power_forecast_service.forecasting import pipeline
    from sklearn.ensemble import HistGradientBoostingRegressor

    task_id = submit(config).task_id
    attempt_id, spec = claim(engine, task_id, config)
    payload, packages = publication(task_id, attempt_id, spec)
    assert publish_result(engine, task_id, attempt_id, payload, "result.json", "a" * 64,
                          packages=packages, artifact_root=config.artifact_root)

    def forbidden(*args, **kwargs):
        pytest.fail("API prediction cannot train")

    monkeypatch.setattr(pipeline, "train_experiment", forbidden)
    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", forbidden)
    responses = []
    # 真实API进程重启另在e2e后执行；这里仅覆盖应用生命周期重新创建。
    for _ in range(2):
        with TestClient(create_app(config), base_url="http://localhost",
                        backend_options={"loop_factory": make_event_loop}) as client:
            artifacts = client.get("/artifacts", params={"run_id": payload["run_id"]})
            assert artifacts.status_code == 200 and len(artifacts.json()) == 2
            current = []
            for package in packages:
                case = json.loads((config.artifact_root / package.path / "verification.json").read_text())["cases"][0]
                body = {"artifact_id": str(package.artifact_id), "observations": case["observations"]}
                response = client.post("/forecasts", json=body)
                assert response.status_code == 200, response.text
                value = response.json()
                assert value["run_id"] == payload["run_id"]
                assert value["prediction"] == pytest.approx(case["expected"][package.model_key],
                                                           abs=1e-8, rel=1e-8)
                assert value["cutoff"] == case["cutoff"]
                assert value["target_time"] == case["target_time"]
                assert value["after_training_cutoff"] is True
                current.append(value)
            responses.append(current)
            assert client.post("/forecasts", json={**body, "artifact_id": str(uuid4())}).status_code == 404
            assert client.post("/forecasts", json={**body, "path": "anywhere"}).status_code == 422
            assert client.post("/forecasts", json={**body, "observations": body["observations"][:12]}).status_code == 422
            invalid = json.loads(json.dumps(body))
            invalid["observations"][0]["temperature"] = float("nan")
            assert client.post("/forecasts", content=json.dumps(invalid),
                               headers={"Content-Type": "application/json"}).status_code == 422
    assert responses[0] == responses[1]

    with Session(engine) as session, session.begin():
        session.get(ModelArtifact, packages[-1].artifact_id).status = "unavailable"
    with TestClient(create_app(config), base_url="http://localhost",
                    backend_options={"loop_factory": make_event_loop}) as client:
        assert len(client.get("/artifacts").json()) == 1
        assert client.post("/forecasts", json=body).json()["detail"] == "artifact_not_ready"
        body["artifact_id"] = str(packages[0].artifact_id)
        directory = config.artifact_root / packages[0].path
        (directory / "model/MLmodel").write_text("damaged")
        failure = client.post("/forecasts", json=body)
        assert failure.status_code == 409 and failure.json()["detail"] == "artifact_integrity_failed"
