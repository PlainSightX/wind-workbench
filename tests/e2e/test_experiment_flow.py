"""真实 HTTP -> PG/outbox -> Redis -> worker -> 结果；会留下真实实验记录。"""

import json
import time
from uuid import uuid4

import pandas as pd
import pytest

from power_forecast_service.forecasting.contracts import ForecastRequest
from power_forecast_service.forecasting.data import load_wind_frame
from power_forecast_service.settings import ROOT
from power_forecast_service.storage.artifacts import source_tree_sha256


@pytest.mark.parametrize("candidate", [None, "ridge_0_1"])
def test_submit_duplicate_wait_and_read_result(http_client, completion_timeout, candidate):
    key = "e2e-" + uuid4().hex
    client = http_client
    record = {"idempotency_key": key, "observed_states": [], "status": "started"}
    request = {"candidate_key": candidate, "training_policy": "fixed_iterations"} if candidate else {}
    model_key = candidate or "hist_gradient_boosting"
    expected_models = {"persistence", "hist_gradient_boosting"}
    if candidate:
        expected_models.add(candidate)
    try:
        health = client.get("/health")
        health.raise_for_status()
        assert health.json()["training_on_read"] is False
        first = client.post("/experiments", json=request, headers={"Idempotency-Key": key})
        first.raise_for_status()
        assert first.status_code == 202
        created = first.json()
        record["submission"] = created
        repeated = client.post("/experiments", json=request, headers={"Idempotency-Key": key})
        repeated.raise_for_status()
        assert repeated.status_code == 202
        assert repeated.json()["task_id"] == created["task_id"]
        record["observed_states"].append(created["status"])
        deadline = time.monotonic() + completion_timeout
        while time.monotonic() < deadline:
            response = client.get(created["status_url"])
            response.raise_for_status()
            task = response.json()
            record["task"] = task
            if task["status"] not in record["observed_states"]:
                record["observed_states"].append(task["status"])
            assert task["status"] != "failed", f"Experiment failed: {task['error_code']}"
            if task["status"] == "succeeded":
                result = client.get(task["result_url"])
                result.raise_for_status()
                run = result.json()
                record["run"] = run
                assert run["task_id"] == created["task_id"]
                assert run["attempt_id"] in {
                    item["attempt_id"] for item in task["attempts"] if item["status"] == "succeeded"
                }
                assert run["result"]["split"]["test_scored"] is False
                assert set(run["result"]["metrics"]) == expected_models
                assert (
                    run["result"]["execution"]["source_fingerprint_version"]
                    == "package-recursive-v2"
                )
                assert len(run["artifact_sha256"]) == 64
                # 防止新 checkout 的测试误验旧镜像；不把旧服务的全绿当成本轮通过。
                assert run["result"]["execution"]["source_tree_sha256"] == source_tree_sha256()
                comparison = client.get("/runs/compare", params={
                    "left_run_id": run["run_id"], "right_run_id": run["run_id"],
                })
                comparison.raise_for_status()
                record["comparison"] = comparison.json()
                assert comparison.json()["status"] == "comparable"
                assert comparison.json()["left"]["metrics"]["samples"] == 3872
                assert comparison.json()["samples_sha256"] == run["result"]["scoring"]["samples_sha256"]
                # 从同一输入的开发期cutoff取历史；不读取封存测试标签评分。
                artifacts = client.get("/artifacts", params={"run_id": run["run_id"]})
                artifacts.raise_for_status()
                versions = artifacts.json()
                assert {v["model_key"] for v in versions} == expected_models
                record["artifacts"] = versions
                frame, _ = load_wind_frame(ROOT / "data/sample/wind_2019_q1.csv", strict=True)
                score_rows = run["result"]["scoring"]["rows"]
                record["forecasts"] = []
                for row in [score_rows[0], score_rows[len(score_rows) // 2], score_rows[-1]]:
                    history = frame.loc[frame.timestamp <= pd.Timestamp(row["cutoff"])].tail(13)
                    for version in versions:
                        body = ForecastRequest(artifact_id=version["artifact_id"],
                                               observations=history.to_dict("records")).model_dump(mode="json")
                        started = time.perf_counter()
                        prediction = client.post("/forecasts", json=body)
                        elapsed = time.perf_counter() - started
                        prediction.raise_for_status()
                        response = prediction.json()
                        assert response["prediction"] == pytest.approx(
                            row["predictions"][version["model_key"]], rel=1e-8, abs=1e-8)
                        assert response["target_time"] == row["target_time"]
                        assert response["cutoff"] == row["cutoff"]
                        assert response["run_id"] == run["run_id"]
                        assert response["unit"] == "MW_source_reported"
                        assert response["after_training_cutoff"] is True
                        record["forecasts"].append({"request": body, "response": response,
                                                    "http_elapsed_seconds": elapsed})
                assert {v["model_key"] for v in run["result"]["model_verification"]} == expected_models
                record["status"] = "passed"
                if candidate == "ridge_0_1":
                    # 浏览器复验使用这次实际训练的身份/数值，不再读取维护者旧 UUID。
                    version = next(v for v in versions if v["model_key"] == candidate)
                    receipt = {"run_id": run["run_id"], "artifact_id": version["artifact_id"],
                        "metrics": run["result"]["metrics"]}
                    path = ROOT / ".local/runtime/ui-delivery.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
                return
            time.sleep(0.5)
        raise TimeoutError(f"Task still unfinished; inspect {created['status_url']}")
    finally:
        # 保留成功或失败时已观察到的回执，不删除业务任务，也不覆盖以往检查结果。
        if record["status"] != "passed":
            record["status"] = "failed_or_interrupted"
        output = ROOT / ".local/runtime/e2e" / f"{key}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
