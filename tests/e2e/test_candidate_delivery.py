"""选定开发候选走真实队列并核对全部3872行；不以离线好分数代替模型交付。"""

import json
import time
from uuid import uuid4

import pandas as pd
import pytest

from power_forecast_service.settings import ROOT
from power_forecast_service.storage.artifacts import source_tree_sha256


def test_selected_candidate_queue(http_client, completion_timeout):
    client = http_client
    evidence_dir = ROOT / "docs/results/wind-diagnosis-20260922"
    summary = json.loads((evidence_dir / "summary.json").read_text())
    key = summary["delivery_candidate"]
    payload = {"candidate_key": key, "training_policy": "fixed_iterations"}
    identity = "round3-" + uuid4().hex
    evidence = {"idempotency_key": identity, "status": "started", "candidate": key}
    try:
        response = client.post("/experiments", json=payload, headers={"Idempotency-Key": identity})
        response.raise_for_status()
        receipt = response.json()
        evidence["receipt"] = receipt
        repeated = client.post("/experiments", json=payload, headers={"Idempotency-Key": identity})
        repeated.raise_for_status()
        assert repeated.json()["task_id"] == receipt["task_id"]
        deadline = time.monotonic() + completion_timeout
        while time.monotonic() < deadline:
            task_response = client.get(receipt["status_url"])
            task_response.raise_for_status()
            task = task_response.json()
            evidence["task"] = task
            assert task["status"] != "failed", task
            if task["status"] == "succeeded":
                break
            time.sleep(0.5)
        else:
            raise TimeoutError("candidate_not_delivered")
        result = client.get(task["result_url"])
        result.raise_for_status()
        run = result.json()
        data = run["result"]
        assert data["execution"]["source_tree_sha256"] == source_tree_sha256()
        assert data["model_set"] == ["persistence", "hist_gradient_boosting", key]
        assert data["split"]["test_scored"] is False
        assert data["split"]["test"] == 3885
        reference = pd.read_csv(evidence_dir / "baseline-main.csv")
        scores = data["scoring"]["rows"]
        assert len(scores) == len(reference) == 3872
        for row, (_, expected) in zip(scores, reference.iterrows(), strict=True):
            assert pd.Timestamp(row["cutoff"]) == pd.Timestamp(expected.timestamp)
            assert row["actual"] == expected.target_power
            for model in data["model_set"]:
                assert row["predictions"][model] == pytest.approx(expected[model], abs=1e-8, rel=1e-8)
        evidence.update(run_id=run["run_id"], attempt_id=run["attempt_id"], metrics=data["metrics"],
                        model_verification=data["model_verification"], model_delivery=data["model_delivery"],
                        model_set=data["model_set"], source=data["execution"],
                        scoring_samples_sha256=data["scoring"]["samples_sha256"],
                        all_3872_predictions_match_offline=True, test_scored=False)
        artifacts = client.get("/artifacts", params={"run_id": run["run_id"]})
        artifacts.raise_for_status()
        artifacts = artifacts.json()
        assert len(artifacts) == 3
        chosen = next(item for item in artifacts if item["model_key"] == key)
        evidence["artifact_id"] = chosen["artifact_id"]
        evidence["forecasts"] = []
        for index in [0, len(scores) // 2, len(scores) - 1]:
            row = scores[index]
            replay = client.post("/replays", json={"artifact_id": chosen["artifact_id"], "cutoff": row["cutoff"]})
            replay.raise_for_status()
            assert replay.json()["forecast"]["prediction"] == pytest.approx(row["predictions"][key], abs=1e-8, rel=1e-8)
            evidence["forecasts"].append(replay.json())
        compare = client.get("/runs/compare", params={"left_run_id": run["run_id"], "right_run_id": run["run_id"],
                                                     "left_model": "persistence", "right_model": key})
        compare.raise_for_status()
        assert compare.json()["status"] == "comparable"
        evidence["comparison"] = compare.json()
        evidence["status"] = "passed"
    finally:
        if evidence["status"] != "passed":
            evidence["status"] = "failed_or_interrupted"
        destination = ROOT / ".local/runtime/e2e" / f"{identity}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(evidence, handle, ensure_ascii=False, indent=2)
