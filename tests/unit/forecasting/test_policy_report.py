"""只验证报告计算/拒绝条件；合成回执不是模型效果或真实HTTP证据。"""

from copy import deepcopy
from datetime import datetime, timedelta
import runpy
from uuid import UUID, uuid4

import pytest

from power_forecast_service.experiments.comparison import compare_results
from power_forecast_service.forecasting.scoring import (
    ScoreRow, ScoringEvidence, metrics_for_rows, sample_fingerprint,
)
from power_forecast_service.forecasting.spec import training_parameters
from power_forecast_service.settings import ROOT

REPORT = runpy.run_path(str(ROOT / "tools/diagnostics/compare_training_policies.py"))
POLICIES = REPORT["POLICIES"]
MODEL = REPORT["MODEL"]


@pytest.fixture
def pair():
    state = {"pair_id": "synthetic", "source_sha256": "a" * 64,
             "input_sha256": "b" * 64, "sides": {}}
    for policy in POLICIES:
        rows = [ScoreRow(
            cutoff=datetime(2020, 1, 1) + timedelta(hours=index * 6),
            target_time=datetime(2020, 1, 1, 1) + timedelta(hours=index * 6),
            actual=10.0, predictions={"persistence": 8.0, MODEL: 9.0 if policy == POLICIES[0] else 10.0},
        ) for index in range(4)]
        scores = ScoringEvidence(
            version="scoring-v1", input_sha256=state["input_sha256"],
            target="wind_power_single_point", unit="source_reported_unit",
            clock="source_time_timezone_unknown", evaluation_split="validation",
            horizon_minutes=60, split_version="fixture", metric_version="unweighted-mae-rmse-v1",
            samples_sha256=sample_fingerprint(rows), rows=rows,
        )
        params = training_parameters(policy)
        spec = {"training_policy": policy, "hgb": params, "random_seed": 42}
        detail = {
            "training_policy": policy, "effective_parameters": {**params, "random_state": 42},
            "input_samples": 64, "n_features_in": 1, "feature_names": ["x"],
            "n_iter": 180, "max_iter": 180, "early_stopping_enabled": False,
            "fit_elapsed_seconds": 0.1, "train_objective_scores": [],
            "internal_validation_objective_scores": [],
            "train_cutoff_start": "2019-01-01", "train_cutoff_end": "2019-01-02",
            "train_target_start": "2019-01-01T01:00", "train_target_end": "2019-01-02T01:00",
        }
        task_id, attempt_id, run_id = (str(uuid4()) for _ in range(3))
        result = {
            "run_id": run_id, "purpose": "development", "evaluation_split": "validation",
            "horizon_steps": 12, "horizon_minutes": 60, "split_version": "fixture",
            "split": {"train": 64, "test_scored": False}, "frozen_spec": deepcopy(spec),
            "input_file_sha256": state["input_sha256"], "feature_columns": ["x"],
            "feature_contract_version": "fixture", "model_version": "fixture",
            "determinism": {"random_seed": 42}, "execution": {"source_tree_sha256": state["source_sha256"]},
            "training": detail, "scoring": scores.model_dump(mode="json"),
            "metrics": {name: metrics_for_rows(rows, name) for name in ("persistence", MODEL)},
        }
        state["sides"][policy] = {
            "submission": {"task_id": task_id},
            "task": {"task_id": task_id, "status": "succeeded", "spec": spec, "attempt_count": 1,
                     "attempts": [{"attempt_id": attempt_id, "status": "succeeded"}]},
            "run": {"task_id": task_id, "attempt_id": attempt_id, "run_id": run_id,
                    "artifact_sha256": "c" * 64, "result": result},
        }
    left, right = [state["sides"][policy]["run"] for policy in POLICIES]
    state["comparison"] = compare_results(UUID(left["run_id"]), left["result"], MODEL,
                                           UUID(right["run_id"]), right["result"], MODEL).model_dump(mode="json")
    return state


def test_report_has_known_signed_errors_and_complete_predeclared_groups(pair):
    report = REPORT["controlled_report"](pair)
    assert report["overall"]["metrics"][POLICIES[0]]["mean_prediction_minus_actual"] == -1
    assert report["overall"]["fixed_minus_auto"]["mae"] == -1
    assert [item["samples"] for item in report["validation_halves"]] == [2, 2]
    assert [item["samples"] for item in report["source_clock_6h"].values()] == [1, 1, 1, 1]
    assert REPORT["metric_table"]([], [], []) == {"samples": 0, "metrics": None}


@pytest.mark.parametrize("mutation,reason", [
    ("source", "source_changed"), ("test", "test_must_remain_unscored"),
    ("seed", "non_strategy_configuration_difference"), ("iteration", "iteration_observation"),
])
def test_report_rejects_uncontrolled_or_invalid_training(pair, mutation, reason):
    side = pair["sides"][POLICIES[1]]
    result = side["run"]["result"]
    if mutation == "source":
        result["execution"]["source_tree_sha256"] = "d" * 64
    elif mutation == "test":
        result["split"]["test_scored"] = True
    elif mutation == "seed":
        result["training"]["effective_parameters"]["random_state"] = 7
        result["frozen_spec"]["random_seed"] = side["task"]["spec"]["random_seed"] = 7
    else:
        result["training"]["n_iter"] = 181
    with pytest.raises(ValueError, match=reason):
        REPORT["controlled_report"](pair)
