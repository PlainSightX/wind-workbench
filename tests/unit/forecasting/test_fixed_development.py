"""固定日期与候选选择只消费开发数据；不以测试集或删行改善结果。"""

from copy import deepcopy
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from power_forecast_service.experiments.contracts import (
    ExperimentRequest, TaskConflict, freeze_spec, submission_fingerprint, validate_spec,
)
from power_forecast_service.forecasting.candidate_models import candidate_predict, fit_candidate
from power_forecast_service.forecasting.data import load_wind_frame
from power_forecast_service.forecasting.development_protocol import WINDOWS, TEST_START
from power_forecast_service.forecasting.features import FEATURE_COLUMNS, build_supervised_frame
from power_forecast_service.forecasting.fixed_evaluation import fixed_window_split
from power_forecast_service.forecasting.model_diagnosis import grouped_errors, measurements
from power_forecast_service.forecasting.model_diagnosis import summarize
from power_forecast_service.settings import ROOT


def test_fixed_windows_do_not_move_when_lookback_changes(sample_path):
    frame, _ = load_wind_frame(sample_path)
    supervised = build_supervised_frame(frame)
    for name in WINDOWS:
        train, evaluation = fixed_window_split(supervised, name)
        later_train, same_evaluation = fixed_window_split(supervised.iloc[11:], name)
        assert len(evaluation) == 3872
        assert train.target_timestamp.max() < evaluation.timestamp.min()
        assert evaluation.target_timestamp.max() < pd.Timestamp(TEST_START)
        pd.testing.assert_frame_equal(evaluation, same_evaluation)
        assert len(later_train) == len(train) - 11
    broken = supervised.drop(supervised.index[supervised.timestamp == WINDOWS["main"][0]])
    with pytest.raises(ValueError, match="coverage"):
        fixed_window_split(broken, "main")


def test_scaler_only_sees_training_and_delta_restores_raw_current(sample_path):
    frame, _ = load_wind_frame(sample_path)
    train = build_supervised_frame(frame).iloc[:300].copy()
    ridge, details = fit_candidate(train, "ridge_1")
    np.testing.assert_allclose(details["scaler"]["mean"], train[FEATURE_COLUMNS].mean())
    assert details["scaler"]["n_samples_seen"] == len(train)
    before = deepcopy(details)
    extreme = train.iloc[:4].copy()
    extreme[FEATURE_COLUMNS] += 1000
    candidate_predict(ridge, extreme)
    assert before == details
    train["target_power"] = train["wind_power_lag_0"] + 7
    delta, _ = fit_candidate(train, "hgb_delta")
    np.testing.assert_allclose(candidate_predict(delta, train), train.wind_power_lag_0 + 7)


def test_groups_use_training_thresholds_and_bad_predictions_fail():
    train = pd.DataFrame({"target_power": [1., 2., 3.], "wind_power_lag_0": [1., 1., 1.]})
    future = pd.DataFrame({"target_power": [100., 200.], "wind_power_lag_0": [1., 1.]})
    groups = grouped_errors(train, future, {"persistence": np.array([1., 1.])})
    np.testing.assert_allclose(groups["power"]["thresholds"], np.quantile([1, 2, 3], [1/3, 2/3]))
    with pytest.raises(ValueError, match="missing_or_nonfinite"):
        measurements(np.array([1., 2.]), np.array([np.nan, 2.]))


@pytest.mark.parametrize("key", ["ridge_0_1", "ridge_1", "ridge_10", "hgb_delta"])
def test_candidate_contract_exact_and_distinct(sample_path, key):
    settings = SimpleNamespace(data_path=sample_path)
    request = ExperimentRequest(candidate_key=key, training_policy="fixed_iterations")
    frozen = freeze_spec(request, settings)
    assert validate_spec(frozen, settings) == frozen
    assert frozen["model_set"][-1] == key
    assert frozen["spec_version"] == "experiment-v3-fixed-q1"
    assert submission_fingerprint(request) != submission_fingerprint(ExperimentRequest(training_policy="fixed_iterations"))
    changed = deepcopy(frozen)
    changed["evaluation_protocol"]["windows"]["main"]["samples"] = 1
    with pytest.raises(TaskConflict):
        validate_spec(changed, settings)
    with pytest.raises(ValidationError):
        ExperimentRequest(candidate_key=key)


@pytest.mark.parametrize("mutation", ["missing_window", "wrong_protocol", "failed_model"])
def test_summary_does_not_promote_incomplete_evidence(tmp_path, mutation):
    source = ROOT / "docs/results/wind-diagnosis-20260922"
    report = json.loads((source / "baseline.json").read_text())
    if mutation == "missing_window":
        del report["windows"]["early_1"]
    elif mutation == "wrong_protocol":
        report["protocol"]["selection"] = "pick_best_window"
    else:
        report["windows"]["early_1"]["models"]["ridge_0_1"]["status"] = "failed"
    (tmp_path / "baseline.json").write_text(json.dumps(report))
    for path in source.glob("baseline-*.csv"):
        shutil.copyfile(path, tmp_path / path.name)
    if mutation == "failed_model":
        result = summarize(tmp_path)
        assert result["models"]["ridge_0_1"]["eligible"] is False
        assert "ridge_0_1" not in result["ranking"]
    else:
        with pytest.raises(ValueError, match="incomplete_or_changed_protocol"):
            summarize(tmp_path)
