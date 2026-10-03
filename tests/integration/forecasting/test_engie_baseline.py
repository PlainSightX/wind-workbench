"""构造数据上真实拟合与完整工件读回；不是场站预测成绩。"""

import joblib
from copy import deepcopy
import json
from pathlib import Path
import runpy
import numpy as np
import pytest

from power_forecast_service.forecasting.engie_baselines import (
    FittedBaseline, TREE_PARAMETERS, fit_lightgbm, fit_ridge,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("family", ["ridge", "lightgbm"])
def test_real_multi_horizon_fit_and_reload(tmp_path, family):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(180, 4, 60))
    y = -200 + 3 * x[:, :, -5, None] + np.arange(6)[None, None, :]
    args = (x[:100], y[:100], x[100:130], y[100:130], x[:130], y[:130])
    if family == "ridge":
        model, details = fit_ridge(*args, alphas=(1.0, 10.0))
        np.testing.assert_allclose(details["scaler_means"], x[:130].mean(axis=0))
        assert details["scaler_samples"] == [130] * 4
    else:
        model, details = fit_lightgbm(*args, parameters={**TREE_PARAMETERS, "n_estimators": 8, "min_child_samples": 5})
        assert np.array(details["selected_iterations"]).shape == (4, 6)
    predicted = model.predict(x[130:])
    assert predicted.shape == (50, 4, 6) and np.isfinite(predicted).all()
    assert (predicted < 0).all()  # ENGIE不能继承旧Q1的非负裁剪。
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(predicted, joblib.load(path).predict(x[130:]))


def test_predictor_refuses_missing_member_history():
    model = FittedBaseline("persistence", [])
    x = np.ones((1, 4, 60))
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="complete_fixed_roster"):
        model.predict(x)


def runner_functions():
    return runpy.run_path(str(Path(__file__).resolve().parents[3] / "tools/diagnostics/run_engie_baseline.py"))


def test_frozen_protocol_survives_json_roundtrip():
    value = runner_functions()["protocol"]()
    assert json.loads(json.dumps(value)) == value


def test_orphan_model_is_rejected_before_fitting(tmp_path):
    runner = runner_functions()
    (tmp_path / "2014-Q2-ridge.joblib").write_bytes(b"unfinished")
    with pytest.raises(ValueError, match="unfinished_attempt"):
        runner["fit_or_resume"]("ridge", "2014-Q2", None, None, tmp_path, tmp_path, "frozen")


def test_mismatched_fit_identity_is_rejected_before_loading(tmp_path):
    runner = runner_functions()
    runner["write_json"](tmp_path / "2014-Q2-ridge-fit.json", {
        "family": "lightgbm", "window": "2014-Q2", "freeze_sha256": "frozen"})
    with pytest.raises(ValueError, match="protocol_mismatch"):
        runner["fit_or_resume"]("ridge", "2014-Q2", None, None, tmp_path, tmp_path, "frozen")


def test_completed_window_is_verified_without_rewriting(tmp_path, monkeypatch):
    runner = runner_functions()
    runner["file_record"].__globals__["ROOT"] = tmp_path
    array = tmp_path / "predictions.npz"
    array.write_bytes(b"recorded")
    value = {"freeze_sha256": "frozen", "predictions": runner["file_record"](array), "fits": {}}
    report = tmp_path / "2014-Q2-result.json"
    runner["write_json"](report, value)
    before = (report.read_bytes(), report.stat().st_mtime_ns, array.stat().st_mtime_ns)
    assert runner["completed_window"](tmp_path, "2014-Q2", "frozen") == value
    assert before == (report.read_bytes(), report.stat().st_mtime_ns, array.stat().st_mtime_ns)
    array.write_bytes(b"modified")
    with pytest.raises(ValueError, match="prediction_hash"):
        runner["completed_window"](tmp_path, "2014-Q2", "frozen")


def verifier_functions():
    return runpy.run_path(str(Path(__file__).resolve().parents[3] / "tools/diagnostics/verify_engie_baseline.py"))


@pytest.mark.parametrize("missing", ["quarter", "model", "freeze"])
def test_verifier_rejects_incomplete_comparison(missing):
    verifier = verifier_functions()
    record = {"freeze_sha256": "frozen", **{key: dict.fromkeys(verifier["FAMILIES"])
              for key in ("fits", "metrics", "actual_outputs", "prediction_cost")}}
    result = {"freeze_sha256": "frozen", "windows": {q: deepcopy(record) for q in verifier["QUARTERS"]}}
    verifier["check_result_members"](result)
    if missing == "quarter":
        del result["windows"]["2014-Q4"]
    elif missing == "model":
        del result["windows"]["2014-Q4"]["metrics"]["ridge"]
    else:
        result["windows"]["2014-Q4"]["freeze_sha256"] = "different"
    with pytest.raises(ValueError):
        verifier["check_result_members"](result)


@pytest.mark.parametrize("corruption", ["curve", "sum", "time", "roster", "fault", "absent_issue"])
def test_verifier_rejects_corrupted_delivered_example(corruption):
    verifier = verifier_functions()
    issue = verifier["pd"].Timestamp("2014-04-01", tz="UTC")
    predictions = {family: np.arange(24).reshape(4, 6) for family in verifier["FAMILIES"]}
    cases = {"normal": {"issue_time": issue.isoformat(),
        "source_cutoff": (issue - verifier["ARRIVAL_LAG"]).isoformat(),
        "valid_times": [(issue + verifier["pd"].Timedelta(minutes=h)).isoformat() for h in verifier["HORIZONS"]],
        "roster": list(verifier["ROSTER"]), "unit": "kW",
        "predictions": {f: p.tolist() for f, p in predictions.items()},
        "farm_predictions": {f: p.sum(axis=0).tolist() for f, p in predictions.items()}},
        "missing_turbine": {"issue_time": issue.isoformat(), "removed_turbine": verifier["ROSTER"][0],
            "input_valid": False, "prediction": None, "synthetic_fault": True, "reason": "incomplete_fixed_roster_history"}}
    verifier["check_cases"](cases, predictions)
    if corruption == "curve":
        cases["normal"]["predictions"]["ridge"][0][0] += 1
    elif corruption == "sum":
        cases["normal"]["farm_predictions"]["ridge"][0] += 1
    elif corruption == "time":
        cases["normal"]["valid_times"][0] = issue.isoformat()
    elif corruption == "roster":
        cases["normal"]["roster"].pop()
    elif corruption == "fault":
        cases["missing_turbine"]["input_valid"] = True
    else:
        predictions = None
    with pytest.raises((AssertionError, ValueError)):
        verifier["check_cases"](cases, predictions)
