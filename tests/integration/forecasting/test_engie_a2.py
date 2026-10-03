"""仅构造数据拟合；不计入144次场站数据估计器预算。"""

from pathlib import Path
import runpy

import joblib
import numpy as np
import pytest

from power_forecast_service.forecasting.engie_l1 import (
    L1_PARAMETERS, fit_inner, refit, select_weight,
)
from power_forecast_service.forecasting.engie_contract import persistence

pytestmark = pytest.mark.integration


def test_l1_inner_selection_refit_and_reload_are_separate(tmp_path):
    rng = np.random.default_rng(31)
    x = rng.normal(size=(150, 4, 60))
    y = -150 + x[:, :, -5, None] * 8 + np.arange(6)[None, None, :]
    parameters = {**L1_PARAMETERS, "n_estimators": 6, "min_child_samples": 5}
    predicted, inner = fit_inner(x[:90], y[:90], x[90:120], y[90:120], parameters=parameters)
    assert predicted.shape == (30, 4, 6) and inner["estimator_fit_count"] == 24
    choice = select_weight(y[90:120], persistence(x[90:120]), predicted)
    saved_inner = predicted.copy()
    model, final = refit(x[:120], y[:120], inner["selected_iterations"], choice["selected_lambda"], parameters=parameters)
    assert final["estimator_fit_count"] == 24
    np.testing.assert_array_equal(predicted, saved_inner)
    assert (model.base.predict(x[120:]) < 0).all()
    path = tmp_path / "l1.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(model.predict(x[120:]), joblib.load(path).predict(x[120:]))


def runner(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "tools/diagnostics"))
    return runpy.run_path(str(root / "tools/diagnostics/run_engie_a2.py"))


def test_partial_fit_refuses_to_restart(tmp_path, monkeypatch):
    functions = runner(monkeypatch)
    functions["start_stage"].__globals__["OUTPUT"] = tmp_path
    artifact = tmp_path / "q-l1.joblib"
    functions["start_stage"]("q", "selection", "frozen", artifact)
    with pytest.raises(ValueError, match="partial_fit"):
        functions["start_stage"]("q", "selection", "frozen", artifact)


def test_resume_checks_identity_before_loading_artifact(tmp_path, monkeypatch):
    functions = runner(monkeypatch)
    functions["stage_receipt"].__globals__["OUTPUT"] = tmp_path
    functions["write_json"](tmp_path / "q-selection.json", {
        "quarter": "q", "stage": "selection", "freeze_sha256": "other"})
    with pytest.raises(ValueError, match="identity_mismatch"):
        functions["stage_receipt"]("q", "selection", "frozen")


def test_refit_rejects_invalid_iterations_before_training():
    with pytest.raises(ValueError, match="invalid_frozen_selection"):
        refit(None, None, np.zeros((4, 6), dtype=int), 0.5)
