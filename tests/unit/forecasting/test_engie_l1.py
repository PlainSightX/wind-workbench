"""共享收缩和开发采用门的反例，避免把赢旧树模型误写成赢强基线。"""

from copy import deepcopy

import numpy as np
import pytest

from power_forecast_service.forecasting.engie_l1 import (
    L1_PARAMETERS, adoption_decision, blend, select_weight,
)
from power_forecast_service.forecasting.engie_baselines import TREE_PARAMETERS

pytestmark = pytest.mark.unit


def test_objective_is_only_changed_training_parameter():
    assert {k for k in TREE_PARAMETERS if TREE_PARAMETERS[k] != L1_PARAMETERS[k]} == {"objective"}
    assert TREE_PARAMETERS["metric"] == L1_PARAMETERS["metric"] == "l1"


def test_one_weight_selected_using_farm_sum_not_individual_error():
    reference = np.zeros((3, 4, 6))
    candidate = np.ones_like(reference) * 4
    actual = np.ones_like(reference)
    result = select_weight(actual, reference, candidate)
    assert result["selected_lambda"] == 0.25
    assert len(result["candidates"]) == 5
    np.testing.assert_array_equal(blend(reference, candidate, 0.25), actual)


def test_equal_scores_choose_zero_and_keep_negative_kw():
    values = np.full((2, 4, 6), -8.)
    assert select_weight(values, values, values)["selected_lambda"] == 0
    np.testing.assert_array_equal(blend(values, values, 1), values)


@pytest.mark.parametrize("defect", ["missing", "nonfinite", "weight"])
def test_bad_curve_or_unregistered_weight_is_rejected(defect):
    a = np.ones((2, 4, 6))
    b = a[:, :3] if defect == "missing" else a.copy()
    if defect == "nonfinite":
        b[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        blend(a, b, 0.1 if defect == "weight" else 0.5)


def good_windows():
    return {q: {"metrics": {"persistence": {"farm": {"mae": 100, "rmse": 130}},
                            "candidate": {"farm": {"mae": 95, "rmse": 129}}},
                "actual_outputs": {"candidate": 10, "persistence": 10},
                "counts": {"input_valid": 10}} for q in ("q2", "q3", "q4")}


@pytest.mark.parametrize("defect", ["small_gain", "rmse", "worst_quarter", "coverage"])
def test_each_adoption_gate_can_reject_a_lower_mean_mae(defect):
    windows = good_windows()
    assert adoption_decision(windows, "candidate")["passed"]
    if defect == "small_gain":
        for w in windows.values():
            w["metrics"]["candidate"]["farm"]["mae"] = 99
    elif defect == "rmse":
        windows["q2"]["metrics"]["candidate"]["farm"]["rmse"] = 140
    elif defect == "worst_quarter":
        windows["q2"]["metrics"]["candidate"]["farm"]["mae"] = 103
        windows["q3"]["metrics"]["candidate"]["farm"]["mae"] = 85
    else:
        windows["q2"]["actual_outputs"]["candidate"] = 9
    assert not adoption_decision(windows, "candidate")["passed"]
