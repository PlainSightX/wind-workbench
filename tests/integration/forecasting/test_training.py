"""小样本真实拟合：auto是请求策略，不代表库一定开启了早停。"""

import json

import pandas as pd
import pytest

from power_forecast_service.forecasting.features import FEATURE_COLUMNS
from power_forecast_service.forecasting.models import fit_models
from power_forecast_service.forecasting.spec import training_parameters


def test_small_auto_records_actual_disabled_state_and_seed():
    frame = pd.DataFrame({name: range(64) for name in FEATURE_COLUMNS})
    frame["target_power"] = [float(index % 13) for index in range(64)]
    fitted = fit_models(frame, random_state=7)
    details = fitted.training_details
    assert details["training_policy"] == "auto_early_stopping"
    assert details["effective_parameters"]["early_stopping"] == "auto"
    assert details["effective_parameters"]["random_state"] == 7
    assert fitted.improved.random_state == 7
    assert details["early_stopping_enabled"] is False
    assert details["stopped_before_max_iter"] is False
    assert details["n_iter"] == 180
    assert details["internal_validation_mode"] == "none"
    assert details["train_objective_scores"] == details["internal_validation_objective_scores"] == []
    json.dumps(details, allow_nan=False)


def test_policy_parameter_mismatch_is_rejected_before_fit():
    with pytest.raises(ValueError, match="training_policy_parameters_mismatch"):
        fit_models(pd.DataFrame(), training_policy="fixed_iterations",
                   model_parameters=training_parameters("auto_early_stopping"))
