"""短构造数据真实拟合验证；不访问数据库，不构成真实预测效果证据。"""

import numpy as np
import pandas as pd
import pytest
import torch

from power_forecast_service.forecasting.features import build_supervised_frame
from power_forecast_service.forecasting.sequence_data import common_history_start
from power_forecast_service.forecasting.method_comparison.data import inner_split, prepare_fit_data
from power_forecast_service.forecasting.method_comparison.neural import NeuralPredictor, fit_neural
from power_forecast_service.forecasting.method_comparison.tabular import fit_tabular
from power_forecast_service.forecasting.method_comparison.protocol import recipes


def data():
    t = np.arange(350)
    frame = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=len(t), freq="5min"),
                          "wind_power": 100 + 20 * np.sin(t / 12), "wind_speed": 5 + np.sin(t / 20),
                          "humidity": 60 + np.cos(t / 15), "temperature": 20 + np.sin(t / 18)})
    train, validation = inner_split(common_history_start(frame, build_supervised_frame(frame)))
    return prepare_fit_data(frame, train, validation)


@pytest.mark.parametrize("key", ["encoder_direct", "encoder_delta", "timexer_0.0001"])
def test_real_neural_fit_and_reload(tmp_path, key):
    recipe = next(r for r in recipes() if r["id"] == key)
    samples = data()
    path = tmp_path / (key + ".pt")
    predictor, details = fit_neural(samples, recipe, path, device="cpu", epochs=3)
    assert details["parameter_max_change"] > 0
    assert details["reload_max_difference"] == 0
    assert len(details["history"]) == 3
    np.testing.assert_array_equal(predictor.predict(samples.validation_windows),
                                  NeuralPredictor.load(path).predict(samples.validation_windows))
    if key.startswith("timexer"):
        assert predictor.target_mean == samples.mean[0]
        assert predictor.target_scale == samples.scale[0]


def test_training_observations_do_not_consume_rng(tmp_path):
    recipe = next(r for r in recipes() if r["id"] == "encoder_direct")
    samples = data()
    for observe in (True, False):
        fit_neural(samples, recipe, tmp_path / f"{observe}.pt", device="cpu", epochs=3,
                   observe_training=observe)
    a = torch.load(tmp_path / "True.pt", weights_only=True)["states"]
    b = torch.load(tmp_path / "False.pt", weights_only=True)["states"]
    for key in a:
        for name in a[key]:
            torch.testing.assert_close(a[key][name], b[key][name], rtol=0, atol=0)


@pytest.mark.parametrize("key", ["ridge_original", "ridge_history_100", "lightgbm_15_50"])
def test_real_tabular_fit_and_reload(tmp_path, key):
    recipe = next(r for r in recipes() if r["id"] == key)
    _, details = fit_tabular(data(), recipe, tmp_path / (key + ".joblib"))
    assert details["reload_max_difference"] == 0
    assert details["validation"]["clipped"]["coverage"] == 1
