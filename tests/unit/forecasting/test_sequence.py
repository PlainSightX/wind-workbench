"""序列边界和真实训练探针；构造数据不作为预测精度证据。"""

import numpy as np
import pandas as pd
import pytest
import torch

from power_forecast_service.forecasting.features import build_supervised_frame
from power_forecast_service.forecasting.sequence_data import common_history_start, history_windows
from power_forecast_service.forecasting.sequence_model import HistoryEncoder, SequenceRegressor
from power_forecast_service.forecasting.sequence_protocol import CONFIG


def synthetic_frame(n=260):
    t = np.arange(n)
    return pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=n, freq="5min"),
                         "wind_power": 100 + 20 * np.sin(t / 15), "wind_speed": 6 + np.sin(t / 15),
                         "humidity": np.full(n, 60.), "temperature": np.full(n, 20.)})


def test_sequence_window_does_not_read_future():
    frame = synthetic_frame()
    cutoffs = frame.timestamp.iloc[[23, 25]]
    first, _ = history_windows(frame, cutoffs)
    assert first.shape == (2, 24, 8)
    assert first[0, -1, 0] == frame.wind_power.iloc[23]
    frame.loc[frame.index > 25, "wind_power"] = 1e8
    np.testing.assert_array_equal(first, history_windows(frame, cutoffs)[0])


def test_sequence_rejects_insufficient_or_broken_history():
    frame = synthetic_frame()
    with pytest.raises(ValueError, match="insufficient"):
        history_windows(frame, [frame.timestamp.iloc[22]])
    with pytest.raises(ValueError, match="continuous"):
        history_windows(frame.drop(index=30), [frame.timestamp.iloc[50]])


def test_encoder_layers_independently_initialized():
    torch.manual_seed(42)
    model = HistoryEncoder(CONFIG)
    assert not torch.equal(model.encoder.layers[0].linear1.weight, model.encoder.layers[1].linear1.weight)
    assert model(torch.zeros(3, 24, 8)).shape == (3,)


@pytest.mark.parametrize("key", ["transformer_direct", "transformer_delta"])
def test_real_training_scaler_and_state_dict(tmp_path, key):
    frame = synthetic_frame()
    supervised = common_history_start(frame, build_supervised_frame(frame))
    train = supervised.iloc[:180]
    estimator = SequenceRegressor(key, {**CONFIG, "epochs": 4, "batch_size": 32}).fit(frame, train)
    assert estimator.details["parameter_max_change"] > 0
    assert estimator.details["max_gradient_norm"] > 0
    assert estimator.details["train_loss"][-1] < estimator.details["train_loss"][0]
    assert estimator.details["scaler_last_time"] == train.timestamp.iloc[-1].isoformat()
    assert not estimator.net.training
    original = estimator.predict_frame(frame, supervised.timestamp.iloc[-3:])
    path = tmp_path / "state.pt"
    estimator.save(path)
    reloaded = SequenceRegressor.load(path)
    np.testing.assert_array_equal(original, reloaded.predict_frame(frame, supervised.timestamp.iloc[-3:]))
    assert "state_dict" in torch.load(path, weights_only=True)
    expected_mean = frame.wind_power.iloc[:203].mean()
    assert estimator.mean[0] == pytest.approx(expected_mean)
