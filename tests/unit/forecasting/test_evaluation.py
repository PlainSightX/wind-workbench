from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from power_forecast_service.forecasting.pipeline import run_experiment, temporal_split


@pytest.mark.parametrize("horizon", [1, 12, 36])
def test_split_isolates_labels_without_moving_scoring_boundaries(horizon: int) -> None:
    cutoffs = pd.date_range("2019-01-01", periods=400, freq="5min")
    targets = pd.date_range(
        datetime(2019, 1, 1) + timedelta(minutes=5 * horizon), periods=400, freq="5min"
    )
    frame = pd.DataFrame({"timestamp": cutoffs, "target_timestamp": targets})
    train, validation, test = temporal_split(frame)
    assert train["target_timestamp"].max() < validation["timestamp"].min()
    assert validation["target_timestamp"].max() < test["timestamp"].min()
    assert validation["timestamp"].iloc[0] == cutoffs[280]
    assert test["timestamp"].iloc[0] == cutoffs[340]
    # 等于边界的标签也不能保留；该断言可发现 < 被误写为 <=。
    assert len(train) == 280 - horizon
    assert len(validation) == 60 - horizon


def test_split_rejects_no_usable_validation_period() -> None:
    cutoffs = pd.date_range("2019-01-01", periods=100, freq="5min")
    targets = pd.date_range("2019-01-01 02:00", periods=100, freq="5min")
    frame = pd.DataFrame({"timestamp": cutoffs, "target_timestamp": targets})
    with pytest.raises(ValueError, match="empty training or validation"):
        temporal_split(frame)


def test_split_rejects_unsorted_cutoffs() -> None:
    cutoffs = pd.date_range("2019-01-01", periods=100, freq="5min")
    targets = pd.date_range("2019-01-01 01:00", periods=100, freq="5min")
    frame = pd.DataFrame({"timestamp": cutoffs, "target_timestamp": targets})
    with pytest.raises(ValueError, match="strictly increasing"):
        temporal_split(frame.iloc[::-1])


def test_development_predictions_never_receive_test_rows(
    monkeypatch: pytest.MonkeyPatch, sample_path
) -> None:
    from power_forecast_service.forecasting import pipeline

    cutoffs = pd.date_range("2019-01-01", periods=200, freq="5min")
    targets = pd.date_range("2019-01-01 01:00", periods=200, freq="5min")
    frame = pd.DataFrame({"timestamp": cutoffs, "target_timestamp": targets, "target_power": 1.0})
    train, validation, test = temporal_split(frame)
    seen = []

    def predict(models, rows):
        seen.append(rows["timestamp"].tolist())
        return rows["target_power"].to_numpy()

    # 仅隔离评分路由；真实模型的训练验证另在 integration/forecasting。
    monkeypatch.setattr(pipeline, "build_supervised_frame", lambda *args, **kwargs: frame)
    monkeypatch.setattr(pipeline, "fit_models", lambda rows, **kwargs: SimpleNamespace(training_details={}))
    monkeypatch.setattr(pipeline, "improved_predict", predict)
    monkeypatch.setattr(pipeline, "persistence_predict", lambda rows: predict(None, rows))
    result = run_experiment(sample_path)
    assert seen == [validation["timestamp"].tolist()] * 2
    assert not set(seen[0]).intersection(test["timestamp"])
    assert result["split"]["train"] == len(train)
