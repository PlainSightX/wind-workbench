"""窗口内的 cutoff 与未来标签必须保持各自的时间语义。"""

import pandas as pd

from power_forecast_service.forecasting.features import build_supervised_frame


def test_supervised_frame_respects_horizon() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2019-01-01", periods=40, freq="5min"),
            "wind_speed": [2.0] * 40,
            "humidity": [50.0] * 40,
            "temperature": [1.0] * 40,
            "wind_power": list(range(40)),
        }
    )
    supervised = build_supervised_frame(frame, horizon_steps=12)
    row = supervised.iloc[0]
    assert row["timestamp"] == pd.Timestamp("2019-01-01 01:00")
    assert row["target_timestamp"] == pd.Timestamp("2019-01-01 02:00")
    assert row["target_power"] == 24
