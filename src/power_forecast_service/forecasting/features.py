"""把历史观测组织为训练窗口或最后一个预测窗口。"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "wind_power_lag_0",
    "wind_power_lag_1",
    "wind_power_lag_12",
    "wind_speed_lag_1",
    "humidity_lag_1",
    "temperature_lag_1",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]


def _add_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    timestamp = data["timestamp"]
    data["wind_power_lag_0"] = data["wind_power"]
    data["wind_power_lag_1"] = data["wind_power"].shift(1)
    data["wind_power_lag_12"] = data["wind_power"].shift(12)
    data["wind_speed_lag_1"] = data["wind_speed"].shift(1)
    data["humidity_lag_1"] = data["humidity"].shift(1)
    data["temperature_lag_1"] = data["temperature"].shift(1)
    minutes = timestamp.dt.hour * 60 + timestamp.dt.minute
    angle = 2 * np.pi * minutes / (24 * 60)
    weekday_angle = 2 * np.pi * timestamp.dt.dayofweek / 7
    data["hour_sin"] = np.sin(angle)
    data["hour_cos"] = np.cos(angle)
    data["weekday_sin"] = np.sin(weekday_angle)
    data["weekday_cos"] = np.cos(weekday_angle)
    return data


def build_supervised_frame(frame: pd.DataFrame, horizon_steps: int = 12) -> pd.DataFrame:
    """构造严格使用 cutoff 时刻及之前信息的监督学习表。"""
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")

    data = _add_feature_columns(frame)
    timestamp = data["timestamp"]
    data["target_power"] = data["wind_power"].shift(-horizon_steps)
    data["target_timestamp"] = timestamp.shift(-horizon_steps)
    return data.dropna(subset=FEATURE_COLUMNS + ["target_power", "target_timestamp"]).reset_index(
        drop=True
    )


def build_inference_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """构造最新 cutoff 行；推理阶段不需要伪造未来标签。"""
    data = _add_feature_columns(frame)
    return data.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
