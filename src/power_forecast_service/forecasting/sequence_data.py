"""24点因果输入；先按日期确定cutoff，再索引完整历史，不重新按行数切分。"""

import numpy as np
import pandas as pd

from .sequence_protocol import LOOKBACK, SEQUENCE_FEATURES


def observation_features(frame):
    times = pd.DatetimeIndex(frame.timestamp)
    if (times.has_duplicates or not times.is_monotonic_increasing
            or times.tz is not None or frame.empty
            or not (np.diff(times.asi8) == pd.Timedelta(minutes=5).value).all()):
        raise ValueError("sequence_history_not_continuous")
    numeric = frame[["wind_power", "wind_speed", "humidity", "temperature"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("sequence_nonfinite_input")
    day = 2 * np.pi * (times.hour * 60 + times.minute) / 1440
    week = 2 * np.pi * times.dayofweek / 7
    return np.column_stack([numeric, np.sin(day), np.cos(day), np.sin(week), np.cos(week)])


def history_windows(frame, cutoffs):
    values = observation_features(frame)
    indices = pd.Index(frame.timestamp).get_indexer(pd.to_datetime(cutoffs))
    if (indices < LOOKBACK - 1).any():
        raise ValueError("sequence_history_insufficient")
    # [样本, 24个已到达时点, 8个数值特征]；不把target或未来观测混入数组。
    return values[indices[:, None] - np.arange(LOOKBACK - 1, -1, -1)], indices


def common_history_start(frame, supervised):
    if len(frame) < LOOKBACK:
        raise ValueError("sequence_history_insufficient")
    return supervised.loc[supervised.timestamp >= frame.timestamp.iloc[LOOKBACK - 1]].copy()
