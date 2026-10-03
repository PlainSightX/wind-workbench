"""按预定日期选择共同样本；窗口长度/模型变化不能移动评分和封存边界。"""

import pandas as pd

from .development_protocol import TEST_START, WINDOWS


def fixed_window_split(supervised, window):
    start, end = map(pd.Timestamp, WINDOWS[window])
    train = supervised.loc[supervised.target_timestamp < start].copy()
    evaluation = supervised.loc[supervised.timestamp.between(start, end)].copy()
    expected = pd.date_range(start, end, freq="5min")
    if (train.empty or not evaluation.timestamp.reset_index(drop=True).equals(pd.Series(expected))
            or not (evaluation.target_timestamp == evaluation.timestamp + pd.Timedelta(hours=1)).all()
            or not (evaluation.target_timestamp < pd.Timestamp(TEST_START)).all()):
        raise ValueError("fixed_evaluation_coverage_or_boundary_invalid")
    return train, evaluation
