"""训练对象只含内层数据；外层标签不进入训练函数的接口。"""

from dataclasses import dataclass

import numpy as np

from ..fixed_evaluation import fixed_window_split
from ..sequence_data import history_windows, observation_features
from ..sequence_protocol import LOOKBACK


@dataclass
class FitData:
    train: object
    validation: object
    train_windows: np.ndarray
    validation_windows: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    identity: dict


def inner_split(outer_train):
    boundary = int(len(outer_train) * 0.8)
    validation = outer_train.iloc[boundary:].copy()
    train = outer_train.loc[outer_train.target_timestamp < validation.timestamp.iloc[0]].copy()
    if train.empty or validation.empty:
        raise ValueError("empty_inner_split")
    return train, validation


def prepare_fit_data(frame, train, validation):
    windows, positions = history_windows(frame, train.timestamp)
    val_windows, _ = history_windows(frame, validation.timestamp)
    unique = np.unique((positions[:, None] - np.arange(LOOKBACK)).ravel())
    raw = observation_features(frame)[unique]
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    scale[scale == 0] = 1.0
    identity = {"train_samples": len(train), "validation_samples": len(validation),
                "train_cutoff_start": train.timestamp.iloc[0].isoformat(),
                "train_cutoff_end": train.timestamp.iloc[-1].isoformat(),
                "train_target_end": train.target_timestamp.iloc[-1].isoformat(),
                "validation_cutoff_start": validation.timestamp.iloc[0].isoformat(),
                "validation_cutoff_end": validation.timestamp.iloc[-1].isoformat(),
                "scaler_observations": len(unique),
                "scaler_last_time": frame.timestamp.iloc[unique[-1]].isoformat()}
    return FitData(train, validation, windows, val_windows, mean, scale, identity)


def window_fit_data(frame, supervised, window):
    outer_train, _ = fixed_window_split(supervised, window)
    train, validation = inner_split(outer_train)
    return prepare_fit_data(frame, train, validation)
