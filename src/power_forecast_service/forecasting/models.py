from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import version
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import FEATURE_COLUMNS
from .spec import DEFAULT_TRAINING_POLICY, TrainingPolicy, training_parameters


@dataclass
class ForecastModels:
    improved: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]
    training_details: dict
    candidates: dict = field(default_factory=dict)
    candidate_details: dict = field(default_factory=dict)


def fit_models(
    train_frame: pd.DataFrame, *, random_state: int = 42,
    training_policy: TrainingPolicy = DEFAULT_TRAINING_POLICY,
    model_parameters: dict | None = None,
) -> ForecastModels:
    """按冻结参数拟合，并记录真实早停行为；内部留出不是外部验证段。"""
    parameters = training_parameters(training_policy)
    if model_parameters is not None:
        if model_parameters != parameters:
            raise ValueError("training_policy_parameters_mismatch")
        parameters = dict(model_parameters)
    model = HistGradientBoostingRegressor(
        **parameters,
        random_state=random_state,
    )
    started = perf_counter()
    model.fit(train_frame[FEATURE_COLUMNS], train_frame["target_power"])
    elapsed = perf_counter() - started
    early_stopping = bool(model.do_early_stopping_)
    details = {
        "training_policy": training_policy,
        "estimator_class": type(model).__name__,
        "sklearn_version": version("scikit-learn"),
        "effective_parameters": model.get_params(deep=False),
        "input_samples": len(train_frame),
        "n_features_in": int(model.n_features_in_),
        "feature_names": list(model.feature_names_in_),
        "n_iter": int(model.n_iter_),
        "max_iter": model.max_iter,
        "early_stopping_enabled": early_stopping,
        "stopped_before_max_iter": early_stopping and model.n_iter_ < model.max_iter,
        "internal_validation_mode": "sklearn_random_holdout" if early_stopping else "none",
        "internal_membership_recorded": False,
        "fit_elapsed_seconds": elapsed,
        # scoring=loss记录负训练目标，当前是half squared error；不是外层MAE。
        "objective_score_definition": "negative_half_squared_error",
        "objective_scores_include_initial_model": early_stopping,
        "train_objective_scores": model.train_score_.tolist(),
        "internal_validation_objective_scores": model.validation_score_.tolist(),
    }
    return ForecastModels(model, tuple(FEATURE_COLUMNS), details)


def persistence_predict(frame: pd.DataFrame) -> np.ndarray:
    """持久性基线：用 cutoff 时刻的最新功率预测未来。"""
    return frame["wind_power_lag_0"].to_numpy(dtype=float)


def improved_predict(models: ForecastModels, frame: pd.DataFrame) -> np.ndarray:
    prediction = models.improved.predict(frame[list(models.feature_columns)])
    return np.maximum(prediction, 0.0)
