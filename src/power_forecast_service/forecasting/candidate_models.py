"""有限简单对照；缩放和增量重构与拟合模型一同保存，推理不重新fit。"""

from time import perf_counter

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .development_protocol import RIDGE_ALPHAS
from .features import FEATURE_COLUMNS
from .spec import training_parameters


class PowerIncrementRegressor(RegressorMixin, BaseEstimator):
    """只替换监督目标；保留输入、HGB参数及seed，用当前功率还原绝对预测。"""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y):
        self.estimator_ = HistGradientBoostingRegressor(
            **training_parameters("fixed_iterations"), random_state=self.random_state)
        self.estimator_.fit(X, np.asarray(y) - X["wind_power_lag_0"].to_numpy())
        return self

    def predict(self, X):
        return X["wind_power_lag_0"].to_numpy() + self.estimator_.predict(X)


def fit_candidate(train, key, *, random_state=42):
    if key in RIDGE_ALPHAS:
        model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHAS[key], solver="svd"))
    elif key == "hgb_delta":
        model = PowerIncrementRegressor(random_state=random_state)
    else:
        raise ValueError("unsupported_candidate_key")
    started = perf_counter()
    model.fit(train[FEATURE_COLUMNS], train["target_power"])
    details = {
        "model_key": key, "fit_elapsed_seconds": perf_counter() - started,
        "input_samples": len(train), "feature_names": list(FEATURE_COLUMNS),
        "preprocessing": "standard_scaler_train_only" if key in RIDGE_ALPHAS else "none_required",
        "target_representation": "increment_from_current" if key == "hgb_delta" else "direct_power",
        "parameters": ({"alpha": RIDGE_ALPHAS[key], "solver": "svd"} if key in RIDGE_ALPHAS
                       else {**training_parameters("fixed_iterations"), "random_state": random_state}),
    }
    if key in RIDGE_ALPHAS:
        scaler = model.named_steps["standardscaler"]
        details["scaler"] = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                             "n_samples_seen": int(scaler.n_samples_seen_)}
    return model, details


def candidate_predict(model, frame):
    prediction = np.maximum(model.predict(frame[FEATURE_COLUMNS]), 0.0)
    if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
        raise ValueError("candidate_prediction_invalid")
    return prediction
