"""树和线性参照；全部在相同内层训练行拟合。"""

from time import perf_counter

import joblib
import numpy as np
from lightgbm import LGBMRegressor, early_stopping
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ..features import FEATURE_COLUMNS
from .neural import prediction_metrics


def input_view(recipe, rows, windows):
    if recipe["family"] == "ridge_original":
        return rows[FEATURE_COLUMNS].to_numpy(float)
    return windows.reshape(len(windows), -1)


def clipped_mae(actual, predicted):
    return "clipped_mae", float(np.mean(np.abs(actual - np.maximum(predicted, 0)))), False


def fit_tabular(data, recipe, artifact, *, deadline=None):
    started = perf_counter()
    x = input_view(recipe, data.train, data.train_windows)
    validation = input_view(recipe, data.validation, data.validation_windows)
    target, val_target = data.train.target_power.to_numpy(float), data.validation.target_power.to_numpy(float)
    if recipe["family"] == "lightgbm":
        model = LGBMRegressor(objective="regression", metric="None", learning_rate=0.03,
                             n_estimators=1000, num_leaves=recipe["num_leaves"],
                             min_child_samples=recipe["min_child_samples"],
                             n_jobs=2, random_state=42, deterministic=True, force_col_wise=True,
                             verbosity=-1)
        def budget_callback(env):
            if deadline is not None and perf_counter() >= deadline:
                raise TimeoutError("comparison_tree_budget_exhausted")
        budget_callback.order = 5
        model.fit(x, target, eval_set=[(validation, val_target)], eval_metric=clipped_mae,
                  callbacks=[budget_callback, early_stopping(50, first_metric_only=True, verbose=False)])
        details = {"selected_iteration": model.best_iteration_, "validation_curve": model.evals_result_}
    else:
        model = make_pipeline(StandardScaler(), Ridge(alpha=recipe["alpha"], solver="svd"))
        with threadpool_limits(limits=2):
            model.fit(x, target)
        scaler = model.named_steps["standardscaler"]
        details = {"scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                   "scaler_samples": int(scaler.n_samples_seen_)}
    fit_seconds = perf_counter() - started
    raw = model.predict(validation)
    details.update({"fit_elapsed_seconds": fit_seconds, "validation": prediction_metrics(val_target, raw),
                    "train_eval": prediction_metrics(target, model.predict(x)),
                    "input_dimensions": int(x.shape[1]), "device": "cpu", "threads": 2})
    if recipe["family"] == "lightgbm":
        details["parameters"] = model.get_params()
    joblib.dump(model, artifact)
    reloaded = joblib.load(artifact).predict(validation)
    np.testing.assert_allclose(raw, reloaded, rtol=1e-10, atol=1e-8)
    details["reload_max_difference"] = float(np.max(np.abs(raw - reloaded)))
    return model, details
