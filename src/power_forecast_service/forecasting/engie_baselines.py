"""同一 ENGIE 合同下的有限基线；内层选型后统一用外层起报前可得数据重拟合。"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from lightgbm import LGBMRegressor, early_stopping
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .engie_contract import FEATURES, HORIZONS, LOOKBACK, ROSTER, persistence


RIDGE_ALPHAS = (1.0, 10.0, 100.0)
TREE_PARAMETERS = {
    "objective": "regression", "metric": "l1", "learning_rate": 0.05,
    "n_estimators": 400, "num_leaves": 31, "min_child_samples": 50,
    "n_jobs": 2, "random_state": 42, "deterministic": True,
    "force_col_wise": True, "verbosity": -1,
}


def raw_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if actual.shape != predicted.shape or not actual.size:
        raise ValueError("engie_score_shape_or_empty")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("engie_incomplete_predictions_cannot_enter_main_ranking")
    error = predicted - actual
    return {"samples": int(error.size), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())), "bias": float(error.mean())}


def score_curves(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.ndim != 3 or actual.shape[1:] != (4, 6):
        raise ValueError("engie_expected_issue_turbine_horizon")
    # 先检查完整预测，再求固定四机组合计，禁止 nansum 把缺失偷换为零。
    all_values = raw_metrics(actual, predicted)
    farm_y, farm_p = actual.sum(axis=1), predicted.sum(axis=1)
    return {
        "farm": raw_metrics(farm_y, farm_p), "all_turbine_values": all_values,
        "farm_by_horizon": {str(h): raw_metrics(farm_y[:, k], farm_p[:, k])
                            for k, h in enumerate(HORIZONS)},
        "turbines": {name: {
            "all_horizons": raw_metrics(actual[:, j], predicted[:, j]),
            "by_horizon": {str(h): raw_metrics(actual[:, j, k], predicted[:, j, k])
                           for k, h in enumerate(HORIZONS)},
        } for j, name in enumerate(ROSTER)},
    }


@dataclass
class FittedBaseline:
    family: str
    models: list

    def predict(self, features):
        features = np.asarray(features, float)
        if features.ndim != 3 or features.shape[1:] != (4, LOOKBACK * len(FEATURES)):
            raise ValueError("engie_prediction_input_shape")
        if not np.isfinite(features).all():
            raise ValueError("engie_prediction_requires_complete_fixed_roster_history")
        if self.family == "persistence":
            return persistence(features)
        result = np.empty((len(features), 4, 6))
        for turbine in range(4):
            x = features[:, turbine]
            if self.family == "ridge":
                result[:, turbine] = self.models[turbine].predict(x)
            elif self.family == "lightgbm":
                for horizon in range(6):
                    # 同一树在拟合和预测使用同一列顺序，保留原始负值输出。
                    model = self.models[turbine][horizon]
                    result[:, turbine, horizon] = model.booster_.predict(x, num_threads=2)
            else:
                raise ValueError(f"engie_unknown_family:{self.family}")
        if not np.isfinite(result).all():
            raise ValueError("engie_model_nonfinite_output")
        return result


def _ridge_models(x, y, alpha):
    models = []
    with threadpool_limits(limits=2):
        for turbine in range(4):
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="svd"))
            model.fit(x[:, turbine], y[:, turbine])
            models.append(model)
    return models


def fit_ridge(x_train, y_train, x_validation, y_validation, x_refit, y_refit,
              *, alphas=RIDGE_ALPHAS):
    started = perf_counter()
    candidates = []
    for alpha in alphas:
        fitted = FittedBaseline("ridge", _ridge_models(x_train, y_train, alpha))
        score = score_curves(y_validation, fitted.predict(x_validation))["farm"]["mae"]
        candidates.append({"alpha": alpha, "inner_farm_mae": score})
    chosen = min(candidates, key=lambda item: (item["inner_farm_mae"], item["alpha"]))
    models = _ridge_models(x_refit, y_refit, chosen["alpha"])
    detail = {
        "selected_alpha": chosen["alpha"], "selection_candidates": candidates,
        "selection_metric": "raw_fixed_four_turbine_sum_six_horizon_mae",
        "fit_seconds": perf_counter() - started,
        "scaler_means": [model.named_steps["standardscaler"].mean_.tolist() for model in models],
        "scaler_samples": [int(model.named_steps["standardscaler"].n_samples_seen_) for model in models],
        "estimator_fit_count": len(alphas) * 4 + 4,
    }
    return FittedBaseline("ridge", models), detail


def fit_lightgbm(x_train, y_train, x_validation, y_validation, x_refit, y_refit,
                 *, parameters=None):
    started = perf_counter()
    parameters = dict(TREE_PARAMETERS if parameters is None else parameters)
    models, iterations, inner_errors = [], [], []
    for turbine in range(4):
        turbine_models, turbine_iterations, turbine_errors = [], [], []
        for horizon in range(6):
            model = LGBMRegressor(**parameters)
            model.fit(x_train[:, turbine], y_train[:, turbine, horizon],
                      eval_set=[(x_validation[:, turbine], y_validation[:, turbine, horizon])],
                      callbacks=[early_stopping(40, first_metric_only=True, verbose=False)])
            iteration = int(model.best_iteration_ or parameters["n_estimators"])
            prediction = model.booster_.predict(x_validation[:, turbine], num_threads=2)
            turbine_errors.append(raw_metrics(y_validation[:, turbine, horizon], prediction))
            # 内层确定迭代数后重拟合；外层标签绝不用于 early stopping。
            final = LGBMRegressor(**{**parameters, "n_estimators": iteration})
            final.fit(x_refit[:, turbine], y_refit[:, turbine, horizon])
            turbine_models.append(final)
            turbine_iterations.append(iteration)
        models.append(turbine_models)
        iterations.append(turbine_iterations)
        inner_errors.append(turbine_errors)
    detail = {
        "selected_iterations": iterations, "inner_errors": inner_errors,
        "selection_metric": "raw_per_turbine_per_horizon_mae_early_stopping",
        "fit_seconds": perf_counter() - started, "parameters": parameters,
        "estimator_fit_count": 48,
    }
    return FittedBaseline("lightgbm", models), detail


def grouped_scores(y_train, x_train, actual, features, predictions):
    """阈值只由训练段得到；未来变化分组只用于事后诊断，不变成起报门控。"""
    train_last = persistence(x_train).sum(axis=1)[:, -1]
    train_target = y_train.sum(axis=1)[:, -1]
    change = np.abs(train_target - train_last)
    threshold = float(np.quantile(change, 0.75))
    power_threshold = float(np.quantile(train_target, 0.25))
    actual_sum = actual.sum(axis=1)[:, -1]
    latest_sum = persistence(features).sum(axis=1)[:, -1]
    masks = {
        "large_realized_change": np.abs(actual_sum - latest_sum) >= threshold,
        "smaller_realized_change": np.abs(actual_sum - latest_sum) < threshold,
        "low_actual_power": actual_sum <= power_threshold,
    }
    return {"threshold_source": "outer_past_refit_training_only", "change_q75_kw": threshold,
            "power_q25_kw": power_threshold, "not_online_gates": True,
            "groups": {name: {"issues": int(mask.sum()), "metrics": {
                family: score_curves(actual[mask], output[mask])["farm"]
                for family, output in predictions.items()
            } if mask.any() else {}} for name, mask in masks.items()}}
