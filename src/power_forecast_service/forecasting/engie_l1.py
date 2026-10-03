"""ENGIE有界L1对照：先保存内层选择，再重拟合；收缩不增加估计器。"""

from dataclasses import dataclass
from time import perf_counter

from lightgbm import LGBMRegressor, early_stopping
import numpy as np

from .engie_baselines import FittedBaseline, TREE_PARAMETERS, score_curves
from .engie_contract import persistence


LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
L1_PARAMETERS = {**TREE_PARAMETERS, "objective": "regression_l1"}
ADOPTION_GATES = {
    "minimum_improved_quarters": 2,
    "minimum_equal_quarter_mae_gain_percent": 3.0,
    "maximum_quarter_mae_regression_percent": 2.0,
    "maximum_equal_quarter_rmse_ratio": 1.0,
    "complete_legal_input_coverage_required": True,
}


def blend(reference, candidate, weight):
    """同一个权重作用于全机组、全时距；不按未来变化决定是否预测。"""
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.shape[1:] != (4, 6):
        raise ValueError("engie_shrinkage_shape")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("engie_shrinkage_nonfinite")
    if weight not in LAMBDAS:
        raise ValueError("engie_shrinkage_unregistered_weight")
    return reference + weight * (candidate - reference)


def select_weight(actual, reference, candidate):
    scores = [{"lambda": weight, "farm_mae": score_curves(
        actual, blend(reference, candidate, weight))["farm"]["mae"]} for weight in LAMBDAS]
    selected = min(scores, key=lambda row: (row["farm_mae"], row["lambda"]))
    return {"selected_lambda": selected["lambda"], "candidates": scores}


def fit_inner(x_train, y_train, x_validation, y_validation, *, parameters=None):
    """返回内层模型和预测；此函数没有outer-refit输入，不能提前消费它。"""
    parameters = dict(L1_PARAMETERS if parameters is None else parameters)
    if parameters["objective"] != "regression_l1":
        raise ValueError("engie_a2_requires_l1_objective")
    started = perf_counter()
    models, iterations = [], []
    for turbine in range(4):
        row, selected = [], []
        for horizon in range(6):
            model = LGBMRegressor(**parameters)
            model.fit(x_train[:, turbine], y_train[:, turbine, horizon],
                      eval_set=[(x_validation[:, turbine], y_validation[:, turbine, horizon])],
                      callbacks=[early_stopping(40, first_metric_only=True, verbose=False)])
            row.append(model)
            selected.append(int(model.best_iteration_ or parameters["n_estimators"]))
        models.append(row)
        iterations.append(selected)
    fitted = FittedBaseline("lightgbm", models)
    predictions = fitted.predict(x_validation)
    return predictions, {
        "selected_iterations": iterations, "parameters": parameters,
        "fit_seconds": perf_counter() - started, "estimator_fit_count": 24,
    }


@dataclass
class ShrunkEngie:
    """完整离线候选包；direct对照由base读取，部署采用仍需独立决定。"""

    base: FittedBaseline
    weight: float

    def predict(self, features):
        return blend(persistence(features), self.base.predict(features), self.weight)


def refit(x, y, iterations, weight, *, parameters=None):
    parameters = dict(L1_PARAMETERS if parameters is None else parameters)
    iterations = np.asarray(iterations)
    if (iterations.shape != (4, 6) or not np.issubdtype(iterations.dtype, np.integer)
            or (iterations < 1).any() or (iterations > parameters["n_estimators"]).any()
            or weight not in LAMBDAS or parameters["objective"] != "regression_l1"):
        raise ValueError("engie_a2_invalid_frozen_selection")
    started = perf_counter()
    models = []
    for turbine in range(4):
        row = []
        for horizon in range(6):
            model = LGBMRegressor(**{**parameters, "n_estimators": int(iterations[turbine, horizon])})
            model.fit(x[:, turbine], y[:, turbine, horizon])
            row.append(model)
        models.append(row)
    return ShrunkEngie(FittedBaseline("lightgbm", models), weight), {
        "fit_seconds": perf_counter() - started, "estimator_fit_count": 24,
    }


def adoption_decision(windows, family):
    """开发投入门槛，不冒充行业标准或独立留出上的统计显著性。"""
    reference = [w["metrics"]["persistence"]["farm"] for w in windows.values()]
    candidate = [w["metrics"][family]["farm"] for w in windows.values()]
    gains = [100 * (1 - c["mae"] / r["mae"]) for c, r in zip(candidate, reference)]
    mae_gain = 100 * (1 - np.mean([c["mae"] for c in candidate])
                      / np.mean([r["mae"] for r in reference]))
    rmse_ratio = np.mean([c["rmse"] for c in candidate]) / np.mean([r["rmse"] for r in reference])
    gates = {
        "two_quarters_improve": sum(g > 0 for g in gains) >= ADOPTION_GATES["minimum_improved_quarters"],
        "mean_mae_gain": mae_gain >= ADOPTION_GATES["minimum_equal_quarter_mae_gain_percent"],
        "worst_quarter": min(gains) >= -ADOPTION_GATES["maximum_quarter_mae_regression_percent"],
        "mean_rmse": rmse_ratio <= ADOPTION_GATES["maximum_equal_quarter_rmse_ratio"],
        "coverage": all(w["actual_outputs"][family] == w["counts"]["input_valid"]
                        == w["actual_outputs"]["persistence"] for w in windows.values()),
    }
    return {"passed": bool(all(gates.values())), "gates": {k: bool(v) for k, v in gates.items()},
            "quarter_mae_gain_percent": dict(zip(windows, gains)),
            "equal_quarter_mae_gain_percent": float(mae_gain),
            "equal_quarter_rmse_ratio": float(rmse_ratio)}
