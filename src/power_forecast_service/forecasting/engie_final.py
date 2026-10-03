"""冻结的ENGIE最终模型与配对时间块区间；不改变开发期序列化对象。"""

from dataclasses import dataclass
from time import perf_counter

from lightgbm import LGBMRegressor, early_stopping
import numpy as np

from .engie_baselines import FittedBaseline, RIDGE_ALPHAS, _ridge_models, score_curves
from .engie_contract import persistence
from .engie_l1 import blend

FAMILIES = ("persistence", "ridge", "lightgbm", "lightgbm_l1", "lightgbm_l1_shrink")


@dataclass
class FinalEngie:
    family: str
    base: FittedBaseline
    weight: float = 1.0

    def predict(self, features):
        prediction = self.base.predict(features)
        if self.family == "lightgbm_l1_shrink":
            return blend(persistence(features), prediction, self.weight)
        if self.family not in FAMILIES:
            raise ValueError("engie_final_unknown_family")
        return prediction


def select_ridge(x, y, vx, vy):
    started, choices, predictions = perf_counter(), [], []
    for alpha in RIDGE_ALPHAS:
        model = FittedBaseline("ridge", _ridge_models(x, y, alpha))
        predicted = model.predict(vx)
        predictions.append(predicted)
        choices.append({"alpha": alpha, "mae": score_curves(vy, predicted)["farm"]["mae"]})
    return np.stack(predictions), {"alpha": min(choices, key=lambda c: (c["mae"], c["alpha"]))["alpha"],
            "choices": choices, "fits": 12, "seconds": perf_counter() - started}


def select_trees(x, y, vx, vy, parameters):
    started, iterations = perf_counter(), []
    predictions = np.empty_like(vy)
    for t in range(4):
        row = []
        for h in range(6):
            model = LGBMRegressor(**parameters)
            model.fit(x[:, t], y[:, t, h], eval_set=[(vx[:, t], vy[:, t, h])],
                      callbacks=[early_stopping(40, first_metric_only=True, verbose=False)])
            row.append(int(model.best_iteration_ or parameters["n_estimators"]))
            predictions[:, t, h] = model.booster_.predict(vx[:, t], num_threads=2)
        iterations.append(row)
    return predictions, {"iterations": iterations, "fits": 24, "seconds": perf_counter() - started}


def refit_trees(x, y, iterations, parameters):
    iterations = np.asarray(iterations)
    if (iterations.shape != (4, 6) or not np.issubdtype(iterations.dtype, np.integer)
            or (iterations < 1).any() or (iterations > parameters["n_estimators"]).any()):
        raise ValueError("engie_final_invalid_iterations")
    started, models = perf_counter(), []
    for t in range(4):
        row = []
        for h in range(6):
            model = LGBMRegressor(**{**parameters, "n_estimators": int(iterations[t, h])})
            model.fit(x[:, t], y[:, t, h])
            row.append(model)
        models.append(row)
    return FittedBaseline("lightgbm", models), {"fits": 24, "seconds": perf_counter() - started}


def paired_blocks(quarters, *, block_steps, samples=2000, seed=42):
    """在未压缩的季度日历上抽连续块；同一索引同时作用于所有策略与缺测掩码。"""
    rng = np.random.default_rng(seed)
    replicates = {f: [] for f in FAMILIES}
    for losses in quarters.values():
        reference = np.asarray(losses["persistence"])
        count = len(reference)
        if count < block_steps or block_steps < 1 or not np.isfinite(reference).any():
            raise ValueError("engie_bootstrap_insufficient_calendar")
        mask = np.isfinite(reference)
        lengths = np.full(int(np.ceil(count / block_steps)), block_steps, dtype=int)
        lengths[-1] = count - int(lengths[:-1].sum())
        starts = rng.integers(0, count - block_steps + 1, size=(samples, len(lengths)))
        ends = starts + lengths
        prefix = np.r_[0, np.cumsum(mask)]
        denominators = (prefix[ends] - prefix[starts]).sum(axis=1)
        if (denominators == 0).any():
            raise ValueError("engie_bootstrap_empty_resample")
        for family in FAMILIES:
            values = np.asarray(losses[family])
            if values.shape != reference.shape or not np.array_equal(np.isfinite(values), mask):
                raise ValueError("engie_bootstrap_unpaired_samples")
            sums = np.r_[0., np.cumsum(np.where(mask, values, 0.))]
            totals = (sums[ends] - sums[starts]).sum(axis=1)
            replicates[family].append(totals / denominators)
    means = {f: np.mean(v, axis=0) for f, v in replicates.items()}
    result = {}
    for family in FAMILIES:
        difference = means[family] - means["persistence"]
        gain = 100 * (1 - means[family] / means["persistence"])
        if not np.isfinite(gain).all():
            raise ValueError("engie_bootstrap_zero_reference")
        result[family] = {
            "delta_mae_kw_95ci": np.quantile(difference, [.025, .975]).tolist(),
            "gain_percent_95ci": np.quantile(gain, [.025, .975]).tolist(),
        }
    return {"block_steps": block_steps, "samples": samples, "seed": seed,
            "method": "paired_non_circular_quarter_stratified_percentile_equal_quarter_mae",
            "families": result}
