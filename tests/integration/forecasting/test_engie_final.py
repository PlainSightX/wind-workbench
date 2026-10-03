"""构造数据实拟合，检查选择/重拟合隔离与实际底层拟合数。"""

import numpy as np
import pytest
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from power_forecast_service.forecasting.engie_baselines import TREE_PARAMETERS, _ridge_models
from power_forecast_service.forecasting.engie_final import FinalEngie, select_ridge, select_trees, refit_trees
from power_forecast_service.forecasting.engie_l1 import L1_PARAMETERS

pytestmark = pytest.mark.integration


def test_selection_and_refit_budget_train_only_scaler_and_shared_base(monkeypatch):
    rng = np.random.default_rng(12)
    x, y = rng.normal(size=(90, 4, 60)), rng.normal(size=(90, 4, 6))
    vx, vy = rng.normal(20, size=(25, 4, 60)), rng.normal(size=(25, 4, 6))
    counts, means = {"ridge": 0, "tree": 0}, []
    original_ridge, original_tree, original_scaler = Ridge.fit, LGBMRegressor.fit, StandardScaler.fit
    def ridge(self, *args, **kwargs):
        counts["ridge"] += 1
        return original_ridge(self, *args, **kwargs)
    def tree(self, *args, **kwargs):
        counts["tree"] += 1
        return original_tree(self, *args, **kwargs)
    def scaler(self, values, *args, **kwargs):
        result = original_scaler(self, values, *args, **kwargs)
        means.append(self.mean_.copy())
        return result
    monkeypatch.setattr(Ridge, "fit", ridge)
    monkeypatch.setattr(LGBMRegressor, "fit", tree)
    monkeypatch.setattr(StandardScaler, "fit", scaler)
    predictions, selection = select_ridge(x, y, vx, vy)
    assert predictions.shape == (3, 25, 4, 6) and counts["ridge"] == 12
    for i, mean in enumerate(means):
        np.testing.assert_allclose(mean, x[:, i % 4].mean(0))
    _ridge_models(np.concatenate([x, vx]), np.concatenate([y, vy]), selection["alpha"])
    assert counts["ridge"] == 16
    for params in (TREE_PARAMETERS, L1_PARAMETERS):
        # 缩短构造数据树数，不改变实际场站协议。
        params = {**params, "n_estimators": 2, "min_child_samples": 5}
        predicted, chosen = select_trees(x, y, vx, vy, params)
        assert predicted.shape == vy.shape and chosen["fits"] == 24
        base, receipt = refit_trees(np.concatenate([x, vx]), np.concatenate([y, vy]), chosen["iterations"], params)
        assert receipt["fits"] == 24
    assert sum(counts.values()) == 112
    direct = FinalEngie("lightgbm_l1", base)
    shrink = FinalEngie("lightgbm_l1_shrink", base, 1)
    assert direct.base is shrink.base
    np.testing.assert_allclose(direct.predict(vx), shrink.predict(vx), atol=1e-12, rtol=1e-12)
