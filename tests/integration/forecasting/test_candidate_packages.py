"""新增候选完整包与v1并存，不以直接pickle加载代替历史输入端到端一致性。"""

from uuid import uuid4

import pytest
from sqlalchemy import URL

from power_forecast_service.experiments.contracts import ExperimentRequest, freeze_spec
from power_forecast_service.forecasting.bundles import save_packages, verify_fresh_process
from power_forecast_service.forecasting.pipeline import train_experiment
from power_forecast_service.settings import Settings
from power_forecast_service.storage.artifacts import execution_provenance
from power_forecast_service.storage.model_packages import checked_manifest


@pytest.mark.parametrize("key", ["ridge_1", "hgb_delta"])
def test_candidate_same_fit_package_reload(sample_path, tmp_path, key):
    config = Settings(database_url=URL.create("postgresql+psycopg"), data_path=sample_path,
                      artifact_root=tmp_path)
    product = train_experiment(sample_path, training_policy="fixed_iterations", candidate_key=key)
    product.result.update(frozen_spec=freeze_spec(ExperimentRequest(
        training_policy="fixed_iterations", candidate_key=key), config), execution=execution_provenance())
    packages = save_packages(product, tmp_path, uuid4(), uuid4())
    assert {p.model_key for p in packages} == {"persistence", "hist_gradient_boosting", key}
    manifests = {p.model_key: checked_manifest(tmp_path, p) for p in packages}
    assert manifests["hist_gradient_boosting"].package_version == "history-pyfunc-v1"
    assert manifests[key].package_version == "history-pyfunc-v2"
    assert manifests[key].preprocessing == ("standard_scaler_train_only" if key == "ridge_1" else "none_required")
    evidence = verify_fresh_process(tmp_path, packages)
    assert all(e["max_absolute_difference"] <= 1e-8 for e in evidence)
