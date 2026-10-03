"""同次拟合完整包、独立重载以及加载前损坏/兼容拒绝。"""

import json
import subprocess
from copy import deepcopy
from uuid import uuid4

import pytest
from sqlalchemy import URL

from power_forecast_service.experiments.contracts import ExperimentRequest, freeze_spec
from power_forecast_service.forecasting.bundles import (
    predict_package, save_packages, verify_fresh_process, verify_packages,
)
from power_forecast_service.forecasting.contracts import ForecastRequest
from power_forecast_service.forecasting.pipeline import ExperimentProduct
from power_forecast_service.settings import Settings
from power_forecast_service.storage.artifacts import execution_provenance, sha256_file
from power_forecast_service.storage.model_packages import PackageError, checked_manifest


@pytest.fixture(scope="module")
def pair(tmp_path_factory, trained_product, sample_path):
    root = tmp_path_factory.mktemp("model-package-verification")
    result = deepcopy(trained_product.result)
    # Settings实例不读取凭据；此组测试不连接任何服务。
    config = Settings(database_url=URL.create("postgresql+psycopg"),
                      data_path=sample_path, artifact_root=root)
    result.update(frozen_spec=freeze_spec(ExperimentRequest(), config), execution=execution_provenance())
    product = ExperimentProduct(result, trained_product.models, trained_product.frame)
    packages = save_packages(product, root, uuid4(), uuid4())
    return root, packages


def test_same_fit_full_prediction_and_fresh_process(pair, monkeypatch):
    from sklearn.ensemble import HistGradientBoostingRegressor

    def forbidden(*args, **kwargs):
        pytest.fail("Packaging verification and prediction cannot refit")

    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", forbidden)
    root, packages = pair
    assert {item.model_key for item in packages} == {"persistence", "hist_gradient_boosting"}
    evidence = verify_packages(root, packages)
    assert evidence == verify_fresh_process(root, packages)
    assert all(item["samples"] == 3 and item["max_absolute_difference"] <= 1e-8
               for item in evidence)
    # 重载不能新增pycache或修改包文件，让下一次加载的哈希检查失效。
    assert evidence == verify_packages(root, packages)


@pytest.mark.parametrize("mutation,reason", [
    ("payload", "artifact_integrity_failed"), ("extra", "artifact_integrity_failed"),
    ("missing", "artifact_integrity_failed"), ("manifest", "artifact_integrity_failed"),
    ("feature", "artifact_incompatible"), ("runtime", "artifact_incompatible"),
    ("version", "artifact_incompatible"), ("input", "artifact_incompatible"),
    ("identity", "artifact_identity_mismatch"), ("escape", "artifact_path_invalid"),
])
def test_reject_before_deserialization(pair, tmp_path, monkeypatch, mutation, reason):
    import shutil
    import mlflow.pyfunc

    root, packages = pair
    item = packages[0].model_copy(deep=True)
    target = tmp_path / item.path
    shutil.copytree(root / item.path, target)
    case = json.loads((target / "verification.json").read_text())["cases"][0]
    body = ForecastRequest(artifact_id=item.artifact_id, observations=case["observations"])
    if mutation == "payload":
        (target / "model/python_model.pkl").write_bytes(b"broken")
    elif mutation == "extra":
        (target / "unexpected").write_text("unexpected")
    elif mutation == "missing":
        (target / "model/MLmodel").unlink()
    elif mutation == "manifest":
        (target / "manifest.json").write_text("{}")
    elif mutation == "identity":
        item.model_key = "other"
    elif mutation == "escape":
        item.path = "../escape"
    else:
        manifest = item.manifest
        if mutation == "feature":
            manifest["feature_contract_version"] = "unknown"
        elif mutation == "version":
            manifest["package_version"] = "future-v9"
        elif mutation == "runtime":
            manifest["runtime"]["python"] = "0.0.0"
        else:
            manifest["input_contract"]["min_observations"] = 1
        (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        item.manifest_sha256 = sha256_file(target / "manifest.json")

    def forbidden(*args, **kwargs):
        pytest.fail("Untrusted/incompatible package reached deserialization")

    monkeypatch.setattr(mlflow.pyfunc, "load_model", forbidden)
    with pytest.raises(PackageError, match=reason):
        predict_package(tmp_path, item, body)


@pytest.mark.parametrize("timeout", [False, True])
def test_verification_failure_preserves_diagnostic(pair, tmp_path, monkeypatch, timeout):
    import shutil
    root, packages = pair
    item = packages[0]
    shutil.copytree(root / item.path, tmp_path / item.path)

    def failed(*args, **kwargs):
        if timeout:
            raise subprocess.TimeoutExpired("verify", 90, stderr=b"unfinished loader")
        return subprocess.CompletedProcess("verify", 1, "", "specific loader failure")

    monkeypatch.setattr(subprocess, "run", failed)
    with pytest.raises(PackageError, match="artifact_verification_timeout" if timeout
                       else "artifact_fresh_process_verification_failed"):
        verify_fresh_process(tmp_path, [item])
    diagnostic = tmp_path / item.path / "../../model-verification-failure.json"
    evidence = json.loads(diagnostic.read_text())
    assert evidence["stderr"] == ("unfinished loader" if timeout else "specific loader failure")
    checked_manifest(tmp_path, item)
