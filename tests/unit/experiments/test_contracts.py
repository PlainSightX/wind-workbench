"""有限训练合同与历史身份；不连接数据库，不用新freeze函数伪造旧配置。"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from power_forecast_service.experiments.contracts import (
    ExperimentRequest, TaskConflict, fingerprint, freeze_spec,
    submission_fingerprint, validate_spec,
)
from power_forecast_service.forecasting import spec as model_spec
from power_forecast_service.storage.artifacts import sha256_file


@pytest.fixture
def settings(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text("contract fixture only", encoding="utf-8")
    return SimpleNamespace(data_path=path)


def historical_spec(path):
    return {
        "dataset_id": "wind-2019-q1", "purpose": "development",
        "input_sha256": sha256_file(path),
        "split_version": "temporal-70-15-15-label-isolated-v0.2",
        "feature_contract_version": "lag-only-source-time-v0.1",
        "model_set": ["persistence", "hist_gradient_boosting"],
        "horizon_steps": 12, "random_seed": 42,
        "hgb": {"learning_rate": 0.06, "max_iter": 180,
                "max_leaf_nodes": 31, "l2_regularization": 0.2},
    }


def test_default_request_keeps_historical_identity():
    old_hash = fingerprint({
        "action": "submit", "request": {"dataset_id": "wind-2019-q1", "purpose": "development"},
    })
    assert submission_fingerprint(ExperimentRequest()) == old_hash
    assert submission_fingerprint(ExperimentRequest(training_policy="auto_early_stopping")) == old_hash
    assert submission_fingerprint(ExperimentRequest(training_policy="fixed_iterations")) != old_hash


@pytest.mark.parametrize("policy", ["auto_early_stopping", "fixed_iterations"])
def test_frozen_configuration_is_independent_and_resolved(settings, monkeypatch, policy):
    frozen = freeze_spec(ExperimentRequest(training_policy=policy), settings)
    before = deepcopy(frozen)
    monkeypatch.setitem(model_spec.HGB_PARAMETERS, "max_iter", 999)
    assert validate_spec(frozen, settings) == before
    other = freeze_spec(ExperimentRequest(training_policy=policy), settings)
    other["hgb"]["max_iter"] = 1
    assert frozen == before
    assert frozen["hgb"]["early_stopping"] == (
        "auto" if policy == "auto_early_stopping" else False
    )


def test_exact_legacy_contract_is_resolved_without_rewriting(settings, monkeypatch):
    frozen = historical_spec(settings.data_path)
    before = deepcopy(frozen)
    monkeypatch.setattr(model_spec, "HGB_PARAMETERS", {"max_iter": 999})
    resolved = validate_spec(frozen, settings)
    assert resolved["training_policy"] == "auto_early_stopping"
    assert resolved["hgb"]["max_iter"] == 180
    assert resolved["hgb"]["early_stopping"] == "auto"
    assert frozen == before and "spec_version" not in frozen


def test_changed_strategy_cannot_reinterpret_legacy_defaults(settings, monkeypatch):
    from power_forecast_service.experiments import contracts

    changed = model_spec.training_parameters()
    changed["early_stopping"] = False
    monkeypatch.setattr(contracts, "training_parameters", lambda policy: changed)
    with pytest.raises(TaskConflict, match="legacy_training_policy_unsupported"):
        validate_spec(historical_spec(settings.data_path), settings)


@pytest.mark.parametrize("change", [
    {"spec_version": "future"}, {"training_policy": "unknown"}, {"unexpected": 1},
    {"random_seed": 7}, {"input_sha256": "0" * 64},
])
def test_invalid_contract_is_rejected(settings, change):
    frozen = freeze_spec(ExperimentRequest(), settings)
    with pytest.raises(TaskConflict):
        validate_spec({**frozen, **change}, settings)


@pytest.mark.parametrize("mutation", ["missing_version", "missing_parameter", "wrong_type", "mismatch"])
def test_missing_version_is_not_a_legacy_escape_hatch(settings, mutation):
    frozen = freeze_spec(ExperimentRequest(training_policy="fixed_iterations"), settings)
    if mutation == "missing_version":
        frozen.pop("spec_version")
    elif mutation == "missing_parameter":
        frozen["hgb"].pop("max_iter")
    elif mutation == "wrong_type":
        frozen["hgb"]["early_stopping"] = 0
    else:
        frozen["hgb"]["early_stopping"] = "auto"
    with pytest.raises(TaskConflict):
        validate_spec(frozen, settings)


@pytest.mark.parametrize("payload", [
    {"training_policy": "unknown"}, {"hgb": {"max_iter": 1}}, {"purpose": "final_evaluation"},
])
def test_request_does_not_open_arbitrary_training(payload):
    with pytest.raises(ValidationError):
        ExperimentRequest.model_validate(payload)
