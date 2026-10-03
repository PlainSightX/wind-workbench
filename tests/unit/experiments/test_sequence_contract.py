"""新增序列/正式协议不能破坏历史身份，也不能提前开放测试集。"""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from power_forecast_service.experiments import contracts
from power_forecast_service.forecasting.sequence_protocol import CONFIG


def test_sequence_freeze_roundtrip_and_tampering(sample_path):
    config = SimpleNamespace(data_path=sample_path)
    request = contracts.ExperimentRequest(sequence_key="transformer_delta", training_policy="fixed_iterations")
    frozen = contracts.freeze_spec(request, config)
    assert contracts.validate_spec(frozen, config) == frozen
    assert len(frozen["model_set"]) == 4
    changed = deepcopy(frozen)
    changed["sequence_recipe"]["epochs"] = 1
    with pytest.raises(contracts.TaskConflict):
        contracts.validate_spec(changed, config)


def test_final_does_not_allow_early_or_arbitrary_selection(sample_path, monkeypatch):
    def unavailable():
        raise ValueError("final_protocol_not_frozen")
    monkeypatch.setattr(contracts, "frozen_selection", unavailable)
    request = contracts.ExperimentRequest(purpose="final_evaluation", training_policy="fixed_iterations")
    with pytest.raises(contracts.TaskConflict, match="not_frozen"):
        contracts.freeze_spec(request, SimpleNamespace(data_path=sample_path))
    with pytest.raises(ValueError):
        contracts.ExperimentRequest(purpose="final_evaluation", sequence_key="transformer_direct", training_policy="fixed_iterations")


def test_final_roundtrip_is_bound_to_frozen_selection(sample_path, monkeypatch):
    # 合同测试不计算正式样本；合成记录不会写入应用内的正式选择文件。
    selection = {"sequence_key": "transformer_delta", "config": CONFIG, "fixture_only": True}
    monkeypatch.setattr(contracts, "frozen_selection", lambda: (selection, "a" * 64))
    settings = SimpleNamespace(data_path=sample_path)
    frozen = contracts.freeze_spec(contracts.ExperimentRequest(purpose="final_evaluation", training_policy="fixed_iterations"), settings)
    assert contracts.validate_spec(frozen, settings) == frozen
    frozen["final_protocol_id"] = "b" * 64
    with pytest.raises(contracts.TaskConflict):
        contracts.validate_spec(frozen, settings)
