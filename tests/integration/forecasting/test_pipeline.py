"""固定样本上的真实 HGB 训练与本地推理；不要求数据库或队列。"""

from uuid import UUID

import numpy as np

from power_forecast_service.forecasting.features import build_inference_frame
from power_forecast_service.forecasting.models import improved_predict
from power_forecast_service.forecasting.pipeline import run_experiment, train_service
from power_forecast_service.forecasting.scoring import ScoringEvidence, metrics_for_rows


def test_experiment_returns_comparable_metrics(sample_path) -> None:
    result = run_experiment(sample_path)
    assert result["horizon_minutes"] == 60
    assert result["feature_contract_version"] == "lag-only-source-time-v0.1"
    assert result["split_version"] == "temporal-70-15-15-label-isolated-v0.2"
    assert result["determinism"]["random_seed"] == 42
    assert result["split"]["test"] > 0
    assert result["purpose"] == "development"
    assert result["evaluation_split"] == "validation"
    assert result["split"]["test_scored"] is False
    assert result["metrics"]["persistence"]["samples"] == result["split"]["validation"]
    assert result["metrics"]["hist_gradient_boosting"]["samples"] == result["split"]["validation"]
    assert UUID(result["run_id"]).version == 4
    assert result["split"]["removed_for_label_isolation"] == {"train": 12, "validation": 12}
    assert result["split"]["train_last_target"] < result["split"]["validation_start"]
    assert result["split"]["validation_last_target"] < result["split"]["test_start"]
    scoring = ScoringEvidence.model_validate(result["scoring"])
    assert len(scoring.rows) == result["split"]["validation"]
    for model in result["model_set"]:
        measured = metrics_for_rows(scoring.rows, model)
        assert np.isclose(measured["mae"], result["metrics"][model]["mae"])
        assert np.isclose(measured["rmse"], result["metrics"][model]["rmse"])


def test_trained_model_predicts_from_history_without_future_labels(sample_path) -> None:
    frame, _, models = train_service(sample_path)
    history = frame.head(13)
    inference = build_inference_frame(history).tail(1)
    assert len(inference) == 1
    assert inference["timestamp"].iloc[0] == history["timestamp"].iloc[-1]
    assert "target_power" not in inference
    prediction = improved_predict(models, inference)
    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()
    assert prediction[0] >= 0
