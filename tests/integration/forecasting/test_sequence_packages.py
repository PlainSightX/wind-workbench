"""真实torch权重和scaler跨进程重载；仅构造数据，不提前评分正式留出。"""

from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from power_forecast_service.forecasting.bundles import save_packages, verify_fresh_process, predict_package
from power_forecast_service.forecasting.contracts import ForecastRequest
from power_forecast_service.forecasting.features import build_supervised_frame
from power_forecast_service.forecasting.sequence_data import common_history_start
from power_forecast_service.forecasting.sequence_model import SequenceRegressor
from power_forecast_service.forecasting.sequence_protocol import CONFIG, SEQUENCE_VERSION
from power_forecast_service.storage.model_packages import PackageError


def test_sequence_complete_package_reloads_without_fit(tmp_path):
    t = np.arange(260)
    frame = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=260, freq="5min"),
                          "wind_power": 100 + 20 * np.sin(t / 15), "wind_speed": 6 + np.sin(t / 15),
                          "humidity": np.full(260, 60.), "temperature": np.full(260, 20.)})
    supervised = common_history_start(frame, build_supervised_frame(frame))
    train, evaluation = supervised.iloc[:180], supervised.iloc[192:]
    key = "transformer_delta"
    fitted = SequenceRegressor(key, {**CONFIG, "epochs": 2}).fit(frame, train)
    values = fitted.predict_frame(frame, evaluation.timestamp)
    rows = [{"cutoff": time.isoformat(), "target_time": (time + pd.Timedelta(hours=1)).isoformat(),
             "predictions": {key: float(value)}} for time, value in zip(evaluation.timestamp, values)]
    result = {"run_id": str(uuid4()), "purpose": "development", "scoring": {"rows": rows},
              "model_set": [key], "model_version": key + "-v1", "horizon_minutes": 60,
              "training": {"train_target_end": train.target_timestamp.iloc[-1].isoformat()},
              "frozen_spec": {"horizon_steps": 12, "sequence_key": key,
                              "sequence_recipe": fitted.config, "feature_contract_version": SEQUENCE_VERSION},
              "execution": {"scope": "synthetic_reload_only"}}
    product = SimpleNamespace(result=result, frame=frame, models=SimpleNamespace(
        candidates={key: fitted}, candidate_details={key: fitted.details}))
    packages = save_packages(product, tmp_path, uuid4(), uuid4())
    assert verify_fresh_process(tmp_path, packages)[0]["max_absolute_difference"] < 1e-3
    short = ForecastRequest(artifact_id=packages[0].artifact_id, observations=frame.iloc[:13].to_dict("records"))
    with pytest.raises(PackageError, match="history_insufficient"):
        predict_package(tmp_path, packages[0], short)
