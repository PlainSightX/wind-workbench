"""工件内完整预测函数：原始历史 -> 特征 -> 指定模型 -> 固定后处理。"""

from datetime import timedelta
from uuid import UUID

import mlflow.pyfunc
import numpy as np
import pandas as pd

from .contracts import ForecastRequest
from .features import FEATURE_COLUMNS, build_inference_frame


class HistoryPredictor(mlflow.pyfunc.PythonModel):
    def __init__(self, model_key: str, estimator=None, *, horizon_minutes: int = 60):
        self.model_key = model_key
        self.estimator = estimator
        self.horizon_minutes = horizon_minutes

    def predict(self, context, model_input, params=None):
        # MLflow签名只负责列类型，时间、允许域和历史长度仍用同一个应用合同。
        history = ForecastRequest(
            artifact_id=UUID(int=0), observations=model_input.to_dict(orient="records")
        )
        frame = pd.DataFrame([item.model_dump() for item in history.observations])
        features = build_inference_frame(frame).tail(1)
        if self.model_key == "persistence":
            value = float(features["wind_power_lag_0"].iloc[0])
        elif self.model_key == "hist_gradient_boosting":
            value = float(np.maximum(self.estimator.predict(features[FEATURE_COLUMNS]), 0.0)[0])
        else:
            raise ValueError("unsupported_model_key")
        if not np.isfinite(value):
            raise ValueError("prediction_not_finite")
        cutoff = history.observations[-1].timestamp
        return pd.DataFrame([{
            "cutoff": cutoff, "target_time": cutoff + timedelta(minutes=self.horizon_minutes),
            "prediction": value,
        }])
