"""新候选的完整历史推理入口；独立版本，不修改已经交付的v1预测代码。"""

from datetime import timedelta
from uuid import UUID

import mlflow.pyfunc
import pandas as pd

from .candidate_models import candidate_predict
from .contracts import ForecastRequest
from .features import build_inference_frame


class CandidateHistoryPredictor(mlflow.pyfunc.PythonModel):
    def __init__(self, estimator, *, horizon_minutes=60):
        self.estimator = estimator
        self.horizon_minutes = horizon_minutes

    def predict(self, context, model_input, params=None):
        history = ForecastRequest(artifact_id=UUID(int=0), observations=model_input.to_dict("records"))
        frame = pd.DataFrame([row.model_dump() for row in history.observations])
        features = build_inference_frame(frame).tail(1)
        value = float(candidate_predict(self.estimator, features)[0])
        cutoff = history.observations[-1].timestamp
        return pd.DataFrame([{"cutoff": cutoff,
                              "target_time": cutoff + timedelta(minutes=self.horizon_minutes),
                              "prediction": value}])
