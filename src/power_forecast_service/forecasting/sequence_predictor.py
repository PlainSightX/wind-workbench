"""序列完整包入口：历史检查后还原state_dict，不调用fit。"""

from uuid import uuid4

import mlflow.pyfunc
import pandas as pd

from .contracts import ForecastRequest
from .sequence_model import SequenceRegressor


class SequenceHistoryPredictor(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.estimator = SequenceRegressor.load(context.artifacts["checkpoint"])

    def predict(self, context, model_input, params=None):
        request = ForecastRequest(artifact_id=uuid4(), observations=model_input.to_dict("records"))
        history = pd.DataFrame([row.model_dump() for row in request.observations])
        cutoff = history.timestamp.iloc[-1]
        value = self.estimator.predict_frame(history, [cutoff])[0]
        return pd.DataFrame({"cutoff": [cutoff], "target_time": [cutoff + pd.Timedelta(hours=1)],
                             "prediction": [float(value)]})
