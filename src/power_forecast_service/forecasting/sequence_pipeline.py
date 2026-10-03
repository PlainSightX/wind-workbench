"""序列开发或已冻结的正式评价；与旧管线共用数据/基线/结果类型。"""

from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from ..storage.artifacts import sha256_file
from .development_protocol import TEST_START
from .fixed_evaluation import fixed_window_split
from .model_diagnosis import grouped_errors
from .pipeline import ExperimentProduct, data_version_for_path, regression_metrics
from .scoring import ScoreRow, ScoringEvidence, sample_fingerprint
from .sequence_experiment import source_data, train_baselines
from .sequence_model import SequenceRegressor
from .sequence_protocol import EVALUATION_VERSION, SEQUENCE_VERSION, SEQUENCE_FEATURES
from .spec import MODEL_VERSION


def train_sequence_experiment(path, *, sequence_key, purpose="development", final_protocol=None):
    frame, supervised, quality = source_data(path)
    if purpose == "final_evaluation":
        if not final_protocol or sequence_key != final_protocol["sequence_key"]:
            raise ValueError("final_protocol_not_frozen")
        train = supervised.loc[supervised.target_timestamp < pd.Timestamp(TEST_START)].copy()
        evaluation = supervised.loc[supervised.timestamp >= pd.Timestamp(TEST_START)].copy()
        if len(evaluation) != 3885:
            raise ValueError("final_samples_changed")
        split_version = EVALUATION_VERSION + ":final-refit-v1"
    else:
        train, evaluation = fixed_window_split(supervised, "main")
        split_version = EVALUATION_VERSION + ":main"
    models, predictions = train_baselines(train, evaluation)
    estimator = SequenceRegressor(sequence_key).fit(frame, train)
    models.candidates[sequence_key] = estimator
    models.candidate_details[sequence_key] = estimator.details
    predictions[sequence_key] = estimator.predict_frame(frame, evaluation.timestamp)
    actual = evaluation.target_power.to_numpy(float)
    rows = [ScoreRow(cutoff=cutoff.to_pydatetime(), target_time=target.to_pydatetime(), actual=value,
                     predictions={key: float(values[i]) for key, values in predictions.items()})
            for i, (cutoff, target, value) in enumerate(zip(evaluation.timestamp, evaluation.target_timestamp, actual))]
    split_name = "test" if purpose == "final_evaluation" else "validation"
    scoring = ScoringEvidence(version="scoring-v1", input_sha256=sha256_file(path),
        target="wind_power_single_point", unit="source_reported_unit", clock="source_time_timezone_unknown",
        evaluation_split=split_name, horizon_minutes=60, split_version=split_version,
        metric_version="unweighted-mae-rmse-v1", samples_sha256=sample_fingerprint(rows), rows=rows)
    result = {"run_id": str(uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
              "purpose": purpose, "evaluation_split": split_name, "model_set": list(predictions),
              "data_version": data_version_for_path(path), "input_file": str(path),
              "input_file_sha256": sha256_file(path), "input_file_bytes": path.stat().st_size,
              "model_version": MODEL_VERSION,
              "model_versions": {key: "persistence-v1" if key == "persistence" else MODEL_VERSION
                                 if key == "hist_gradient_boosting" else key + "-v1" for key in predictions},
              "training": {**models.training_details,
                           "train_cutoff_start": train.timestamp.iloc[0].isoformat(),
                           "train_cutoff_end": train.timestamp.iloc[-1].isoformat(),
                           "train_target_start": train.target_timestamp.iloc[0].isoformat(),
                           "train_target_end": train.target_timestamp.iloc[-1].isoformat()},
              "feature_contract_version": SEQUENCE_VERSION, "feature_columns": SEQUENCE_FEATURES,
              "split_version": split_version, "determinism": {"random_seed": 42},
              "horizon_steps": 12, "horizon_minutes": 60, "quality": quality.as_dict(),
              "split": {"train": len(train), "validation": len(evaluation) if split_name == "validation" else 0,
                        "test": 3885, "test_start": TEST_START, "test_scored": split_name == "test",
                        "train_last_target": train.target_timestamp.iloc[-1].isoformat(),
                        "evaluation_start": evaluation.timestamp.iloc[0].isoformat(),
                        "evaluation_last_target": evaluation.target_timestamp.iloc[-1].isoformat()},
              "metrics": {key: regression_metrics(actual, values) for key, values in predictions.items()},
              "grouped_errors": grouped_errors(train, evaluation, predictions),
              "scoring": scoring.model_dump(mode="json"), "candidate_training": models.candidate_details}
    if final_protocol:
        result["final_selection"] = final_protocol
    return ExperimentProduct(result, models, frame)
