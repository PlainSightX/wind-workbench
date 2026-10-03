"""离线开发实验：输入检查、时间隔离、真实训练与验证集评分的显式主流程。"""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from ..storage.artifacts import sha256_file
from .data import load_wind_frame
from .features import FEATURE_COLUMNS, build_supervised_frame
from .models import ForecastModels, fit_models, improved_predict, persistence_predict
from .candidate_models import candidate_predict, fit_candidate
from .development_protocol import INPUT_SHA256, PROTOCOL_VERSION, TEST_START
from .fixed_evaluation import fixed_window_split
from .scoring import ScoreRow, ScoringEvidence, sample_fingerprint
from .spec import (
    DEFAULT_TRAINING_POLICY, FEATURE_CONTRACT_VERSION, HORIZON_STEPS, MODEL_SEED,
    MODEL_VERSION, SPLIT_VERSION, TrainingPolicy,
)


def data_version_for_path(path: Path) -> str:
    """为本地输入生成可读版本名；正式数据版本仍由来源卡负责。"""
    if path.name == "wind_2019_q1.csv":
        return "mendeley-fdfftr3tc2-v1-wind-2019-q1-sample"
    return f"local-{path.stem}"


def temporal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """保持评分起点，排除答案时间已进入下一段的前段样本。

    cutoff 排在前面，不代表它的一小时后标签也排在前面。按真实时间过滤，
    不硬编码删除 12 行；更改预测跨度后仍使用同一隔离规则。
    """
    if len(frame) < 100:
        raise ValueError("At least 100 supervised rows are required")
    times = frame[["timestamp", "target_timestamp"]]
    if times.isna().any().any():
        raise ValueError("Split timestamps must not be missing")
    if not frame["timestamp"].is_monotonic_increasing or frame["timestamp"].duplicated().any():
        raise ValueError("Split cutoffs must be strictly increasing")
    if not (frame["target_timestamp"] > frame["timestamp"]).all():
        raise ValueError("Every target must be later than its cutoff")
    train_end = int(len(frame) * 0.70)
    validation_end = int(len(frame) * 0.85)
    train = frame.iloc[:train_end]
    validation = frame.iloc[train_end:validation_end]
    test = frame.iloc[validation_end:]
    train = train.loc[train["target_timestamp"] < validation["timestamp"].iloc[0]]
    validation = validation.loc[validation["target_timestamp"] < test["timestamp"].iloc[0]]
    if train.empty or validation.empty:
        raise ValueError("Label isolation leaves an empty training or validation split")
    return train.copy(), validation.copy(), test.copy()


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    error = actual - prediction
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "samples": int(len(actual)),
    }


@dataclass
class ExperimentProduct:
    """结果和同次拟合对象一起交给worker；打包不能再训练一个替代模型。"""

    result: dict
    models: ForecastModels
    frame: pd.DataFrame


def train_experiment(
    path: Path, *, horizon_steps: int = HORIZON_STEPS,
    training_policy: TrainingPolicy = DEFAULT_TRAINING_POLICY,
    model_parameters: dict | None = None, random_seed: int = MODEL_SEED,
    candidate_key: str = "none",
) -> ExperimentProduct:
    """运行开发实验；只评分验证段，正式测试需后续冻结协议入口。"""
    frame, quality = load_wind_frame(path, strict=True)
    supervised = build_supervised_frame(frame, horizon_steps=horizon_steps)
    train, validation, test = temporal_split(supervised)
    if candidate_key != "none":
        if (sha256_file(path) != INPUT_SHA256 or training_policy != "fixed_iterations"
                or test["timestamp"].iloc[0] != pd.Timestamp(TEST_START)):
            raise ValueError("candidate_evaluation_protocol_mismatch")
        train, validation = fixed_window_split(supervised, "main")
    models = fit_models(
        train, random_state=random_seed, training_policy=training_policy,
        model_parameters=model_parameters,
    )

    actual = validation["target_power"].to_numpy(dtype=float)
    baseline = persistence_predict(validation)
    improved = improved_predict(models, validation)
    predictions = {"persistence": baseline, "hist_gradient_boosting": improved}
    if candidate_key != "none":
        estimator, details = fit_candidate(train, candidate_key, random_state=random_seed)
        models.candidates[candidate_key] = estimator
        models.candidate_details[candidate_key] = details
        predictions[candidate_key] = candidate_predict(estimator, validation)
    split_version = SPLIT_VERSION if candidate_key == "none" else PROTOCOL_VERSION + ":main"
    # 留存同一批 cutoff 的两模型预测，后续比较无需重新训练或猜测样本身份。
    rows = [
        ScoreRow(
            cutoff=cutoff.to_pydatetime(), target_time=target.to_pydatetime(), actual=value,
            predictions={key: values[index] for key, values in predictions.items()},
        )
        for index, (cutoff, target, value) in enumerate(
            zip(validation["timestamp"], validation["target_timestamp"], actual, strict=True)
        )
    ]
    input_sha256 = sha256_file(path)
    scoring = ScoringEvidence(
        version="scoring-v1", target="wind_power_single_point", unit="source_reported_unit",
        clock="source_time_timezone_unknown", evaluation_split="validation",
        metric_version="unweighted-mae-rmse-v1",
        input_sha256=input_sha256, horizon_minutes=horizon_steps * 5,
        split_version=split_version, samples_sha256=sample_fingerprint(rows), rows=rows,
    )
    result = {
        "run_id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "development",
        "evaluation_split": "validation",
        "model_set": list(predictions),
        "data_version": data_version_for_path(path),
        "input_file": str(path),
        "input_file_sha256": input_sha256,
        "input_file_bytes": path.stat().st_size,
        "model_version": MODEL_VERSION,
        "model_versions": {key: ("persistence-v1" if key == "persistence" else MODEL_VERSION
                                  if key == "hist_gradient_boosting" else key + "-v1")
                           for key in predictions},
        "training": {
            **models.training_details,
            "train_cutoff_start": train["timestamp"].iloc[0].isoformat(),
            "train_cutoff_end": train["timestamp"].iloc[-1].isoformat(),
            "train_target_start": train["target_timestamp"].iloc[0].isoformat(),
            "train_target_end": train["target_timestamp"].iloc[-1].isoformat(),
        },
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "split_version": split_version,
        "determinism": {
            "random_seed": random_seed,
            "note": "HistGradientBoosting is fit with a fixed seed; hardware/runtime drift is not benchmarked.",
        },
        "metric_definition": {
            "mae": "mean(abs(actual - prediction))",
            "rmse": "sqrt(mean((actual - prediction)^2))",
        },
        "horizon_steps": horizon_steps,
        "horizon_minutes": horizon_steps * 5,
        "feature_columns": list(FEATURE_COLUMNS),
        "quality": quality.as_dict(),
        "split": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "removed_for_label_isolation": {
                "train": int(len(supervised) * 0.70) - len(train),
                "validation": (
                    int(len(supervised) * 0.85) - int(len(supervised) * 0.70) - len(validation)
                ),
            },
            "train_last_target": train["target_timestamp"].iloc[-1].isoformat(),
            "validation_start": validation["timestamp"].iloc[0].isoformat(),
            "validation_last_target": validation["target_timestamp"].iloc[-1].isoformat(),
            "test_start": test["timestamp"].iloc[0].isoformat(),
            "test_end": test["target_timestamp"].iloc[-1].isoformat(),
            "test_scored": False,
        },
        "metrics": {key: regression_metrics(actual, values) for key, values in predictions.items()},
        "scoring": scoring.model_dump(mode="json"),
    }
    if candidate_key != "none":
        result["candidate_training"] = models.candidate_details
    return ExperimentProduct(result=result, models=models, frame=frame)


def run_experiment(path: Path, **kwargs) -> dict:
    """保留CLI/离线结果调用合同；worker改用包含同次拟合对象的入口。"""
    return train_experiment(path, **kwargs).result


def train_service(path: Path, *, horizon_steps: int = HORIZON_STEPS):
    """本地模型探针复用的训练入口；HTTP 应用不得在启动或查询时调用。"""
    frame, quality = load_wind_frame(path, strict=True)
    supervised = build_supervised_frame(frame, horizon_steps=horizon_steps)
    train, _, _ = temporal_split(supervised)
    models = fit_models(train, random_state=MODEL_SEED)
    return frame, quality, models
