"""可按窗口恢复的序列开发实验；正式测试由另一个冻结入口负责。"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ..storage.artifacts import execution_provenance, sha256_file
from .candidate_models import candidate_predict, fit_candidate
from .data import load_wind_frame
from .development_protocol import INPUT_SHA256, WINDOWS
from .features import build_supervised_frame
from .fixed_evaluation import fixed_window_split
from .model_diagnosis import grouped_errors, measurements, write_json
from .models import fit_models, improved_predict, persistence_predict
from .sequence_data import common_history_start, history_windows
from .sequence_model import SequenceRegressor
from .sequence_protocol import CONFIG, SEQUENCE_KEYS, sequence_protocol


def source_data(path):
    if sha256_file(path) != INPUT_SHA256:
        raise ValueError("sequence_input_changed")
    frame, quality = load_wind_frame(path, strict=True)
    return frame, common_history_start(frame, build_supervised_frame(frame)), quality


def train_baselines(train, evaluation):
    hgb = fit_models(train, training_policy="fixed_iterations")
    ridge, detail = fit_candidate(train, "ridge_0_1")
    hgb.candidates["ridge_0_1"], hgb.candidate_details["ridge_0_1"] = ridge, detail
    return hgb, {"persistence": persistence_predict(evaluation),
                 "hist_gradient_boosting": improved_predict(hgb, evaluation),
                 "ridge_0_1": candidate_predict(ridge, evaluation)}


def run_development(path, output, checkpoints, *, key, window, seed=42, device="cpu"):
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    stem = f"{key}-{seed}-{window}"
    destination = output / (stem + ".json")
    if destination.exists():
        raise FileExistsError("Completed experiment cannot be overwritten")
    frame, supervised, _ = source_data(path)
    train, evaluation = fixed_window_split(supervised, window)
    estimator = SequenceRegressor(key).fit(frame, train, seed=seed, device=device)
    started = perf_counter()
    prediction = estimator.predict_frame(frame, evaluation.timestamp)
    inference_seconds = perf_counter() - started
    checkpoint = checkpoints / (stem + ".pt")
    if checkpoint.exists():
        raise FileExistsError("Retain incomplete checkpoint and choose a new output for recovery")
    estimator.save(checkpoint)
    restored = SequenceRegressor.load(checkpoint).predict_frame(frame, evaluation.timestamp)
    np.testing.assert_allclose(restored, prediction, rtol=1e-6, atol=1e-3)
    baselines, predictions = train_baselines(train, evaluation)
    predictions[key] = prediction
    scores = evaluation[["timestamp", "target_timestamp", "target_power", "wind_power_lag_0"]].copy()
    for name, values in predictions.items():
        scores[name] = values
    csv_path = output / (stem + ".csv")
    scores.to_csv(csv_path, index=False, mode="x")
    report = {"protocol": sequence_protocol(), "key": key, "window": window, "seed": seed,
              "test_scored": False, "execution": execution_provenance(),
              "train_samples": len(train), "train_target_end": train.target_timestamp.iloc[-1].isoformat(),
              "training": estimator.details, "batch_inference_seconds": inference_seconds,
              "checkpoint_sha256": sha256_file(checkpoint), "prediction_file": csv_path.name,
              "prediction_sha256": sha256_file(csv_path),
              "reload_max_difference": float(np.max(np.abs(restored - prediction))),
              "metrics": {name: measurements(evaluation.target_power.to_numpy(), values)
                          for name, values in predictions.items()},
              "groups": grouped_errors(train, evaluation, predictions),
              "baseline_training": {"hist_gradient_boosting": baselines.training_details,
                                    "ridge_0_1": baselines.candidate_details["ridge_0_1"]}}
    write_json(destination, report)
    print(json.dumps({"completed": stem, "metrics": report["metrics"]}), flush=True)
    return report


def select_development(output):
    reports = {}
    for key in SEQUENCE_KEYS:
        reports[key] = {}
        for window in WINDOWS:
            path = output / f"{key}-42-{window}.json"
            data = json.loads(path.read_text())
            if data["protocol"] != json.loads(json.dumps(sequence_protocol())):
                raise ValueError("sequence_protocol_changed")
            if sha256_file(output / data["prediction_file"]) != data["prediction_sha256"]:
                raise ValueError("sequence_prediction_changed")
            reports[key][window] = data
    means = {name: float(np.mean([r["metrics"][name]["mae"] for r in records.values()]))
             for key, records in reports.items() for name in [key]}
    first = reports[SEQUENCE_KEYS[0]]
    for name in ("persistence", "hist_gradient_boosting", "ridge_0_1"):
        means[name] = float(np.mean([r["metrics"][name]["mae"] for r in first.values()]))
        # 两配置必须仍然评分同一基线；不能在后一次换了数据/评价才声称改进。
        for window in WINDOWS:
            if first[window]["metrics"][name] != reports[SEQUENCE_KEYS[1]][window]["metrics"][name]:
                raise ValueError("baseline_changed_between_configurations")
    selected_sequence = min(SEQUENCE_KEYS, key=lambda name: means[name])
    chosen = min(means, key=means.get)
    return {"protocol": sequence_protocol(), "mean_mae": means,
            "selected_sequence": selected_sequence, "selected_model": chosen,
            "needs_seed_confirmation": chosen in SEQUENCE_KEYS,
            "test_scored": False,
            "reports": {path.name: sha256_file(path) for path in sorted(output.glob("*-42-*.json"))}}
