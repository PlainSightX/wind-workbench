"""开发期有限对照及误差审查；复用正式数据、特征、拟合和评分，不读测试统计。"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ..storage.artifacts import execution_provenance, sha256_file
from .candidate_models import candidate_predict, fit_candidate
from .data import load_wind_frame
from .development_protocol import INPUT_SHA256, RIDGE_ALPHAS, WINDOWS, protocol_document
from .features import FEATURE_COLUMNS, build_supervised_frame
from .fixed_evaluation import fixed_window_split
from .models import fit_models, improved_predict, persistence_predict
from .pipeline import regression_metrics


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def measurements(actual, predicted):
    if len(actual) != len(predicted) or not np.isfinite(predicted).all():
        raise ValueError("missing_or_nonfinite_predictions")
    return {**regression_metrics(actual, predicted),
            "bias": float(np.mean(predicted - actual)), "coverage": 1.0}


def distribution(train, validation):
    report = {}
    for column in [*FEATURE_COLUMNS, "target_power"]:
        a, b = train[column], validation[column]
        report[column] = {
            "train": {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean())},
            "validation": {"min": float(b.min()), "max": float(b.max()), "mean": float(b.mean())},
            "below_train_fraction": float((b < a.min()).mean()),
            "above_train_fraction": float((b > a.max()).mean()),
        }
    return report


def grouped_errors(train, evaluation, predictions):
    variables = {
        "power": (train.target_power.to_numpy(), evaluation.target_power.to_numpy(), [1 / 3, 2 / 3]),
        "change": (np.abs(train.target_power - train.wind_power_lag_0).to_numpy(),
                   np.abs(evaluation.target_power - evaluation.wind_power_lag_0).to_numpy(), [0.5, 0.9]),
    }
    groups = {}
    for name, (training, values, quantiles) in variables.items():
        thresholds = np.quantile(training, quantiles)
        membership = np.searchsorted(thresholds, values, side="right")
        entries = []
        for index in range(3):
            mask = membership == index
            entries.append({"group": index, "samples": int(mask.sum()), "metrics": {
                key: measurements(evaluation.target_power.to_numpy()[mask], pred[mask])
                for key, pred in predictions.items()} if mask.any() else {}})
        groups[name] = {"train_quantiles": quantiles, "thresholds": thresholds.tolist(),
                        "groups": entries}
    return groups


def _record_model(key, train, evaluation, predict, details):
    started = perf_counter()
    output = predict(evaluation)
    elapsed = perf_counter() - started
    report = {"status": "succeeded", "training": details,
              "validation": measurements(evaluation.target_power.to_numpy(), output),
              "train": measurements(train.target_power.to_numpy(), predict(train)),
              "batch_inference_seconds": elapsed}
    return output, report


def run_diagnosis(path: Path, output: Path, *, phase: str):
    """baseline先落盘；delta必须已有独立决定和hash锚点，不覆写首次结果。"""
    if sha256_file(path) != INPUT_SHA256:
        raise ValueError("diagnosis_input_changed")
    output.mkdir(parents=True, exist_ok=True)
    protocol = protocol_document()
    if phase == "baseline":
        write_json(output / "protocol.json", protocol)
    else:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("diagnosis_protocol_changed")
        decision = json.loads((output / "delta-decision.json").read_text(encoding="utf-8"))
        if (decision.get("baseline_sha256") != sha256_file(output / "baseline.json")
                or decision.get("execute") is not True or not decision.get("rationale")):
            raise ValueError("delta_requires_prior_evidence_decision")
    frame, _ = load_wind_frame(path, strict=True)
    supervised = build_supervised_frame(frame)
    report = {"phase": phase, "protocol": protocol, "execution": execution_provenance(),
              "test_scored": False, "windows": {}}
    for window in WINDOWS:
        train, evaluation = fixed_window_split(supervised, window)
        predictions, models = {}, {}
        keys = ["persistence", "hist_gradient_boosting", *RIDGE_ALPHAS] if phase == "baseline" else ["hgb_delta"]
        for key in keys:
            try:
                if key == "persistence":
                    predict, details = persistence_predict, {"fit_elapsed_seconds": 0.0, "fitted": False}
                elif key == "hist_gradient_boosting":
                    fitted = fit_models(train, training_policy="fixed_iterations")
                    predict = lambda values, fitted=fitted: improved_predict(fitted, values)
                    details = fitted.training_details
                else:
                    fitted, details = fit_candidate(train, key)
                    predict = lambda values, fitted=fitted: candidate_predict(fitted, values)
                predictions[key], models[key] = _record_model(key, train, evaluation, predict, details)
            except Exception as exc:
                # 不删失败窗口、不用另一个样本集补出好分数；仅该候选失去选择资格。
                models[key] = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
        csv = evaluation[["timestamp", "target_timestamp", "target_power", "wind_power_lag_0"]].copy()
        for key, values in predictions.items():
            csv[key] = values
        destination = output / f"{phase}-{window}.csv"
        csv.to_csv(destination, index=False, mode="x")
        report["windows"][window] = {
            "train_samples": len(train), "validation_samples": len(evaluation),
            "train_label_end": train.target_timestamp.iloc[-1].isoformat(),
            "validation_first_cutoff": evaluation.timestamp.iloc[0].isoformat(),
            "distribution": distribution(train, evaluation), "models": models,
            "grouped_errors": grouped_errors(train, evaluation, predictions),
            "predictions_file": destination.name, "predictions_sha256": sha256_file(destination),
        }
        print(f"{phase} {window}: " + json.dumps({k: v.get('validation', {}).get('mae')
                                                 for k, v in models.items()}), flush=True)
    write_json(output / f"{phase}.json", report)
    return report


def summarize(output: Path):
    baseline = json.loads((output / "baseline.json").read_text(encoding="utf-8"))
    reports = [baseline]
    if (output / "delta.json").exists():
        reports.append(json.loads((output / "delta.json").read_text(encoding="utf-8")))
    merged = {}
    for report in reports:
        if report["protocol"] != protocol_document() or set(report["windows"]) != set(WINDOWS):
            raise ValueError("diagnosis_incomplete_or_changed_protocol")
        for window, record in report["windows"].items():
            path = output / record["predictions_file"]
            if sha256_file(path) != record["predictions_sha256"]:
                raise ValueError("diagnosis_predictions_changed")
            frame = pd.read_csv(path)
            expected = pd.date_range(*WINDOWS[window], freq="5min")
            if (not pd.to_datetime(frame.timestamp).equals(pd.Series(expected))
                    or frame.isna().any().any()):
                raise ValueError("diagnosis_scoring_samples_changed")
            if window in merged:
                identity = ["timestamp", "target_timestamp", "target_power", "wind_power_lag_0"]
                pd.testing.assert_frame_equal(merged[window][identity], frame[identity])
                merged[window] = pd.concat([merged[window], frame.drop(columns=identity)], axis=1)
            else:
                merged[window] = frame
    keys = list(dict.fromkeys(k for r in reports for w in r["windows"].values() for k in w["models"]))
    summary = {}
    pooled = pd.concat(list(merged.values()), ignore_index=True)
    for key in keys:
        records = {window: record["models"][key] for report in reports
                   for window, record in report["windows"].items() if key in record["models"]}
        if (set(records) != set(WINDOWS) or any(r["status"] != "succeeded" for r in records.values())
                or any(key not in frame for frame in merged.values())):
            summary[key] = {"status": "failed_in_at_least_one_window", "eligible": False}
            continue
        per_window = {window: measurements(frame.target_power.to_numpy(), frame[key].to_numpy())
                      for window, frame in merged.items()}
        for window, stats in per_window.items():
            base = measurements(merged[window].target_power.to_numpy(), merged[window].persistence.to_numpy())["mae"]
            stats["skill"] = 1 - stats["mae"] / base if base else None
        mean_mae = float(np.mean([item["mae"] for item in per_window.values()]))
        base_mean = float(np.mean([measurements(f.target_power.to_numpy(), f.persistence.to_numpy())["mae"]
                                   for f in merged.values()]))
        summary[key] = {"eligible": True, "mean_window_mae": mean_mae,
                        "mean_mae_skill": 1 - mean_mae / base_mean if base_mean else None,
                        "windows": per_window,
                        "pooled": measurements(pooled.target_power.to_numpy(), pooled[key].to_numpy()),
                        "costs": {window: {"fit_seconds": r["training"]["fit_elapsed_seconds"],
                                           "batch_inference_seconds": r["batch_inference_seconds"]}
                                  for window, r in records.items()}}
    ranked = sorted((key for key in keys if summary[key]["eligible"]),
                    key=lambda key: (summary[key]["mean_window_mae"], keys.index(key)))
    chosen = ranked[0]
    delivery = next((key for key in ranked if key in (*RIDGE_ALPHAS, "hgb_delta")), None)
    # 案例由共同主窗和固定准则选取；正负效果均保留，不暗挑有利日期。
    cases = []
    if delivery:
        main = merged["main"]
        gain = (main.persistence - main.target_power).abs() - (main[delivery] - main.target_power).abs()
        for label, index in [("largest_gain", gain.idxmax()), ("largest_regression", gain.idxmin())]:
            cases.append({"selection_rule": label, "candidate": delivery,
                          "error_reduction": float(gain.loc[index]), **main.loc[index].to_dict()})
    value = {"protocol": baseline["protocol"], "models": summary, "ranking": ranked,
             "selected_by_primary_metric": chosen, "delivery_candidate": delivery, "cases": cases,
             "test_scored": False, "source_reports": {r["phase"]: sha256_file(output / (r["phase"] + ".json"))
                                                        for r in reports}}
    write_json(output / "summary.json", value)
    return value
