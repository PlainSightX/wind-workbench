"""显式离线 ENGIE 准入、冻结和基线运行；不连接数据库或读取2015评价标签。"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from power_forecast_service.forecasting.engie_contract import (
    ARRIVAL_LAG, ARCHIVE_SHA256, DEVELOPMENT_START, FEATURES, HOLDOUT_START,
    HORIZONS, LOOKBACK, MEMBER_SHA256, QUARTERS, ROSTER, STEP, VERSION,
    admit, before_training_boundary, load_source, make_batch,
)
from power_forecast_service.forecasting.engie_baselines import (
    FittedBaseline, RIDGE_ALPHAS, TREE_PARAMETERS, fit_lightgbm, fit_ridge,
    grouped_scores, score_curves,
)

ROOT = Path(__file__).resolve().parents[2]
CODE_FILES = ["src/power_forecast_service/forecasting/engie_contract.py",
              "src/power_forecast_service/forecasting/engie_baselines.py",
              "tools/diagnostics/run_engie_baseline.py"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_record(path):
    return {"path": path.resolve().relative_to(ROOT).as_posix(),
            "sha256": sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def protocol():
    return {
        "version": VERSION, "source_archive_sha256": ARCHIVE_SHA256,
        "scada_sha256": MEMBER_SHA256, "roster": list(ROSTER), "features": list(FEATURES),
        "lookback_records": LOOKBACK, "arrival_lag_minutes": 20,
        "target_minutes_after_issue": list(HORIZONS),
        "quarters": {name: list(bounds) for name, bounds in QUARTERS.items()},
        "inner_validation_days": 28, "label_purge": "last_target_plus_20min_strictly_before_boundary",
        "ridge_alphas": list(RIDGE_ALPHAS), "lightgbm": TREE_PARAMETERS,
        "refit": "all_complete_samples_with_labels_available_before_outer_start",
        "holdout": "2015_unscored_and_not_used_for_selection",
        "quarter_boundary": "targets_may_cross_development_quarters_but_never_2015",
        "ranking": "complete_coverage_only_equal_quarter_mean_farm_six_horizon_raw_mae",
        "units": "raw_kw_no_clipping_or_capacity_normalization", "seed": 42,
        "code": [file_record(ROOT / path) for path in CODE_FILES],
        "environment": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "joblib")},
        "uv_lock_sha256": sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
    }


def load_development(source_path):
    source = load_source(source_path)
    issues = pd.date_range(DEVELOPMENT_START, HOLDOUT_START, freq=STEP, inclusive="left")
    return source, make_batch(source, issues)


def split_masks(batch, start):
    validation_start = start - pd.Timedelta(days=28)
    inner_train = before_training_boundary(batch, validation_start)
    refit = before_training_boundary(batch, start)
    validation = refit & (batch.issues >= validation_start)
    return inner_train, validation, refit, validation_start


def fit_or_resume(family, window, batch, masks, output, artifacts, freeze_hash):
    receipt_path = output / f"{window}-{family}-fit.json"
    artifact = artifacts / f"{window}-{family}.joblib"
    started_path = output / f"{window}-{family}-started.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["freeze_sha256"] != freeze_hash or receipt["family"] != family or receipt["window"] != window:
            raise ValueError("engie_resume_frozen_protocol_mismatch")
        path = ROOT / receipt["artifact"]["path"]
        if path.resolve() != artifact.resolve() or file_record(path) != receipt["artifact"]:
            raise ValueError("engie_resume_artifact_mismatch")
        model = joblib.load(path)
        if model.family != family:
            raise ValueError("engie_resume_model_family_mismatch")
        return model, receipt
    if artifact.exists() or started_path.exists():
        raise ValueError("engie_unfinished_attempt_requires_review_before_refit")
    train, validation, refit = masks
    if not all(mask.any() for mask in masks):
        raise ValueError("engie_empty_training_partition")
    started = {"family": family, "window": window, "freeze_sha256": freeze_hash,
               "started_at": datetime.now(timezone.utc).isoformat()}
    write_json(started_path, started)
    x, y = batch.features, batch.targets
    if family == "persistence":
        model, detail = FittedBaseline(family, []), {"fit_seconds": 0.0, "estimator_fit_count": 0}
    else:
        fit = fit_ridge if family == "ridge" else fit_lightgbm
        model, detail = fit(x[train], y[train], x[validation], y[validation], x[refit], y[refit])
    joblib.dump(model, artifact)
    probe = x[validation]
    before = model.predict(probe)
    reloaded = joblib.load(artifact).predict(probe)
    np.testing.assert_allclose(before, reloaded, rtol=1e-12, atol=1e-8)
    receipt = {**started,
               "finished_at": datetime.now(timezone.utc).isoformat(), "artifact": file_record(artifact),
               "details": detail, "reload_max_difference": float(np.max(np.abs(before - reloaded)))}
    write_json(receipt_path, receipt)
    print(json.dumps({"completed_fit": f"{window}/{family}", "seconds": detail["fit_seconds"]}), flush=True)
    return model, receipt


def completed_window(output, window, freeze_hash):
    """完成回执是恢复边界；读取时先核对关联工件，不重新生成或改写耗时。"""
    path = output / f"{window}-result.json"
    if not path.exists():
        return None
    value = read_json(path)
    if value["freeze_sha256"] != freeze_hash:
        raise ValueError("engie_completed_window_freeze_mismatch")
    if file_record(ROOT / value["predictions"]["path"]) != value["predictions"]:
        raise ValueError("engie_completed_prediction_hash_mismatch")
    for family, receipt in value["fits"].items():
        if receipt != read_json(output / f"{window}-{family}-fit.json"):
            raise ValueError("engie_completed_fit_receipt_mismatch")
        if receipt["family"] != family or receipt["window"] != window or receipt["freeze_sha256"] != freeze_hash:
            raise ValueError("engie_completed_fit_identity_mismatch")
        if file_record(ROOT / receipt["artifact"]["path"]) != receipt["artifact"]:
            raise ValueError("engie_completed_model_hash_mismatch")
    return value


def run(args, source, batch):
    frozen = read_json(args.output / "protocol.json")
    if frozen["protocol"] != protocol():
        raise ValueError("engie_code_or_contract_changed_after_freeze")
    if read_json(args.output / "admission.json")["status"] != "admitted":
        raise ValueError("engie_data_not_admitted")
    if (args.output / "baseline.json").exists():
        raise ValueError("engie_completed_result_exists_use_verifier_not_refit")
    freeze_hash = sha256((args.output / "protocol.json").read_bytes()).hexdigest()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    summaries = {}
    cases = read_json(args.output / "cases.json") if (args.output / "cases.json").exists() else {}
    for window, (start, end) in QUARTERS.items():
        existing = completed_window(args.output, window, freeze_hash)
        if existing is not None:
            if not cases:
                raise ValueError("engie_completed_window_cases_missing")
            summaries[window] = existing
            print(json.dumps({"resumed_completed_window": window}), flush=True)
            continue
        archive = args.artifacts / f"{window}-predictions.npz"
        if archive.exists():
            raise ValueError("engie_orphan_predictions_require_review_before_overwrite")
        start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        outer = (batch.issues >= start) & (batch.issues < end)
        train, validation, refit, val_start = split_masks(batch, start)
        counts = batch.counts(outer)
        models, receipts = {}, {}
        for family in ("persistence", "ridge", "lightgbm"):
            models[family], receipts[family] = fit_or_resume(
                family, window, batch, (train, validation, refit), args.output, args.artifacts, freeze_hash)
        # 所有选择/重拟合完成后才评分外层；输入合法但无标签的轮次仍实际产生预测。
        x, y = batch.features[outer], batch.targets[outer]
        input_valid, scoreable = batch.input_valid[outer], batch.scoreable[outer]
        predictions, costs = {}, {}
        for family, model in models.items():
            predicted = np.full_like(y, np.nan)
            clock = perf_counter()
            predicted[input_valid] = model.predict(x[input_valid])
            costs[family] = {"batch_prediction_seconds": perf_counter() - clock,
                             "input_valid_issues": int(input_valid.sum())}
            predictions[family] = predicted
        scored_predictions = {family: value[scoreable] for family, value in predictions.items()}
        metrics = {family: score_curves(y[scoreable], value) for family, value in scored_predictions.items()}
        arrays = {"issue_ns": batch.issues[outer].asi8, "targets": y,
                  "input_valid": input_valid, "label_valid": batch.label_valid[outer],
                  "boundary_valid": batch.boundary_valid[outer], "scoreable": scoreable,
                  **{f"prediction_{family}": value for family, value in predictions.items()}}
        np.savez_compressed(archive, **arrays)
        summaries[window] = {
            "freeze_sha256": freeze_hash,
            "counts": counts, "inner_train_issues": int(train.sum()),
            "inner_validation_issues": int(validation.sum()), "refit_issues": int(refit.sum()),
            "validation_start": val_start.isoformat(), "outer_start": start.isoformat(),
            "last_inner_training_label_available": (batch.issues[train][-1] + pd.Timedelta(minutes=80)).isoformat(),
            "last_refit_label_available": (batch.issues[refit][-1] + pd.Timedelta(minutes=80)).isoformat(),
            "metrics": metrics, "prediction_cost": costs,
            "actual_outputs": {family: int(np.isfinite(value).all(axis=(1, 2)).sum())
                               for family, value in predictions.items()},
            "groups": grouped_scores(batch.targets[refit], batch.features[refit], y[scoreable],
                                     x[scoreable], scored_predictions),
            "fits": receipts, "predictions": file_record(archive),
        }
        if not cases:
            index = int(np.flatnonzero(input_valid)[0])
            issue = batch.issues[outer][index]
            # 删除固定机组的一条最新记录，检验拒绝而非悄悄改为三台合计。
            damaged = source.values.copy()
            row = source.times.get_loc(issue - ARRIVAL_LAG)
            damaged[row, 0, :] = np.nan
            rejected = make_batch(replace(source, values=damaged), pd.DatetimeIndex([issue]))
            if rejected.input_valid[0]:
                raise ValueError("engie_missing_turbine_case_was_not_rejected")
            cases = {"normal": {"issue_time": issue.isoformat(),
                                "source_cutoff": (issue - ARRIVAL_LAG).isoformat(),
                                "valid_times": [(issue + pd.Timedelta(minutes=h)).isoformat() for h in HORIZONS],
                                "roster": list(ROSTER), "unit": "kW",
                                "predictions": {family: value[index].tolist() for family, value in predictions.items()},
                                "farm_predictions": {family: value[index].sum(axis=0).tolist() for family, value in predictions.items()}},
                     "missing_turbine": {"issue_time": issue.isoformat(), "removed_turbine": ROSTER[0],
                                         "input_valid": False, "prediction": None,
                                         "reason": "incomplete_fixed_roster_history", "synthetic_fault": True}}
        if not (args.output / "cases.json").exists():
            write_json(args.output / "cases.json", cases)
        write_json(args.output / f"{window}-result.json", summaries[window])
        print(json.dumps({"scored_window": window, "farm_mae": {key: val["farm"]["mae"] for key, val in metrics.items()}}), flush=True)
    mean_mae = {family: float(np.mean([value["metrics"][family]["farm"]["mae"] for value in summaries.values()]))
                for family in ("persistence", "ridge", "lightgbm")}
    summary = {"equal_quarter_farm_mae": mean_mae,
               "skill_vs_persistence_percent": {family: 100 * (1 - value / mean_mae["persistence"])
                                                 for family, value in mean_mae.items()},
               "recommended_development_baseline": min(mean_mae, key=mean_mae.get),
               "formal_adoption": False, "holdout_scored": False, "service_changed": False}
    report = {"version": VERSION, "freeze_sha256": freeze_hash, "source": file_record(args.source),
              "completed_at": datetime.now(timezone.utc).isoformat(), "windows": summaries, "summary": summary}
    write_json(args.output / "baseline.json", report)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("admit", "freeze", "run"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    if args.phase == "run" and not args.artifacts:
        parser.error("run requires --artifacts")
    source, batch = load_development(args.source)
    admission = admit(source, batch)
    if args.phase == "admit":
        write_json(args.output / "admission.json", admission)
        print(json.dumps({"status": admission["status"], "quarters": admission["quarters"]}, indent=2))
    elif args.phase == "freeze":
        if admission["status"] != "admitted":
            raise ValueError("engie_data_admission_failed")
        if (args.output / "protocol.json").exists():
            raise ValueError("engie_protocol_already_frozen")
        write_json(args.output / "admission.json", admission)
        write_json(args.output / "protocol.json", {
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "protocol": protocol(),
        })
        print("ENGIE protocol frozen before fitting", flush=True)
    else:
        if admission["status"] != "admitted":
            raise ValueError("engie_live_data_admission_failed")
        run(args, source, batch)


if __name__ == "__main__":
    main()
