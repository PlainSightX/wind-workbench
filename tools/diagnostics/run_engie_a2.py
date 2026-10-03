"""A2显式离线入口：冻结 -> 三季度内层选择封存 -> 重拟合 -> 外层同样本比较。"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import subprocess
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from run_engie_baseline import (
    ROOT, file_record, load_development, read_json, split_masks, write_json,
)
from power_forecast_service.forecasting.engie_baselines import grouped_scores, score_curves
from power_forecast_service.forecasting.engie_contract import QUARTERS, persistence
from power_forecast_service.forecasting.engie_l1 import (
    ADOPTION_GATES, L1_PARAMETERS, LAMBDAS, adoption_decision, blend, fit_inner, refit,
    select_weight,
)


BASELINE = ROOT / "docs/results/wind-engie-baseline-20260923/baseline.json"
OUTPUT = ROOT / "docs/results/wind-engie-a2-20260923"
ARTIFACTS = ROOT / ".local/work-packages/wind-engie-a2-20260923/artifacts"
CODE_FILES = (
    "src/power_forecast_service/forecasting/engie_l1.py",
    "src/power_forecast_service/forecasting/engie_contract.py",
    "src/power_forecast_service/forecasting/engie_baselines.py",
    "tools/diagnostics/run_engie_baseline.py",
    "tools/diagnostics/run_engie_a2.py",
)
FAMILIES = ("persistence", "ridge", "lightgbm", "lightgbm_l1", "lightgbm_l1_shrink")


def now():
    return datetime.now(timezone.utc).isoformat()


def checked_file(record):
    path = ROOT / record["path"]
    if file_record(path) != record:
        raise ValueError(f"engie_a2_artifact_mismatch:{record['path']}")
    return path


def protocol():
    baseline = read_json(BASELINE)
    old_protocol = ROOT / "docs/results/wind-engie-baseline-20260923/protocol.json"
    if sha256(old_protocol.read_bytes()).hexdigest() != baseline["freeze_sha256"]:
        raise ValueError("engie_a2_baseline_protocol_mismatch")
    original = read_json(old_protocol)["protocol"]
    for record in original["code"]:
        checked_file(record)
    return {
        "version": "engie-l1-shrinkage-a2-v1", "baseline": file_record(BASELINE),
        "baseline_protocol": file_record(old_protocol), "source": baseline["source"],
        "reference_predictions": {q: w["predictions"] for q, w in baseline["windows"].items()},
        "parameters": L1_PARAMETERS, "lambda_grid": list(LAMBDAS),
        "selection": "one shared lambda per quarter by inner farm MAE; exact ties choose smaller",
        "fit_order": "all three inner selections sealed before any outer refit or scoring",
        "inner_validation_days": 28, "field_data_estimator_fit_budget": 144,
        "early_stopping": {"rounds": 40, "metric": "l1", "scope": "per turbine and horizon"},
        "adoption_gates": ADOPTION_GATES, "families": list(FAMILIES),
        "holdout": "2015 labels never enter model/selection/scoring arrays",
        "outer_usage": "already seen 2014 development; not unseen generalization",
        "refit": "same purged masks as baseline; source label availability strictly before boundaries",
        "counterfactual_limits": "no L2 shrinkage ablation; no new features; no causal proof from error groups",
        "code": [file_record(ROOT / name) for name in CODE_FILES],
        "environment": {name: version(name) for name in original["environment"]},
        "uv_lock_sha256": sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
    }


def frozen_protocol():
    path = OUTPUT / "protocol.json"
    frozen = read_json(path)
    if frozen["protocol"] != protocol():
        raise ValueError("engie_a2_contract_or_environment_changed_after_freeze")
    return sha256(path.read_bytes()).hexdigest()


def stage_receipt(quarter, stage, freeze_hash):
    path = OUTPUT / f"{quarter}-{stage}.json"
    if not path.exists():
        return None
    record = read_json(path)
    if (record["quarter"], record["stage"], record["freeze_sha256"]) != (quarter, stage, freeze_hash):
        raise ValueError("engie_a2_resume_identity_mismatch")
    checked_file(record["artifact"])
    return record


def start_stage(quarter, stage, freeze_hash, artifact):
    path = OUTPUT / f"{quarter}-{stage}-started.json"
    if path.exists() or artifact.exists():
        raise ValueError("engie_a2_partial_fit_requires_diagnosis_not_automatic_refit")
    record = {"quarter": quarter, "stage": stage, "freeze_sha256": freeze_hash, "started_at": now()}
    write_json(path, record)
    return record


def masks_for(batch, quarter):
    start, end = (pd.Timestamp(t, tz="UTC") for t in QUARTERS[quarter])
    train, validation, past, validation_start = split_masks(batch, start)
    if not all(m.any() for m in (train, validation, past)):
        raise ValueError("engie_a2_empty_partition")
    outer = (batch.issues >= start) & (batch.issues < end)
    return train, validation, past, outer, {
        "inner_train_issues": int(train.sum()), "inner_validation_issues": int(validation.sum()),
        "refit_issues": int(past.sum()), "validation_start": validation_start.isoformat(),
        "outer_start": start.isoformat(),
        "last_inner_training_label_available": (batch.issues[train][-1] + pd.Timedelta(minutes=80)).isoformat(),
        "last_refit_label_available": (batch.issues[past][-1] + pd.Timedelta(minutes=80)).isoformat(),
    }


def select_all(batch, freeze_hash):
    selections = {}
    for quarter in QUARTERS:
        receipt = stage_receipt(quarter, "selection", freeze_hash)
        if receipt is None:
            train, validation, _, _, boundaries = masks_for(batch, quarter)
            path = ARTIFACTS / f"{quarter}-inner.npz"
            started = start_stage(quarter, "selection", freeze_hash, path)
            predicted, fit_details = fit_inner(batch.features[train], batch.targets[train],
                                               batch.features[validation], batch.targets[validation])
            reference = persistence(batch.features[validation])
            choice = select_weight(batch.targets[validation], reference, predicted)
            # 这里的逐点预测来自未重拟合的内层模型；不能用最终包重新生成。
            np.savez_compressed(path, issue_ns=batch.issues[validation].asi8,
                                targets=batch.targets[validation], reference=reference, candidate=predicted)
            receipt = {**started, "finished_at": now(), "artifact": file_record(path),
                       "boundaries": boundaries, "fit": fit_details, **choice}
            write_json(OUTPUT / f"{quarter}-selection.json", receipt)
        selections[quarter] = receipt
        print(f"inner selection {quarter}: lambda={receipt['selected_lambda']}", flush=True)
    path = OUTPUT / "selections-sealed.json"
    content = {"freeze_sha256": freeze_hash, "selections": selections}
    if path.exists():
        sealed = read_json(path)
        if any(sealed[k] != value for k, value in content.items()):
            raise ValueError("engie_a2_sealed_selections_changed")
    else:
        write_json(path, {**content, "sealed_at": now()})
    return selections, sha256(path.read_bytes()).hexdigest()


def refit_all(batch, selections, freeze_hash, seal_hash):
    receipts = {}
    for quarter, choice in selections.items():
        receipt = stage_receipt(quarter, "refit", freeze_hash)
        if receipt is not None:
            if receipt["selections_sha256"] != seal_hash:
                raise ValueError("engie_a2_refit_selection_mismatch")
        else:
            _, _, past, _, _ = masks_for(batch, quarter)
            path = ARTIFACTS / f"{quarter}-l1.joblib"
            started = start_stage(quarter, "refit", freeze_hash, path)
            model, fit_details = refit(batch.features[past], batch.targets[past],
                                      choice["fit"]["selected_iterations"], choice["selected_lambda"])
            joblib.dump(model, path)
            receipt = {**started, "finished_at": now(), "artifact": file_record(path),
                       "selections_sha256": seal_hash, "fit": fit_details}
            write_json(OUTPUT / f"{quarter}-refit.json", receipt)
        receipts[quarter] = receipt
        print(f"refit complete {quarter}", flush=True)
    return receipts


def score_all(batch, selections, refits, freeze_hash, seal_hash):
    baseline = read_json(BASELINE)
    windows = {}
    for quarter in QUARTERS:
        result_path = OUTPUT / f"{quarter}-result.json"
        if result_path.exists():
            result = read_json(result_path)
            if (result["freeze_sha256"] != freeze_hash or result["selections_sha256"] != seal_hash
                    or result["refit"] != refits[quarter]):
                raise ValueError("engie_a2_completed_result_identity")
            checked_file(result["predictions"])
            windows[quarter] = result
            continue
        _, _, past, outer, _ = masks_for(batch, quarter)
        original = baseline["windows"][quarter]
        old_path = checked_file(original["predictions"])
        with np.load(old_path, allow_pickle=False) as old:
            # 逐项验证起报/标签/输入掩码，不能靠样本数相等认定比较公平。
            arrays = {k: old[k].copy() for k in old.files}
        for name, actual in {"issue_ns": batch.issues[outer].asi8, "targets": batch.targets[outer],
                             "input_valid": batch.input_valid[outer], "label_valid": batch.label_valid[outer],
                             "boundary_valid": batch.boundary_valid[outer], "scoreable": batch.scoreable[outer]}.items():
            np.testing.assert_array_equal(arrays[name], actual)
        counts = batch.counts(outer)
        if counts != original["counts"]:
            raise ValueError("engie_a2_coverage_denominator_changed")
        model = joblib.load(checked_file(refits[quarter]["artifact"]))
        x, y = batch.features[outer], batch.targets[outer]
        valid, scored = batch.input_valid[outer], batch.scoreable[outer]
        started = perf_counter()
        direct = np.full_like(y, np.nan)
        shrunk = np.full_like(y, np.nan)
        direct[valid] = model.base.predict(x[valid])
        shrunk[valid] = blend(persistence(x[valid]), direct[valid], model.weight)
        arrays["prediction_lightgbm_l1"] = direct
        arrays["prediction_lightgbm_l1_shrink"] = shrunk
        predictions = {family: arrays[f"prediction_{family}"] for family in FAMILIES}
        path = ARTIFACTS / f"{quarter}-predictions.npz"
        if path.exists():
            raise ValueError("engie_a2_orphan_predictions_require_review")
        np.savez_compressed(path, **arrays)
        windows[quarter] = {
            "quarter": quarter, "freeze_sha256": freeze_hash, "selections_sha256": seal_hash,
            "scored_at": now(), "counts": counts, "selection": selections[quarter], "refit": refits[quarter],
            "metrics": {family: score_curves(y[scored], value[scored]) for family, value in predictions.items()},
            "actual_outputs": {family: int(np.isfinite(value).all(axis=(1, 2)).sum())
                               for family, value in predictions.items()},
            "groups": grouped_scores(batch.targets[past], batch.features[past], y[scored], x[scored],
                                      {f: p[scored] for f, p in predictions.items()}),
            "predictions": file_record(path), "batch_predict_and_score_seconds": perf_counter() - started,
        }
        write_json(result_path, windows[quarter])
        print(f"scored {quarter}: " + str({f: w["farm"]["mae"] for f, w in windows[quarter]["metrics"].items()}), flush=True)
    mean = {f: {key: float(np.mean([w["metrics"][f]["farm"][key] for w in windows.values()]))
                for key in ("mae", "rmse", "bias")} for f in FAMILIES}
    count = sum(w["selection"]["fit"]["estimator_fit_count"] + w["refit"]["fit"]["estimator_fit_count"]
                for w in windows.values())
    if count != 144:
        raise ValueError("engie_a2_estimator_fit_budget_mismatch")
    report = {
        "freeze_sha256": freeze_hash, "selections_sha256": seal_hash, "completed_at": now(),
        "windows": windows, "summary": {
            "equal_quarter_farm": mean,
            "direct_mae_gain_over_l2_percent": 100 * (1 - mean["lightgbm_l1"]["mae"] / mean["lightgbm"]["mae"]),
            "adoption": {f: adoption_decision(windows, f) for f in FAMILIES[-2:]},
            "estimator_fit_count": count,
            "fit_seconds": sum(w[stage]["fit"]["fit_seconds"] for w in windows.values() for stage in ("selection", "refit")),
            "all_weights_zero": all(w["selected_lambda"] == 0 for w in selections.values()),
            "holdout_scored": False, "service_changed": False,
        },
    }
    write_json(OUTPUT / "result.json", report)
    print(report["summary"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run"))
    args = parser.parse_args()
    if args.phase == "freeze":
        if (OUTPUT / "protocol.json").exists():
            raise ValueError("engie_a2_protocol_already_frozen")
        value = protocol()
        checked_file(value["source"])
        for record in value["reference_predictions"].values():
            checked_file(record)
        write_json(OUTPUT / "protocol.json", {
            "frozen_at": now(), "git_base": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "protocol": value})
        print("A2 protocol frozen before field-data fitting", flush=True)
        return
    freeze_hash = frozen_protocol()
    if (OUTPUT / "result.json").exists():
        raise ValueError("engie_a2_complete_use_verifier_not_refit")
    source_path = checked_file(read_json(BASELINE)["source"])
    _, batch = load_development(source_path)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    selections, seal_hash = select_all(batch, freeze_hash)
    refits = refit_all(batch, selections, freeze_hash, seal_hash)
    score_all(batch, selections, refits, freeze_hash, seal_hash)


if __name__ == "__main__":
    main()
