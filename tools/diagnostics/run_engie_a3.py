"""最终2015评价：只在2014选型与重拟合，全部包封存后一次评分。"""

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

from run_engie_baseline import ROOT, file_record, read_json, write_json, load_development, split_masks
from power_forecast_service.forecasting.engie_contract import (
    HOLDOUT_START, SOURCE_END, STEP, make_batch, persistence,
)
from power_forecast_service.forecasting.engie_baselines import (
    TREE_PARAMETERS, RIDGE_ALPHAS, FittedBaseline, _ridge_models, score_curves, grouped_scores,
)
from power_forecast_service.forecasting.engie_l1 import L1_PARAMETERS, LAMBDAS, select_weight
from power_forecast_service.forecasting.engie_final import (
    FAMILIES, FinalEngie, select_ridge, select_trees, refit_trees, paired_blocks,
)

OUTPUT = ROOT / "docs/results/wind-engie-a3-20260923"
ARTIFACTS = ROOT / ".local/work-packages/wind-engie-a3-20260923/artifacts"
BASELINE = ROOT / "docs/results/wind-engie-baseline-20260923/baseline.json"
QUARTERS = {f"2015-Q{i+1}": (a, b) for i, (a, b) in enumerate(zip(
    ("2015-01-01", "2015-04-01", "2015-07-01", "2015-10-01"),
    ("2015-04-01", "2015-07-01", "2015-10-01", "2016-01-01")))}
CODE = (
    "src/power_forecast_service/forecasting/engie_final.py",
    "src/power_forecast_service/forecasting/engie_contract.py",
    "src/power_forecast_service/forecasting/engie_baselines.py",
    "src/power_forecast_service/forecasting/engie_l1.py",
    "tools/diagnostics/run_engie_baseline.py", "tools/diagnostics/run_engie_a3.py",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def checked(record):
    path = ROOT / record["path"]
    if file_record(path) != record:
        raise ValueError("engie_final_source_identity_mismatch")
    return path


def protocol():
    previous = read_json(BASELINE)
    return {
        "version": "engie-final-2015-v1", "source": previous["source"],
        "development_result": file_record(ROOT / "docs/results/wind-engie-a2-20260923/result.json"),
        "families": list(FAMILIES), "primary_candidate": "lightgbm_l1_shrink",
        "training_boundary": HOLDOUT_START.isoformat(), "inner_days": 28,
        "label_availability": "last_target_plus_20min_strictly_before_training_boundary",
        "ridge_alphas": list(RIDGE_ALPHAS), "l2": TREE_PARAMETERS, "l1": L1_PARAMETERS,
        "lambda_grid": list(LAMBDAS), "fit_budget": 112, "seed": 42,
        "quarters": {q: list(bounds) for q, bounds in QUARTERS.items()},
        "label_end": SOURCE_END.isoformat(), "primary": "equal_quarter_farm_raw_kw_mae",
        "secondary": "pooled_year_mae_rmse_bias_and_all_quarters_horizons_groups",
        "bootstrap": {"main_block_steps": 1008, "sensitivity_block_steps": 144,
                      "samples": 2000, "seed": 42, "confidence": .95,
                      "resampling": "noncircular paired blocks on full calendar within each quarter; truncate final block"},
        "adoption": "development_3percent_gate_not_passed; no automatic adoption or tuning from holdout",
        "delivery_comparison": {"cases_per_kind": 5, "seed": 42,
            "kinds": ["normal", "missing_turbine", "stale_input", "late_result", "duplicate_result"],
            "normal_budget_ms": 60000, "late_budget_ms": 1,
            "positions": "five evenly spaced legal issue indexes, determined without scores",
            "identity": "one logical request_key and delivery_id; not cross-key business dedup",
            "baseline": "same input validator/model plus in-memory identical-key/result dedup; no durable deadline publication",
            "clock": "real elapsed compute against declared accelerated budget; not production issue deadline or SLA"},
        "code": [file_record(ROOT / name) for name in CODE],
        "environment": {n: version(n) for n in ("numpy", "pandas", "lightgbm", "scikit-learn", "joblib")},
        "uv_lock_sha256": sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
    }


def freeze_hash():
    path = OUTPUT / "protocol.json"
    if read_json(path)["protocol"] != protocol():
        raise ValueError("engie_final_frozen_protocol_changed")
    return sha256(path.read_bytes()).hexdigest()


def start(stage, frozen):
    path = OUTPUT / f"{stage}-started.json"
    if path.exists():
        raise ValueError("engie_final_partial_stage_requires_diagnosis")
    value = {"stage": stage, "freeze_sha256": frozen, "started_at": now()}
    write_json(path, value)
    return value


def training(source, batch, frozen):
    ensure_unexposed()
    train, validation, past, inner_start = split_masks(batch, HOLDOUT_START)
    if not all(mask.any() for mask in (train, validation, past)):
        raise ValueError("engie_final_empty_partition")
    boundaries = {"inner_start": inner_start.isoformat(), "inner_train": int(train.sum()),
        "validation": int(validation.sum()), "refit": int(past.sum()),
        "last_inner_label_available": (batch.issues[train][-1] + pd.Timedelta(minutes=80)).isoformat(),
        "last_refit_label_available": (batch.issues[past][-1] + pd.Timedelta(minutes=80)).isoformat()}
    selection_path = OUTPUT / "selection.json"
    if selection_path.exists():
        selection = read_json(selection_path)
        if selection["freeze_sha256"] != frozen or selection["boundaries"] != boundaries:
            raise ValueError("engie_final_selection_changed")
        checked(selection["inner_predictions"])
        validate_stage("selection", selection, frozen)
    else:
        started = start("selection", frozen)
        x, y = batch.features, batch.targets
        ridge_predictions, ridge = select_ridge(x[train], y[train], x[validation], y[validation])
        l2p, l2 = select_trees(x[train], y[train], x[validation], y[validation], TREE_PARAMETERS)
        l1p, l1 = select_trees(x[train], y[train], x[validation], y[validation], L1_PARAMETERS)
        reference = persistence(x[validation])
        choice = select_weight(y[validation], reference, l1p)
        artifact = ARTIFACTS / "inner.npz"
        np.savez_compressed(artifact, issue_ns=batch.issues[validation].asi8, targets=y[validation],
                            persistence=reference, ridge=ridge_predictions, lightgbm=l2p, lightgbm_l1=l1p)
        selection = {**started, "sealed_at": now(), "boundaries": boundaries,
                     "ridge": ridge, "l2": l2, "l1": l1, "shrink": choice,
                     "inner_predictions": file_record(artifact)}
        write_json(selection_path, selection)
    print(f"Final selection sealed: alpha={selection['ridge']['alpha']}, lambda={selection['shrink']['selected_lambda']}", flush=True)
    refit_path = OUTPUT / "refit.json"
    if refit_path.exists():
        receipt = read_json(refit_path)
        if receipt["selection_sha256"] != sha256(selection_path.read_bytes()).hexdigest():
            raise ValueError("engie_final_refit_selection_mismatch")
        for record in receipt["artifacts"].values():
            checked(record)
        validate_seals(selection, receipt, frozen)
        return selection, receipt
    started = start("refit", frozen)
    x, y = batch.features[past], batch.targets[past]
    timer = perf_counter()
    ridge = FittedBaseline("ridge", _ridge_models(x, y, selection["ridge"]["alpha"]))
    ridge_detail = {"fits": 4, "seconds": perf_counter() - timer}
    l2, l2_detail = refit_trees(x, y, selection["l2"]["iterations"], TREE_PARAMETERS)
    l1, l1_detail = refit_trees(x, y, selection["l1"]["iterations"], L1_PARAMETERS)
    models = {"persistence": FinalEngie("persistence", FittedBaseline("persistence", [])),
              "ridge": FinalEngie("ridge", ridge), "lightgbm": FinalEngie("lightgbm", l2),
              "lightgbm_l1": FinalEngie("lightgbm_l1", l1),
              "lightgbm_l1_shrink": FinalEngie("lightgbm_l1_shrink", l1, selection["shrink"]["selected_lambda"])}
    artifacts = {}
    for family, model in models.items():
        path = ARTIFACTS / f"2015-{family}.joblib"
        if path.exists():
            raise ValueError("engie_final_orphan_model")
        joblib.dump(model, path)
        artifacts[family] = file_record(path)
    receipt = {**started, "sealed_at": now(), "selection_sha256": sha256(selection_path.read_bytes()).hexdigest(),
               "artifacts": artifacts, "ridge": ridge_detail, "l2": l2_detail, "l1": l1_detail}
    write_json(refit_path, receipt)
    validate_seals(selection, receipt, frozen)
    print("All final model bytes sealed; no holdout scoring yet", flush=True)
    return selection, receipt


def ensure_unexposed():
    if any((OUTPUT / name).exists() for name in ("holdout-exposure-started.json", "result.json")):
        raise ValueError("engie_final_already_exposed_use_readonly_verifier")


def validate_stage(stage, record, frozen):
    marker = read_json(OUTPUT / f"{stage}-started.json")
    if (record["stage"] != stage or record["freeze_sha256"] != frozen
            or any(record[k] != marker[k] for k in marker)
            or datetime.fromisoformat(record["started_at"]) > datetime.fromisoformat(record["sealed_at"])):
        raise ValueError("engie_final_invalid_seal")


def validate_seals(selection, receipt, frozen):
    for stage, record in (("selection", selection), ("refit", receipt)):
        validate_stage(stage, record, frozen)
    if (datetime.fromisoformat(selection["sealed_at"]) > datetime.fromisoformat(receipt["started_at"])
            or set(receipt["artifacts"]) != set(FAMILIES)
            or any(selection[f]["fits"] != n for f, n in (("ridge", 12), ("l2", 24), ("l1", 24)))
            or any(receipt[f]["fits"] != n for f, n in (("ridge", 4), ("l2", 24), ("l1", 24)))):
        raise ValueError("engie_final_invalid_refit_contract")
    for family, record in receipt["artifacts"].items():
        path = checked(record)
        if path.resolve() != (ARTIFACTS / f"2015-{family}.joblib").resolve():
            raise ValueError("engie_final_unexpected_artifact_path")
        model = joblib.load(path)
        expected = "lightgbm" if family.startswith("lightgbm") else family
        if model.family != family or model.base.family != expected:
            raise ValueError("engie_final_model_identity_mismatch")
        if family == "lightgbm_l1_shrink" and model.weight != selection["shrink"]["selected_lambda"]:
            raise ValueError("engie_final_weight_mismatch")
        if family == "ridge":
            if len(model.base.models) != 4 or any(m.named_steps["ridge"].alpha != selection["ridge"]["alpha"] for m in model.base.models):
                raise ValueError("engie_final_ridge_parameters_changed")
        if family.startswith("lightgbm"):
            key = "l2" if family == "lightgbm" else "l1"
            parameters = TREE_PARAMETERS if key == "l2" else L1_PARAMETERS
            if len(model.base.models) != 4 or any(len(row) != 6 for row in model.base.models):
                raise ValueError("engie_final_tree_shape_changed")
            for t, row in enumerate(model.base.models):
                for h, tree in enumerate(row):
                    expected_params = {**parameters, "n_estimators": selection[key]["iterations"][t][h]}
                    if any(tree.get_params()[k] != value for k, value in expected_params.items()):
                        raise ValueError("engie_final_tree_parameters_changed")


def evaluate(source, development, selection, receipt, frozen):
    validate_seals(selection, receipt, frozen)
    exposure = start("holdout-exposure", frozen)
    # 仅此处创建2015目标数组；此前最终模型和选择已落盘封存。
    batch = make_batch(source, pd.date_range(HOLDOUT_START, SOURCE_END, freq=STEP, inclusive="left"), label_end=SOURCE_END)
    arrays = {"issue_ns": batch.issues.asi8, "targets": batch.targets, "input_valid": batch.input_valid,
              "label_valid": batch.label_valid, "boundary_valid": batch.boundary_valid, "scoreable": batch.scoreable}
    cost = {}
    for family in FAMILIES:
        model = joblib.load(checked(receipt["artifacts"][family]))
        if model.family != family:
            raise ValueError("engie_final_family_mismatch")
        prediction = np.full_like(batch.targets, np.nan)
        timer = perf_counter()
        prediction[batch.input_valid] = model.predict(batch.features[batch.input_valid])
        cost[family] = perf_counter() - timer
        if not np.isfinite(prediction[batch.input_valid]).all():
            raise ValueError("engie_final_incomplete_predictions")
        arrays[f"prediction_{family}"] = prediction
    path = ARTIFACTS / "2015-predictions.npz"
    if path.exists():
        raise ValueError("engie_final_orphan_predictions")
    np.savez_compressed(path, **arrays)
    _, _, past, _ = split_masks(development, HOLDOUT_START)
    windows, losses = {}, {}
    for quarter, (start_time, end_time) in QUARTERS.items():
        mask = (batch.issues >= pd.Timestamp(start_time, tz="UTC")) & (batch.issues < pd.Timestamp(end_time, tz="UTC"))
        scored = mask & batch.scoreable
        predictions = {f: arrays[f"prediction_{f}"][scored] for f in FAMILIES}
        windows[quarter] = {"counts": batch.counts(mask),
            "actual_outputs": {f: int(np.isfinite(arrays[f"prediction_{f}"][mask]).all(axis=(1, 2)).sum()) for f in FAMILIES},
            "metrics": {f: score_curves(batch.targets[scored], p) for f, p in predictions.items()},
            "groups": grouped_scores(development.targets[past], development.features[past], batch.targets[scored], batch.features[scored], predictions)}
        losses[quarter] = {}
        for family in FAMILIES:
            values = np.full(int(mask.sum()), np.nan)
            values[batch.scoreable[mask]] = np.abs(predictions[family].sum(1) - batch.targets[scored].sum(1)).mean(1)
            losses[quarter][family] = values
    scoreable = batch.scoreable
    pooled = {f: score_curves(batch.targets[scoreable], arrays[f"prediction_{f}"][scoreable]) for f in FAMILIES}
    means = {f: {m: float(np.mean([w["metrics"][f]["farm"][m] for w in windows.values()])) for m in ("mae", "rmse", "bias")} for f in FAMILIES}
    ci = {name: paired_blocks(losses, block_steps=steps) for name, steps in (("seven_day", 1008), ("one_day_sensitivity", 144))}
    fits = sum(selection[f]["fits"] + receipt[f]["fits"] for f in ("ridge", "l2", "l1"))
    assert fits == 112
    report = {**exposure, "completed_at": now(), "version": "engie-final-2015-v1", "source": read_json(BASELINE)["source"],
        "selection": selection, "refit": receipt, "predictions": file_record(path), "windows": windows,
        "counts": batch.counts(np.ones(len(batch.issues), dtype=bool)), "pooled_year": pooled,
        "actual_outputs": {f: int(batch.input_valid.sum()) for f in FAMILIES}, "prediction_seconds": cost,
        "summary": {"equal_quarter_farm": means, "bootstrap": ci, "field_estimator_fits": fits,
                    "fit_seconds": sum(selection[f]["seconds"] + receipt[f]["seconds"] for f in ("ridge", "l2", "l1")),
                    "primary_candidate": "lightgbm_l1_shrink", "development_adoption_gate_passed": False,
                    "automatic_adoption": False, "model_updated_in_2015": False}}
    write_json(OUTPUT / "result.json", report)
    print(report["summary"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run"))
    args = parser.parse_args()
    if args.phase == "freeze":
        if (OUTPUT / "protocol.json").exists():
            raise ValueError("engie_final_already_frozen")
        value = protocol()
        checked(value["source"])
        write_json(OUTPUT / "protocol.json", {"frozen_at": now(), "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "protocol": value})
        print("Final protocol frozen; holdout remains unopened", flush=True)
        return
    frozen = freeze_hash()
    ensure_unexposed()
    if (OUTPUT / "result.json").exists():
        raise ValueError("engie_final_complete_use_readonly_verifier")
    source, development = load_development(checked(read_json(BASELINE)["source"]))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    selection, receipt = training(source, development, frozen)
    evaluate(source, development, selection, receipt, frozen)


if __name__ == "__main__":
    main()
