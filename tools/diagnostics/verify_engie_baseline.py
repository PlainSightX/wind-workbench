"""新进程只读验收：复算样本、指标和同包预测，明确禁止重新拟合。"""

import argparse
from dataclasses import replace
from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from power_forecast_service.forecasting.engie_contract import (
    ARRIVAL_LAG, DEVELOPMENT_START, HOLDOUT_START, HORIZONS, QUARTERS, ROSTER, STEP,
    admit, before_training_boundary, load_source, make_batch,
)

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = {"persistence", "ridge", "lightgbm"}


def require(condition, name):
    if not condition:
        raise ValueError(name)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_file(record):
    path = ROOT / record["path"]
    require(path.stat().st_size == record["bytes"], f"size:{path.name}")
    require(sha256(path.read_bytes()).hexdigest() == record["sha256"], f"sha256:{path.name}")
    return path


def compare_metric(actual, predicted, expected):
    error = predicted - actual
    computed = [np.abs(error).mean(), np.sqrt(np.square(error).mean()), error.mean()]
    np.testing.assert_allclose(computed, [expected[k] for k in ("mae", "rmse", "bias")], rtol=1e-10, atol=1e-8)
    require(expected["samples"] == actual.size, "score_denominator_mismatch")


def check_result_members(result):
    """缺失整窗或整种模型不能通过缩小验收遍历范围来消失。"""
    require(set(result["windows"]) == set(QUARTERS), "complete_quarter_set")
    for record in result["windows"].values():
        require(record["freeze_sha256"] == result["freeze_sha256"], "window_frozen_contract")
        for key in ("fits", "metrics", "actual_outputs", "prediction_cost"):
            require(set(record[key]) == FAMILIES, f"complete_family_set:{key}")


def check_cases(cases, predictions):
    """示例是交付内容，也必须对应真实保存的同次预测，而非只检查字段存在。"""
    normal, missing = cases["normal"], cases["missing_turbine"]
    issue = pd.Timestamp(normal["issue_time"])
    require(predictions is not None, "example_issue_in_legal_development_outputs")
    require(normal["roster"] == list(ROSTER) and normal["unit"] == "kW", "example_roster_unit")
    require(pd.Timestamp(normal["source_cutoff"]) == issue - ARRIVAL_LAG, "example_source_cutoff")
    require([pd.Timestamp(t) for t in normal["valid_times"]] ==
            [issue + pd.Timedelta(minutes=h) for h in HORIZONS], "example_valid_times")
    require(set(normal["predictions"]) == set(normal["farm_predictions"]) == FAMILIES, "example_families")
    for family in FAMILIES:
        np.testing.assert_allclose(normal["predictions"][family], predictions[family], rtol=1e-12, atol=1e-8)
        np.testing.assert_allclose(normal["farm_predictions"][family], predictions[family].sum(axis=0), rtol=1e-12, atol=1e-8)
    require(pd.Timestamp(missing["issue_time"]) == issue and missing["removed_turbine"] == ROSTER[0], "fault_identity")
    require(missing["input_valid"] is False and missing["prediction"] is None, "fault_no_output")
    require(missing["synthetic_fault"] is True and missing["reason"] == "incomplete_fixed_roster_history", "fault_scope")


def verify(result_path):
    result = read_json(result_path)
    check_result_members(result)
    folder = result_path.parent
    freeze_path = folder / "protocol.json"
    frozen = read_json(freeze_path)
    require(sha256(freeze_path.read_bytes()).hexdigest() == result["freeze_sha256"], "freeze_hash")
    require(result["version"] == frozen["protocol"]["version"], "result_contract_version")
    require(frozen["protocol"]["environment"] == {name: version(name) for name in frozen["protocol"]["environment"]}, "frozen_environment")
    require(sha256((ROOT / "uv.lock").read_bytes()).hexdigest() == frozen["protocol"]["uv_lock_sha256"], "frozen_lock")
    for record in frozen["protocol"]["code"]:
        check_file(record)
    source = load_source(check_file(result["source"]))
    batch = make_batch(source, pd.date_range(DEVELOPMENT_START, HOLDOUT_START, freq=STEP, inclusive="left"))
    require(admit(source, batch) == read_json(folder / "admission.json"), "admission_recomputed")
    require(not np.isfinite(batch.targets[~batch.boundary_valid, :, -1]).any(), "holdout_never_in_target_array")
    cases = read_json(folder / "cases.json")
    example_issue = pd.Timestamp(cases["normal"]["issue_time"])
    example_predictions = None

    def forbidden(*args, **kwargs):
        raise AssertionError("verification_must_not_fit")

    Ridge.fit = forbidden
    StandardScaler.fit = forbidden
    LGBMRegressor.fit = forbidden
    verified_rows, reload_differences, fit_counts = 0, {}, {}
    for window, record in result["windows"].items():
        require(record == read_json(folder / f"{window}-result.json"), "window_receipt_identity")
        start, end = [pd.Timestamp(value, tz="UTC") for value in QUARTERS[window]]
        outer = (batch.issues >= start) & (batch.issues < end)
        refit = before_training_boundary(batch, start)
        inner_start = start - pd.Timedelta(days=28)
        train = before_training_boundary(batch, inner_start)
        validation = refit & (batch.issues >= inner_start)
        require(record["counts"] == batch.counts(outer), "all_issue_denominators")
        require(record["refit_issues"] == int(refit.sum()), "refit_count")
        require(record["inner_train_issues"] == int(train.sum()), "inner_train_count")
        require(record["inner_validation_issues"] == int(validation.sum()), "validation_count")
        require(batch.issues[refit][-1] + pd.Timedelta(minutes=80) < start, "refit_label_availability")
        require(batch.issues[train][-1] + pd.Timedelta(minutes=80) < inner_start, "inner_label_availability")
        with np.load(check_file(record["predictions"]), allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["issue_ns"], batch.issues[outer].asi8)
            for key in ("input_valid", "label_valid", "boundary_valid", "scoreable"):
                np.testing.assert_array_equal(arrays[key], getattr(batch, key)[outer])
            np.testing.assert_array_equal(arrays["targets"], batch.targets[outer])
            scoreable = arrays["scoreable"]
            actual = arrays["targets"][scoreable]
            valid = arrays["input_valid"]
            example_indices = np.flatnonzero((batch.issues[outer] == example_issue) & valid)
            if example_indices.size:
                example_predictions = {family: arrays[f"prediction_{family}"][example_indices[0]].copy() for family in FAMILIES}
            for family in ("persistence", "ridge", "lightgbm"):
                receipt = record["fits"][family]
                require(receipt == read_json(folder / f"{window}-{family}-fit.json"), "fit_receipt_identity")
                require(receipt["freeze_sha256"] == result["freeze_sha256"], "fit_frozen_contract")
                require(receipt["family"] == family and receipt["window"] == window, "fit_identity")
                require(datetime.fromisoformat(frozen["frozen_at"]) < datetime.fromisoformat(receipt["started_at"]) <= datetime.fromisoformat(receipt["finished_at"]), "freeze_before_fit_start")
                model = joblib.load(check_file(receipt["artifact"]))
                require(model.family == family, "model_family")
                predicted = arrays[f"prediction_{family}"]
                require(np.isfinite(predicted[valid]).all(), "complete_outputs_for_legal_inputs")
                require(np.isnan(predicted[~valid]).all(), "no_output_for_rejected_inputs")
                require(record["actual_outputs"][family] == int(valid.sum()), "actual_output_count")
                loaded = model.predict(batch.features[outer][valid])
                np.testing.assert_allclose(loaded, predicted[valid], rtol=1e-12, atol=1e-8)
                reload_differences[f"{window}/{family}"] = float(np.max(np.abs(loaded - predicted[valid])))
                fit_counts[f"{window}/{family}"] = receipt["details"]["estimator_fit_count"]
                expected = record["metrics"][family]
                p = predicted[scoreable]
                compare_metric(actual.sum(axis=1), p.sum(axis=1), expected["farm"])
                compare_metric(actual, p, expected["all_turbine_values"])
                for h, minutes in enumerate(HORIZONS):
                    compare_metric(actual.sum(axis=1)[:, h], p.sum(axis=1)[:, h], expected["farm_by_horizon"][str(minutes)])
                for t, name in enumerate(frozen["protocol"]["roster"]):
                    compare_metric(actual[:, t], p[:, t], expected["turbines"][name]["all_horizons"])
                    for h, minutes in enumerate(HORIZONS):
                        compare_metric(actual[:, t, h], p[:, t, h], expected["turbines"][name]["by_horizon"][str(minutes)])
                if family == "ridge":
                    scalers = [m.named_steps["standardscaler"] for m in model.models]
                    np.testing.assert_allclose(np.array([m.mean_ for m in scalers]), batch.features[refit].mean(axis=0), rtol=1e-10, atol=1e-8)
                    require(all(m.n_samples_seen_ == int(refit.sum()) for m in scalers), "train_only_scaler_samples")
                if family == "lightgbm":
                    selected = receipt["details"]["selected_iterations"]
                    require(all(m.n_estimators == selected[t][h] for t, ms in enumerate(model.models) for h, m in enumerate(ms)), "fixed_inner_selected_iterations")
                # 分组阈值必须来自过去训练，不依赖外层误差调节。
                group = record["groups"]
                latest = batch.features[refit, :, -5].sum(axis=1)
                past = batch.targets[refit].sum(axis=1)[:, -1]
                np.testing.assert_allclose([np.quantile(np.abs(past - latest), .75), np.quantile(past, .25)],
                                           [group["change_q75_kw"], group["power_q25_kw"]])
                current_last = batch.features[outer][scoreable, :, -5].sum(axis=1)
                current_y = actual.sum(axis=1)[:, -1]
                masks = {"large_realized_change": np.abs(current_y - current_last) >= group["change_q75_kw"],
                         "smaller_realized_change": np.abs(current_y - current_last) < group["change_q75_kw"],
                         "low_actual_power": current_y <= group["power_q25_kw"]}
                for name, mask in masks.items():
                    require(group["groups"][name]["issues"] == int(mask.sum()), "group_denominator")
                    if mask.any():
                        compare_metric(actual[mask].sum(axis=1), p[mask].sum(axis=1), group["groups"][name]["metrics"][family])
            verified_rows += int(scoreable.sum())
    means = {family: float(np.mean([w["metrics"][family]["farm"]["mae"] for w in result["windows"].values()]))
             for family in ("persistence", "ridge", "lightgbm")}
    require(means == result["summary"]["equal_quarter_farm_mae"], "equal_quarter_mean")
    require(result["summary"]["recommended_development_baseline"] == min(means, key=means.get), "recommended_baseline")
    require(all(result["summary"][key] is False for key in ("formal_adoption", "holdout_scored", "service_changed")), "development_only_claims")
    for family in means:
        np.testing.assert_allclose(100 * (1 - means[family] / means["persistence"]), result["summary"]["skill_vs_persistence_percent"][family])
    check_cases(cases, example_predictions)
    issue = pd.Timestamp(cases["normal"]["issue_time"])
    damaged = source.values.copy()
    damaged[source.times.get_loc(issue - ARRIVAL_LAG), 0] = np.nan
    require(not make_batch(replace(source, values=damaged), pd.DatetimeIndex([issue])).input_valid[0], "fixed_roster_fault_rejected")
    require(cases["missing_turbine"]["prediction"] is None, "no_three_turbine_substitution")
    return {"result": "passed", "scoreable_issue_count": verified_rows,
            "scored_turbine_horizon_values_per_model": verified_rows * 24,
            "reload_max_differences": reload_differences, "estimator_fit_counts": fit_counts,
            "verification_fits": 0, "holdout_scored": False,
            "scope": "same_source_contract_denominators_raw_metrics_frozen_selections_fresh_process_no_fit_reload"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = verify(args.result)
    text = json.dumps(value, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
