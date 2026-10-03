"""第一轮有界真实HTTP对照；固定请求身份恢复，不直接调用训练或替换失败任务。"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlencode, urlsplit
from urllib.request import ProxyHandler, Request, build_opener
from uuid import UUID, uuid4

from power_forecast_service.experiments.comparison import compare_results
from power_forecast_service.experiments.contracts import ExperimentRequest, fingerprint
from power_forecast_service.forecasting.scoring import ScoringEvidence, metrics_for_rows
from power_forecast_service.settings import ROOT
from power_forecast_service.storage.artifacts import sha256_file, source_tree_sha256

POLICIES = ("auto_early_stopping", "fixed_iterations")
MODEL = "hist_gradient_boosting"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def save(path: Path, value: dict):
    """同目录临时文件+替换；中断不会留下半份恢复JSON。"""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def metric_table(left, right, indices):
    if not indices:
        return {"samples": 0, "metrics": None}
    a, b = [left[i] for i in indices], [right[i] for i in indices]
    metrics = {}
    for label, rows, model in (("persistence", a, "persistence"),
                               (POLICIES[0], a, MODEL), (POLICIES[1], b, MODEL)):
        metrics[label] = {
            **metrics_for_rows(rows, model),
            "mean_prediction_minus_actual": math.fsum(
                row.predictions[model] - row.actual for row in rows
            ) / len(rows),
        }
    return {
        "samples": len(a), "cutoff_start": a[0].cutoff.isoformat(),
        "cutoff_end": a[-1].cutoff.isoformat(), "metrics": metrics,
        "fixed_minus_auto": {name: metrics[POLICIES[1]][name] - metrics[POLICIES[0]][name]
                             for name in ("mae", "rmse", "mean_prediction_minus_actual")},
    }


def controlled_report(state):
    """评分可比之外，再核验这对实验预先固定的控制条件。"""
    runs, scores, identities, training = [], [], {}, {}
    for policy in POLICIES:
        side = state["sides"][policy]
        task, run = side["task"], side["run"]
        result = run["result"]
        require(task["status"] == "succeeded", "task_not_succeeded")
        require(run["task_id"] == task["task_id"] == side["submission"]["task_id"], "task_identity")
        require(run["run_id"] == result["run_id"], "run_identity")
        require(run["attempt_id"] in {item["attempt_id"] for item in task["attempts"]
                                     if item["status"] == "succeeded"}, "attempt_identity")
        require(fingerprint(result["frozen_spec"]) == fingerprint(task["spec"]), "frozen_spec")
        require(result["split"]["test_scored"] is False, "test_must_remain_unscored")
        require(result["execution"]["source_tree_sha256"] == state["source_sha256"], "source_changed")
        require(result["input_file_sha256"] == state["input_sha256"], "input_changed")
        detail = result["training"]
        require(detail["training_policy"] == task["spec"]["training_policy"] == policy, "policy_route")
        expected_early = "auto" if policy == POLICIES[0] else False
        effective = detail["effective_parameters"]
        require(fingerprint({"early": effective["early_stopping"]}) ==
                fingerprint({"early": expected_early}), "early_stopping_route")
        require(fingerprint({key: effective[key] for key in task["spec"]["hgb"]}) ==
                fingerprint(task["spec"]["hgb"]), "effective_parameters")
        require(effective["random_state"] == task["spec"]["random_seed"], "seed_route")
        require(0 < detail["n_iter"] <= detail["max_iter"], "iteration_observation")
        require(math.isfinite(detail["fit_elapsed_seconds"]) and detail["fit_elapsed_seconds"] >= 0,
                "fit_time_observation")
        require(detail["input_samples"] == result["split"]["train"], "training_samples")
        require(detail["feature_names"] == result["feature_columns"], "training_features")
        if policy == POLICIES[1]:
            require(detail["early_stopping_enabled"] is False and
                    detail["n_iter"] == detail["max_iter"], "fixed_iterations_not_executed")
        for name in ("train_objective_scores", "internal_validation_objective_scores"):
            curve = detail[name]
            require(len(curve) == (detail["n_iter"] + 1 if detail["early_stopping_enabled"] else 0)
                    and all(math.isfinite(value) for value in curve), "objective_curve_invalid")
        score = ScoringEvidence.model_validate(result["scoring"])
        self_check = compare_results(UUID(run["run_id"]), result, "persistence",
                                     UUID(run["run_id"]), result, MODEL)
        require(self_check.status == "comparable", "scoring_summary_invalid")
        runs.append(result)
        scores.append(score)
        identities[policy] = {"task_id": run["task_id"], "run_id": run["run_id"],
                              "attempt_id": run["attempt_id"], "attempt_count": task["attempt_count"],
                              "artifact_hash_reported": run["artifact_sha256"],
                              "artifact_bytes_verified_by_this_runner": False}
        # 完整曲线在checkpoint中的原始回执；摘要不复制几百个数值。
        training[policy] = {key: value for key, value in detail.items() if not key.endswith("_scores")}

    a, b = runs
    require(fingerprint(a["execution"]) == fingerprint(b["execution"]), "execution_context_differs")
    for key in ("split", "feature_columns", "feature_contract_version", "model_version", "determinism"):
        require(fingerprint({key: a[key]}) == fingerprint({key: b[key]}), f"different:{key}")
    for key in ("input_samples", "n_features_in", "feature_names", "train_cutoff_start",
                "train_cutoff_end", "train_target_start", "train_target_end"):
        require(a["training"][key] == b["training"][key], f"different:training:{key}")
    normalized = []
    for result in runs:
        spec = deepcopy(result["frozen_spec"])
        spec.pop("training_policy")
        spec["hgb"].pop("early_stopping")
        params = dict(result["training"]["effective_parameters"])
        params.pop("early_stopping")
        normalized.append(fingerprint({"spec": spec, "parameters": params}))
    require(normalized[0] == normalized[1], "non_strategy_configuration_difference")
    comparison = compare_results(UUID(a["run_id"]), a, MODEL, UUID(b["run_id"]), b, MODEL)
    require(comparison.status == "comparable", "scoring_not_comparable")
    require(state["comparison"]["status"] == "comparable", "http_comparison_not_comparable")
    require(state["comparison"]["samples_sha256"] == comparison.samples_sha256, "http_samples_differ")
    for metric in ("mae", "rmse"):
        require(math.isclose(state["comparison"]["delta"][metric], comparison.delta[metric],
                             abs_tol=1e-10, rel_tol=1e-10), "http_metric_difference")
    left, right = scores[0].rows, scores[1].rows
    require(all(x.predictions["persistence"] == y.predictions["persistence"]
                for x, y in zip(left, right, strict=True)), "persistence_changed")
    count = len(left)
    halves = [list(range(count // 2)), list(range(count // 2, count))]
    clock_groups = [[i for i, row in enumerate(left) if start <= row.cutoff.hour < start + 6]
                    for start in (0, 6, 12, 18)]
    for groups in (halves, clock_groups):
        require(sorted(i for group in groups for i in group) == list(range(count)), "group_coverage")
    return {
        "protocol": "round-1-policy-pair-v1", "pair_id": state["pair_id"],
        "status": "controlled_pair_verified", "source_sha256": state["source_sha256"],
        "input_sha256": state["input_sha256"], "samples_sha256": comparison.samples_sha256,
        "identities": identities, "training": training,
        "overall": metric_table(left, right, list(range(count))),
        "validation_halves": [metric_table(left, right, group) for group in halves],
        "source_clock_6h": {f"{start:02d}-{start+6:02d}": metric_table(left, right, group)
                            for start, group in zip((0, 6, 12, 18), clock_groups, strict=True)},
        "limits": ["development_only_test_unscored", "auto_changes_internal_holdout_and_stopping",
                   "one_pair_no_significance_or_stable_speedup_claim", "source_timezone_unknown",
                   "internal_holdout_membership_not_recorded"],
    }


def run_pair(base_url: str, directory: Path, timeout: float):
    require(urlsplit(base_url).hostname in ("localhost", "127.0.0.1"), "local_service_only")
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.json"
    current_source = source_tree_sha256()
    current_input = sha256_file(ROOT / "data/sample/wind_2019_q1.csv")
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        require(state["base_url"] == base_url, "resume_service_mismatch")
        require(state["source_sha256"] == current_source, "resume_source_mismatch")
        require(state["input_sha256"] == current_input, "resume_input_mismatch")
    else:
        pair_id = "round-1-" + uuid4().hex
        state = {
            "pair_id": pair_id, "base_url": base_url, "created_at": utc_now(),
            "source_sha256": current_source, "input_sha256": current_input,
            "status": "registered_before_submission", "sides": {
                policy: {"body": ExperimentRequest(training_policy=policy).model_dump(),
                         "idempotency_key": f"{pair_id}:{policy}", "observed_states": []}
                for policy in POLICIES
            },
        }
        save(checkpoint, state)

    # 本机回环不走环境代理；此工具不调用外部provider，也不打印凭据。
    opener = build_opener(ProxyHandler({}))

    def request(path, body=None, key=None):
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Idempotency-Key"] = key
        data = json.dumps(body).encode() if body is not None else None
        with opener.open(Request(base_url + path, data=data, headers=headers), timeout=20) as response:
            require(response.status == (202 if body is not None else 200), "unexpected_http_status")
            return json.load(response)

    try:
        require(request("/health")["training_on_read"] is False, "health_contract")
        for policy in POLICIES:
            side = state["sides"][policy]
            if "submission" not in side:
                side["submitted_at"] = utc_now()
                save(checkpoint, state)
                side["submission"] = request("/experiments", side["body"], side["idempotency_key"])
                save(checkpoint, state)
            deadline = time.monotonic() + timeout
            while True:
                task = request(side["submission"]["status_url"])
                side["task"] = task
                if not side["observed_states"] or side["observed_states"][-1]["status"] != task["status"]:
                    side["observed_states"].append({"status": task["status"], "observed_at": utc_now()})
                    save(checkpoint, state)
                    print(f"{policy}: {task['status']}", flush=True)
                if task["status"] == "failed":
                    save(checkpoint, state)
                    raise RuntimeError(f"{policy}:task_failed:{task['error_code']}")
                if task["status"] == "succeeded":
                    side["run"] = request(task["result_url"])
                    side.setdefault("completion_observed_at", utc_now())
                    save(checkpoint, state)
                    break
                if time.monotonic() >= deadline:
                    save(checkpoint, state)
                    raise TimeoutError(f"{policy}:waiting_timeout; resume with same output directory")
                time.sleep(0.5)
        state["comparison"] = request("/runs/compare?" + urlencode({
            "left_run_id": state["sides"][POLICIES[0]]["run"]["run_id"],
            "right_run_id": state["sides"][POLICIES[1]]["run"]["run_id"],
            "left_model": MODEL, "right_model": MODEL,
        }))
        save(checkpoint, state)
        report = controlled_report(state)
        save(directory / "report.json", report)
        state["status"] = "controlled_pair_verified"
        state.pop("last_error", None)
        save(checkpoint, state)
        print(json.dumps(report["overall"], ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        state["status"] = "waiting_timeout" if isinstance(exc, TimeoutError) else "needs_inspection"
        state["last_error"] = {"type": type(exc).__name__, "message": str(exc), "at": utc_now()}
        save(checkpoint, state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/runtime/round-1-pair")
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    run_pair(args.base_url.rstrip("/"), args.output, args.timeout)
