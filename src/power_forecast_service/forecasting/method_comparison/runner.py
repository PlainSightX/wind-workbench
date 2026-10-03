"""完整比较的可恢复入口：逐配方拟合 -> 内层选择锁 -> 外层统一揭示。"""

import json
import platform
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import torch

from ...storage.artifacts import execution_provenance, sha256_file, source_tree_sha256
from ..development_protocol import INPUT_SHA256, WINDOWS
from ..features import FEATURE_COLUMNS
from ..fixed_evaluation import fixed_window_split
from ..model_diagnosis import grouped_errors, write_json
from ..sequence_data import history_windows
from ..sequence_experiment import source_data
from .data import window_fit_data
from .neural import NeuralPredictor, fit_neural, prediction_metrics
from .protocol import EXPECTED_COUNTS, fingerprint, protocol, recipes
from .tabular import fit_tabular, input_view


def now():
    return datetime.now(timezone.utc).isoformat()


def environment():
    return {"python": platform.python_version(),
            "packages": {key: version(key) for key in ("torch", "lightgbm", "numpy", "pandas", "scikit-learn")},
            "lock_sha256": sha256_file(Path("uv.lock")), "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def probe(data_path, output, *, device):
    """只用首窗内层的少量样本测运行成本，不输出/查看外层精度。"""
    output.mkdir(parents=True, exist_ok=False)
    frame, supervised, _ = source_data(data_path)
    data = window_fit_data(frame, supervised, "early_1")
    from .data import prepare_fit_data
    subset = prepare_fit_data(frame, data.train.iloc[:1024], data.validation.iloc[:256])
    records = {}
    for recipe in [item for item in recipes() if item["id"] in ("encoder_direct", "timexer_0.0001")]:
        _, details = fit_neural(subset, recipe, output / (recipe["id"] + ".pt"), device=device, epochs=2)
        records[recipe["id"]] = {key: details[key] for key in
                                 ("fit_elapsed_seconds", "parameters", "reload_max_difference")}
    result = {"purpose": "resource_only_inner_subset_no_outer_scoring", "at": now(),
              "device": device, "environment": environment(), "source_sha256": source_tree_sha256(),
              "measurements": records, "training_samples": 1024, "validation_samples": 256,
              "epochs_per_method": 2}
    write_json(output / "probe.json", result)
    return result


def freeze(data_path, output, artifacts, probe_path, *, device):
    if sha256_file(data_path) != INPUT_SHA256:
        raise ValueError("comparison_data_changed")
    result = json.loads(probe_path.read_text(encoding="utf-8"))
    if result["device"] != device or result["source_sha256"] != source_tree_sha256():
        raise ValueError("probe_device_or_source_changed")
    if result["environment"] != environment():
        raise ValueError("probe_environment_changed")
    output.mkdir(parents=True, exist_ok=False)
    artifacts.mkdir(parents=True, exist_ok=False)
    payload = {"protocol": protocol(device), "frozen_at": now(), "execution": execution_provenance(),
               "environment": environment(), "probe_sha256": sha256_file(probe_path),
               "artifacts_root": str(artifacts.resolve()), "data_path": str(data_path.resolve()),
               "recipe_order": [r["id"] for r in recipes()]}
    write_json(output / "protocol.json", payload)
    return payload


def load_frozen(output, data_path):
    frozen = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    if (frozen["protocol"] != json.loads(json.dumps(protocol(frozen["protocol"]["device"])))
            or frozen["execution"]["source_tree_sha256"] != source_tree_sha256()
            or frozen["environment"] != environment()
            or sha256_file(data_path) != frozen["protocol"]["input_sha256"]):
        raise ValueError("comparison_frozen_identity_changed")
    return frozen


def completed_attempt(output, window, recipe, protocol_hash):
    paths = sorted((output / "attempts" / window / recipe["id"]).glob("attempt-*/receipt.json"))
    if not paths:
        return None
    path = paths[-1]
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["protocol_sha256"] != protocol_hash or record["recipe"] != recipe or record["window"] != window:
        raise ValueError("attempt_identity_changed")
    if record["status"] not in ("succeeded", "failed"):
        return None
    for item in record.get("files", []):
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError("attempt_artifact_changed")
    return path, record


def run_fits(data_path, output):
    frozen = load_frozen(output, data_path)
    if (output / "selection.json").exists():
        raise ValueError("selection_already_locked_no_more_fits")
    protocol_hash = sha256_file(output / "protocol.json")
    frame, supervised, _ = source_data(data_path)
    spent = sum(json.loads(p.read_text(encoding="utf-8")).get("elapsed_seconds", 0)
                for p in (output / "attempts").glob("*/*/attempt-*/receipt.json"))
    unfinished = [p for p in (output / "attempts").glob("*/*/attempt-*/started.json")
                  if not p.with_name("receipt.json").exists()]
    if unfinished:
        # 缺少终结时间时不能假装成本为零，也不能声称精确恢复optimizer/RNG。
        raise ValueError("unfinished_attempt_requires_explicit_recovery_and_budget_accounting")
    for window in WINDOWS:
        data = window_fit_data(frame, supervised, window)
        if (len(data.train), len(data.validation)) != EXPECTED_COUNTS[window]:
            raise ValueError("comparison_inner_sample_identity_changed")
        for recipe in recipes():
            previous = completed_attempt(output, window, recipe, protocol_hash)
            if previous:
                print(f"retain {window}/{recipe['id']} {previous[1]['status']}", flush=True)
                continue
            if spent >= frozen["protocol"]["fit_budget_seconds"]:
                raise TimeoutError("comparison_fit_budget_exhausted_pending_attempts_retained")
            parent = output / "attempts" / window / recipe["id"]
            attempts = sorted(parent.glob("attempt-*"))
            number = len(attempts) + 1
            directory = parent / f"attempt-{number:02d}"
            directory.mkdir(parents=True, exist_ok=False)
            artifact_dir = Path(frozen["artifacts_root"]) / window / recipe["id"] / directory.name
            artifact_dir.mkdir(parents=True, exist_ok=False)
            neural = recipe["family"] in ("encoder", "timexer")
            artifact = artifact_dir / ("model.pt" if neural else "model.joblib")
            start = {"started_at": now(), "window": window, "recipe": recipe,
                     "protocol_sha256": protocol_hash, "data_identity": data.identity,
                     "prior_unfinished_attempts": [p.name for p in attempts]}
            write_json(directory / "started.json", start)
            started = perf_counter()
            deadline = started + frozen["protocol"]["fit_budget_seconds"] - spent
            try:
                if neural:
                    _, details = fit_neural(data, recipe, artifact, device=frozen["protocol"]["device"],
                                            history_path=directory / "epochs.jsonl", deadline=deadline)
                else:
                    _, details = fit_tabular(data, recipe, artifact, deadline=deadline)
                files = [{"path": str(artifact), "sha256": sha256_file(artifact),
                          "bytes": artifact.stat().st_size}]
                if (directory / "epochs.jsonl").exists():
                    files.append({"path": str((directory / "epochs.jsonl").resolve()),
                                  "sha256": sha256_file(directory / "epochs.jsonl")})
                record = {**start, "status": "succeeded", "details": details, "files": files,
                          "artifact": str(artifact)}
            except Exception as exc:
                partials = [p for p in (artifact, directory / "epochs.jsonl") if p.exists()]
                record = {**start, "status": "interrupted" if isinstance(exc, TimeoutError) else "failed",
                          "error_type": type(exc).__name__, "error": str(exc),
                          "traceback": traceback.format_exc(),
                          "files": [{"path": str(p.resolve()), "sha256": sha256_file(p)} for p in partials]}
            record.update({"finished_at": now(), "elapsed_seconds": perf_counter() - started})
            spent += record["elapsed_seconds"]
            write_json(directory / "receipt.json", record)
            if record["status"] == "interrupted":
                raise TimeoutError("comparison_budget_interrupted_attempt_retained")
            print(json.dumps({"completed": f"{window}/{recipe['id']}", "status": record["status"],
                              "elapsed_seconds": record["elapsed_seconds"]}), flush=True)


def selection_contents(output, data_path):
    load_frozen(output, data_path)
    protocol_hash = sha256_file(output / "protocol.json")
    selected, attempts = {}, []
    for window in WINDOWS:
        by_family = {}
        for recipe in recipes():
            found = completed_attempt(output, window, recipe, protocol_hash)
            if not found:
                raise ValueError("all_39_terminal_attempts_required_before_selection")
            path, record = found
            relative = path.relative_to(output).as_posix()
            attempts.append({"receipt": relative, "sha256": sha256_file(path), "status": record["status"]})
            if record["status"] == "succeeded":
                family = recipe["family"]
                score = record["details"]["validation"]["clipped"]["mae"]
                epoch = record["details"].get("selected_epoch", 0)
                if family not in by_family or (score, epoch) < (by_family[family]["inner_mae"], by_family[family]["selected_epoch"]):
                    by_family[family] = {"recipe_id": recipe["id"], "inner_mae": score,
                                         "selected_epoch": epoch, "receipt": relative}
        selected[window] = by_family
    return {"protocol_sha256": protocol_hash, "attempts": attempts, "selected": selected,
            "outer_scored": False,
            "selection_rule": "clipped_inner_mae_earlier_epoch_then_preregistered_recipe_order"}


def lock_selection(output, data_path):
    result = {"locked_at": now(), **selection_contents(output, data_path)}
    write_json(output / "selection.json", result)
    return result


def verified_selection(output, data_path):
    frozen = load_frozen(output, data_path)
    selected = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    if selected["protocol_sha256"] != sha256_file(output / "protocol.json") or len(selected["attempts"]) != 39:
        raise ValueError("selection_identity_or_attempt_count")
    for item in selected["attempts"]:
        if sha256_file(output / item["receipt"]) != item["sha256"]:
            raise ValueError("selection_receipt_changed")
        receipt = json.loads((output / item["receipt"]).read_text(encoding="utf-8"))
        completed_attempt(output, receipt["window"], receipt["recipe"], selected["protocol_sha256"])
    expected = selection_contents(output, data_path)
    if any(selected[key] != value for key, value in expected.items()):
        raise ValueError("selection_not_exact_inner_winners_or_cartesian_attempt_set")
    return frozen, selected


def evaluate(data_path, output):
    """全部配方和epoch诊断只在选择锁之后评分；不可反向修改该锁。"""
    frozen, selected = verified_selection(output, data_path)
    if (output / "results.json").exists():
        saved = json.loads((output / "results.json").read_text(encoding="utf-8"))
        if saved["selection_sha256"] != sha256_file(output / "selection.json"):
            raise ValueError("completed_evaluation_selection_changed")
        for window in saved["windows"].values():
            if sha256_file(output / window["predictions"]) != window["predictions_sha256"]:
                raise ValueError("completed_evaluation_predictions_changed")
        return saved
    frame, supervised, _ = source_data(data_path)
    result = {"evaluated_at": now(), "selection_sha256": sha256_file(output / "selection.json"),
              "protocol_sha256": sha256_file(output / "protocol.json"), "windows": {},
              "evidence_scope": frozen["protocol"]["evidence_scope"]}
    for window in WINDOWS:
        window_receipt = output / f"outer-{window}.json"
        if window_receipt.exists():
            saved = json.loads(window_receipt.read_text(encoding="utf-8"))
            if (saved["selection_sha256"] != result["selection_sha256"]
                    or sha256_file(output / saved["predictions"]) != saved["predictions_sha256"]):
                raise ValueError("completed_outer_window_changed")
            result["windows"][window] = saved
            continue
        fit_data = window_fit_data(frame, supervised, window)
        _, outer = fixed_window_split(supervised, window)
        windows, _ = history_windows(frame, outer.timestamp)
        target = outer.target_power.to_numpy(float)
        csv = outer[["timestamp", "target_timestamp", "target_power", "wind_power_lag_0"]].copy()
        raw_predictions = {"persistence": outer.wind_power_lag_0.to_numpy(float)}
        costs, appendix = {}, {}
        for item in selected["attempts"]:
            receipt = json.loads((output / item["receipt"]).read_text(encoding="utf-8"))
            if receipt["window"] != window or receipt["status"] != "succeeded":
                continue
            recipe, artifact = receipt["recipe"], Path(receipt["artifact"])
            neural = recipe["family"] in ("encoder", "timexer")
            labels = ("best60", "best20", "epoch20") if neural else ("selected",)
            for label in labels:
                started = perf_counter()
                if neural:
                    raw = NeuralPredictor.load(artifact, label).predict(windows, device=frozen["protocol"]["device"])
                else:
                    raw = joblib.load(artifact).predict(input_view(recipe, outer, windows))
                elapsed = perf_counter() - started
                key = recipe["id"] + "/" + label
                raw_predictions[key] = raw
                appendix[key] = prediction_metrics(target, raw)
                costs[key] = {"load_and_batch_inference_seconds": elapsed,
                              "fit_elapsed_seconds": receipt["details"]["fit_elapsed_seconds"],
                              "artifact_bytes": artifact.stat().st_size}
            chosen = selected["selected"][window].get(recipe["family"])
            if chosen and chosen["recipe_id"] == recipe["id"]:
                raw_predictions[recipe["family"]] = raw_predictions[recipe["id"] + "/" + labels[0]]
        main = {key: raw_predictions[key] for key in ["persistence", *selected["selected"][window]]}
        for key, raw in raw_predictions.items():
            csv[key + "__raw"] = raw
            csv[key + "__clipped"] = np.maximum(raw, 0)
        destination = output / f"outer-{window}.csv"
        if destination.exists():
            # CSV已写但回执未写的中断：只接受同次权重重算的相同字节，不覆盖先前曝光。
            expected_bytes = csv.to_csv(index=False, lineterminator="\n").encode("utf-8")
            if destination.read_bytes() != expected_bytes:
                raise ValueError("partial_outer_csv_does_not_match_recomputed_prediction")
        else:
            csv.to_csv(destination, index=False, mode="x", lineterminator="\n")
        result["windows"][window] = {
            "selection_sha256": result["selection_sha256"],
            "samples": len(outer), "metrics": {key: prediction_metrics(target, raw) for key, raw in main.items()},
            "groups": grouped_errors(fit_data.train, outer, {key: np.maximum(raw, 0) for key, raw in main.items()}),
            "appendix": appendix, "costs": costs, "data_identity": fit_data.identity,
            "predictions": destination.name, "predictions_sha256": sha256_file(destination)}
        write_json(window_receipt, result["windows"][window])
        print(f"outer {window}: " + json.dumps({k: v["clipped"]["mae"] for k, v in result["windows"][window]["metrics"].items()}), flush=True)
    families = ["persistence", "ridge_original", "ridge_history", "lightgbm", "encoder", "timexer"]
    summary = {}
    for family in families:
        records = [w["metrics"].get(family) for w in result["windows"].values()]
        if any(record is None for record in records):
            summary[family] = {"eligible": False, "reason": "missing_at_least_one_window"}
            continue
        summary[family] = {"eligible": True, "mean_mae": float(np.mean([r["clipped"]["mae"] for r in records])),
                           "mean_rmse": float(np.mean([r["clipped"]["rmse"] for r in records])),
                           "mean_bias": float(np.mean([r["clipped"]["bias"] for r in records]))}
    result["summary"] = summary
    write_json(output / "results.json", result)
    return result
