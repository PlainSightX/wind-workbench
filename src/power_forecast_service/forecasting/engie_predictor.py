"""已登记ENGIE包的只读预测/回放；实况始终在预测之后组装。"""

from datetime import datetime, timedelta
from hashlib import sha256
from io import BytesIO
import json

import numpy as np
import pandas as pd

from .engie_service_contract import EngieForecastRequest, HORIZONS, ROSTER
from ..storage.engie_packages import checked_bytes
from ..storage.model_packages import PackageError


def check_issue(registration, issue):
    manifest = registration["run"]
    if not (datetime.fromisoformat(manifest["start"]) <= issue < datetime.fromisoformat(manifest["end"])):
        raise PackageError("engie_issue_not_open")
    if issue <= datetime.fromisoformat(manifest["training_label_available"]):
        raise PackageError("engie_issue_before_training")


def prediction_arrays(root, registration):
    run = registration["run"]
    payload = checked_bytes(root, run["predictions_path"], run["predictions_sha256"])
    with np.load(BytesIO(payload), allow_pickle=False) as arrays:
        return {key: arrays[key] for key in ("issue_ns", "input_valid", "targets")}


def windows(root, registration):
    checked_bytes(root, registration["path"], registration["manifest"]["sha256"])
    run = registration["run"]
    arrays = prediction_arrays(root, registration)
    issues = pd.to_datetime(arrays["issue_ns"][arrays["input_valid"]], utc=True)
    return {"issue_times": [stamp.isoformat() for stamp in issues], "count": len(issues),
            "quarter": run["quarter"], "training_label_available": run["training_label_available"]}


def feature_array(body):
    raw = np.array([[[getattr(row.turbines[name], field) for field in
                      ("power_kw", "wind_speed", "direction_degrees", "temperature")]
                     for row in body.history] for name in ROSTER])
    angle = np.deg2rad(raw[..., 2])
    encoded = np.stack([raw[..., 0], raw[..., 1], np.sin(angle), np.cos(angle), raw[..., 3]], axis=-1)
    return encoded.reshape(1, 4, 60)


def forecast(root, registration, body):
    import joblib

    check_issue(registration, body.issue_time)
    payload = checked_bytes(root, registration["path"], registration["manifest"]["sha256"])
    try:
        model = joblib.load(BytesIO(payload))
        if model.family != registration["family"]:
            raise PackageError("engie_model_identity_mismatch")
        output = np.asarray(model.predict(feature_array(body)))
    except PackageError:
        raise
    except Exception as exc:
        raise PackageError("engie_package_incompatible") from exc
    if output.shape != (1, 4, 6) or not np.isfinite(output).all():
        raise PackageError("engie_model_invalid_output")
    with np.errstate(over="ignore"):
        farm = output[0].sum(axis=0)
    if not np.isfinite(farm).all():
        raise PackageError("engie_model_invalid_output")
    history = [row.model_dump(mode="json") for row in body.history]
    return {
        "artifact_id": str(body.artifact_id), "import_id": str(registration["import_id"]),
        "family": registration["family"], "model_version": registration["manifest"]["model_version"],
        "issue_time": body.issue_time.isoformat(), "source_cutoff": body.history[-1].timestamp.isoformat(),
        "target_times": [(body.issue_time + timedelta(minutes=h)).isoformat() for h in HORIZONS],
        "roster": list(ROSTER), "predictions": output[0].tolist(), "farm_predictions": farm.tolist(),
        "unit": "kW", "clock": "UTC", "training_label_available": registration["run"]["training_label_available"],
        "input_sha256": sha256(json.dumps(history, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }


def replay_input(root, registration, issue):
    check_issue(registration, issue)
    arrays = prediction_arrays(root, registration)
    matches = np.flatnonzero(arrays["issue_ns"] == pd.Timestamp(issue).value)
    if len(matches) != 1 or not arrays["input_valid"][matches[0]]:
        raise PackageError("engie_history_unavailable")
    run = registration["run"]
    payload = checked_bytes(root, run["history_path"], run["history_sha256"])
    expected = pd.date_range(issue - timedelta(minutes=130), periods=12, freq="10min")
    with np.load(BytesIO(payload), allow_pickle=False) as source:
        positions = pd.Index(source["time_ns"]).get_indexer(expected.asi8)
        if (positions < 0).any():
            raise PackageError("engie_history_unavailable")
        raw = source["values"][positions]
    if not np.isfinite(raw).all():
        raise PackageError("engie_history_unavailable")
    fields = ("power_kw", "wind_speed", "direction_degrees", "temperature")
    history = [{"timestamp": stamp.isoformat(), "turbines": {
        name: dict(zip(fields, raw[i, j].tolist(), strict=True)) for j, name in enumerate(ROSTER)
    }} for i, stamp in enumerate(expected)]
    body = EngieForecastRequest(artifact_id=registration["id"], issue_time=issue, history=history)
    actual = arrays["targets"][matches[0]]
    # 未来标签缺失/2015保留期均为null；不用零填充或三台实况冒充全场。
    actual_json = [[float(value) if np.isfinite(value) else None for value in row] for row in actual]
    return body, actual_json


def replay(root, registration, issue):
    body, actual = replay_input(root, registration, issue)
    result = forecast(root, registration, body)
    return {"mode": "imported_final_replay" if registration["run"].get("scope") == "final_2015" else "imported_development_replay", "forecast": result, "actual": actual,
            "history": [row.model_dump(mode="json") for row in body.history]}
