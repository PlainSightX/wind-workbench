"""已评分的最终模型原字节接入；不重训、不改写开发期信任锚。"""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pandas as pd

from .engie_imports import runtime_identity, ATOL, RTOL
from .engie_packages import checked_bytes, write_verified
from .model_packages import PackageError

FAMILIES = ("persistence", "ridge", "lightgbm", "lightgbm_l1", "lightgbm_l1_shrink")


def trust():
    return json.loads(Path(__file__).with_name("engie_final_source.json").read_text(encoding="utf-8"))


def stage_sources(repository, destination):
    approved = trust()
    files = {"result.json": checked_bytes(repository, approved["result_path"], approved["result_sha256"]),
             "protocol.json": checked_bytes(repository, approved["protocol_path"], approved["protocol_sha256"])}
    result = json.loads(files["result.json"])
    for name, record in {"source.zip": result["source"], "predictions.npz": result["predictions"],
                         **{f"{f}.joblib": r for f, r in result["refit"]["artifacts"].items()}}.items():
        files[name] = checked_bytes(repository, record["path"], record["sha256"])
    for name, payload in files.items():
        write_verified(Path(destination) / name, payload)
    return {"files": len(files), "result_sha256": approved["result_sha256"]}


def prepare_import(source_dir, artifact_root):
    import joblib
    from ..forecasting.engie_contract import HOLDOUT_START, SOURCE_END, STEP, load_source, make_batch

    approved = trust()
    result = json.loads(checked_bytes(source_dir, "result.json", approved["result_sha256"]))
    frozen = json.loads(checked_bytes(source_dir, "protocol.json", approved["protocol_sha256"]))
    if result["freeze_sha256"] != approved["protocol_sha256"] or set(result["refit"]["artifacts"]) != set(FAMILIES):
        raise PackageError("engie_final_contract_conflict")
    checked_bytes(source_dir, "source.zip", result["source"]["sha256"])
    source = load_source(Path(source_dir) / "source.zip")
    batch = make_batch(source, pd.date_range(HOLDOUT_START, SOURCE_END, freq=STEP, inclusive="left"), label_end=SOURCE_END)
    relative = Path("engie-final-imports") / approved["result_sha256"]
    destination = Path(artifact_root) / relative
    buffer = BytesIO()
    # 只供此最终导入身份使用；旧季度仍指向原先仅含2014的历史文件。
    np.savez_compressed(buffer, time_ns=source.times.asi8, values=source.values)
    payload = buffer.getvalue()
    write_verified(destination / "history.npz", payload)
    predictions = checked_bytes(source_dir, "predictions.npz", result["predictions"]["sha256"])
    import_id = uuid5(NAMESPACE_URL, approved["result_sha256"] + "2015-final")
    manifest = {
        "origin": "imported_offline", "quarter": "2015-final", "scope": "final_2015", "families": list(FAMILIES),
        "protocol_version": result["version"], "protocol_sha256": result["freeze_sha256"],
        "source_sha256": result["source"]["sha256"], "result_sha256": approved["result_sha256"],
        "training_label_available": result["selection"]["boundaries"]["last_refit_label_available"],
        "original_environment": frozen["protocol"]["environment"], "inference_environment": runtime_identity(),
        "start": HOLDOUT_START.isoformat(), "end": SOURCE_END.isoformat(),
        "history_path": (relative / "history.npz").as_posix(), "history_sha256": sha256(payload).hexdigest(),
        "predictions_path": (relative / "predictions.npz").as_posix(), "predictions_sha256": result["predictions"]["sha256"],
        "counts": result["counts"], "metrics": {},
        "evaluation": {"metric": "equal_quarter_farm_raw_kw_mae", "result_sha256": approved["result_sha256"],
            "primary_candidate": result["summary"]["primary_candidate"], "bootstrap": result["summary"]["bootstrap"],
            "development_adoption_gate_passed": False, "automatic_adoption": False,
            "pooled_year": {f: result["pooled_year"][f]["farm"] for f in FAMILIES}},
    }
    artifacts, verification = [], {}
    with np.load(BytesIO(predictions), allow_pickle=False) as saved:
        for name, value in (("issue_ns", batch.issues.asi8), ("targets", batch.targets), ("input_valid", batch.input_valid),
                            ("scoreable", batch.scoreable), ("label_valid", batch.label_valid), ("boundary_valid", batch.boundary_valid)):
            np.testing.assert_array_equal(saved[name], value)
        for family in FAMILIES:
            record = result["refit"]["artifacts"][family]
            payload = checked_bytes(source_dir, f"{family}.joblib", record["sha256"])
            model = joblib.load(BytesIO(payload))
            if model.family != family:
                raise PackageError("engie_model_identity_mismatch")
            predicted = model.predict(batch.features[batch.input_valid])
            expected = saved[f"prediction_{family}"][batch.input_valid]
            np.testing.assert_allclose(predicted, expected, atol=ATOL, rtol=RTOL)
            verification[family] = {"issues": int(batch.input_valid.sum()), "max_abs_difference_kw": float(np.abs(predicted - expected).max())}
            model_path = relative / f"{family}.joblib"
            write_verified(Path(artifact_root) / model_path, payload)
            metrics = result["summary"]["equal_quarter_farm"][family]
            manifest["metrics"][family] = metrics
            artifacts.append({"id": uuid5(import_id, family + record["sha256"]), "import_id": import_id, "family": family,
                "path": model_path.as_posix(), "status": "ready", "manifest": {
                    "sha256": record["sha256"], "model_version": f"2015-final-{family}-{record['sha256'][:12]}",
                    "metrics": metrics, "coverage": {**result["counts"], "output_count": result["actual_outputs"][family]}}})
    write_verified(destination / "predictions.npz", predictions)
    return [({"id": import_id, "source_sha256": approved["result_sha256"], "quarter": "2015-final", "manifest": manifest}, artifacts)], {
        "atol_kw": ATOL, "rtol": RTOL, "runtime": runtime_identity(), "packages": verification, "model_bytes": "same_as_final_scored"}
