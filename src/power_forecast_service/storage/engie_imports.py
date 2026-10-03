"""受控离线导入：可信来源先于反序列化，CPU复核先于一次事务发布。"""

from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import platform
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from .model_packages import PackageError
from .engie_packages import checked_bytes, trust, write_verified
from .models import ImportedArtifact, ImportedRun

FAMILIES = ("persistence", "ridge", "lightgbm")
ATOL, RTOL = 1e-6, 1e-8


def runtime_identity():
    return {"python": platform.python_version(), "platform": platform.system(),
            **{name: version(name) for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "joblib")}}


def stage_sources(repository, destination):
    """只打包已核实来源；不反序列化、不复制工作区或2015的派生分析。"""
    approved = trust()
    baseline = checked_bytes(repository, approved["baseline_path"], approved["baseline_sha256"])
    protocol = checked_bytes(repository, approved["protocol_path"], approved["protocol_sha256"])
    result = json.loads(baseline)
    files = {"baseline.json": baseline, "protocol.json": protocol}
    record = result["source"]
    files["source.zip"] = checked_bytes(repository, record["path"], record["sha256"])
    for quarter, window in result["windows"].items():
        for family in FAMILIES:
            record = window["fits"][family]["artifact"]
            files[f"{quarter}-{family}.joblib"] = checked_bytes(repository, record["path"], record["sha256"])
        record = window["predictions"]
        files[f"{quarter}.npz"] = checked_bytes(repository, record["path"], record["sha256"])
    for name, payload in files.items():
        write_verified(Path(destination) / name, payload)
    return {"files": len(files), "source_sha256": approved["baseline_sha256"]}


def prepare_import(source_dir, artifact_root):
    """从可信源在当前CPU进程复核九包；只有2014历史会进入线上回放存储。"""
    import joblib
    from ..forecasting.engie_contract import (
        DEVELOPMENT_START, HOLDOUT_START, QUARTERS, STEP, load_source, make_batch,
    )

    approved = trust()
    result = json.loads(checked_bytes(source_dir, "baseline.json", approved["baseline_sha256"]))
    frozen = json.loads(checked_bytes(source_dir, "protocol.json", approved["protocol_sha256"]))
    checked_bytes(source_dir, "source.zip", result["source"]["sha256"])
    source = load_source(Path(source_dir) / "source.zip")
    issues = pd.date_range(DEVELOPMENT_START, HOLDOUT_START, freq=STEP, inclusive="left")
    batch = make_batch(source, issues)
    relative = Path("engie-imports") / approved["baseline_sha256"]
    destination = Path(artifact_root) / relative
    # NPZ只含已开放2014原始历史。线上不需要、也不持有2015实况。
    dev = source.times < HOLDOUT_START
    buffer = BytesIO()
    np.savez_compressed(buffer, time_ns=source.times[dev].asi8, values=source.values[dev])
    source_payload = buffer.getvalue()
    write_verified(destination / "history.npz", source_payload)
    history_sha = sha256(source_payload).hexdigest()
    prepared, verification = [], {}
    for quarter, (start, end) in QUARTERS.items():
        window = result["windows"][quarter]
        selected = (batch.issues >= pd.Timestamp(start, tz="UTC")) & (batch.issues < pd.Timestamp(end, tz="UTC"))
        prediction_bytes = checked_bytes(source_dir, f"{quarter}.npz", window["predictions"]["sha256"])
        with np.load(BytesIO(prediction_bytes), allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["issue_ns"], batch.issues[selected].asi8)
            np.testing.assert_array_equal(saved["input_valid"], batch.input_valid[selected])
            np.testing.assert_array_equal(saved["targets"], batch.targets[selected])
            legal = saved["input_valid"]
            features = batch.features[selected][legal]
            import_id = uuid5(NAMESPACE_URL, approved["baseline_sha256"] + quarter)
            manifest = {
                "origin": "imported_offline", "quarter": quarter,
                "protocol_version": result["version"], "protocol_sha256": result["freeze_sha256"],
                "source_sha256": result["source"]["sha256"],
                "baseline_sha256": approved["baseline_sha256"],
                "training_label_available": window["last_refit_label_available"],
                "original_environment": frozen["protocol"]["environment"],
                "inference_environment": runtime_identity(),
                "start": pd.Timestamp(start, tz="UTC").isoformat(),
                "end": pd.Timestamp(end, tz="UTC").isoformat(),
                "history_path": (relative / "history.npz").as_posix(), "history_sha256": history_sha,
                "predictions_path": (relative / f"{quarter}.npz").as_posix(),
                "predictions_sha256": window["predictions"]["sha256"],
                "counts": window["counts"], "metrics": {},
            }
            artifacts = []
            for family in FAMILIES:
                record = window["fits"][family]["artifact"]
                payload = checked_bytes(source_dir, f"{quarter}-{family}.joblib", record["sha256"])
                # pickle只从上方固定baseline的哈希链加载；没有HTTP上传接口。
                model = joblib.load(BytesIO(payload))
                if model.family != family:
                    raise PackageError("engie_model_identity_mismatch")
                predicted = model.predict(features)
                expected = saved[f"prediction_{family}"][legal]
                np.testing.assert_allclose(predicted, expected, atol=ATOL, rtol=RTOL)
                difference = float(np.abs(predicted - expected).max())
                verification[f"{quarter}-{family}"] = {"issues": int(legal.sum()), "max_abs_difference_kw": difference}
                model_path = relative / f"{quarter}-{family}.joblib"
                write_verified(Path(artifact_root) / model_path, payload)
                artifact_id = uuid5(import_id, family + record["sha256"])
                artifact_manifest = {
                    "sha256": record["sha256"], "model_version": f"{quarter}-{family}-{record['sha256'][:12]}",
                    "metrics": window["metrics"][family]["farm"],
                    "coverage": {**window["counts"], "output_count": window["actual_outputs"][family]},
                }
                manifest["metrics"][family] = artifact_manifest["metrics"]
                artifacts.append({"id": artifact_id, "import_id": import_id, "family": family,
                                  "path": model_path.as_posix(), "manifest": artifact_manifest, "status": "ready"})
            write_verified(destination / f"{quarter}.npz", prediction_bytes)
            prepared.append(({"id": import_id, "source_sha256": approved["baseline_sha256"],
                              "quarter": quarter, "manifest": manifest}, artifacts))
    return prepared, {"atol_kw": ATOL, "rtol": RTOL, "runtime": runtime_identity(), "packages": verification}


def publish_import(engine, prepared):
    """九包全验完后一次提交；锁保护并发重入，异常回滚不暴露半组版本。"""
    imported = []
    with Session(engine) as session, session.begin():
        session.execute(text("SELECT pg_advisory_xact_lock(461923004)"))
        for run, artifacts in prepared:
            existing = session.get(ImportedRun, run["id"])
            if existing is None:
                session.add(ImportedRun(**run))
                session.flush()
                session.add_all(ImportedArtifact(**item) for item in artifacts)
            else:
                # 重导不能把unavailable自动恢复，也不能悄悄替换既有版本。
                rows = session.scalars(select(ImportedArtifact).where(ImportedArtifact.import_id == run["id"])).all()
                expected = {item["id"]: item for item in artifacts}
                if existing.manifest != run["manifest"] or {row.id for row in rows} != set(expected):
                    raise PackageError("engie_import_registration_conflict")
                for row in rows:
                    item = expected[row.id]
                    if row.manifest != item["manifest"] or row.path != item["path"] or row.status != "ready":
                        raise PackageError("engie_import_registration_conflict")
            imported.append(str(run["id"]))
    return imported
