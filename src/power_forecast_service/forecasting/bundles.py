"""完整预测包的生成与独立验证；模型文件不是任务成功或可用登记的权威。"""

import json
import os
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

from ..storage.artifacts import PACKAGE_ROOT, sha256_file
from ..storage.model_packages import (
    ATOL, RTOL, PACKAGE_VERSION, PackageError, PackageManifest, PackageRegistration,
    checked_manifest, confined_path, inference_code_sha256, inference_runtime,
    CANDIDATE_PACKAGE_VERSION,
    SEQUENCE_PACKAGE_VERSION,
)
from .contracts import ForecastRequest
from .features import FEATURE_COLUMNS
from .predictor import HistoryPredictor
from .candidate_predictor import CandidateHistoryPredictor
from .spec import FEATURE_CONTRACT_VERSION
from .sequence_protocol import SEQUENCE_KEYS, SEQUENCE_VERSION, SEQUENCE_FEATURES


def write_json(path: Path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def local_packaging_environment():
    name = "MLFLOW_UV_AUTO_DETECT"
    previous = os.environ.get(name)
    os.environ[name] = "false"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def history_frame(observations) -> pd.DataFrame:
    # JSON整数与CSV整型均在共同Pydantic边界变为float，符合pyfunc显式double签名。
    request = ForecastRequest(artifact_id=uuid4(), observations=observations)
    return pd.DataFrame([row.model_dump() for row in request.observations])


def save_packages(product, root: Path, task_id, attempt_id) -> list[PackageRegistration]:
    """使用同次拟合对象和开发期参考预测；不训练、不读取测试标签验证模型。"""
    result = product.result
    rows = result["scoring"]["rows"]
    selected = [rows[index] for index in sorted({0, len(rows) // 2, len(rows) - 1})]
    cases = []
    for row in selected:
        cutoff = pd.Timestamp(row["cutoff"])
        history = product.frame.loc[product.frame.timestamp <= cutoff].tail(
            24 if any(key in SEQUENCE_KEYS for key in result["model_set"]) else 13)
        request = ForecastRequest(artifact_id=uuid4(), observations=history.to_dict("records"))
        cases.append({"observations": request.model_dump(mode="json")["observations"],
                      "expected": row["predictions"], "cutoff": cutoff.isoformat(),
                      "target_time": row["target_time"]})
    signature = ModelSignature(
        inputs=Schema([ColSpec("datetime", "timestamp"), *[
            ColSpec("double", name) for name in
            ("wind_power", "wind_speed", "humidity", "temperature")]]),
        outputs=Schema([ColSpec("datetime", "cutoff"), ColSpec("datetime", "target_time"),
                        ColSpec("double", "prediction")]),
    )
    runtime = inference_runtime()
    requirements = [f"{name}=={value}" for name, value in runtime["packages"].items()]
    registrations = []
    # 不把项目的CUDA/Agent开发环境自动打入CPU推理工件。
    with local_packaging_environment():
        for key in result["model_set"]:
            identity = uuid4()
            relative = f"{task_id}/{attempt_id}/models/{identity}"
            directory = confined_path(root, relative)
            directory.mkdir(parents=True, exist_ok=False)
            estimator = product.models.improved if key == "hist_gradient_boosting" else None
            candidate = key in product.models.candidates
            if candidate:
                estimator = product.models.candidates[key]
            sequence = key in SEQUENCE_KEYS
            extra = {}
            if sequence:
                from .sequence_predictor import SequenceHistoryPredictor
                checkpoint = directory / "state.pt"
                estimator.save(checkpoint)
                extra["artifacts"] = {"checkpoint": str(checkpoint)}
                predictor = SequenceHistoryPredictor()
            else:
                predictor = (CandidateHistoryPredictor(estimator) if candidate
                         else HistoryPredictor(key, estimator))
            package_version = (SEQUENCE_PACKAGE_VERSION if sequence else
                               CANDIDATE_PACKAGE_VERSION if candidate else PACKAGE_VERSION)
            runtime = inference_runtime(package_version)
            requirements = [f"{name}=={value}" for name, value in runtime["packages"].items()]
            training = product.models.candidate_details[key] if candidate else result["training"]
            mlflow.pyfunc.save_model(
                str(directory / "model"), python_model=predictor,
                signature=signature, input_example=history_frame(cases[0]["observations"]),
                pip_requirements=requirements, code_paths=[str(PACKAGE_ROOT)],
                **extra,
            )
            write_json(directory / "verification.json", {"cases": cases})
            # MLflow已关闭文件；补持久化再建立清单。数据库登记仍在全部校验之后。
            files = {}
            for item in sorted(directory.rglob("*")):
                if item.is_file():
                    with item.open("rb") as handle:
                        os.fsync(handle.fileno())
                    files[item.relative_to(directory).as_posix()] = sha256_file(item)
            manifest = PackageManifest(
                package_version=package_version, artifact_id=identity, run_id=result["run_id"],
                task_id=task_id, attempt_id=attempt_id, model_key=key,
                model_version=(key + "-v1" if candidate else result["model_version"]
                               if estimator is not None else "persistence-v1"),
                horizon_minutes=result["horizon_minutes"],
                feature_contract_version=SEQUENCE_VERSION if sequence else FEATURE_CONTRACT_VERSION,
                feature_columns=SEQUENCE_FEATURES if sequence else FEATURE_COLUMNS,
                input_contract={"min_observations": 24 if sequence else 13, "max_observations": 288,
                                "frequency_minutes": 5, "fields": list(history_frame(
                                    cases[0]["observations"]).columns)},
                preprocessing=training.get("preprocessing", "none_required"),
                postprocessing="nonnegative_clip" if estimator is not None else "identity",
                unit="MW_source_reported", clock="source_time_timezone_unknown",
                training_label_end=result["training"]["train_target_end"],
                training=training if estimator is not None else {"fitted": False},
                frozen_spec=result["frozen_spec"], provenance=result["execution"],
                inference_code_sha256=inference_code_sha256(package_version), runtime=runtime, files=files,
                verification={"protocol": result["purpose"] + "-raw-history-v1",
                              "atol": 1e-3 if sequence else ATOL, "rtol": 1e-6 if sequence else RTOL,
                              "samples": len(cases), "cutoffs": [c["cutoff"] for c in cases]},
            ).model_dump(mode="json")
            write_json(directory / "manifest.json", manifest)
            registrations.append(PackageRegistration(
                artifact_id=identity, model_key=key, path=relative, manifest=manifest,
                manifest_sha256=sha256_file(directory / "manifest.json"),
            ))
    return registrations


def predict_package(root: Path, registration: PackageRegistration, request: ForecastRequest):
    if request.artifact_id != registration.artifact_id:
        raise PackageError("artifact_identity_mismatch")
    manifest = checked_manifest(root, registration)
    if len(request.observations) < manifest.input_contract["min_observations"]:
        raise PackageError("artifact_history_insufficient")
    directory = confined_path(root, registration.path)
    try:
        model = mlflow.pyfunc.load_model(str(directory / "model"))
        values = model.predict(history_frame(request.observations))
        if len(values) != 1 or not np.isfinite(values["prediction"].iloc[0]):
            raise PackageError("artifact_prediction_invalid")
        return manifest, values.iloc[0]
    except PackageError:
        raise
    except Exception as exc:
        raise PackageError("artifact_load_or_prediction_failed") from exc


def verify_packages(root: Path, registrations: list[PackageRegistration]) -> list[dict]:
    evidence = []
    for registration in registrations:
        manifest = checked_manifest(root, registration)
        directory = confined_path(root, registration.path)
        cases = json.loads((directory / "verification.json").read_text(encoding="utf-8"))["cases"]
        errors = []
        for case in cases:
            request = ForecastRequest(artifact_id=registration.artifact_id,
                                      observations=case["observations"])
            _, prediction = predict_package(root, registration, request)
            expected = case["expected"][registration.model_key]
            np.testing.assert_allclose(prediction["prediction"], expected,
                                       atol=manifest.verification["atol"], rtol=manifest.verification["rtol"])
            assert pd.Timestamp(prediction["cutoff"]) == pd.Timestamp(case["cutoff"])
            assert pd.Timestamp(prediction["target_time"]) == pd.Timestamp(case["target_time"])
            errors.append(abs(float(prediction["prediction"]) - expected))
        evidence.append({"artifact_id": str(registration.artifact_id),
                         "manifest_sha256": registration.manifest_sha256,
                         "model_key": registration.model_key, "samples": len(cases),
                         "max_absolute_difference": max(errors), "atol": manifest.verification["atol"],
                         "rtol": manifest.verification["rtol"]})
    return evidence


def verify_fresh_process(root: Path, registrations: list[PackageRegistration]) -> list[dict]:
    """训练进程不代替加载验收；子进程禁止连接网络和重新fit。"""
    def failed(code, stdout="", stderr=""):
        def bounded(value):
            return (value.decode("utf-8", errors="replace") if isinstance(value, bytes)
                    else value or "")[-12000:]

        # 诊断放在attempt目录，不改已建立清单的模型包；保留原因但不向HTTP泄露堆栈。
        if registrations:
            directory = confined_path(root, registrations[0].path).parent.parent
            write_json(directory / "model-verification-failure.json",
                       {"reason": code, "stdout": bounded(stdout), "stderr": bounded(stderr)})
        raise PackageError(code)

    try:
        child = subprocess.run(
            [sys.executable, "-m", "power_forecast_service.forecasting.bundles", str(root)],
            input=json.dumps([item.model_dump(mode="json") for item in registrations]),
            capture_output=True, text=True, timeout=90,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except subprocess.TimeoutExpired as exc:
        failed("artifact_verification_timeout", exc.stdout, exc.stderr)
    if child.returncode:
        failed("artifact_fresh_process_verification_failed", child.stdout, child.stderr)
    try:
        return json.loads(child.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        failed("artifact_verification_response_invalid", child.stdout, child.stderr)


if __name__ == "__main__":
    import socket
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def forbidden(*args, **kwargs):
        raise RuntimeError("Verification must not train or connect to external services")

    socket.socket.connect = forbidden
    HistGradientBoostingRegressor.fit = forbidden
    Ridge.fit = forbidden
    Pipeline.fit = forbidden
    StandardScaler.fit = forbidden
    records_raw = json.load(sys.stdin)
    if any(item.get("manifest", {}).get("package_version") == SEQUENCE_PACKAGE_VERSION
           for item in records_raw):
        from .sequence_model import SequenceRegressor
        SequenceRegressor.fit = forbidden
    records = [PackageRegistration.model_validate(item) for item in records_raw]
    print(json.dumps(verify_packages(Path(sys.argv[1]), records), allow_nan=False))
