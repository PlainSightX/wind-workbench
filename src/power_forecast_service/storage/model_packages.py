"""模型包完整性与兼容边界；先校验数据库锚定的清单，再允许反序列化。"""

import hashlib
import json
import platform
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..forecasting.spec import FEATURE_CONTRACT_VERSION
from .artifacts import PACKAGE_ROOT, sha256_file

PACKAGE_VERSION = "history-pyfunc-v1"
CANDIDATE_PACKAGE_VERSION = "history-pyfunc-v2"
SEQUENCE_PACKAGE_VERSION = "history-pyfunc-v3"
INFERENCE_DEPENDENCIES = (
    "mlflow-skinny", "numpy", "pandas", "scikit-learn", "pydantic", "cloudpickle",
)
ATOL = RTOL = 1e-8


class PackageError(ValueError):
    """稳定原因码，不把内部路径或第三方反序列化异常回传给浏览器。"""


class PackageManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    package_version: Literal["history-pyfunc-v1", "history-pyfunc-v2", "history-pyfunc-v3"]
    artifact_id: UUID
    run_id: UUID
    task_id: UUID
    attempt_id: UUID
    model_key: Literal["persistence", "hist_gradient_boosting", "ridge_0_1", "ridge_1", "ridge_10", "hgb_delta", "transformer_direct", "transformer_delta"]
    model_version: str
    horizon_minutes: Literal[60]
    feature_contract_version: str
    feature_columns: list[str]
    input_contract: dict
    preprocessing: Literal["none_required", "standard_scaler_train_only", "sequence_train_only_standardization"]
    postprocessing: Literal["identity", "nonnegative_clip"]
    unit: Literal["MW_source_reported"]
    clock: Literal["source_time_timezone_unknown"]
    training_label_end: datetime
    training: dict
    frozen_spec: dict
    provenance: dict
    inference_code_sha256: str
    runtime: dict
    files: dict[str, str]
    verification: dict


class PackageRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_id: UUID
    model_key: str
    path: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest: dict


def inference_code_sha256(package_version=PACKAGE_VERSION) -> str:
    """只绑定影响预测的实现，不让无关API修改废掉旧模型。"""
    digest = hashlib.sha256()
    names = ["forecasting/predictor.py", "forecasting/features.py",
             "forecasting/contracts.py", "forecasting/spec.py"]
    if package_version == CANDIDATE_PACKAGE_VERSION:
        names += ["forecasting/candidate_predictor.py", "forecasting/candidate_models.py",
                  "forecasting/development_protocol.py"]
    elif package_version == SEQUENCE_PACKAGE_VERSION:
        names += ["forecasting/sequence_protocol.py", "forecasting/sequence_data.py",
                  "forecasting/sequence_model.py", "forecasting/sequence_predictor.py"]
    elif package_version != PACKAGE_VERSION:
        raise PackageError("artifact_incompatible")
    for name in names:
        digest.update(name.encode())
        digest.update(bytes.fromhex(sha256_file(PACKAGE_ROOT / name)))
    return digest.hexdigest()


def inference_runtime(package_version=PACKAGE_VERSION) -> dict:
    return {"python": platform.python_version(),
            "packages": {name: version(name) for name in (*INFERENCE_DEPENDENCIES,
                         *(("torch",) if package_version == SEQUENCE_PACKAGE_VERSION else ()))}}


def confined_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise PackageError("artifact_path_invalid")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise PackageError("artifact_path_invalid")
    return resolved


def checked_manifest(root: Path, registration: PackageRegistration) -> PackageManifest:
    """哈希不是信任来源；expected hash必须来自本库登记或本次可信worker。"""
    directory = confined_path(root, registration.path)
    try:
        manifest_path = directory / "manifest.json"
        if manifest_path.is_symlink() or sha256_file(manifest_path) != registration.manifest_sha256:
            raise PackageError("artifact_integrity_failed")
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and raw.get("package_version") not in (PACKAGE_VERSION, CANDIDATE_PACKAGE_VERSION, SEQUENCE_PACKAGE_VERSION):
            raise PackageError("artifact_incompatible")
        manifest = PackageManifest.model_validate(raw)
        if manifest.model_dump(mode="json") != registration.manifest:
            raise PackageError("artifact_identity_mismatch")
        if (manifest.artifact_id != registration.artifact_id
                or manifest.model_key != registration.model_key):
            raise PackageError("artifact_identity_mismatch")
        expected_path = f"{manifest.task_id}/{manifest.attempt_id}/models/{manifest.artifact_id}"
        if registration.path != expected_path:
            raise PackageError("artifact_identity_mismatch")
        # 拒绝附加文件和链接，防止清单只验证部分载荷后加载另一份代码/权重。
        actual_files = set()
        for item in directory.rglob("*"):
            if item.is_symlink():
                raise PackageError("artifact_integrity_failed")
            if item.is_file() and item != manifest_path:
                actual_files.add(item.relative_to(directory).as_posix())
        if actual_files != set(manifest.files) or not manifest.files:
            raise PackageError("artifact_integrity_failed")
        for name, digest in manifest.files.items():
            if sha256_file(confined_path(directory, name)) != digest:
                raise PackageError("artifact_integrity_failed")
        from ..forecasting.sequence_protocol import SEQUENCE_KEYS, SEQUENCE_VERSION, SEQUENCE_FEATURES
        sequence = manifest.model_key in SEQUENCE_KEYS
        expected_feature_version = SEQUENCE_VERSION if sequence else FEATURE_CONTRACT_VERSION
        if (manifest.feature_contract_version != expected_feature_version
                or manifest.inference_code_sha256 != inference_code_sha256(manifest.package_version)
                or manifest.runtime != inference_runtime(manifest.package_version)):
            raise PackageError("artifact_incompatible")
        from ..forecasting.features import FEATURE_COLUMNS

        if manifest.feature_columns != (SEQUENCE_FEATURES if sequence else FEATURE_COLUMNS):
            raise PackageError("artifact_incompatible")
        from ..forecasting.development_protocol import RIDGE_ALPHAS

        candidate = manifest.model_key in (*RIDGE_ALPHAS, "hgb_delta")
        if (candidate != (manifest.package_version == CANDIDATE_PACKAGE_VERSION)
                or sequence != (manifest.package_version == SEQUENCE_PACKAGE_VERSION)
                or manifest.preprocessing != ("sequence_train_only_standardization" if sequence else "standard_scaler_train_only"
                                               if manifest.model_key in RIDGE_ALPHAS else "none_required")):
            raise PackageError("artifact_incompatible")
        sequence_run = manifest.frozen_spec.get("spec_version") == "experiment-v4-sequence"
        if candidate and ((not sequence_run and manifest.frozen_spec.get("candidate_key") != manifest.model_key)
                          or manifest.training.get("model_key") != manifest.model_key):
            raise PackageError("artifact_identity_mismatch")
        if candidate:
            target = "direct_power" if manifest.model_key in RIDGE_ALPHAS else "increment_from_current"
            if (manifest.training.get("target_representation") != target
                    or (not sequence_run and manifest.frozen_spec.get("candidate_recipe", {}).get("target") != target)
                    or manifest.model_version != manifest.model_key + "-v1"):
                raise PackageError("artifact_incompatible")
        if sequence and (manifest.frozen_spec.get("sequence_key") != manifest.model_key
                         or manifest.training.get("model_key") != manifest.model_key
                         or manifest.training.get("config") != manifest.frozen_spec.get("sequence_recipe")
                         or manifest.model_version != manifest.model_key + "-v1"):
            raise PackageError("artifact_incompatible")
        expected_input = {"min_observations": 24 if sequence else 13, "max_observations": 288,
                          "frequency_minutes": 5,
                          "fields": ["timestamp", "wind_power", "wind_speed", "humidity", "temperature"]}
        if (manifest.input_contract != expected_input
                or manifest.postprocessing != ("identity" if manifest.model_key == "persistence"
                                                else "nonnegative_clip")
                or manifest.frozen_spec.get("horizon_steps") != 12
                or manifest.frozen_spec.get("model_feature_contracts", {}).get(manifest.model_key,
                    manifest.frozen_spec.get("feature_contract_version")) != expected_feature_version):
            raise PackageError("artifact_incompatible")
        return manifest
    except PackageError:
        raise
    except FileNotFoundError as exc:
        raise PackageError("artifact_missing") from exc
    except (ValidationError, json.JSONDecodeError, UnicodeError) as exc:
        raise PackageError("artifact_manifest_invalid") from exc
