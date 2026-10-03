"""实验输入、冻结配置与回执类型；API 和 worker 共用，不引入训练实现。"""

import hashlib
import json
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, model_validator

from ..forecasting.development_protocol import (
    CandidateKey, INPUT_SHA256, PROTOCOL_VERSION, RIDGE_ALPHAS,
    candidate_model_set, protocol_document,
)

from ..forecasting.spec import (
    DEFAULT_TRAINING_POLICY,
    EXPERIMENT_SPEC_VERSION,
    FEATURE_CONTRACT_VERSION,
    HORIZON_STEPS,
    MODEL_SEED,
    SPLIT_VERSION,
    TrainingPolicy,
    legacy_execution_parameters,
    legacy_hgb_parameters,
    training_parameters,
)
from ..settings import Settings
from ..forecasting.sequence_protocol import (
    CONFIG, EVALUATION_VERSION, SEQUENCE_VERSION, sequence_model_set, sequence_protocol,
)
from ..forecasting.final_protocol import frozen_selection
from ..storage.artifacts import sha256_file

DATASET_ID = "wind-2019-q1"


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_id: Literal["wind-2019-q1"] = DATASET_ID
    purpose: Literal["development", "final_evaluation"] = "development"
    training_policy: TrainingPolicy = DEFAULT_TRAINING_POLICY
    candidate_key: CandidateKey = "none"
    sequence_key: Literal["none", "transformer_direct", "transformer_delta"] = "none"

    @model_validator(mode="after")
    def candidate_policy(self):
        if (self.candidate_key != "none" or self.sequence_key != "none" or self.purpose == "final_evaluation") and self.training_policy != "fixed_iterations":
            raise ValueError("candidate_requires_fixed_iterations")
        if self.sequence_key != "none" and self.candidate_key != "none":
            raise ValueError("sequence_already_includes_ridge")
        if self.purpose == "final_evaluation" and (self.sequence_key != "none" or self.candidate_key != "none"):
            raise ValueError("final_evaluation_uses_frozen_selection")
        return self


class TaskReceipt(BaseModel):
    task_id: UUID
    status: str
    status_url: str
    source_task_id: UUID | None


class TaskConflict(Exception):
    pass


def fingerprint(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def submission_fingerprint(request: ExperimentRequest) -> str:
    """默认策略保持旧POST{}身份；省略与显式默认等价，不回写旧请求。"""
    payload = {"dataset_id": request.dataset_id, "purpose": request.purpose}
    if request.training_policy != "auto_early_stopping":
        payload["training_policy"] = request.training_policy
    if request.candidate_key != "none":
        payload["candidate_key"] = request.candidate_key
    if request.sequence_key != "none":
        payload["sequence_key"] = request.sequence_key
    return fingerprint({"action": "submit", "request": payload})


def freeze_spec(request: ExperimentRequest, settings: Settings) -> dict:
    """冻结实际数据和模型配置；后续默认值不能悄悄改变已接受的实验。"""
    frozen = {
        **request.model_dump(exclude={"candidate_key", "sequence_key"}),
        "spec_version": EXPERIMENT_SPEC_VERSION,
        "input_sha256": sha256_file(settings.data_path),
        "split_version": SPLIT_VERSION,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "model_set": ["persistence", "hist_gradient_boosting"],
        "horizon_steps": HORIZON_STEPS,
        "random_seed": MODEL_SEED,
        "hgb": training_parameters(request.training_policy),
    }
    if request.sequence_key != "none" or request.purpose == "final_evaluation":
        if frozen["input_sha256"] != INPUT_SHA256:
            raise TaskConflict("candidate_dataset_not_registered")
        key = request.sequence_key
        if request.purpose == "final_evaluation":
            try:
                selection, identity = frozen_selection()
            except ValueError as exc:
                raise TaskConflict(str(exc)) from exc
            key = selection["sequence_key"]
            frozen.update(final_selection=selection, final_protocol_id=identity)
        frozen.update(spec_version="experiment-v4-sequence", sequence_key=key,
                      model_set=sequence_model_set(key), feature_contract_version=SEQUENCE_VERSION,
                      split_version=EVALUATION_VERSION + (":final-refit-v1" if request.purpose == "final_evaluation" else ":main"),
                      evaluation_protocol=sequence_protocol(), sequence_recipe=dict(CONFIG))
        frozen["model_feature_contracts"] = {name: SEQUENCE_VERSION if name == key else FEATURE_CONTRACT_VERSION
                                            for name in frozen["model_set"]}
    elif request.candidate_key != "none":
        if frozen["input_sha256"] != INPUT_SHA256:
            raise TaskConflict("candidate_dataset_not_registered")
        frozen.update(
            spec_version="experiment-v3-fixed-q1", candidate_key=request.candidate_key,
            model_set=candidate_model_set(request.candidate_key),
            split_version=PROTOCOL_VERSION + ":main", evaluation_protocol=protocol_document(),
            candidate_recipe=({"alpha": RIDGE_ALPHAS[request.candidate_key], "solver": "svd",
                               "preprocessing": "standard_scaler_train_only", "target": "direct_power"}
                              if request.candidate_key in RIDGE_ALPHAS else
                              {"target": "increment_from_current", "preprocessing": "none_required"}),
        )
    return frozen


def validate_spec(spec: dict, settings: Settings) -> dict:
    """验证后返回可执行配置；旧精确合同可解析，但永不原地升级历史记录。"""
    try:
        if not isinstance(spec, dict):
            raise TaskConflict("experiment_contract_changed")
        version = spec.get("spec_version", "legacy-v1")
        if version not in ("legacy-v1", EXPERIMENT_SPEC_VERSION, "experiment-v3-fixed-q1", "experiment-v4-sequence"):
            raise TaskConflict("experiment_contract_version_unsupported")
        request = ExperimentRequest(
            dataset_id=spec["dataset_id"], purpose=spec["purpose"],
            training_policy=(
                "auto_early_stopping" if version == "legacy-v1" else spec["training_policy"]
            ),
            candidate_key=spec["candidate_key"] if version == "experiment-v3-fixed-q1" else "none",
            sequence_key=(spec["sequence_key"] if version == "experiment-v4-sequence" and spec["purpose"] == "development" else "none"),
        )
        resolved = freeze_spec(request, settings)
        expected = dict(resolved)
        if version == "legacy-v1":
            # 若当前实现已不支持历史隐含语义，明确拒绝而不是给旧任务换策略。
            if fingerprint(resolved["hgb"]) != fingerprint(legacy_execution_parameters()):
                raise TaskConflict("legacy_training_policy_unsupported")
            expected.pop("spec_version")
            expected.pop("training_policy")
            expected["hgb"] = legacy_hgb_parameters()
        # 规范JSON还区分False/0等类型；多字段、缺字段和任意参数都不是已支持合同。
        if fingerprint(spec) != fingerprint(expected):
            raise TaskConflict("experiment_contract_changed")
        return resolved
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise TaskConflict("experiment_contract_unavailable") from exc
