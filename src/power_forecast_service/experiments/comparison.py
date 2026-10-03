"""比较已登记的实验结果；可比较与可复现是不同结论。"""

from math import isclose
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError

from ..forecasting.scoring import ScoringEvidence, metrics_for_rows
from ..forecasting.development_protocol import RIDGE_ALPHAS


def selected_model_version(result: dict, model: str):
    """旧聚合字段只命名HGB；按所选模型解析，不改写已经登记的运行记录。"""
    if model in result.get("model_versions", {}):
        return result["model_versions"][model]
    if model == "persistence":
        return "persistence-v1"
    if model == "hist_gradient_boosting":
        return result.get("model_version")
    frozen = result.get("frozen_spec", {})
    if (frozen.get("spec_version") == "experiment-v3-fixed-q1"
            and frozen.get("candidate_key") == model and model in (*RIDGE_ALPHAS, "hgb_delta")):
        return model + "-v1"
    return None


class ComparisonSide(BaseModel):
    run_id: UUID
    model: str
    metrics: dict[str, float | int] | None = None
    context: dict = Field(default_factory=dict)


class ComparisonResult(BaseModel):
    status: Literal["comparable", "not_comparable"]
    left: ComparisonSide
    right: ComparisonSide
    reasons: list[str]
    warnings: list[str]
    samples_sha256: str | None = None
    delta: dict[str, float] | None = None
    interpretation: str = "差值为右侧减左侧；MAE/RMSE 越小越好。相同评分条件不证明因果或跨平台复现。"


def compare_results(
    left_id: UUID, left_result: dict, left_model: str,
    right_id: UUID, right_result: dict, right_model: str,
) -> ComparisonResult:
    sides = []
    evidence = []
    reasons = []
    warnings = []
    for label, run_id, result, model in (
        ("left", left_id, left_result, left_model),
        ("right", right_id, right_result, right_model),
    ):
        side = ComparisonSide(
            run_id=run_id, model=model,
            context={key: result.get(key) for key in (
                "feature_contract_version", "determinism", "execution", "split", "evaluation_split"
            )},
        )
        side.context["model_version"] = selected_model_version(result, model)
        sides.append(side)
        if "scoring" not in result:
            reasons.append(f"{label}:scoring_evidence_missing")
            continue
        try:
            scores = ScoringEvidence.model_validate(result["scoring"])
        except ValidationError:
            reasons.append(f"{label}:scoring_evidence_invalid")
            continue
        evidence.append(scores)
        if scores.input_sha256 != result.get("input_file_sha256"):
            reasons.append(f"{label}:input_identity_mismatch")
        if (
            (result.get("purpose"), scores.evaluation_split) not in (("development", "validation"), ("final_evaluation", "test"))
            or result.get("horizon_steps") != scores.horizon_minutes / 5
            or any(result.get(key) != getattr(scores, key) for key in (
                "evaluation_split", "horizon_minutes", "split_version"
            ))
        ):
            reasons.append(f"{label}:scoring_protocol_mismatch")
        if model not in scores.rows[0].predictions:
            reasons.append(f"{label}:model_not_scored")
            continue
        if scores.evaluation_split == "test" and not result.get("frozen_spec", {}).get("final_protocol_id"):
            reasons.append(f"{label}:final_protocol_missing")
        measured = metrics_for_rows(scores.rows, model)
        side.metrics = measured
        summaries = result.get("metrics")
        summary = summaries.get(model, {}) if isinstance(summaries, dict) else {}
        try:
            matching = summary["samples"] == measured["samples"] and all(
                isclose(summary[key], measured[key], rel_tol=1e-10, abs_tol=1e-10)
                for key in ("mae", "rmse")
            )
        except (KeyError, TypeError, ValueError):
            matching = False
        if not matching:
            reasons.append(f"{label}:metric_summary_mismatch")

    if len(evidence) == 2:
        a, b = evidence
        # 训练方法/seed/特征可不同，这是对照对象；评分协议与真实答案必须一致。
        for field in (
            "input_sha256", "target", "unit", "clock", "evaluation_split", "horizon_minutes",
            "split_version", "metric_version", "samples_sha256",
        ):
            if getattr(a, field) != getattr(b, field):
                reasons.append(f"different:{field}")
    if left_result.get("execution") != right_result.get("execution"):
        warnings.append("execution_context_differs:运行环境或源码不同，差值不能单独归因于模型。")
    if not left_result.get("execution") or not right_result.get("execution"):
        warnings.append("execution_context_missing:无法核验完整执行来源。")
    return ComparisonResult(
        status="not_comparable" if reasons else "comparable",
        left=sides[0], right=sides[1], reasons=reasons, warnings=warnings,
        samples_sha256=evidence[0].samples_sha256 if not reasons else None,
        delta={key: sides[1].metrics[key] - sides[0].metrics[key] for key in ("mae", "rmse")}
        if not reasons else None,
    )
