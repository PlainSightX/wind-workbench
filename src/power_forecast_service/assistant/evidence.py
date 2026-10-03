"""投影已登记结果并计算同合同差值；不加载模型、不训练、不修改发布状态。"""

from datetime import datetime, timezone
import hashlib
import json

from sqlalchemy import select

from ..storage.models import Run, ImportedRun, ImportedArtifact
from .contracts import AssistantError, Fact


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


async def get_results(sessions, contexts, executor=None):
    facts, records, scopes = [], [], []
    async with sessions() as session:
        # 明确只读事务，未来工具若误加UPDATE也会由PG拒绝。
        from sqlalchemy import text
        await session.execute(text("SET TRANSACTION READ ONLY"))
        for index, ref in enumerate(contexts):
            prefix = f"c{index}"
            if ref.kind == "engie_import":
                row = await session.get(ImportedRun, ref.id)
                if row is None:
                    raise AssistantError("context_not_found")
                artifacts = (await session.scalars(select(ImportedArtifact).where(
                    ImportedArtifact.import_id == row.id, ImportedArtifact.status == "ready"
                ).order_by(ImportedArtifact.family))).all()
                if not artifacts or (ref.model and ref.model not in {a.family for a in artifacts}):
                    raise AssistantError("model_not_in_context")
                manifest = row.manifest
                if {a.family for a in artifacts} != set(manifest.get("families", ["persistence", "ridge", "lightgbm"])):
                    raise AssistantError("context_incomplete")
                coverage = [a.manifest["coverage"] for a in artifacts]
                if any(c != coverage[0] for c in coverage[1:]):
                    raise AssistantError("context_coverage_conflict")
                final = manifest.get("scope") == "final_2015"
                aggregation = "四台合计、六时距、四季度等权" if final else "四台合计、六时距、单季度"
                scope = "engie_final" if final else "engie_development"
                source_hash = manifest.get("result_sha256", row.source_sha256)
                data = {"kind": ref.kind, "id": str(row.id), "scope": scope,
                        "protocol": manifest["protocol_version"], "quarter": row.quarter,
                        "aggregation": aggregation, "unit": "kW", "selected_model": ref.model,
                        "models": {a.family: a.manifest["metrics"] for a in artifacts},
                        "coverage": artifacts[0].manifest["coverage"],
                        "training_label_available": manifest["training_label_available"],
                        "evaluation": manifest.get("evaluation", {})}
                data.update(result_sha256=manifest.get("result_sha256"), protocol_sha256=manifest.get("protocol_sha256"))
                data["interpretation_constraints"] = {
                    "input_domain": "原始功率允许负值。缺失值未知，补零虚构观测并改变目标，不能断言误差方向或系统性压低。",
                    "coverage": "输入拒绝与输出后无法评分不同。标签缺测、冲突或超出评价边界均可能不可评分；历史结果不能笼统说等标签到期就能评分。末六个越界只是其中一部分，不能解释整个差额。",
                    "publication": "迟到结果保留expired持久登记，只拒绝有效发布，不是不登记。幂等是同一请求身份唯一有效结果。",
                    "comparison": "只陈述数字支持的高低/差值。负偏差减小不等于消失或接近零，不擅自作量级相近判断。",
                }
                if final:
                    data["adoption_boundary"] = "开发采用门未过，默认持续性。2015最终结果不得用于重新决定开发采用门，区间也不是采用门。"
                def add(key, label, value, unit="", pointer="", artifact=None):
                    facts.append(Fact(id=f"{prefix}.{key}", label=label, value=value, unit=unit,
                        object_id=str(row.id), artifact_id=artifact, aggregation=aggregation,
                        source_sha256=source_hash, pointer=pointer or key))
                for a in artifacts:
                    for metric, value in a.manifest["metrics"].items():
                        if metric in {"mae", "rmse", "bias"}:
                            add(f"{a.family}.{metric}", f"{a.family} {metric.upper()}", value, "kW",
                                f"/artifacts/{a.id}/manifest/metrics/{metric}", str(a.id))
                for key in ("planned", "input_valid", "output_count", "scoreable"):
                    add(key, {"planned":"计划起报", "input_valid":"合法输入", "output_count":"实际输出", "scoreable":"可评分起报"}[key], data["coverage"][key], "次", f"/coverage/{key}")
                timestamp = datetime.fromisoformat(data["training_label_available"]).astimezone(timezone.utc).isoformat()
                add("training_label_available", "最终训练标签可得截止", timestamp, "UTC", "/training_label_available")
                if final:
                    evaluation = manifest["evaluation"]
                    add("adopted", "开发采用门通过", evaluation["development_adoption_gate_passed"], pointer="/evaluation/development_adoption_gate_passed")
                    ci = evaluation["bootstrap"]["seven_day"]["families"]["lightgbm_l1_shrink"]["gain_percent_95ci"]
                    for suffix, value in zip(("low", "high"), ci):
                        add(f"gain_ci_{suffix}", f"七天配对时间块MAE改善区间{suffix}", value, "%", f"/evaluation/bootstrap/seven_day/families/lightgbm_l1_shrink/gain_percent_95ci/{0 if suffix == 'low' else 1}")
            else:
                row = await session.get(Run, ref.id)
                if row is None:
                    raise AssistantError("context_not_found")
                result = row.result
                metrics = result.get("metrics", {})
                if ref.model and ref.model not in metrics:
                    raise AssistantError("model_not_in_context")
                scoring = result.get("scoring", {})
                # 复用既有评分校验，不能绕过逐点证据与摘要一致性。
                import asyncio
                from ..experiments.comparison import compare_results as compare_q1
                for model in metrics:
                    checked = await asyncio.get_running_loop().run_in_executor(
                        executor, compare_q1, row.id, result, model, row.id, result, model)
                    if checked.status != "comparable":
                        raise AssistantError("context_unverified")
                # docs/results/wind-sequence-round4/summary.json的正式结果身份。
                report_matches = (str(row.id) == "e0f98578-fbca-49a1-9269-07be2ccc2b57" and
                    result.get("frozen_spec", {}).get("final_protocol_id") == "177366d00abe00776bd9fbad65fedef6912aec00a06f67bc3f0d362dc94a329e")
                scope = "q1" if report_matches else "q1_unbound"
                aggregation = "Q1单点、同运行评分集"
                data = {"kind": ref.kind, "id": str(row.id), "scope": scope, "aggregation": aggregation,
                    "unit": scoring.get("unit", "数据来源标注单位"), "selected_model": ref.model,
                    "protocol": result.get("feature_contract_version"), "models": metrics,
                    "purpose": result.get("purpose"), "evaluation_split": result.get("evaluation_split"),
                    "horizon_minutes": result.get("horizon_minutes"),
                    "final_protocol_id": result.get("frozen_spec", {}).get("final_protocol_id"),
                    "comparison_identity": {k: scoring.get(k) for k in ("input_sha256", "target", "unit", "clock", "evaluation_split", "horizon_minutes", "split_version", "metric_version", "samples_sha256")}}
                for model, values in metrics.items():
                    for metric in ("mae", "rmse"):
                        if metric in values:
                            facts.append(Fact(id=f"{prefix}.{model}.{metric}", label=f"{model} {metric.upper()}",
                                value=values[metric], unit=data["unit"], object_id=str(row.id), aggregation=aggregation,
                                source_sha256=row.artifact_sha256, pointer=f"/result/metrics/{model}/{metric}"))
            records.append(data)
            scopes.append(scope)
    from .stages import bind_stages
    return bind_stages({"facts": [f.model_dump() for f in facts], "records": records, "scopes": sorted(set(scopes))})


def compare_results(evidence):
    """同一导入/运行内比较；跨运行必须具有完整且相同的评分身份。"""
    facts = list(evidence["facts"])
    for i, record in enumerate(evidence["records"]):
        models = record["models"]
        baseline = models.get("persistence")
        if not baseline:
            continue
        for model, values in models.items():
            for metric in ("mae", "rmse"):
                if metric not in values or not baseline.get(metric):
                    continue
                source = next(f for f in facts if f["id"] == f"c{i}.{model}.{metric}")
                for suffix, value, unit in (
                    ("delta", values[metric] - baseline[metric], source["unit"]),
                    ("gain", 100 * (1 - values[metric] / baseline[metric]), "%"),
                ):
                    facts.append({**source, "id": f"c{i}.{model}.{metric}_{suffix}",
                        "label": f"{model} 相对持续性{metric.upper()}{'差值(候选减基线)' if suffix == 'delta' else '改善'}",
                        "value": value, "unit": unit, "derived": True,
                        "pointer": f"computed:{source['pointer']};baseline=persistence;formula={'candidate-baseline' if suffix == 'delta' else '100*(1-candidate/baseline)'}"})
    records = evidence["records"]
    comparable = len(records) == 1
    if len(records) == 2:
        a, b = records
        comparable = a["kind"] == b["kind"] and a["id"] == b["id"]
        if a["kind"] == b["kind"] == "q1_run":
            identity = a["comparison_identity"]
            comparable = all(v is not None for v in identity.values()) and identity == b["comparison_identity"]
    return {**evidence, "facts": facts, "cross_context_comparable": comparable,
        "comparison_note": "只在同一对象内计算相对持续性差值。跨合同不可比较。" if not comparable else "同合同结果；均值差不证明因果或泛化。"}
