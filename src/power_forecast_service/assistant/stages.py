"""已冻结报告的阶段投影与定向回答约束；不重算训练，也不判断任意自然语言。"""

import json
from pathlib import Path
import re

from .contracts import AssistantError, Fact


def stage_sources():
    # 镜像只携带这份最小投影；原报告仍是权威，定向测试核对其hash、指针与值。
    return json.loads(Path(__file__).with_name("stage_sources.json").read_text(encoding="utf-8"))


def bind_stages(evidence):
    sources = stage_sources()
    for index, record in enumerate(evidence["records"]):
        prefix = f"c{index}"
        scope = record["scope"]
        if scope == "q1":
            source = sources["q1"]
            if record["id"] != source["object_id"] or record["final_protocol_id"] != source["final_protocol_id"]:
                raise AssistantError("context_stage_conflict")
            for model, value in source["mean_mae"].items():
                evidence["facts"].append(Fact(id=f"{prefix}.development.{model}.mae",
                    label=f"开发 {model} MAE", value=value, unit=record["unit"], object_id=record["id"],
                    aggregation="Q1三个开发窗口等权平均", stage="development_selection",
                    source_sha256=source["selection_source"]["sha256"], source_path=source["selection_source"]["path"],
                    pointer=f"/mean_mae/{model}").model_dump())
            record["pre_selection_evidence"] = source["pre_selection_evidence"]
            record["final_holdout_evidence"] = source["final_holdout_evidence"]
        elif scope == "engie_final":
            source = sources["engie"]
            # 任意单季度导入不能继承A2汇总；只有同一冻结A3结果/协议拥有该来源关联。
            if (record["result_sha256"], record["protocol_sha256"]) != (source["result_sha256"], source["protocol_sha256"]):
                continue
            if record["evaluation"].get("development_adoption_gate_passed") is not source["development_adopted"]:
                raise AssistantError("context_stage_conflict")
            for key, label, value, source_key, pointer, stage in (
                ("development.lightgbm_l1_shrink.mae_gain", "开发共享收缩MAE改善", source["development_mae_gain_percent"], "development_source", "/summary/adoption/lightgbm_l1_shrink/equal_quarter_mae_gain_percent", "development_summary"),
                ("development.mae_gain_gate", "开发MAE改善采用门", source["minimum_mae_gain_percent"], "gate_source", "/protocol/adoption_gates/minimum_equal_quarter_mae_gain_percent", "adoption_gate"),
            ):
                evidence["facts"].append(Fact(id=f"{prefix}.{key}", label=label, value=value, unit="%", object_id=record["id"],
                    aggregation="四台合计、六时距、2014三个开发季度等权", stage=stage,
                    source_sha256=source[source_key]["sha256"], source_path=source[source_key]["path"], pointer=pointer).model_dump())
            record["development_evidence"] = source["development_evidence"]
            record["final_holdout_evidence"] = source["final_holdout_evidence"]
        else:
            continue
        record["stage_boundary"] = "事前开发选型/采用决定先于正式留出；正式结果只作事后核验，不得反向充当选择或采用依据。"
        for fact in evidence["facts"]:
            if fact["id"].startswith(prefix + ".") and fact["stage"] == "selected_result":
                fact["stage"] = "adoption_gate" if fact["id"] == f"{prefix}.adopted" else "final_holdout"
                if fact["stage"] == "adoption_gate":
                    fact["aggregation"] = "开发采用决定（2014三开发季度）"
                if fact["stage"] == "final_holdout":
                    fact["label"] = "正式 " + fact["label"]
    return evidence


def prepare_stage_requirements(evidence, documents, question):
    """只识别本轮有证据的选型/开发-最终问题；无关问答保持原路径。"""
    options, required, seen_objects = [], [], set()
    query = question.lower()
    causal = any(word in query for word in ("为什么", "为何", "依据", "理由", "原因", "选型", "事前", "当时"))
    gain = any(word in query for word in ("改善", "提升", "收益")) and not (
        any(word in query for word in ("rmse", "偏差", "bias")) and "mae" not in query)
    for index, record in enumerate(evidence["records"]):
        if "stage_boundary" not in record or record["id"] in seen_objects:
            continue
        # 同一对象可选两个模型；阶段仍取第一份完整投影，避免c1覆盖c0规则。
        seen_objects.add(record["id"])
        prefix = f"c{index}."
        facts = [f for f in evidence["facts"] if f["id"].startswith(prefix)]
        dev = [f["id"] for f in facts if f["stage"] in {"development_selection", "development_summary"}]
        final = [f["id"] for f in facts if f["stage"] == "final_holdout"]
        def add(role, label, allowed, needed, citations, needed_citations, mandatory):
            option = {"object_id": record["id"], "role": role, "label": label,
                "allowed_fact_ids": allowed, "required_fact_ids": needed,
                "allowed_citations": citations, "required_citations": needed_citations}
            options.append(option)
            if mandatory:
                required.append({"object_id": record["id"], "role": role})
        if record["scope"] == "q1":
            needed = [prefix + "development." + model + ".mae" for model in ("ridge_0_1", "persistence", "transformer_delta")]
            selection_question = causal and any(word in query for word in ("采用", "选择", "选型", "推荐", "放弃", "不选", "没选", "选ridge", "选用")) and (
                any(word in query for word in ("模型", "ridge", "transformer", "选型", "推荐")) and
                not any(word in query for word in ("训练损失", "损失函数", "mse", "优化器", "学习率")))
            add("selection_basis", "事前开发选型", dev, needed, record["pre_selection_evidence"], record["pre_selection_evidence"], selection_question)
        else:
            dev_question = gain and any(word in query for word in ("开发", "a2", "事前", "当时"))
            gate_question = any(word in query for word in ("采用门", "开发门", "采用门槛")) or (
                causal and any(word in query for word in ("默认", "不采用", "采用候选", "采用模型", "采用共享收缩")))
            # 事前门槛可用于解释开发改善，不等于允许正式结果或采用状态混入开发指标。
            thresholds = [f["id"] for f in facts if f["stage"] == "adoption_gate"
                          and f["id"] == prefix + "development.mae_gain_gate"]
            add("development_result", "开发评价（2014三季度汇总）", dev + thresholds, [prefix + "development.lightgbm_l1_shrink.mae_gain"],
                record["development_evidence"], record["development_evidence"], dev_question or gate_question)
            gates = [f["id"] for f in facts if f["stage"] == "adoption_gate"]
            add("adoption_decision", "开发采用决定（未通过，默认持久性）", dev + gates,
                [prefix + "development.mae_gain_gate", prefix + "adopted"], record["development_evidence"], record["development_evidence"], gate_question)
        final_question = gain and any(word in query for word in ("最终", "正式", "留出", "a3", "2015", "后来"))
        needed = [prefix + "lightgbm_l1_shrink.mae_gain"] if record["scope"] == "engie_final" else []
        add("final_holdout_result", "正式留出核验（不用于事前选择）", final, needed,
            record["final_holdout_evidence"], record["final_holdout_evidence"][:1], final_question)
    # 强制阶段证据优先进入同一版本上下文；不改语料、不借机会替换检索算法。
    if required:
        from .retrieval import corpus
        by_id = {d["id"]: d for d in corpus()["chunks"]}
        anchors = dict.fromkeys(c for option in options for c in option["allowed_citations"])
        if any(c not in by_id for c in anchors):
            raise AssistantError("corpus_stage_evidence_missing")
        documents = [by_id[c] for c in anchors] + [d for d in documents if d["id"] not in anchors]
    return {**evidence, "stage_options": options, "stage_requirements": required}, documents


def restrict_adoption_output(evidence, question):
    """可选 references 输出范围；完整事实/文档输入与默认旧合同均保持不变。"""
    objects = {item["object_id"] for item in evidence.get("stage_requirements", [])
               if item["role"] == "adoption_decision"}
    if not objects or any(word in question.lower() for word in (
            "区间", "置信", "不确定", "bootstrap", "confidence", "时间块")):
        return evidence
    options = []
    for option in evidence["stage_options"]:
        if option["object_id"] in objects and option["role"] == "final_holdout_result":
            # 只限制不相关的输出绑定；不让模型失去看到原始区间证据的机会。
            option = {**option, "allowed_fact_ids": [key for key in option["allowed_fact_ids"]
                                                      if ".gain_ci_" not in key]}
        options.append(option)
    return {**evidence, "stage_options": options}


def validate_stage_claims(draft, evidence):
    required = {(r["object_id"], r["role"]) for r in evidence.get("stage_requirements", [])}
    if not required and not draft.stage_claims:
        return
    options = {(r["object_id"], r["role"]): r for r in evidence.get("stage_options", [])}
    if required and (draft.status != "answered" or not draft.stage_claims):
        raise AssistantError("answer_stage_incomplete", "证据已齐备，必须分阶段直接回答并提供stage_claims。")
    compact = lambda text: re.sub(r"\s+", "", text)
    if draft.answer.strip() and compact(draft.answer) != compact("\n".join(c.text for c in draft.stage_claims)):
        raise AssistantError("answer_stage_body_mismatch", "answer必须依序等于stage_claims的text，不可另加未绑定的理由。")
    seen, cited = set(), set()
    for claim in draft.stage_claims:
        identity = (str(claim.object_id), claim.role)
        if identity not in options or identity in seen:
            raise AssistantError("answer_stage_identity_invalid")
        seen.add(identity)
        option = options[identity]
        if claim.role == "final_holdout_result" and any(".gain_ci_" in key for key in option["allowed_fact_ids"]):
            # 合法引用身份不证明区间与采用门的关系；只拦截本项目有确切反证的混淆。
            pattern = r"(?:开发采用门|采用门槛|采用门|开发门)(?:的|为|是|区间为|范围为|：)[^。！？;；\n]{0,60}\{\{c\d+\.gain_ci_(?:low|high)\}\}"
            for match in re.finditer(pattern, claim.text):
                before = re.sub(r"\s+", "", claim.text[:match.start()])
                if before.endswith(("不是", "并非", "不属于", "不同于", "不等于", "不能作为", "不能当作")):
                    continue
                raise AssistantError("answer_stage_interval_gate_conflict", "置信区间是正式结果的不确定性，不是开发采用门；不能用区间判断采用。")
        used = set(re.findall(r"\{\{([^{}]+)\}\}", claim.text))
        invalid = used - set(option["allowed_fact_ids"])
        if invalid:
            stages = {f["id"]: f.get("stage", "unknown") for f in evidence["facts"]}
            bad = ",".join(f"{key}({stages.get(key, 'unknown')})" for key in sorted(invalid))
            raise AssistantError("answer_stage_fact_conflict",
                f"{claim.role}不能使用{bad}；该段必须使用" + ",".join(option["required_fact_ids"]))
        if set(option["required_fact_ids"]) - used:
            missing = sorted(set(option["required_fact_ids"]) - used)
            raise AssistantError("answer_stage_metric_missing",
                claim.role + "的text缺少" + ",".join("{{" + key + "}}" for key in missing) +
                "；请在该段用占位符解释事实，纯文字结论不代替绑定。")
        if set(claim.citations) - set(option["allowed_citations"]) or set(option["required_citations"]) - set(claim.citations):
            raise AssistantError("answer_stage_citation_conflict", "此阶段必须引用对应阶段原文，不能只附正式结果。")
        cited.update(claim.citations)
    if required - seen:
        raise AssistantError("answer_stage_incomplete", "缺少所问阶段：" + ",".join(role for _, role in sorted(required - seen)))
    if cited != set(draft.citations):
        raise AssistantError("answer_stage_citation_conflict", "全局引用必须与正文阶段引用一致。")


def stage_body(draft, evidence):
    if not draft.stage_claims:
        return draft.answer
    options = {(r["object_id"], r["role"]): r for r in evidence["stage_options"]}
    return "\n\n".join(options[(str(c.object_id), c.role)]["label"] + "：" + c.text for c in draft.stage_claims)
