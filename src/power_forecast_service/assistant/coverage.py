"""从用户所问的覆盖数量推导正文要求；不读取题号或独立评价答案。"""

import re

from .contracts import AssistantError


def coverage_requirements(evidence, question):
    """只处理已有数量事实的明确问法；方法解释和未知意图保持原校验。"""
    keys = set()
    for query in re.split(r"[，。；！？,;!?]", question.lower()):
        counting = any(word in query for word in ("多少", "数量", "次数", "计数", "几次", "how many", "count"))
        relation = any(word in query for word in ("全部", "所有", "是否都", "都输出", "都产出", "all"))
        if counting and any(word in query for word in ("计划起报", "计划预测", "planned")):
            keys.add("planned")
        inputs = any(word in query for word in ("合法输入", "有效输入", "input_valid"))
        outputs = any(word in query for word in ("输出", "产出", "output_count"))
        if inputs and (counting or relation):
            keys.add("input_valid")
        output_count = any(word in query for word in ("实际输出", "合法输出", "输出起报", "输出数量", "输出次数", "output_count"))
        output_count = output_count or bool(re.search(r"(?:输出|产出)(?:了|有|共)?(?:多少|几次)", query))
        if outputs and counting and output_count:
            keys.add("output_count")
        if inputs and outputs and relation:
            keys.update(("input_valid", "output_count"))
        if counting and any(word in query for word in ("可评分", "能够评分", "能评分", "scoreable")):
            keys.add("scoreable")
    facts = {fact["id"]: fact for fact in evidence["facts"]}
    required = []
    for index, record in enumerate(evidence["records"]):
        coverage = record.get("coverage", {})
        for key in sorted(keys):
            identity = f"c{index}.{key}"
            fact = facts.get(identity)
            # 数量必须属于所选记录及其已验证投影，不能从另一对象借用同名字段。
            if (type(coverage.get(key)) is int and fact is not None
                    and fact.get("object_id") == record.get("id")
                    and fact["value"] == coverage[key]):
                required.append(identity)
    return required


def validate_coverage(draft, required):
    """事实仅列在附件里不算回答；正文必须使用同一绑定，拒答也不能绕过。"""
    if not required:
        return
    body = "\n".join(claim.text for claim in draft.stage_claims) if draft.stage_claims else draft.answer
    used = set(re.findall(r"\{\{([^{}]+)\}\}", body)) & set(draft.fact_ids)
    missing = sorted(set(required) - used)
    if draft.status != "answered" or missing:
        raise AssistantError("answer_coverage_incomplete", "所问覆盖数量需在正文绑定：" + ",".join(missing or required))
