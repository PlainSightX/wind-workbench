"""有明确证据的局部语义防线；不声称识别任意自然语言的蕴含关系。"""

import re

from .contracts import AssistantError

QUARTERS = tuple(f"Q{number}" for number in range(1, 5))


def bind_nominal_identifiers(evidence, question):
    """标识不等于测量值：仅放行真实季度及问题原有的合法分位数标签。"""
    identifiers = {str(record.get("quarter", "")) for record in evidence["records"]
                   if re.fullmatch(r"\d{4}-Q[1-4]", str(record.get("quarter", "")))}
    for label in re.findall(r"(?<![A-Za-z0-9_.])P\d{1,3}(?:\.\d+)?(?![A-Za-z0-9_.%])", question, flags=re.IGNORECASE):
        if 0 <= float(label[1:]) <= 100:
            identifiers.add(label.upper())
    result = {key: value for key, value in evidence.items() if key != "nominal_identifiers"}
    if identifiers:
        result["nominal_identifiers"] = sorted(identifiers)
    return result


def validate_refusal(draft, evidence, question):
    if draft.status != "insufficient_evidence":
        return
    if draft.stage_claims:
        raise AssistantError("answer_refusal_structure", "证据不足应用普通解释，不列强制采用阶段。")
    if not draft.missing_evidence or any(not item.strip() for item in draft.missing_evidence):
        raise AssistantError("answer_refusal_missing_reason", "明确列出本请求所缺的证据，并在正文说明不能推出什么。")
    # 只认完整事实身份，不把“某指标的原因/证明”误当成已有指标本身。
    # False 和零同样是已提供的证据；不能用真值判断把否定结论改成证据缺失。
    present = {fact["id"] for fact in evidence["facts"] if fact.get("value") is not None}
    for missing in draft.missing_evidence:
        identity = missing.strip().strip("`")
        if identity.startswith("{{") and identity.endswith("}}"):
            identity = identity[2:-2].strip()
        if identity in present:
            raise AssistantError("answer_refusal_present_fact",
                "列为缺失的事实已在所选对象facts中：" + identity +
                "。仅从missing_evidence移除这个已有事实，改列真正缺少的资料；这不证明整个问题可答。"
                "收入、因果或实测证据仍缺少时保持insufficient_evidence。仅当问题已经能够回答时才用answered。False/零值也是证据。")
    if not any(term in draft.answer for term in ("缺", "没有", "未提供", "不包含", "未包含", "不足", "无法", "不能", "不支持")):
        raise AssistantError("answer_refusal_missing_reason", "正文需说明证据缺口，不能只罗列不相关的已有事实。")
    # 只拦截明确询问已提供指标原值的拒答；分组、原因、收入等复合需求不能据此推断可答。
    raw = question.lower()
    extra = ("为什么", "原因", "因果", "收入", "收益", "金额", "每", "各", "分别", "工况", "时距", "置信", "区间", "其他", "另一")
    if any(word in raw for word in extra):
        return
    metric = next((name for name in ("mae", "rmse", "bias") if name in raw), None)
    direct = any(word in raw for word in ("多少", "原值", "数值"))
    if metric and direct:
        for record in evidence["records"]:
            for model in record["models"]:
                requested = [fact for fact in evidence["facts"] if
                    fact["id"].endswith("." + model + "." + metric)
                    and fact.get("object_id") == record.get("id")]
                if model.lower() in raw and requested and not any("{{" + fact["id"] + "}}" in draft.answer for fact in requested):
                    raise AssistantError("answer_refusal_answerable", "所问模型指标原值已在所选对象facts中，需直接回答，不应拒答。")


def validate_quarter_assignment(body, evidence):
    """限定单季度对象的直接归属句；不把否定、假设或跨季度讨论当作肯定归属。"""
    quarters = {str(record.get("quarter", ""))[-2:].upper() for record in evidence["records"]}
    if len(quarters) != 1 or next(iter(quarters)) not in QUARTERS:
        return
    expected = next(iter(quarters))
    pattern = r"(?:所选(?:对象|结果|季度)|当前(?:对象|结果|季度)|该(?:对象|结果|季度)|开发季度|评价季度)(?:\s*(?:属于|对应|是|为|即|的|（即|\(即|：|:|\s))*\s*(?:\d{4}[-年])?(Q[1-4]|第[一二三四]季度)"
    for match in re.finditer(pattern, body, flags=re.IGNORECASE):
        prefix = body[:match.start()].rsplit("。", 1)[-1]
        if any(term in prefix for term in ("如果", "假设", "若", "并非", "不是", "不能称")):
            continue
        actual = match[1].upper()
        if actual.startswith("第"):
            actual = "Q" + str("一二三四".index(actual[1]) + 1)
        if actual != expected:
            raise AssistantError("answer_object_quarter_conflict", "所选对象季度为" + expected + "，直接归属句不得写成" + actual + "。")
