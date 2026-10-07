"""把模型选中的版本引用解析成既有验收输入；不生成结论或读取评测答案。"""

from copy import deepcopy
import re

from .contracts import (AssistantError, DraftAnswer, ReferenceAnswer, StageClaim, StagedReferenceBody,
                        MAX_ANSWER_FACTS, MAX_ANSWER_CITATIONS)
from .validation import FACT_TOKEN, normalize_fact_tokens


def reference_schema(evidence, documents, required_facts=()):
    """把已有请求边界编译成生成约束；不选择结论，也不读取外部验收答案。"""
    schema = deepcopy(ReferenceAnswer.model_json_schema())
    definitions = schema["$defs"]
    doc_ids = sorted({d["id"] for d in documents}) or ["__no_document__"]
    definitions["CitationSelection"]["properties"]["document_id"]["enum"] = doc_ids
    models = sorted({name for record in evidence["records"] for name in record["models"]})
    facts = {fact["id"]: fact for fact in evidence["facts"]}

    def text_schema(allowed, required, max_length=3500):
        # 自由文字可用中文表达方法数字；阿拉伯指标及计算表达式不能绕过真实fact ID。
        # 这只约束可生成的字节，不验证模型选择的事实是否支持其自然语言结论。
        tokens = [r"\{\{" + re.escape(identity) + r"\}\}" for identity in allowed]
        tokens += [re.escape(name) for name in models] + ["Q1", "L1", "L2"]
        # XGrammar 0.1.25把pattern直接放入字符串语法，不与JSON转义规则求交。
        # 排除原始引号、反斜杠和控制字符，否则成功编译仍可生成非法JSON。
        atom = r'(?:[^0-9{}"\\\x00-\x1f]|' + "|".join(tokens) + r")"
        if set(required) - set(allowed):
            raise AssistantError("context_required_fact_conflict")
        # 缺项交给原业务校验及一次纠错；强制正文事实次序会阻止模型及时结束，
        # 把可纠错的遗漏变成重复生成/截断。解码只约束合法身份，不代替覆盖验收。
        pattern = "^" + atom + "+$"
        # 把当前片段可用事实的类型放在同一个局部合同中；布尔决定、时间和指标
        # 不能仅因ID合法便相互替代。值仍来自完整evidence，不添加题目答案。
        meanings = []
        for identity in allowed:
            fact = facts[identity]
            value = fact["value"]
            kind = "是否事实" if isinstance(value, bool) else "数值" if isinstance(value, (int, float)) else "文本/时刻"
            meanings.append(identity + "=" + fact["label"] + "（" + kind + "，" + fact.get("unit", "") + "）")
        return {"type": "string", "minLength": 1, "maxLength": max_length,
                "pattern": pattern,
                "description": "正文直接解释所问。阿拉伯数值仅由事实占位符呈现，禁止abs等表达式。以下绑定均须在正文解释，可按自然顺序使用；不要遗漏或改变含义：" +
                    "、".join("{{" + key + "}}" for key in required) +
                    "。是否事实只能说明是否，文本/时刻不能当改善率或阈值。可用事实含义：" + "；".join(meanings)}

    stages = evidence.get("stage_requirements", [])
    if not stages:
        body = definitions["PlainReferenceBody"]
        required = list(dict.fromkeys([*required_facts, *evidence.get("temporal_requirements", [])]))
        body["properties"]["text"] = text_schema([f["id"] for f in evidence["facts"]], required)
        schema["properties"]["body"] = {"$ref": "#/$defs/PlainReferenceBody"}
    else:
        options = {(o["object_id"], o["role"]): o for o in evidence["stage_options"]}
        claims = []
        for stage in stages:
            option = options[(stage["object_id"], stage["role"])]
            claim = deepcopy(definitions["ReferenceStageClaim"])
            claim["description"] = option.get("label", stage["role"]) + "；只解释本阶段事实，其他阶段的结果不能作为本阶段原因。"
            if stage["role"] == "final_holdout_result":
                claim["description"] += "正式改善只作事后核验；本段不能宣称达到或未达到开发采用门。置信区间不是开发门槛。"
            fields = claim["properties"]
            fields["object_id"] = {"type": "string", "const": stage["object_id"]}
            fields["role"] = {"type": "string", "const": stage["role"]}
            fields["text"] = text_schema(option["allowed_fact_ids"], option["required_fact_ids"], 1500)
            fields["citations"]["description"] = "引用当前阶段，必须覆盖：" + ",".join(option["required_citations"])
            fields["citations"]["items"] = {"type": "object", "additionalProperties": False,
                "properties": {"document_id": {"type": "string", "enum": option["allowed_citations"]}},
                "required": ["document_id"]}
            claims.append(claim)
        definitions["StagedReferenceBody"]["properties"]["claims"] = {
            "type": "array", "prefixItems": claims, "items": False, "minItems": len(claims), "maxItems": len(claims)}
        schema["properties"]["body"] = {"$ref": "#/$defs/StagedReferenceBody"}
    # 不呈现无法被当前body分支使用的另一种回答，减少无关模式选择。
    unused = "StagedReferenceBody" if not stages else "PlainReferenceBody"
    definitions.pop(unused)
    return schema


def resolve_reference_answer(response, documents):
    """仅绑定当前请求文档；不接受模型指定版本，也不从跨请求缓存猜来源。"""
    docs = {}
    for document in documents:
        previous = docs.get(document["id"])
        if previous is not None and any(previous.get(key) != document.get(key)
                                        for key in ("text", "source_sha256", "title")):
            raise AssistantError("answer_citation_ambiguous", "同一引用ID对应不同原文或版本。")
        docs[document["id"]] = document
    body = response.body
    staged = isinstance(body, StagedReferenceBody)
    parts = body.claims if staged else [body]
    citations, quotes, claims, texts = [], {}, [], []
    for part in parts:
        selected = []
        for ref in part.citations:
            document = docs.get(ref.document_id)
            if document is None:
                raise AssistantError("answer_citation_invalid", "只能选择本次提供的文档ID。")
            selected.append(ref.document_id)
            quotes[ref.document_id] = document["text"]
        citations.extend(selected)
        text = normalize_fact_tokens(part.text)
        texts.append(text)
        if staged:
            claims.append(StageClaim(object_id=part.object_id, role=part.role, text=text,
                                     citations=list(dict.fromkeys(selected))))
    # 正文中实际出现的ID仍需既有validate_answer校验，不能靠附件增加事实覆盖。
    fact_ids = list(dict.fromkeys(token[1] for text in texts for token in FACT_TOKEN.finditer(text)))
    citations = list(dict.fromkeys(citations))
    # 各阶段独立合法不代表合并后仍满足原交付上限；超限属于已返回回答的可纠错错误。
    if len(fact_ids) > MAX_ANSWER_FACTS or len(citations) > MAX_ANSWER_CITATIONS:
        raise AssistantError("answer_reference_limit", "同一回答最多绑定十五项事实、八份文档，请精简到所问内容。")
    return DraftAnswer(status=response.status, answer="" if staged else texts[0],
                       fact_ids=fact_ids, citations=citations,
                       quotes=quotes, stage_claims=claims)
