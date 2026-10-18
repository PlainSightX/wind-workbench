"""只替换已定位的正文片段；不让修复响应修改对象、阶段、来源或其他片段。"""

from copy import deepcopy
import json

from pydantic import BaseModel, ConfigDict, Field

from .contracts import AssistantError, ReferenceAnswer, StagedReferenceBody
from .quality import text_findings
from .validation import FACT_TOKEN

LOCAL_ERRORS = frozenset({"answer_stage_metric_missing"})


class TextPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int = Field(ge=0, le=7)
    text: str = Field(min_length=1, max_length=3500)


class TextRepairs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patches: list[TextPatch] = Field(min_length=1, max_length=8)


def repair_targets(response, evidence, required, errors):
    """无法确认错误属于哪些已存在片段时，保留原整篇修复路径。"""
    if response.status != "answered" or not errors or any(error.code not in LOCAL_ERRORS for error in errors):
        return []
    body = response.body
    staged = isinstance(body, StagedReferenceBody)
    # 真实试验中，单段局部修复没有可靠收益；存在语义矛盾时局部补值可能保留错误解释。
    # 只保留多段回答中的纯覆盖遗漏，其余继续已有的整篇纠错，不增加第三次调用。
    if not staged or len(body.claims) < 2:
        return []
    parts = body.claims if staged else [body]
    options = {(item["object_id"], item["role"]): item for item in evidence.get("stage_options", [])}
    targets = []
    for index, part in enumerate(parts):
        if staged:
            identity = str(part.object_id)
            option = options.get((identity, part.role))
            if option is None:
                return []
            needed = option["required_fact_ids"]
            facts = [f for f in evidence["facts"] if f.get("object_id") == identity]
        else:
            needed = [*required, *evidence.get("temporal_requirements", [])]
            facts = evidence["facts"]
        missing = sorted(set(needed) - set(FACT_TOKEN.findall(part.text)))
        problems = text_findings(part.text, facts)
        if missing or problems:
            targets.append({"index": index, "missing": missing, "problems": [error.detail for error in problems]})
    return targets


def repair_schema(original, targets):
    """复用原片段的合法事实约束；只缩小输出范围，不删模型输入证据。"""
    schema = TextRepairs.model_json_schema()
    definitions = original["$defs"]
    body_schema = original.get("properties", {}).get("body")
    if body_schema is None:
        body_schema = original["oneOf"][0]["properties"]["body"]
    staged = body_schema["$ref"].endswith("StagedReferenceBody")
    items = []
    for target in targets:
        if staged:
            source = definitions["StagedReferenceBody"]["properties"]["claims"]["prefixItems"][target["index"]]
        else:
            source = definitions["PlainReferenceBody"]
        items.append({"type": "object", "additionalProperties": False,
                      "properties": {"index": {"type": "integer", "const": target["index"]},
                                     "text": deepcopy(source["properties"]["text"])}, "required": ["index", "text"]})
    schema["properties"]["patches"] = {"type": "array", "prefixItems": items, "items": False,
                                         "minItems": len(items), "maxItems": len(items)}
    return schema


def apply_repairs(response, repairs, targets):
    """在深拷贝上替换准确位置；整份回答随后仍需重新校验。"""
    if [patch.index for patch in repairs.patches] != [target["index"] for target in targets]:
        raise AssistantError("answer_repair_target_invalid", "修复片段与已定位的范围不一致。")
    data = response.model_dump(mode="json")
    staged = data["body"]["kind"] == "staged"
    for patch in repairs.patches:
        part = data["body"]["claims"][patch.index] if staged else data["body"]
        part["text"] = patch.text
    return ReferenceAnswer.model_validate(data)


def repair_feedback(targets):
    instructions = [{"index": item["index"], "must_include_tokens": ["{{" + key + "}}" for key in item["missing"]],
                     "diagnostics": item["problems"]} for item in targets]
    return ("本次仅修复下列正文片段，输出patches数组，每项只含index和text，禁止重写整份回答。"
            "前文status/body格式只属于首稿；本次服从这个局部修复格式。保留完整原证据的含义，"
            "既补齐缺失绑定，也纠正片段中错误的解释，不能只追加正确事实却保留错误结论。"
            "引用、对象、阶段及未列片段由程序原样保留。must_include_tokens中的每一个完整占位符"
            "必须实际写入对应text的自然解释，不能只用中文是/否替代。diagnostics是错误诊断，"
            "不是可照抄的答案；不要只抄诊断而遗漏占位符。数值继续用完整占位符，不写空位或只罗列事实。"
            + json.dumps(instructions, ensure_ascii=False))
