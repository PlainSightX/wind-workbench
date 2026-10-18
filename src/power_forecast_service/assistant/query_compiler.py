"""将完整且无歧义的简单查询编译为既有只读操作，其他问题保留模型路径。"""

import math
import re

from .contracts import AssistantError
from .program import RequestedPlan, comparison_facts


# 只允许词语之间的空白和明确句读，不删除任意标点后再猜测用户意图。
_SPACE = r"[ \t]*"
_MODEL = r"[a-z][a-z0-9_]*"
_END = rf"{_SPACE}[。.？?]?{_SPACE}"
_READ = re.compile(
    rf"(?:请{_SPACE})?(?:(?:读取|查询|给出|提供){_SPACE})?"
    rf"(?P<model>{_MODEL}){_SPACE}的{_SPACE}(?P<metric>mae|rmse)"
    rf"{_SPACE}(?:是多少)?"
    rf"(?P<period>{_SPACE}[，,]{_SPACE}(?:并{_SPACE})?(?:说明|注明){_SPACE}"
    rf"单位{_SPACE}(?:和|与){_SPACE}(?:所选{_SPACE})?评价期间)?{_END}",
    re.IGNORECASE,
)
_COMPARE = re.compile(
    rf"(?:请{_SPACE})?(?P<left>{_MODEL}){_SPACE}对比{_SPACE}(?P<right>{_MODEL})"
    rf"{_SPACE}[，,]{_SPACE}mae{_SPACE}和{_SPACE}rmse{_SPACE}两项是否"
    rf"(?:均|同时)严格更小{_SPACE}(?:[？?]{_SPACE}(?:请{_SPACE})?"
    rf"(?:给|给出|提供){_SPACE}原值)?{_END}",
    re.IGNORECASE,
)
_COVERAGE = re.compile(
    rf"(?:请{_SPACE})?(?:说明|解释|给出|提供|报告){_SPACE}合法输入"
    rf"{_SPACE}(?:与|和){_SPACE}实际输出{_SPACE}的{_SPACE}(?:次数|计数|数量)"
    rf"(?:{_SPACE}(?:和|及|与){_SPACE}各自(?:的)?定义)?{_END}"
)
_COVERAGE_COUNT = re.compile(
    rf"(?:请{_SPACE})?合法输入{_SPACE}(?:与|和){_SPACE}实际输出"
    rf"{_SPACE}的{_SPACE}(?:次数|计数|数量){_SPACE}(?:分别)?是多少{_END}"
)


def _fact_map(evidence):
    """重复身份存在歧义时不选择最后一条，回到原路径处理。"""
    facts = evidence.get("facts", [])
    if not isinstance(facts, list) or any(not isinstance(fact, dict) for fact in facts):
        return None
    identifiers = [fact.get("id") for fact in facts]
    if any(not isinstance(key, str) for key in identifiers) or len(set(identifiers)) != len(identifiers):
        return None
    return dict(zip(identifiers, facts))


def _selected_fact(facts, record, key):
    """检查所选记录与事实身份；编译器不从模型名称、文档或数字相似度借用结果。"""
    fact = facts.get(key)
    if fact is None or fact.get("object_id") != record.get("id"):
        return None
    value = fact.get("value")
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        return None
    if fact.get("stage") not in {"selected_result", "final_holdout"} or fact.get("derived"):
        return None
    if any(not isinstance(fact.get(axis), str) or not fact[axis] for axis in ("aggregation", "unit")):
        return None
    axes = ("aggregation", "unit") if len(key.split(".")) == 3 else ("aggregation",)
    if any(record.get(axis) is not None and record[axis] != fact[axis] for axis in axes):
        return None
    return fact


def _same_scope(facts):
    axes = ("object_id", "stage", "aggregation", "unit")
    return len({tuple(fact.get(axis) for axis in axes) for fact in facts}) == 1


def _rendering_available(evidence, documents, selected):
    """有阶段呈现义务时沿用原容器及引用；普通指标读取不追加新义务。"""
    if not evidence.get("stage_hints"):
        return True
    keys = {fact["id"] for fact in selected}
    option = next((item for item in evidence.get("stage_options", [])
        if item.get("object_id") == selected[0]["object_id"]
        and keys <= set(item.get("allowed_fact_ids", []))), None)
    if option is None:
        return False
    from .retrieval import corpus
    canonical = {item["id"]: item for item in corpus()["chunks"]}
    required, available = set(option.get("required_citations", [])), {}
    for document in documents:
        identity = document.get("id")
        if identity not in required:
            continue
        # 同ID的第二份文档会改变原执行器的映射，不能只按ID集合证明来源一致。
        if identity in available:
            return False
        available[identity] = document
    for identity in required:
        document, original = available.get(identity), canonical.get(identity)
        if document is None or original is None or any(
                document.get(key) != original.get(key) for key in ("source_sha256", "text")):
            return False
    return True


def _plan(question, operations, *, comparison_summary=False):
    request = {"request": question, "resolution": {"kind": "supported", "operations": operations}}
    if comparison_summary:
        request["presentation"] = {"comparison_summary": True}
    return RequestedPlan.model_validate({"requests": [request]})


def compile_structured_query(evidence, documents, question) -> RequestedPlan | None:
    """仅覆盖三类完整问法；任何未消费子句、未知模型或不兼容来源都返回None。

    返回值只有事实身份和操作，无结果数值。评价期间、比较结论、计数定义仍由
    execute_program生成，并由validate_answer走原合同校验。
    """
    if not isinstance(question, str) or not 1 <= len(question) <= 300:
        return None
    records = evidence.get("records", [])
    if len(records) != 1 or evidence.get("temporal_requirements") or evidence.get("stage_requirements"):
        return None
    record = records[0]
    facts = _fact_map(evidence)
    if not isinstance(record, dict) or not record.get("id") or facts is None:
        return None
    text = question.strip()
    read, compare = _READ.fullmatch(text), _COMPARE.fullmatch(text)
    coverage = _COVERAGE.fullmatch(text) or _COVERAGE_COUNT.fullmatch(text)
    if not any((read, compare, coverage)):
        return None
    models = record.get("models", {})
    if not isinstance(models, dict):
        return None
    names = {name.lower(): name for name in models if isinstance(name, str) and re.fullmatch(_MODEL, name, re.IGNORECASE)}
    if len(names) != len(models):
        return None
    if read:
        model = names.get(read["model"].lower())
        if model is None or (read["period"] and not record.get("quarter")):
            return None
        key = f"c0.{model}.{read['metric'].lower()}"
        fact = _selected_fact(facts, record, key)
        if fact is None or not _rendering_available(evidence, documents, [fact]):
            return None
        return _plan(question, [{"kind": "values", "fact_ids": [key]}])
    if compare:
        pair = tuple(names.get(compare[axis].lower()) for axis in ("left", "right"))
        if None in pair or pair[0] == pair[1]:
            return None
        operations, selected = [], []
        for metric in ("mae", "rmse"):
            left, right = (f"c0.{model}.{metric}" for model in pair)
            a, b = (_selected_fact(facts, record, key) for key in (left, right))
            if a is None or b is None:
                return None
            try:
                comparison_facts(left, right, facts)
            except AssistantError:
                return None
            operations.append({"kind": "compare", "left": left, "right": right})
            selected.extend((a, b))
        if not _same_scope(selected) or not _rendering_available(evidence, documents, selected):
            return None
        return _plan(question, operations, comparison_summary=True)
    counts = ("input_valid", "output_count")
    selected = [_selected_fact(facts, record, "c0." + key) for key in counts]
    if any(fact is None for fact in selected) or not _same_scope(selected):
        return None
    projected = record.get("coverage", {})
    if any(type(projected.get(key)) is not int or projected[key] < 0
           or selected[index]["value"] != projected[key] or selected[index]["unit"] != "次"
           for index, key in enumerate(counts)):
        return None
    try:
        comparison_facts("c0.input_valid", "c0.output_count", facts)
    except AssistantError:
        return None
    if not _rendering_available(evidence, documents, selected):
        return None
    return _plan(question, [{"kind": "coverage", "context": "c0", "counts": list(counts)}])
