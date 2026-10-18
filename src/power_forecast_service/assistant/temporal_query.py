"""将完整、来源匹配的时序查询编译到既有操作；未识别的问法交回模型。"""

from decimal import Decimal
import re

from .contracts import AssistantError
from .program import RequestedPlan, delay_policy_contexts
from .retrieval import corpus
from .temporal import prepare_temporal_evidence


_CONNECTOR = r"(?:并且|同时|然后|并请|再请|请再|并再|也请|请|并|再)?"
_CLOCK = r"(?P<hour>\d{1,2})[:：](?P<minute>\d{2})"
_SCENARIO = re.compile(r"(?:设|假设|以)?" + _CLOCK + r"起报(?:时)?")
_POLICY = re.compile(
    r"(?:解释|说明)(?:当前(?:实验|版本)?|本实验|现行|所选实验)?"
    r"(?:模拟(?:的)?到达延(?:迟|时)(?:的)?(?:规则|假设|参数)?"
    r"|到达延(?:迟|时)(?:的)?模拟(?:规则|假设|参数)|模拟输入规则)")
_INPUT = r"(?:最晚输入(?:观测)?标签(?:时刻|时间)?|输入(?:观测标签)?截止(?:时刻|时间)?)"
_TARGET = r"(?:预测)?目标(?:时刻|时间)?"
_DELAY = r"(?:模拟)?(?:到达)?延(?:迟|时)"
_TIME_FIELDS = re.compile(
    r"(?P<fields>(?:" + _INPUT + "|" + _TARGET + "|" + _DELAY + ")"
    r"(?:(?:和|与|及|以及|、)(?:" + _INPUT + "|" + _TARGET + "|" + _DELAY + "))*)"
    r"(?:各是什么|分别是什么|是什么|分别是多少|各是多少|是多少|有哪些)?")
_MEASUREMENT = re.compile(
    r"(?:给出|提供|报告|列出|查询|告诉我)"
    r"(?P<subject>(?:(?:生产|现场|线上)?(?:实测|实际|真实)|生产|现场|线上)"
    r"(?:到达)?延(?:迟|时))"
    r"(?P<quantiles>[Pp](?:100|[1-9]?\d)(?:\.\d+)?"
    r"(?:(?:和|与|及|以及|、)[Pp](?:100|[1-9]?\d)(?:\.\d+)?)*)"
    r"(?:的(?:分位数|数据|数值)|(?:分位数|数据|数值))?")
_GAP_INSTRUCTION = re.compile(
    r"(?:缺少资料|资料不足|缺少证据|证据不足|没有资料)(?:时|的话)?"
    r"(?:请)?(?:列出|说明|报告|给出)(?:具体)?(?:资料|证据)?缺口")


def _qualified_source(evidence, documents):
    """缺口仅适用于这份冻结实验；新来源或测量事实必须交回原路径。"""
    if len(evidence.get("records", [])) != 1 or delay_policy_contexts(evidence, documents) != ["c0"]:
        return False
    canonical = {item["id"]: item for item in corpus()["chunks"]}
    seen = set()
    for document in documents:
        identity = document.get("id")
        original = canonical.get(identity)
        if identity in seen or original is None or any(
                document.get(key) != original.get(key) for key in ("source_sha256", "text")):
            return False
        seen.add(identity)
    for fact in evidence.get("facts", []):
        if fact.get("id", "").startswith("c0.time."):
            continue
        subject = " ".join(str(fact.get(key, "")) for key in ("id", "label", "pointer"))
        if re.search(r"到达|延迟|延时|实测|latency|arrival|delay|[Pp]\d", subject, re.I):
            return False
    return True


def _bound_time(evidence, documents, question, requested_components):
    """重新推导但不修改输入，核对同问题、同协议、同对象的已准备事实。"""
    existing = [fact for fact in evidence.get("facts", []) if ".time." in fact.get("id", "")]
    base = {**evidence, "facts": [fact for fact in evidence["facts"] if fact not in existing],
            "temporal_requirements": []}
    try:
        expected, _ = prepare_temporal_evidence(base, documents, question)
    except AssistantError:
        return False
    required = expected.get("temporal_requirements", [])
    additions = [fact for fact in expected["facts"] if fact["id"] in required]
    # 共同准备器若没有识别某个时间同义词，不能把未准备的分量默默略去。
    return (bool(required) and {"c0.time." + item for item in requested_components} <= set(required)
            and evidence.get("temporal_requirements") == required and existing == additions)


def compile_temporal_query(evidence, documents, question) -> RequestedPlan | None:
    """只接收整问均由时序合同覆盖的查询；返回计划仍须经原执行器和校验器。"""
    if not isinstance(question, str) or len(question) > 300 or re.search(r"[\"'`“”‘’「」『』{}\[\]]|[\x00-\x1f]", question):
        return None
    if not _qualified_source(evidence, documents):
        return None
    clauses = [item.strip() for item in re.split(r"[，。；！？,;!?]|\.(?!\d)|(?<!\d)\.", question) if item.strip()]
    requests, scenario, time_requested, policy_requested = [], None, False, False
    time_components = {"issue_time"}
    measurement_requested, gap_instruction = False, False
    for original in clauses:
        clause = re.sub(r"\s+", "", original)
        clause = re.sub(r"^" + _CONNECTOR, "", clause, count=1)
        issue = _SCENARIO.match(clause)
        if issue:
            hour, minute = int(issue["hour"]), int(issue["minute"])
            if scenario is not None or hour > 23 or minute > 59:
                return None
            scenario = f"{hour:02d}:{minute:02d}"
            clause = clause[issue.end():]
            if not clause:
                continue
        if _GAP_INSTRUCTION.fullmatch(clause):
            if gap_instruction:
                return None
            gap_instruction = True
            continue
        if _POLICY.fullmatch(clause):
            if policy_requested:
                return None
            policy_requested = True
            requests.append({"request": "解释所选实验的模拟到达延迟规则", "resolution": {
                "kind": "supported", "operations": [{"kind": "delay_policy", "context": "c0"}]}})
            continue
        measured = _MEASUREMENT.fullmatch(clause)
        if measured:
            quantiles = re.findall(r"[Pp]\d+(?:\.\d+)?", measured["quantiles"])
            if any(not 0 <= Decimal(item[1:]) <= 100 for item in quantiles):
                return None
            measurement_requested = True
            # 每个请求保留指标与所缺资料的归属；不把未知推广成所有生产系统都无资料。
            for quantile in quantiles:
                quantile = quantile.upper()
                requests.append({"request": measured["subject"] + quantile, "resolution": {
                    "kind": "unavailable", "missing": "当前所选冻结实验的实际到达日志或测量记录"}})
            continue
        fields = _TIME_FIELDS.fullmatch(re.sub(r"^(?:给出|提供|报告|查询|列出|告诉我)", "", clause))
        if fields and re.search(_INPUT + "|" + _TARGET, fields["fields"]):
            if time_requested:
                return None
            time_requested = True
            for field in re.findall(_INPUT + "|" + _TARGET + "|" + _DELAY, fields["fields"]):
                if re.fullmatch(_INPUT, field):
                    time_components.add("input_cutoff")
                elif re.fullmatch(_TARGET, field):
                    time_components.add("targets")
                else:
                    time_components.update(("arrival_lag", "delay_basis"))
            requests.append({"request": "情景起报的" + fields["fields"], "resolution": {
                "kind": "supported", "operations": [{"kind": "time", "context": "c0"}]}})
            continue
        # 子句即使含有效关键词，仍不可悄悄略去其未知语义或其他对象。
        return None
    if not requests or (scenario is not None) != time_requested or gap_instruction and not measurement_requested:
        return None
    if time_requested and not _bound_time(evidence, documents, question, time_components):
        return None
    if not time_requested and any(".time." in fact.get("id", "") for fact in evidence.get("facts", [])):
        return None
    unique = []
    for item in requests:
        if item not in unique:
            unique.append(item)
    if len(unique) > 12 or sum(item["resolution"]["kind"] == "unavailable" for item in unique) > 6:
        return None
    return RequestedPlan.model_validate({"requests": unique})
