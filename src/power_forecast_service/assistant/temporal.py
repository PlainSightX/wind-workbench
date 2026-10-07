"""从匹配的冻结协议推导情景时刻；不读取评测题号、答案或任意时区。"""

from datetime import datetime, timedelta
from hashlib import sha256
import re

from .contracts import AssistantError, Fact
from .stages import stage_sources


def prepare_temporal_evidence(evidence, documents, question):
    """仅处理一个明确HH:mm起报情景；其他问法交给原路径，不猜日期或起报点。"""
    clocks = re.findall(r"(?<![\d:])(\d{1,2})[:：](\d{2})(?![\d:])", question)
    issue = re.search(r"(?<![\d:])(\d{1,2})[:：](\d{2})(?![\d:])\s*起报", question)
    if len(clocks) != 1 or issue is None:
        return evidence, documents
    hour, minute = map(int, issue.groups())
    if hour > 23 or minute > 59:
        return evidence, documents
    requested = []
    if any(word in question for word in ("输入", "截止", "观测")):
        requested.append("input_cutoff")
    if any(word in question for word in ("时距", "目标", "预测哪些", "预测时刻")):
        requested.append("targets")
    if any(word in question for word in ("延迟", "实测", "测量")):
        requested.extend(("arrival_lag", "delay_basis"))
    if not requested:
        return evidence, documents
    source = stage_sources()["engie"]
    protocol = source["time_contract"]
    origin = datetime(2000, 1, 2, hour, minute)

    def clock(offset):
        result = origin + timedelta(minutes=offset)
        day = (result.date() - origin.date()).days
        prefix = "前一日 " if day == -1 else "次日 " if day == 1 else ""
        return prefix + result.strftime("%H:%M")

    additions, requirements, seen = [], [], set()
    for index, record in enumerate(evidence["records"]):
        if record.get("scope") != "engie_final" or record.get("id") in seen or (
            record.get("result_sha256"), record.get("protocol_sha256")
        ) != (source["result_sha256"], source["protocol_sha256"]):
            continue
        seen.add(record["id"])
        values = {
            "input_cutoff": ("输入观测标签最晚时刻", clock(-protocol["arrival_lag_minutes"]), "", "/protocol/arrival_lag_minutes"),
            "targets": ("预测目标时刻", "、".join(clock(n) for n in protocol["target_minutes_after_issue"]), "", "/protocol/target_minutes_after_issue"),
            "arrival_lag": ("模拟到达延迟", protocol["arrival_lag_minutes"], "分钟", "/protocol/arrival_lag_minutes"),
            "delay_basis": ("延迟依据", protocol["delay_basis"], "", "#2-数据合同与真实问题"),
        }
        # 起报时刻是请求情景输入，不伪装成协议观测；与推导的截止时刻分开绑定。
        origin_id = f"c{index}.time.issue_time"
        additions.append(Fact(id=origin_id, label="用户设定的情景起报时刻", value=origin.strftime("%H:%M"),
            object_id=record["id"], aggregation="用户情景参数；未指定日期和时区，不是真实起报观测",
            source_sha256=sha256(question.encode("utf-8")).hexdigest(), pointer="request:/question/issue_time",
            derived=True, stage="question_scenario").model_dump())
        requirements.append(origin_id)
        for name in requested:
            label, value, unit, pointer = values[name]
            identity = f"c{index}.time.{name}"
            descriptive = name == "delay_basis"
            additions.append(Fact(id=identity, label=label, value=value, unit=unit,
                object_id=record["id"], aggregation="情景起报 " + origin.strftime("%H:%M") + "；未指定日期和时区",
                source_sha256=protocol["method_source_sha256"] if descriptive else protocol["source"]["sha256"],
                source_path="docs/results/wind-engie-baseline-20260923/README.md" if descriptive else protocol["source"]["path"],
                pointer=pointer, derived=name in {"input_cutoff", "targets"}, stage="forecast_time_contract").model_dump())
            requirements.append(identity)
    if not additions:
        return evidence, documents
    # 从单一版本语料补齐方法出处；来源冲突不能靠本工具覆盖掉。
    from .retrieval import corpus
    by_id = {d["id"]: d for d in corpus()["chunks"]}
    anchors = protocol["citations"]
    if any(c not in by_id or by_id[c]["source_sha256"] != protocol["method_source_sha256"] for c in anchors):
        raise AssistantError("corpus_time_evidence_missing")
    known = {f["id"]: f for f in evidence["facts"]}
    for fact in additions:
        if fact["id"] in known and known[fact["id"]] != fact:
            raise AssistantError("context_time_conflict")
    data = {**evidence, "facts": [*evidence["facts"], *(f for f in additions if f["id"] not in known)],
            "temporal_requirements": requirements}
    present = {d["id"] for d in documents}
    return data, [*documents, *(by_id[c] for c in anchors if c not in present)]


def validate_temporal_coverage(draft, evidence):
    """只保证所问时间分量进入正文；否定/因果等语义仍须独立审阅。"""
    required = set(evidence.get("temporal_requirements", []))
    body = "\n".join(c.text for c in draft.stage_claims) if draft.stage_claims else draft.answer
    used = set(re.findall(r"\{\{([^{}]+)\}\}", body))
    missing = required - used
    if missing:
        raise AssistantError("answer_time_fact_missing", "所问时刻及延迟依据必须在正文绑定：" + ",".join(sorted(missing)))
