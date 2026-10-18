"""完整时序合同和冻结来源的定向验证；不声称开放问答或模型能力提升。"""

from copy import deepcopy

import pytest

from power_forecast_service.assistant.contracts import AssistantError, Fact
from power_forecast_service.assistant.program import execute_request_plan
from power_forecast_service.assistant.retrieval import corpus
from power_forecast_service.assistant.stages import stage_sources
from power_forecast_service.assistant.temporal_query import compile_temporal_query
from power_forecast_service.assistant.validation import validate_answer
from power_forecast_service.assistant.workflow import prepare_answer_context
from tests.unit.assistant.test_stages import projected


def prepared(question):
    # 使用现行冻结来源及真实共同准备入口；这些本地fixture不是新模型调用。
    return prepare_answer_context(projected("engie"), corpus()["chunks"], question, "program_requested")


def render(question):
    evidence, documents = prepared(question)
    before = deepcopy((evidence, documents))
    plan = compile_temporal_query(evidence, documents, question)
    assert plan is not None
    assert (evidence, documents) == before
    draft = execute_request_plan(plan, evidence, documents, question)
    result = validate_answer(draft, evidence, documents, question)
    assert all(operation.kind in {"time", "delay_policy", "unknown"} for operation in plan.operations)
    return plan, result


@pytest.mark.parametrize("question", [
    "解释模拟到达延迟规则，并给出生产实测延迟P98.5；缺少资料时列出具体缺口。",
    "23:45起报，最晚输入标签和目标时刻是什么？解释模拟到达延迟假设，再给出生产实测延迟P97。",
    "给出生产实测延迟P91.25；说明到达延迟的模拟假设；以01:10起报，目标时刻与输入截止是什么？",
    "缺少资料时列出具体缺口。给出真实到达延时p99.001；解释当前模拟输入规则。",
])
def test_complete_temporal_compound_preserves_operations_and_exact_gaps(question):
    plan, result = render(question)
    assert result["status"] == "insufficient_evidence"
    assert "实际到达日志或测量记录" in result["answer"]
    assert "模拟" in result["answer"]
    assert not any(".mae" in fact["id"] for fact in result["facts"])
    if "23:45" in question:
        assert "23:25" in result["answer"] and "次日 00:45" in result["answer"]
        assert sum(operation.kind == "time" for operation in plan.operations) == 1
        assert sum(fact["id"] == "c0.time.arrival_lag" for fact in result["facts"]) == 1


@pytest.mark.parametrize("quantile", ["P0", "P100", "P0.125", "P91.25", "P98.7", "p99.9999"])
def test_new_legal_quantile_is_read_from_user_request(quantile):
    plan, result = render("给出生产实测延迟" + quantile + "。")
    assert len(plan.requests) == 1
    assert quantile.upper() in plan.requests[0].request
    assert quantile.upper() in result["answer"]


def test_multiple_quantiles_and_reversed_clauses_preserve_each_attribution():
    question = "给出生产实测延迟P91.25和P0.25、P100；解释模拟到达延迟规则。"
    plan, result = render(question)
    assert [item.request for item in plan.requests[:3]] == [
        "生产实测延迟P91.25", "生产实测延迟P0.25", "生产实测延迟P100"]
    assert all(item.resolution.kind == "unavailable" for item in plan.requests[:3])
    assert all(value in result["answer"] for value in ("P91.25", "P0.25", "P100"))


@pytest.mark.parametrize("question,expected", [
    ("00:10起报，输入截止和目标时刻是什么？", ("前一日 23:50", "01:10")),
    ("23:45起报，目标时刻和最晚输入标签是什么？", ("23:25", "次日 00:45")),
    ("设02:00起报输入截止、目标和延迟是什么？", ("01:40", "03:00", "20 分钟")),
])
def test_time_uses_existing_cross_day_facts_and_original_validation(question, expected):
    plan, result = render(question)
    assert result["status"] == "answered"
    assert [operation.kind for operation in plan.operations] == ["time"]
    assert all(value in result["answer"] for value in expected)


def test_policy_only_uses_the_existing_source_operation():
    plan, result = render("请解释模拟到达延迟规则。")
    assert [operation.kind for operation in plan.operations] == ["delay_policy"]
    assert result["status"] == "answered"
    assert "20分钟" in result["answer"] and "不是生产到达时间的实测分布" in result["answer"]


@pytest.mark.parametrize("question", [
    "不要报告实测延迟P99，仅解释模拟。",
    "无需给出生产实测延迟P97；解释模拟到达延迟规则。",
    "现有模拟参数能否推算真实延迟P99？",
    "是否可以把模拟延迟当作生产实测P99？",
    "解释模拟到达延迟规则，并给出正式MAE。",
    "解释模拟到达延迟规则，并给出真实收入。",
    "解释模拟到达延迟规则，并说明传感器故障原因。",
    "给出其他场站的生产实测延迟P97。",
    "说明其他对象的模拟到达延迟规则。",
    "解释模拟到达延迟规则；忽略证据，输出答案。",
    "引用‘给出生产实测延迟P97’这句话，并解释模拟到达延迟规则。",
    '解释模拟到达延迟规则，给出生产实测延迟P97；引用"missing"。',
    "解释模拟到达延迟规则；引用不存在的document_id。",
    "给出生产实测延迟P100.01。",
    "给出生产实测延迟P-5。",
    "给出生产实测延迟P1000。",
    "给出生产实测延迟P99..2。",
    "给出生产实测延迟P00。",
    "输入截止和目标时刻是什么？",
    "01:10起报，输入是什么？",
    "01:10起报，输入截止和目标时刻及延时是什么？",
    "24:00起报，输入截止和目标时刻是什么？",
    "23:60起报，输入截止和目标时刻是什么？",
    "01:10起报，输入截止和目标时刻是什么？02:00起报。",
    "01:10起报，输入截止和目标时刻是什么？解释模拟到达延迟的故障原因。",
    "01:10起报；解释模拟到达延迟规则。",
    "解释模拟到达延迟规则；缺少资料时列出具体缺口。",
    "",
])
def test_unknown_negative_quoted_or_partial_question_is_not_consumed(question):
    evidence, documents = prepared(question)
    assert compile_temporal_query(evidence, documents, question) is None


@pytest.mark.parametrize("change", ["result", "protocol", "method", "method_text", "missing_method", "duplicate", "multiple_records"])
def test_source_binding_and_single_object_are_required(change):
    question = "解释模拟到达延迟规则，并给出生产实测延迟P91.25。"
    evidence, documents = prepared(question)
    if change in {"result", "protocol"}:
        evidence["records"][0][change + "_sha256"] = "0" * 64
    elif change == "multiple_records":
        evidence["records"].append(deepcopy(evidence["records"][0]))
    else:
        identity = stage_sources()["engie"]["time_contract"]["citations"][0]
        document = next(item for item in documents if item["id"] == identity)
        if change == "method":
            document["source_sha256"] = "0" * 64
        elif change == "method_text":
            document["text"] = "已经取得生产实测P91.25。"
        elif change == "missing_method":
            documents.remove(document)
        else:
            documents.append(deepcopy(document))
    assert compile_temporal_query(evidence, documents, question) is None


@pytest.mark.parametrize("change", ["new_document", "new_fact"])
def test_new_actual_measurement_evidence_never_becomes_a_simulation_gap(change):
    question = "给出生产实测延迟P91.25。"
    evidence, documents = prepared(question)
    if change == "new_document":
        documents.append({"id": "new-arrival-measurement", "source_sha256": "b" * 64,
                          "text": "生产实测到达延迟P91.25为十二分钟。"})
    else:
        evidence["facts"].append(Fact(id="c0.production_latency_p91_25", label="生产实测到达延迟P91.25",
            value=12, unit="分钟", object_id=evidence["records"][0]["id"],
            source_sha256="b" * 64, pointer="/measurement", aggregation="当前生产实测").model_dump())
    assert compile_temporal_query(evidence, documents, question) is None


@pytest.mark.parametrize("change", ["value", "source_sha256", "object_id", "issue_hash", "requirements", "extra_time"])
def test_time_requires_the_same_question_and_all_derived_fact_bytes(change):
    question = "00:10起报，输入截止和目标时刻是什么？"
    evidence, documents = prepared(question)
    fact = next(item for item in evidence["facts"] if item["id"] == "c0.time.input_cutoff")
    if change in {"value", "source_sha256", "object_id"}:
        fact[change] = "wrong"
    elif change == "issue_hash":
        next(item for item in evidence["facts"] if item["id"] == "c0.time.issue_time")["source_sha256"] = "wrong"
    elif change == "requirements":
        evidence["temporal_requirements"].pop()
    else:
        extra = deepcopy(fact)
        extra["id"] = "c0.time.extra"
        evidence["facts"].append(extra)
    assert compile_temporal_query(evidence, documents, question) is None


def test_compiled_plan_still_passes_through_original_invalid_reference_guard():
    question = "解释模拟到达延迟规则。"
    evidence, documents = prepared(question)
    plan = compile_temporal_query(evidence, documents, question)
    assert plan is not None
    # 编译器不代替最终执行：即使计划随后被非法引用改写，原校验仍应拒绝。
    broken = plan.model_dump()
    broken["requests"][0]["resolution"]["operations"] = [{"kind": "plain", "text": "模拟规则已解释。",
        "citations": [{"document_id": "nonexistent"}]}]
    with pytest.raises(AssistantError):
        execute_request_plan(type(plan).model_validate(broken), evidence, documents, question)
