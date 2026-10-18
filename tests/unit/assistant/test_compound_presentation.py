"""复合呈现只去除可证明的重复；真实缺口、来源限制和原计划保持可观察。"""
from copy import deepcopy

import pytest

from power_forecast_service.assistant.contracts import AssistantError, DraftAnswer, RequestAnswerPart
from power_forecast_service.assistant.program import (
    ProgramAnswer, RequestedPlan, RequestPlan, compose_parts, execute_program, missing_evidence_text,
    validate_requested_measurement_gap,
)
from power_forecast_service.assistant.validation import validate_answer
from power_forecast_service.assistant.workflow import prepare_answer_context
from power_forecast_service.assistant.retrieval import corpus
from tests.unit.assistant.test_program import evidence
from tests.unit.assistant.test_stages import setup_question


def supported(request, operations):
    return {"request": request, "resolution": {"kind": "supported", "operations": operations}}


def gap(name):
    return {"request": name, "resolution": {"kind": "unavailable", "missing": name}}


def temporal_context(question):
    data, _ = setup_question("engie", question)
    return prepare_answer_context(data, corpus()["chunks"], question, "program_requested")


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("same_request", [False, True])
def test_separate_complete_time_removes_only_automatic_temporal_closure(reverse, same_request):
    question = "23:45起报，列出最晚输入标签和目标时刻，解释模拟到达延迟，并给出生产实测延迟P97。"
    data, docs = temporal_context(question)
    time = {"kind": "time", "context": "c0"}
    policy = {"kind": "delay_policy", "context": "c0"}
    requests = ([supported("情景时间与规则", [time, policy])] if same_request else
        [supported("情景时间", [time]), supported("模拟规则", [policy])]) + [gap("生产实测延迟P97")]
    if reverse:
        requests.reverse()
    raw = {"requests": requests}
    plan = RequestedPlan.model_validate(raw)
    before = plan.model_dump()
    draft = execute_program(plan, data, docs, question)
    result = validate_answer(draft, data, docs, question)
    assert plan.model_dump() == before
    for label in ("用户设定的情景起报时刻：", "输入观测标签最晚时刻：", "预测目标时刻：", "延迟依据："):
        assert result["answer"].count(label) == 1
    # 规则仍读取并说明同一延迟参数；不是通过删除实际依据来取得干净文本。
    assert "不是生产到达时间的实测分布" in result["answer"]
    assert {f["id"] for f in result["facts"]} >= set(data["temporal_requirements"])
    assert "P97" in result["answer"] and result["status"] == "insufficient_evidence"
    legacy = validate_answer(execute_program(RequestPlan.model_validate(raw), data, docs, question), data, docs, question)
    assert legacy["answer"].count("预测目标时刻：") == 2


def test_policy_without_separate_time_keeps_full_temporal_obligations():
    question = "23:45起报，列出最晚输入标签和目标时刻，解释模拟到达延迟。"
    data, docs = temporal_context(question)
    plan = RequestedPlan.model_validate({"requests": [supported("模拟规则", [{"kind": "delay_policy", "context": "c0"}])]})
    draft = execute_program(plan, data, docs, question)
    result = validate_answer(draft, data, docs, question)
    assert {f["id"] for f in result["facts"]} >= set(data["temporal_requirements"])
    assert result["answer"].count("预测目标时刻：") == 1


def test_time_for_invalid_context_cannot_silently_satisfy_real_obligation():
    question = "23:45起报，列出最晚输入标签和目标时刻，解释模拟到达延迟。"
    data, docs = temporal_context(question)
    plan = RequestedPlan.model_validate({"requests": [supported("规则", [{"kind": "delay_policy", "context": "c0"}]),
        supported("别的对象时间", [{"kind": "time", "context": "c1"}])]})
    with pytest.raises(AssistantError):
        execute_program(plan, data, docs, question)


@pytest.mark.parametrize("reverse", [False, True])
def test_grouped_gaps_keep_every_resource_and_original_request_attribution(reverse):
    requests = [supported("指标", [{"kind": "values", "fact_ids": ["c0.ridge.mae"]}]),
        gap("实际收入凭证"), gap("结算凭证"), gap("故障凭证")]
    if reverse:
        requests.reverse()
    plan = RequestedPlan.model_validate({"requests": requests})
    original = plan.model_dump()
    draft = execute_program(plan, evidence(), [])
    before = deepcopy(draft.parts)
    result = validate_answer(draft, evidence(), [])
    assert plan.model_dump() == original and draft.parts == before
    assert result["answer"].count("缺少以下所问信息的证据：") == 1
    assert result["answer"].count("不能用现有的其他指标") == 1
    assert set(draft.missing_evidence) == {"实际收入凭证", "结算凭证", "故障凭证"}
    assert [part.request_index for part in draft.parts] == list(range(4))
    for name in draft.missing_evidence:
        assert result["answer"].count(name) == 1
    legacy = execute_program(RequestPlan.model_validate({"requests": requests}), evidence(), [])
    assert validate_answer(legacy, evidence(), [])["answer"].count("缺少以下所问信息的证据：") == 3
    draft.answer = draft.answer.replace("结算凭证", "")
    with pytest.raises(AssistantError, match="answer_program_composition_mismatch"):
        validate_answer(draft, evidence(), [])


def test_grouping_does_not_swallow_noncanonical_gap_body():
    canonical = DraftAnswer(status="insufficient_evidence", answer=missing_evidence_text(["资料甲"]), missing_evidence=["资料甲"])
    noncanonical = DraftAnswer(status="insufficient_evidence", answer="这项说明必须保留。", missing_evidence=["资料乙"])
    parts = [RequestAnswerPart(request_index=0, draft=canonical), RequestAnswerPart(request_index=1, draft=noncanonical)]
    original = deepcopy(parts)
    composed = compose_parts(parts, evidence(), group_gaps=True)
    assert "这项说明必须保留。" in composed.answer and parts == original
    assert composed.missing_evidence == ["资料甲", "资料乙"]


def doc(identity, text):
    return {"id": identity, "text": text, "title": "已选原文", "source_sha256": "a" * 64}


def reference_plan(text, resource="结算凭证", *, staged=False):
    operation = {"kind": "plain", "text": text, "citations": [{"document_id": "doc-a"}]}
    if staged:
        operation = {"kind": "staged", "claims": [{"object_id": evidence()["records"][0]["id"],
            "role": "development_result", "text": text, "citations": [{"document_id": "doc-a"}]}]}
    return RequestedPlan.model_validate({"requests": [gap(resource), supported("资料说明", [operation])]})


@pytest.mark.parametrize("text", ["结算凭证缺失", "结算凭证尚不可获取", "缺少结算凭证", "结算凭证只是未来可能提供的资料"])
def test_general_method_citation_cannot_support_declared_gap_restatement(text):
    plan = reference_plan(text)
    before = plan.model_dump()
    with pytest.raises(AssistantError, match="answer_program_gap_reference"):
        execute_program(plan, evidence(), [doc("doc-a", "预测误差改善不代表收入。")])
    assert plan.model_dump() == before


@pytest.mark.parametrize("source", ["不能认定结算凭证缺失。", "若结算凭证缺失，需要另行查询。"])
def test_gap_reference_cannot_truncate_negation_or_condition(source):
    with pytest.raises(AssistantError, match="answer_program_gap_reference"):
        execute_program(reference_plan("结算凭证缺失"), evidence(), [doc("doc-a", source)])


def test_gap_reference_requires_its_own_citation_not_another_available_document():
    with pytest.raises(AssistantError, match="answer_program_gap_reference"):
        execute_program(reference_plan("结算凭证缺失"), evidence(), [doc("doc-a", "常规方法说明。"), doc("doc-b", "结算凭证缺失。")])


def test_gap_reference_accepts_full_source_sentence_and_keeps_unavailable():
    text = "结算凭证的用途是核对金额"
    data, docs = evidence(), [doc("doc-a", text + "。另有方法说明。")]
    draft = execute_program(reference_plan(text), data, docs)
    result = validate_answer(draft, data, docs)
    assert text in result["answer"] and draft.missing_evidence == ["结算凭证"]
    assert result["status"] == "insufficient_evidence" and len(result["citations"]) == 1


def test_staged_reference_cannot_borrow_source_for_declared_gap():
    with pytest.raises(AssistantError, match="answer_program_gap_reference"):
        execute_program(reference_plan("结算凭证缺失", staged=True), evidence(), [doc("doc-a", "一般方法说明。")])


@pytest.mark.parametrize("staged", [False, True])
def test_gap_feedback_locates_bad_operation_before_workflow_truncation(staged):
    plan = reference_plan("结算凭证缺失", staged=staged)
    with pytest.raises(AssistantError) as caught:
        execute_program(plan, evidence(), [doc("doc-a", "一般方法说明。")])
    path = "requests[1].resolution.operations[0]" + (".claims[0]" if staged else "")
    feedback = caught.value.detail[:240]
    assert path in feedback and all(text in feedback for text in ("结算凭证", "删除", "完整原文句", "unavailable", "已有事实"))


@pytest.mark.parametrize("staged", [False, True])
def test_pure_coverage_repetition_requests_repair_and_keeps_raw_plan(staged):
    question = "请说明合法输入与实际输出的次数和各自定义。"
    op = {"kind":"plain", "text":"合法输入与实际输出具有各自定义。", "citations":[{"document_id":"doc-a"}]}
    if staged:
        op = {"kind":"staged", "claims":[{"object_id":evidence()["records"][0]["id"],
            "role":"development_result", "text":"合法输入与实际输出具有各自定义。", "citations":[{"document_id":"doc-a"}]}]}
    plan = RequestedPlan.model_validate({"requests":[
        supported(question, [{"kind":"coverage", "context":"c0", "counts":["input_valid", "output_count"]}]),
        supported(question, [op]), gap("实际收入凭证")]})
    before = plan.model_dump()
    with pytest.raises(AssistantError) as caught:
        execute_program(plan, evidence(), [doc("doc-a", "合法输入与实际输出具有各自定义。")], question)
    assert caught.value.code == "answer_program_coverage_repetition"
    assert "requests[1].resolution.operations[0]" in caught.value.detail[:240]
    assert all(text in caught.value.detail[:240] for text in ("删除", "coverage", "各自定义", "真实缺口"))
    assert plan.model_dump() == before
    repaired = RequestedPlan.model_validate({"requests":[before["requests"][0], before["requests"][2]]})
    result = validate_answer(execute_program(repaired, evidence(), [], question), evidence(), [], question)
    assert {f["id"] for f in result["facts"]} == {"c0.input_valid", "c0.output_count"}
    assert result["status"] == "insufficient_evidence" and "实际收入凭证" in result["answer"]


@pytest.mark.parametrize("change", ["extra_question", "different_request", "all_counts", "wrong_context", "multiple_contexts"])
def test_coverage_guard_does_not_reject_distinct_or_unqualified_explanations(change):
    from power_forecast_service.assistant.program import validate_requested_coverage_repetition
    question = "请说明合法输入与实际输出的次数和各自定义。"
    source = evidence()
    coverage = {"kind":"coverage", "context":"c0", "counts":["input_valid", "output_count"]}
    request = question
    if change == "extra_question":
        question += "并解释质量控制方法。"
    elif change == "different_request":
        request = "解释质量控制方法"
    elif change == "all_counts":
        coverage.pop("counts")
    elif change == "wrong_context":
        coverage["context"] = "c1"
    else:
        source["records"].append({**source["records"][0], "id":"another"})
    response = RequestedPlan.model_validate({"requests":[supported(question, [coverage]),
        supported(request, [{"kind":"plain", "text":"质量控制有明确方法。", "citations":[{"document_id":"doc-a"}]}])]})
    validate_requested_coverage_repetition(response, source, question)


def test_unrelated_method_paraphrase_is_not_forced_into_extractive_format():
    data, docs = evidence(), [doc("doc-a", "模型用于预测而非结算。")]
    draft = execute_program(reference_plan("预测实验与业务核算的目的不同", resource="故障诊断记录"), data, docs)
    assert "预测实验与业务核算的目的不同" in validate_answer(draft, data, docs)["answer"]


def test_distinct_gap_and_measurement_obligations_are_not_repaired_by_grouping():
    question = "解释模拟到达延迟，并给出生产实测延迟P98.5。"
    data, docs = temporal_context(question)
    plan = RequestedPlan.model_validate({"requests": [supported("规则", [{"kind": "delay_policy", "context": "c0"}]), gap("实际收入凭证")]})
    with pytest.raises(AssistantError, match="answer_program_measurement_gap"):
        execute_program(plan, data, docs, question)


@pytest.mark.parametrize("plan_type", [RequestPlan, RequestedPlan])
@pytest.mark.parametrize("missing", [None, "实际收入凭证", "生产实测延迟P97的到达日志"])
def test_measurement_repair_feedback_uses_current_request_contract_without_relaxing_gap(plan_type, missing):
    question = "解释模拟到达延迟，并给出生产实测延迟P98.5。"
    data, docs = temporal_context(question)
    requests = [supported("模拟规则", [{"kind": "delay_policy", "context": "c0"}])]
    if missing:
        requests.append(gap(missing))
    plan = plan_type.model_validate({"requests": requests})
    with pytest.raises(AssistantError) as caught:
        execute_program(plan, data, docs, question)
    assert caught.value.code == "answer_program_measurement_gap"
    feedback = caught.value.detail
    assert all(term in feedback for term in ("P98.5", "requests", "request", "resolution.kind", "unavailable", "resolution.missing", "到达日志", "supported"))
    assert "unknown" not in feedback and "body" not in feedback


@pytest.mark.parametrize("plan_type", [RequestPlan, RequestedPlan])
def test_supported_policy_and_attributed_measurement_gap_still_deliver(plan_type):
    question = "解释模拟到达延迟，并给出生产实测延迟P98.5。"
    data, docs = temporal_context(question)
    plan = plan_type.model_validate({"requests": [
        supported("模拟规则", [{"kind": "delay_policy", "context": "c0"}]),
        {"request": "生产实测延迟P98.5", "resolution": {"kind": "unavailable", "missing": "实际到达日志"}},
    ]})
    original = plan.model_dump()
    result = validate_answer(execute_program(plan, data, docs, question), data, docs, question)
    assert result["status"] == "insufficient_evidence"
    assert "P98.5" in result["answer"] and "实际到达日志" in result["answer"]
    assert "不是生产到达时间的实测分布" in result["answer"]
    assert plan.model_dump() == original


def test_legacy_measurement_feedback_keeps_body_contract():
    question = "解释模拟到达延迟，并给出生产实测延迟P98.5。"
    data, docs = temporal_context(question)
    plan = ProgramAnswer.model_validate({"request_summary": question, "evidence_assessment": "只有模拟协议",
        "status": "answered", "body": {"kind": "program", "operations": [{"kind": "delay_policy", "context": "c0"}]}})
    with pytest.raises(AssistantError) as caught:
        validate_requested_measurement_gap(plan, data, docs, question)
    assert caught.value.code == "answer_program_measurement_gap"
    assert all(term in caught.value.detail for term in ("body", "partial", "supported", "missing_evidence", "gap"))
    assert "resolution" not in caught.value.detail
