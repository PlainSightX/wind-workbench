"""合法/非法相邻输入走同一合同；合成回答不代表模型质量。"""

import re

import pytest
from pydantic import ValidationError

from power_forecast_service.assistant.contracts import AssistantError, ReferenceAnswer
from power_forecast_service.assistant.references import reference_schema, resolve_reference_answer
from power_forecast_service.assistant.validation import validate_answer
from tests.unit.assistant.test_stages import setup_question


def gap(text="所选结果缺少电价、交易量和收益核算，无法从预测误差推出交易收入。", **changes):
    body = {"kind": "insufficient", "text": text, "missing_evidence": ["电价和交易量", "收入核算"], "citations": []}
    body.update(changes)
    return ReferenceAnswer(status="insufficient_evidence", body=body)


def test_refusal_has_legal_generation_parse_and_render_path_for_incidental_stage_keywords():
    question = "只依据所选结果，能给出采用共享收缩后真实电力交易收入增加的金额吗？"
    evidence, documents = setup_question("engie", question)
    assert evidence["stage_requirements"]
    schema = reference_schema(evidence, documents)
    assert schema["oneOf"][1]["properties"]["body"]["$ref"].endswith("InsufficientReferenceBody")
    payload = gap()
    pattern = schema["$defs"]["InsufficientReferenceBody"]["properties"]["text"]["pattern"]
    assert re.fullmatch(pattern, payload.body.text)
    parsed = ReferenceAnswer.model_validate_json(payload.model_dump_json())
    draft = resolve_reference_answer(parsed, documents)
    result = validate_answer(draft, evidence, documents, question)
    assert result["status"] == "insufficient_evidence" and "缺少电价" in result["answer"]


@pytest.mark.parametrize("status,kind", [("answered", "insufficient"), ("insufficient_evidence", "plain")])
def test_status_cannot_disagree_with_body(status, kind):
    body = gap().body.model_dump()
    body["kind"] = kind
    if kind == "plain":
        body.pop("missing_evidence")
    with pytest.raises(ValidationError):
        ReferenceAnswer(status=status, body=body)


@pytest.mark.parametrize("text,code", [
    ("没有依据，但可报告{{unknown.money}}。", "answer_fact_invalid"),
    ("证据不足。{{c0.persistence.mae}} GW", "answer_fact_unit_conflict"),
    ("开发阶段保持原模型。", "answer_refusal_missing_reason"),
])
def test_refusal_does_not_bypass_shared_checks(text, code):
    evidence, documents = setup_question("engie")
    draft = resolve_reference_answer(gap(text), documents)
    with pytest.raises(AssistantError, match=code):
        validate_answer(draft, evidence, documents, "收入增加多少？")


def test_available_metric_cannot_be_evaded_by_refusal():
    evidence, documents = setup_question("engie")
    draft = resolve_reference_answer(gap("没有MAE依据，无法回答。", missing_evidence=["MAE原值"]), documents)
    with pytest.raises(AssistantError, match="answer_refusal_answerable"):
        validate_answer(draft, evidence, documents, "persistence MAE原值是多少？")
    # 已有总体指标不证明按工况细分的值存在。
    assert validate_answer(draft, evidence, documents, "各工况的persistence MAE是多少？")["status"] == "insufficient_evidence"


@pytest.mark.parametrize("value", [False, 0, 0.0, "否"])
@pytest.mark.parametrize("identity", ["c0.adopted", "`c0.adopted`", "{{c0.adopted}}"])
def test_present_fact_is_not_missing_even_when_false_or_zero(value, identity):
    evidence, documents = setup_question("engie")
    fact = next(item for item in evidence["facts"] if item["id"] == "c0.adopted")
    fact["value"] = value
    draft = resolve_reference_answer(gap("缺少采用状态，不能判断。", missing_evidence=[identity]), documents)
    with pytest.raises(AssistantError, match="answer_refusal_present_fact"):
        validate_answer(draft, evidence, documents, "宣布采用门通过，默认已替换。")


@pytest.mark.parametrize("missing", ["收入核算", "c0.unavailable_revenue", "c0.adopted 的业务收益证明"])
def test_existing_adoption_fact_does_not_prove_unavailable_business_evidence(missing):
    evidence, documents = setup_question("engie")
    draft = resolve_reference_answer(gap(missing_evidence=[missing]), documents)
    assert validate_answer(draft, evidence, documents, "采用后收入多少？")["status"] == "insufficient_evidence"


@pytest.mark.parametrize("quarter", ["Q1", "Q2", "Q3", "Q4"])
def test_all_quarters_are_reachable_through_generation_and_runtime(quarter):
    evidence = {"records": [{"quarter": "2014-" + quarter, "models": {"ridge": {}}}],
                "facts": [{"id": "c0.ridge.mae", "label": "ridge MAE", "value": 4.5, "unit": "kW"}]}
    text = "所选对象是" + quarter + "，MAE为{{c0.ridge.mae}}。"
    schema = reference_schema(evidence, [])
    assert re.fullmatch(schema["$defs"]["PlainReferenceBody"]["properties"]["text"]["pattern"], text)
    payload = ReferenceAnswer(status="answered", body={"kind": "plain", "text": text, "citations": []})
    assert validate_answer(resolve_reference_answer(payload, []), evidence, [])["status"] == "answered"
    wrong = payload.model_copy(deep=True)
    wrong.body.text = "所选对象是" + ("Q2" if quarter == "Q1" else "Q1") + "，MAE为{{c0.ridge.mae}}。"
    with pytest.raises(AssistantError, match="answer_object_quarter_conflict"):
        validate_answer(resolve_reference_answer(wrong, []), evidence, [])


@pytest.mark.parametrize("text", ["所选对象不是Q1，而是Q2。", "如果所选对象是Q1，就需要另一份数据。", "Q1不能替代当前结果。"])
def test_negation_and_hypothetical_quarter_discussion_remain_legal(text):
    evidence = {"records": [{"quarter": "2014-Q2", "models": {}}], "facts": []}
    doc = {"id": "d", "title": "边界", "text": "只讨论当前对象。", "source_sha256": "a" * 64}
    payload = ReferenceAnswer(status="answered", body={"kind": "plain", "text": text, "citations": [{"document_id": "d"}]})
    assert validate_answer(resolve_reference_answer(payload, [doc]), evidence, [doc])["status"] == "answered"


def test_nominal_identifiers_are_bound_to_request_not_arbitrary_numbers():
    from power_forecast_service.assistant.answerability import bind_nominal_identifiers
    from power_forecast_service.assistant.contracts import DraftAnswer

    question = "有P95与P99.9实测值吗？"
    data = bind_nominal_identifiers({"facts": [], "records": [{"quarter": "2014-Q3", "models": {}}]}, question)
    assert data["nominal_identifiers"] == ["2014-Q3", "P95", "P99.9"]
    assert "nominal_identifiers" not in bind_nominal_identifiers({"records": []}, "P999和P99.9%及P95foo")
    untouched = {"facts": [], "records": [{"models": {}}]}
    assert bind_nominal_identifiers(untouched, "说明MAE") == untouched
    pattern = reference_schema(data, [])["$defs"]["InsufficientReferenceBody"]["properties"]["text"]["pattern"]
    text = "所选期间2014-Q3没有P95或P99.9生产测量证据。"
    assert re.fullmatch(pattern, text)
    assert not re.fullmatch(pattern, "没有P96测量证据。")
    draft = DraftAnswer(status="insufficient_evidence", answer=text, missing_evidence=["生产分位数测量"])
    assert validate_answer(draft, data, [], question)["status"] == "insufficient_evidence"
