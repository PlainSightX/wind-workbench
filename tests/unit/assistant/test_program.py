"""比较结果随输入变化；执行正确性与模型选对操作是两层证据。"""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from power_forecast_service.assistant.contracts import AssistantError
from power_forecast_service.assistant.program import (EvidencePlan, ProgramAnswer, RequestPlan,
    execute_program, program_schema, request_program_schema, prepare_program_evidence)
from power_forecast_service.assistant.validation import validate_answer
from power_forecast_service.assistant.workflow import response_contract, output_schema

IDENTITY = "00000000-0000-0000-0000-000000000001"
ASSESSMENT = {"request_summary":"查询所选对象已有指标。", "evidence_assessment":"所选对象提供对应指标。"}


def evidence():
    facts=[]
    for key,value in (("ridge.mae",12.0),("persistence.mae",10.0),("ridge.rmse",13.0),("persistence.rmse",15.0),
                      ("planned",100),("input_valid",90),("output_count",90),("scoreable",80)):
        facts.append({"id":"c0."+key,"value":value,"label":key,"unit":"次" if "." not in key else "kW",
            "object_id":IDENTITY,"aggregation":"same","stage":"selected_result","source_sha256":"a"*64,"pointer":"/"+key})
    return {"records":[{"id":IDENTITY,"quarter":"2014-Q4","models":{"ridge":{},"persistence":{}},
        "interpretation_constraints":{"coverage":"输入拒绝与输出后无法评分不同。"}}],
        "facts":facts,"nominal_identifiers":["2014-Q4"],"stage_options":[],"stage_requirements":[]}


def plan(*operations):
    return ProgramAnswer(**ASSESSMENT,status="answered",body={"kind":"program","operations":list(operations)})


def test_opt_in_comparison_summary_computes_conjunction_without_changing_default():
    from power_forecast_service.assistant.program import PresentationPolicy
    source = evidence()
    response = plan({"kind":"compare", "left":"c0.ridge.mae", "right":"c0.persistence.mae"},
                    {"kind":"compare", "left":"c0.ridge.rmse", "right":"c0.persistence.rmse"})
    legacy = execute_program(response, source, []).answer
    policy = PresentationPolicy(requested_only=True, comparison_summary=True)
    assert "不能认定前者各项误差同时严格更小" in execute_program(response, source, [], presentation=policy).answer
    assert execute_program(response, source, []).answer == legacy
    source["facts"][0]["value"] = 9
    assert "前者各项误差均严格更小" in execute_program(response, source, [], presentation=policy).answer
    source["facts"][0]["value"] = 10
    assert "不能认定" in execute_program(response, source, [], presentation=policy).answer


@pytest.mark.parametrize("left_value,expected", [(9, "前者各项误差均严格更小"), (10, "不能认定"), (12, "不能认定")])
def test_explicit_question_recovers_missing_conjunction_without_mutating_plan(left_value, expected):
    from power_forecast_service.assistant.program import RequestedPlan
    source = evidence()
    source["facts"][0]["value"] = left_value
    response = RequestedPlan.model_validate({"requests":[{"request":"两项比较", "resolution":{"kind":"supported", "operations":[
        {"kind":"compare", "left":"c0.ridge.mae", "right":"c0.persistence.mae"},
        {"kind":"compare", "left":"c0.ridge.rmse", "right":"c0.persistence.rmse"} ]}}]})
    original = response.model_dump()
    question = "ridge对比persistence，MAE和RMSE两项是否均严格更小？请给原值。"
    result = validate_answer(execute_program(response, source, [], question), source, [], question)
    assert expected in result["answer"] and len(result["facts"]) == 4
    assert response.model_dump() == original


@pytest.mark.parametrize("question", [
    "ridge对比persistence，MAE和RMSE分别是多少？",
    "ridge对比persistence，MAE和RMSE两项是否均严格更小？不要总结。",
    "ridge对比persistence，MAE和RMSE两项是否均严格更小？只给原值。",
    "ridge对比persistence，MAE和RMSE任一项是否严格更小？",
    "persistence对比ridge，MAE和RMSE两项是否均严格更小？",
    "ridge对比persistence，不要判断MAE和RMSE两项是否均严格更小，仅列出原值。",
    "旧问题是‘ridge对比persistence，MAE和RMSE两项是否均严格更小’，现在仅列出原值。",
    "ridge对比persistence，MAE和RMSE两项是否均严格更小？另说明统计方法。",
])
def test_question_does_not_add_unrequested_or_reversed_conjunction(question):
    from power_forecast_service.assistant.program import RequestedPlan
    response = RequestedPlan.model_validate(presented([
        {"kind":"compare", "left":"c0.ridge.mae", "right":"c0.persistence.mae"},
        {"kind":"compare", "left":"c0.ridge.rmse", "right":"c0.persistence.rmse"}]))
    result = execute_program(response, evidence(), [], question)
    assert "各项误差" not in result.answer


@pytest.mark.parametrize("change", ["unit", "stage", "aggregation", "duplicate", "missing", "bias", "other_context"])
def test_question_conjunction_requires_compatible_two_metrics(change):
    from power_forecast_service.assistant.program import RequestedPlan, requested_comparison_policy
    source = requested_evidence()
    operations = [
        {"kind":"compare", "left":"c0.ridge.mae", "right":"c0.persistence.mae"},
        {"kind":"compare", "left":"c0.ridge.rmse", "right":"c0.persistence.rmse"}]
    if change in {"unit", "stage", "aggregation"}:
        # 每对本身合法，但两对不属于同一适用范围。
        for fact in source["facts"]:
            if fact["id"].endswith(".rmse"):
                fact[change] = "other"
    elif change == "duplicate":
        operations[1] = deepcopy(operations[0])
    elif change == "missing":
        operations.pop()
    elif change == "bias":
        operations[1] = {"kind":"compare", "left":"c0.ridge.bias", "right":"c0.persistence.bias"}
    else:
        operations[1] = {"kind":"compare", "left":"c1.ridge.rmse", "right":"c1.persistence.rmse"}
    response = RequestedPlan.model_validate(presented(operations))
    policy = requested_comparison_policy(response.requests[0], response.presentation_policies()[0], source,
        "ridge对比persistence，MAE和RMSE两项是否同时严格更小？")
    assert not policy.comparison_summary


def test_comparison_summary_stays_on_comparison_when_coverage_follows():
    from power_forecast_service.assistant.program import PresentationPolicy
    response = plan(
        {"kind":"compare", "left":"c0.ridge.mae", "right":"c0.persistence.mae"},
        {"kind":"compare", "left":"c0.ridge.rmse", "right":"c0.persistence.rmse"},
        {"kind":"coverage", "context":"c0", "counts":["input_valid", "output_count"]})
    draft = execute_program(response, evidence(), [], presentation=PresentationPolicy(comparison_summary=True))
    assert draft.answer.index("不能认定") < draft.answer.index("{{c0.input_valid}}")


def test_extra_metric_whitelist_is_explicit_and_does_not_allow_mixed_metrics():
    from power_forecast_service.assistant.program import comparison_facts
    facts = {f["id"]:f for f in evidence()["facts"]}
    for model in ("ridge", "persistence"):
        facts[f"c0.{model}.MASE"] = {**facts[f"c0.{model}.mae"], "id":f"c0.{model}.MASE", "unit":"dimensionless"}
    with pytest.raises(AssistantError):
        comparison_facts("c0.ridge.MASE", "c0.persistence.MASE", facts)
    comparison_facts("c0.ridge.MASE", "c0.persistence.MASE", facts, extra_metrics=frozenset({"MASE"}))
    with pytest.raises(AssistantError):
        comparison_facts("c0.ridge.MASE", "c0.persistence.mae", facts, extra_metrics=frozenset({"MASE"}))


def requested_evidence():
    source = evidence()
    for model, value in (("ridge", -2.0), ("persistence", 0.0)):
        source["facts"].append({**source["facts"][0], "id":f"c0.{model}.bias",
                                "label":f"{model} BIAS", "value":value})
    return source


def presented(operations, options=None):
    return {"requests":[{"request":"按所问解释", "resolution":{"kind":"supported", "operations":operations},
                          **({"presentation":options} if options is not None else {})}]}


def test_requested_contract_reaches_rendering_with_bias_scope_and_raw_plan_intact():
    import jsonschema
    response_type, _, version = response_contract("program_requested")
    source = requested_evidence()
    raw = presented([{"kind":"values", "fact_ids":["c0.ridge.bias", "c0.persistence.bias"]}],
                    {"bias_fact_ids":["c0.ridge.bias"]})
    jsonschema.validate(raw, output_schema("program_requested", source, [], []))
    response = response_type.model_validate(raw)
    before = response.model_dump()
    result = validate_answer(execute_program(response, source, []), source, [])
    assert "-2 kW" in result["answer"] and "0 kW" in result["answer"]
    assert "ridge BIAS" in result["answer"] and "原值不为零" in result["answer"]
    assert response.model_dump() == before and version.endswith("requested-presentation")
    assert len(result["facts"]) == 2
    without = response_type.model_validate(presented(raw["requests"][0]["resolution"]["operations"]))
    answer = validate_answer(execute_program(without, source, []), source, [])
    assert "原值不为零" not in answer["answer"]
    with pytest.raises(ValidationError):
        RequestPlan.model_validate(raw)


@pytest.mark.parametrize("options", [
    {"bias_fact_ids":["c0.persistence.bias"]},
    {"bias_fact_ids":["c0.ridge.bias", "c0.ridge.bias"]},
    {"bias_fact_ids":["c0.ridge.mae"]},
    {"bias_fact_ids":["c9.ridge.bias"]},
    {"comparison_summary":True},
    {"comparison_summary":"true"},
    {"extra_metrics":["MASE"]},
    {"requested_only":False},
])
def test_requested_presentation_cannot_fabricate_or_expand_policy(options):
    response_type, _, _ = response_contract("program_requested")
    with pytest.raises(ValidationError):
        response_type.model_validate(presented([{"kind":"values", "fact_ids":["c0.ridge.bias"]}], options))


@pytest.mark.parametrize("options", [{}, {"comparison_summary":False}, {"bias_fact_ids":[]}])
def test_unavailable_rejects_presentation_in_both_schema_and_runtime(options):
    from jsonschema import Draft202012Validator
    response_type, _, _ = response_contract("program_requested")
    raw = {"requests":[{"request":"真实收入", "resolution":{"kind":"unavailable", "missing":"实际收入凭证"},
                        "presentation":options}]}
    assert not Draft202012Validator(output_schema("program_requested", evidence(), [], [])).is_valid(raw)
    with pytest.raises(ValidationError):
        response_type.model_validate(raw)


def test_requested_conjunction_changes_with_facts_and_rejects_wrong_direction_metric():
    response_type, _, _ = response_contract("program_requested")
    raw = presented([{"kind":"compare", "left":"c0.ridge.mae", "right":"c0.persistence.mae"},
                     {"kind":"compare", "left":"c0.ridge.rmse", "right":"c0.persistence.rmse"}],
                    {"comparison_summary":True})
    response = response_type.model_validate(raw)
    source = evidence()
    result = validate_answer(execute_program(response, source, []), source, [])
    assert "不能认定前者各项误差同时严格更小" in result["answer"]
    source["facts"][0]["value"] = 9
    assert "前者各项误差均严格更小" in validate_answer(execute_program(response, source, []), source, [])["answer"]
    raw["requests"][0]["resolution"]["operations"][1] = {
        "kind":"compare", "left":"c0.ridge.bias", "right":"c0.persistence.bias"}
    with pytest.raises(AssistantError, match="answer_program_summary"):
        execute_program(response_type.model_validate(raw), requested_evidence(), [])


def test_requested_policy_cannot_be_overridden_by_research_auxiliary_metrics():
    from power_forecast_service.assistant.program import PresentationPolicy
    response_type, _, _ = response_contract("program_requested")
    response = response_type.model_validate(presented([{"kind":"values", "fact_ids":["c0.ridge.mae"]}]))
    with pytest.raises(ValueError, match="requested_plan_owns_presentation"):
        execute_program(response, evidence(), [], presentation=[PresentationPolicy(extra_metrics=frozenset({"MASE"}))])


def test_requested_compare_repair_without_any_valid_pair_keeps_noncomparison_options():
    from jsonschema import Draft202012Validator
    from power_forecast_service.assistant.program import comparison_repair_schema
    response_type, _, _ = response_contract("program_requested")
    source = evidence()
    source["facts"] = source["facts"][:1]
    raw = presented([{"kind":"compare", "left":"c0.ridge.mae", "right":"c0.ridge.mae"}])
    schema, invalid = comparison_repair_schema(output_schema("program_requested", source, [], []),
                                               response_type.model_validate(raw), source)
    assert invalid and "Compare" not in schema["$defs"]
    validator = Draft202012Validator(schema)
    assert not validator.is_valid(raw)
    assert validator.is_valid(presented([{"kind":"values", "fact_ids":["c0.ridge.mae"]}]))


def test_requested_reconciliation_preserves_presentation_and_true_missing_request():
    response_type, _, _ = response_contract("program_requested")
    source = requested_evidence()
    for name, value in (("gain_ci_low", -2.), ("gain_ci_high", 3.)):
        source["facts"].append({**source["facts"][0], "id":"c0." + name, "value":value, "unit":"%"})
    raw = presented([{"kind":"values", "fact_ids":["c0.gain_ci_low", "c0.gain_ci_high", "c0.ridge.bias"]}],
                    {"bias_fact_ids":["c0.ridge.bias"]})
    raw["requests"] += [
        {"request":"适用范围", "resolution":{"kind":"unavailable", "missing":"区间是否能代表换场站"}},
        {"request":"实际收入", "resolution":{"kind":"unavailable", "missing":"实际售电收入凭证"}}]
    response = response_type.model_validate(raw)
    before, events = response.model_dump_json(), []
    result = validate_answer(execute_program(response, source, [], reconciliation_log=events), source, [])
    assert events[0]["rule"] == "selected_interval_scope"
    assert result["status"] == "insufficient_evidence"
    assert all(text in result["answer"] for text in ("不能代表换场站", "原值不为零", "实际售电收入"))
    assert response.model_dump_json() == before


def request_plan(legacy):
    return RequestPlan(requests=[{"request":"查询所选对象或请求缺少的资料。", "resolution":(
        {"kind":"unavailable", "missing":op.request} if op.kind == "unknown"
        else {"kind":"supported", "operations":[op.model_dump()]})} for op in legacy.operations])


def test_request_binding_keeps_missing_first_without_losing_supported_content():
    response = RequestPlan(requests=[
        {"request":"收入", "resolution":{"kind":"unavailable", "missing":"实际售电收入"}},
        {"request":"MAE", "resolution":{"kind":"supported", "operations":[{"kind":"values", "fact_ids":["c0.ridge.mae"]}]}}])
    result = validate_answer(execute_program(response.as_answer(), evidence(), []), evidence(), [])
    assert result["status"] == "insufficient_evidence"
    assert "12" in result["answer"] and "实际售电收入" in result["answer"]
    assert response.requests[0].resolution.kind == "unavailable"


def test_request_schema_and_runtime_both_accept_bound_queries_and_gaps():
    import jsonschema
    response = request_plan(EvidencePlan(**ASSESSMENT, operations=[
        {"kind":"values", "fact_ids":["c0.ridge.mae"]}, {"kind":"unknown", "request":"真实收入"}]))
    jsonschema.validate(response.model_dump(mode="json"), request_program_schema(evidence(), []))
    assert RequestPlan.model_validate_json(response.model_dump_json()) == response
    with pytest.raises(ValidationError):
        RequestPlan(requests=[{"request":"运行命令", "resolution":{"kind":"supported", "operations":[{"kind":"exec", "code":"bad"}]}}])


def test_request_schema_accepts_cross_request_but_rejects_local_rendering_mix():
    import jsonschema
    schema = request_program_schema(evidence(), [])
    mixed = {"requests":[
        {"request":"数值", "resolution":{"kind":"supported", "operations":[{"kind":"values", "fact_ids":["c0.ridge.mae"]}]}},
        {"request":"解释", "resolution":{"kind":"supported", "operations":[{"kind":"plain", "text":"说明方法。", "citations":[]}]}}]}
    jsonschema.validate(mixed, schema)
    RequestPlan.model_validate(mixed)
    for item in mixed["requests"]:
        jsonschema.validate({"requests":[item]}, schema)
    joined = {"requests":[{"request":"混在同一请求", "resolution":{"kind":"supported", "operations":[
        *mixed["requests"][0]["resolution"]["operations"], *mixed["requests"][1]["resolution"]["operations"]]}}]}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(joined, schema)
    with pytest.raises(ValidationError):
        RequestPlan.model_validate(joined)


def composed_plan(*operations):
    return RequestPlan(requests=[{"request":"保留当前分项的依据。", "resolution":
        {"kind":"unavailable", "missing":op["missing"]} if op["kind"] == "unavailable" else
        {"kind":"supported", "operations":[op]}} for op in operations])


def method_document(identity, text):
    return {"id":identity, "title":identity, "text":text, "source_sha256":"a"*64}


def test_multiple_explanations_render_with_their_own_sources_and_real_gap():
    import jsonschema
    docs = [method_document("method-a", "未来标签只用于评分。"), method_document("method-b", "结果只适用于当前对象。")]
    response = composed_plan(
        {"kind":"plain", "text":docs[0]["text"], "citations":[{"document_id":"method-a"}]},
        {"kind":"values", "fact_ids":["c0.ridge.mae"]},
        {"kind":"plain", "text":docs[1]["text"], "citations":[{"document_id":"method-b"}]},
        {"kind":"unavailable", "missing":"实际售电收入"})
    before = response.model_dump()
    jsonschema.validate(before, request_program_schema(evidence(), docs))
    draft = execute_program(response, evidence(), docs)
    assert [part.request_index for part in draft.parts] == [0, 1, 2, 3]
    result = validate_answer(draft, evidence(), docs)
    assert result["status"] == "insufficient_evidence"
    assert all(text in result["answer"] for text in (docs[0]["text"], docs[1]["text"], "12 kW", "实际售电收入"))
    assert {c["id"] for c in result["citations"]} == {"method-a", "method-b"}
    assert response.model_dump() == before


@pytest.mark.parametrize("addition,code", [
    ("已测量999 kW。", "answer_number_unbound"),
    ("指标{{c0.ridge.mae}}。", "answer_program_fact_requires_operation"),
    ("资料没有实测记录。", "answer_program_gap_in_explanation"),
])
def test_composition_cannot_hide_invalid_explanation_behind_valid_metric(addition, code):
    docs = [method_document("method", "未来标签只用于评分。")]
    response = composed_plan({"kind":"values", "fact_ids":["c0.ridge.mae"]},
        {"kind":"plain", "text":addition, "citations":[{"document_id":"method"}]})
    with pytest.raises(AssistantError, match=code):
        validate_answer(execute_program(response,evidence(),docs),evidence(),docs)


def test_composition_cannot_borrow_other_parts_citation_to_validate_a_number():
    docs = [method_document("method-a", "方法参数为87分钟。"), method_document("method-b", "未来标签只用于评分。")]
    response = composed_plan(
        {"kind":"plain", "text":"方法参数来自这份资料。", "citations":[{"document_id":"method-a"}]},
        {"kind":"plain", "text":"窗口为87分钟。", "citations":[{"document_id":"method-b"}]})
    with pytest.raises(AssistantError, match="answer_number_unbound"):
        validate_answer(execute_program(response,evidence(),docs),evidence(),docs)


def test_composition_stages_remain_separate_from_methods_and_missing_evidence():
    from tests.unit.assistant.test_stages import setup_question
    data, docs = setup_question("engie", "开发改善和采用门是多少，通过了吗？")
    docs.append(method_document("method-extra", "未来标签只用于评分。"))
    response = composed_plan(
        {"kind":"values", "fact_ids":["c0.development.lightgbm_l1_shrink.mae_gain"]},
        {"kind":"values", "fact_ids":["c0.development.mae_gain_gate", "c0.adopted"]},
        {"kind":"plain", "text":"未来标签只用于评分。", "citations":[{"document_id":"method-extra"}]},
        {"kind":"unavailable", "missing":"实际售电收入"})
    draft = execute_program(response,data,docs)
    result = validate_answer(draft,data,docs)
    assert result["status"] == "insufficient_evidence"
    assert all(text in result["answer"] for text in ("开发评价", "开发采用决定", "未来标签只用于评分", "实际售电收入"))
    assert "method-extra" not in {c for p in draft.parts for claim in p.draft.stage_claims for c in claim.citations}
    incomplete = composed_plan(response.requests[0].resolution.operations[0].model_dump(),
        {"kind":"unavailable", "missing":"实际售电收入"})
    with pytest.raises(AssistantError, match="answer_stage_incomplete"):
        validate_answer(execute_program(incomplete,data,docs),data,docs)


@pytest.mark.parametrize("count,valid", [(1, True), (10, True), (11, False)])
def test_request_operation_count_schema_and_parser_agree(count, valid):
    import jsonschema
    payload = {"requests":[{"request":"读取", "resolution":{"kind":"supported",
        "operations":[{"kind":"values", "fact_ids":["c0.ridge.mae"]}]*count}}]}
    assert jsonschema.Draft202012Validator(request_program_schema(evidence(),[])).is_valid(payload) == valid
    if valid:
        RequestPlan.model_validate(payload)
    else:
        with pytest.raises(ValidationError):
            RequestPlan.model_validate(payload)


@pytest.mark.parametrize("count,valid", [(12, True), (13, False)])
def test_missing_request_count_has_no_hidden_flattened_limit(count, valid):
    import jsonschema
    payload = {"requests":[{"request":"所问缺失资料", "resolution":{"kind":"unavailable", "missing":"实际售电收入"}}]*count}
    assert jsonschema.Draft202012Validator(request_program_schema(evidence(),[])).is_valid(payload) == valid
    if valid:
        response = RequestPlan.model_validate(payload)
        assert validate_answer(execute_program(response,evidence(),[]),evidence(),[])["status"] == "insufficient_evidence"
    else:
        with pytest.raises(ValidationError):
            RequestPlan.model_validate(payload)


def test_composed_aggregate_body_cannot_be_replaced_after_execution():
    draft = execute_program(composed_plan({"kind":"values", "fact_ids":["c0.ridge.mae"]}),evidence(),[])
    draft.answer += "未校验的结论。"
    with pytest.raises(AssistantError, match="answer_program_composition_mismatch"):
        validate_answer(draft,evidence(),[])


def test_evaluation_period_operation_needs_no_unrequested_metric_or_cutoff():
    data = evidence()
    data["facts"].append({**data["facts"][0], "id":"c0.training_label_available", "value":"2014-09-30T23:50:00+00:00",
        "unit":"UTC", "pointer":"/training_label_available"})
    before = deepcopy(data)
    bound = prepare_program_evidence(data)
    response = plan({"kind":"period", "context":"c0"})
    result = validate_answer(execute_program(response, bound, []), bound, [])
    assert [f["id"] for f in result["facts"]] == ["c0.evaluation_period"]
    assert result["answer"] == "评价期间：2014-Q4。"
    assert data == before and prepare_program_evidence(bound) == bound
    assert request_program_schema(bound, [])["$defs"]["EvaluationPeriod"]["properties"]["context"]["enum"] == ["c0"]
    with pytest.raises(AssistantError, match="answer_program_fact"):
        execute_program(response, data, [])


def test_period_and_selected_metric_have_one_period_and_no_extra_facts():
    data = evidence()
    data["facts"].append({**data["facts"][0], "id":"c0.training_label_available", "value":"prior", "unit":"UTC"})
    data = prepare_program_evidence(data)
    response = plan({"kind":"period", "context":"c0"}, {"kind":"values", "fact_ids":["c0.ridge.rmse"]})
    result = validate_answer(execute_program(response, data, []), data, [])
    assert result["answer"].count("2014-Q4") == 1
    assert {f["id"] for f in result["facts"]} == {"c0.evaluation_period", "c0.ridge.rmse"}


@pytest.mark.parametrize("left,right,expected",[(8.,10.,"低于"),(12.,10.,"高于"),(10.,10.,"等于"),(-2.,-1.,"低于")])
def test_comparison_direction_is_executed_not_generated(left,right,expected):
    data=evidence()
    data["facts"][0]["value"],data["facts"][1]["value"]=left,right
    draft=execute_program(plan({"kind":"compare","left":"c0.ridge.mae","right":"c0.persistence.mae"}),data,[])
    assert "前者"+expected+"后者" in draft.answer
    result=validate_answer(draft,data,[])
    assert result["status"]=="answered" and "2014-Q4" in result["answer"]
    assert len(result["facts"])==2


def test_two_metrics_keep_opposite_directions():
    result=execute_program(plan(*({"kind":"compare","left":"c0.ridge."+metric,"right":"c0.persistence."+metric} for metric in ("mae","rmse"))),evidence(),[])
    assert "前者高于后者" in result.answer and "前者低于后者" in result.answer
    assert len(result.fact_ids)==4


@pytest.mark.parametrize("field,value",[("object_id","other"),("stage","other"),("aggregation","other"),("unit","MW"),("value",True),("value",float("nan"))])
def test_invalid_comparison_identities_and_values_rejected(field,value):
    data=evidence(); data["facts"][1][field]=value
    with pytest.raises(AssistantError):
        execute_program(plan({"kind":"compare","left":"c0.ridge.mae","right":"c0.persistence.mae"}),data,[])


def test_mixed_metric_and_unknown_fact_rejected():
    for left,right in (("c0.ridge.mae","c0.persistence.rmse"),("c0.absent.mae","c0.ridge.mae")):
        with pytest.raises(AssistantError):
            execute_program(plan({"kind":"compare","left":left,"right":right}),evidence(),[])


def test_counts_explain_same_values_not_same_meaning():
    data=evidence()
    draft=execute_program(plan({"kind":"coverage","context":"c0"}),data,[])
    result=validate_answer(draft,data,[])
    assert "合法输入与实际输出次数相等" in result["answer"]
    assert "实际输出与可评分次数不相等" in result["answer"]
    assert "数量相等也不代表集合定义相同" in result["answer"]


def test_count_comparison_is_legal_and_does_not_confuse_definitions():
    data=evidence()
    draft=execute_program(plan({"kind":"compare","left":"c0.output_count","right":"c0.scoreable"}),data,[])
    assert "前者高于后者" in validate_answer(draft,data,[])["answer"]


@pytest.mark.parametrize("counts", [["output_count", "scoreable"], ["scoreable", "output_count", "scoreable"]])
def test_coverage_projection_keeps_only_selected_counts_and_their_definitions(counts):
    data = evidence()
    response = plan({"kind":"coverage", "context":"c0", "counts":counts})
    draft = execute_program(response, data, [])
    result = validate_answer(draft, data, [])
    assert set(draft.fact_ids) == {"c0.output_count", "c0.scoreable"}
    assert draft.answer.count("{{c0.output_count}}") == draft.answer.count("{{c0.scoreable}}") == 1
    assert "实际输出与可评分次数不相等" in result["answer"]
    assert "实际输出是已经产出预测" in result["answer"]
    assert "可评分还要求评价标签可用" in result["answer"]
    assert "计划起报是" not in result["answer"] and "合法输入是" not in result["answer"]
    assert "不能据集合差额判断具体故障原因" in result["answer"]


def test_separate_count_reads_do_not_omit_later_definitions():
    draft = execute_program(plan(
        {"kind":"coverage", "context":"c0", "counts":["input_valid"]},
        {"kind":"coverage", "context":"c0", "counts":["output_count", "scoreable"]}), evidence(), [])
    assert "合法输入是" in draft.answer and "实际输出是" in draft.answer and "可评分还要求" in draft.answer
    assert draft.answer.count("不能据集合差额判断具体故障原因") == 1


@pytest.mark.parametrize("operation", [
    {"kind":"coverage", "context":"c0", "counts":[]},
    {"kind":"coverage", "context":"c0", "counts":["fault_count"]},
    {"kind":"time", "context":"c0", "counts":["scoreable"]},
])
def test_count_projection_rejects_invalid_sets_and_time_operations(operation):
    with pytest.raises(ValidationError):
        plan(operation)


def test_coverage_projection_does_not_mask_missing_source_fact():
    data = evidence()
    data["facts"] = [fact for fact in data["facts"] if fact["id"] != "c0.scoreable"]
    with pytest.raises(AssistantError, match="answer_program_fact"):
        execute_program(plan({"kind":"coverage", "context":"c0", "counts":["output_count", "scoreable"]}), data, [])


@pytest.mark.parametrize("direction,first",[("ascending","c0.persistence.mae"),("descending","c0.ridge.mae")])
def test_sort_computes_order_independent_of_model_input_order(direction,first):
    data=evidence()
    for keys in (["c0.ridge.mae","c0.persistence.mae"],["c0.persistence.mae","c0.ridge.mae"]):
        draft=execute_program(plan({"kind":"sort","fact_ids":keys,"direction":direction}),data,[])
        assert draft.answer.index("{{"+first+"}}") < draft.answer.index("{{"+next(k for k in keys if k!=first)+"}}")


def test_mixed_program_binds_output_to_actual_final_stage():
    data=evidence()
    keys=[f["id"] for f in data["facts"]]
    data["records"][0]["stage_boundary"]="正式结果不能反向作为采用理由。"
    data["stage_options"]=[{"object_id":IDENTITY,"role":"final_holdout_result","label":"正式结果",
        "allowed_fact_ids":keys,"required_fact_ids":[],"required_citations":[],"allowed_citations":[]}]
    draft=execute_program(plan({"kind":"stage","context":"c0","role":"final_holdout_result"},{"kind":"coverage","context":"c0"}),data,[])
    assert draft.answer=="" and len(draft.stage_claims)==1
    assert "实际输出与可评分次数不相等" in validate_answer(draft,data,[])["answer"]


def test_value_list_spanning_real_stages_is_grouped_without_changing_identity():
    from tests.unit.assistant.test_stages import setup_question
    data, docs = setup_question("engie", "开发改善和采用状态？")
    operations=[{"kind":"values", "fact_ids":["c0.lightgbm_l1_shrink.mae_gain", "c0.development.mae_gain_gate", "c0.adopted"]}]
    operations.extend({"kind":"stage", "context":"c0", "role":item["role"]} for item in data["stage_requirements"])
    draft=execute_program(plan(*operations),data,docs)
    result=validate_answer(draft,data,docs)
    assert result["status"]=="answered"
    final=next(c for c in draft.stage_claims if c.role=="final_holdout_result")
    assert "{{c0.lightgbm_l1_shrink.mae_gain}}" in final.text
    assert all("{{c0.lightgbm_l1_shrink.mae_gain}}" not in c.text for c in draft.stage_claims if c.role!="final_holdout_result")


@pytest.mark.parametrize("question,rejected",[
    ("报告RMSE，不要报告训练标签截止时间。",True),
    ("无需列出训练截止日期，仅给指标。",True),
    ("不要遗漏训练标签截止时间。",False),
    ("请报告训练标签截止时间。",False),
    ("不用训练截止时间回答季度。",True),
])
def test_explicit_training_time_exclusion_is_not_a_negated_fact(question,rejected):
    data=evidence()
    item=deepcopy(data["facts"][0])
    item.update(id="c0.training_label_available",label="训练标签截止",value="2014-06-30T23:50:00+00:00",unit="UTC")
    data["facts"].append(item)
    response=plan({"kind":"values","fact_ids":[item["id"]]})
    if rejected:
        with pytest.raises(AssistantError,match="answer_program_excluded_fact"):
            execute_program(response,data,[],question)
    else:
        assert item["id"] in execute_program(response,data,[],question).fact_ids


def test_program_cannot_append_unchecked_prose_or_code():
    for field in ("text","code"):
        with pytest.raises(ValidationError):
            ProgramAnswer(**ASSESSMENT,status="answered",body={"kind":"program","operations":[{"kind":"values","fact_ids":["c0.ridge.mae"]}],field:"wrong prose"})
    with pytest.raises(ValidationError):
        plan({"kind":"exec","code":"print('unsafe')"})


def test_program_status_and_missing_temporal_context_rejected():
    with pytest.raises(ValidationError):
        ProgramAnswer(**ASSESSMENT,status="insufficient_evidence",body={"kind":"program","operations":[{"kind":"values","fact_ids":["c0.ridge.mae"]}]})
    with pytest.raises(AssistantError,match="answer_program_time"):
        execute_program(plan({"kind":"time","context":"c0"}),evidence(),[])


def test_plain_fact_reads_require_operations_without_changing_reference_mode():
    from power_forecast_service.assistant.contracts import ReferenceAnswer
    from power_forecast_service.assistant.references import resolve_reference_answer
    response=ProgramAnswer(**ASSESSMENT,status="answered",body={"kind":"plain","text":"{{c0.ridge.mae}}","citations":[]})
    with pytest.raises(AssistantError,match="answer_program_fact_requires_operation"):
        execute_program(response,evidence(),[])
    original=ReferenceAnswer(status="answered",body=response.body)
    result=validate_answer(resolve_reference_answer(original,[]),evidence(),[])
    assert result["status"]=="answered"
    assert response_contract("references")[2] != response_contract("program")[2]


def test_plain_cannot_relabel_training_cutoff_as_evaluation_quarter():
    data=evidence()
    fact=deepcopy(data["facts"][0])
    fact.update(id="c0.training_label_available",value="2014-06-30T23:50:00+00:00",unit="UTC")
    data["facts"].append(fact)
    bad=ProgramAnswer(**ASSESSMENT,status="answered",body={"kind":"plain",
        "text":"评价季度是{{c0.training_label_available}}，MAE为{{c0.ridge.mae}}。","citations":[]})
    with pytest.raises(AssistantError,match="answer_program_fact_requires_operation"):
        execute_program(bad,data,[])
    good=execute_program(plan({"kind":"values","fact_ids":["c0.ridge.mae"]}),data,[])
    assert "2014-Q4" in good.answer and fact["id"] not in good.fact_ids
    schema=program_schema(data,[])
    assert "不包含" in schema["$defs"]["PlainReferenceBody"]["properties"]["text"]["description"]


def test_runtime_and_freeze_schema_entry_are_identical_and_do_not_mutate_input():
    data=evidence(); before=deepcopy(data)
    schema=output_schema("program",data,[],[])
    assert schema==request_program_schema(data,[],[]) and data==before
    assert schema["$defs"]["Compare"]["properties"]["left"]=={"$ref":"#/$defs/FactId"}
    assert schema["$defs"]["FactId"]["enum"]==[f["id"] for f in data["facts"]]
    assert set(schema["properties"]) == {"requests"}


def test_present_fact_refusal_feedback_does_not_force_answered():
    response=ProgramAnswer(**ASSESSMENT,status="insufficient_evidence",body={"kind":"insufficient","text":"没有账单，不能计算收入。","missing_evidence":["c0.ridge.mae"],"citations":[]})
    with pytest.raises(AssistantError) as error:
        validate_answer(execute_program(response,evidence(),[]),evidence(),[],"实际收入是多少？")
    assert error.value.code=="answer_refusal_present_fact"
    assert "这不证明整个问题可答" in error.value.detail
    assert "保持insufficient_evidence" in error.value.detail


def test_program_prompt_is_not_original_prose_instruction_plus_conflicting_appendix():
    _,instructions,_=response_contract("program")
    assert "每个子问题由恰当操作覆盖一次" in instructions
    assert "普通回答body.kind为plain" not in instructions
    assert program_schema(evidence(),[])["properties"]["operations"]["anyOf"][0]["items"]["anyOf"][0]=={"$ref":"#/$defs/Values"}


def test_stage_program_consumes_actual_bound_source_and_preserves_boundary():
    from tests.unit.assistant.test_stages import setup_question
    data,docs=setup_question("engie","先给开发改善、门槛和采用状态。再解释最终成绩是否可以修改事前门槛，不要求报告正式改善原值。")
    assert any(item["role"]=="adoption_decision" for item in data["stage_requirements"])
    operations=[{"kind":"stage","context":"c0","role":item["role"]} for item in data["stage_requirements"]]
    result=validate_answer(execute_program(plan(*operations),data,docs),data,docs)
    assert result["status"]=="answered"
    assert "正式结果只作事后核验" in result["answer"]
    assert "低于已设定" in result["answer"]
    assert any(f["id"]=="c0.adopted" and f["value"] is False for f in result["facts"])


def test_development_threshold_comparison_is_valid_but_final_threshold_is_not():
    from tests.unit.assistant.test_stages import setup_question
    from power_forecast_service.assistant.stages import stage_body
    data,docs=setup_question("engie")
    valid=plan({"kind":"compare","left":"c0.development.lightgbm_l1_shrink.mae_gain","right":"c0.development.mae_gain_gate"})
    assert "前者低于后者" in stage_body(execute_program(valid,data,docs),data)
    invalid=plan({"kind":"compare","left":"c0.lightgbm_l1_shrink.mae_gain","right":"c0.development.mae_gain_gate"})
    with pytest.raises(AssistantError,match="answer_program_comparison"):
        execute_program(invalid,data,docs)


@pytest.mark.parametrize("first_invalid",[False,True])
def test_workflow_repairs_only_invalid_drafts_without_exceeding_original_budget(monkeypatch,first_invalid):
    import asyncio
    from langchain_core.messages import AIMessage
    from power_forecast_service.assistant import workflow
    from power_forecast_service.assistant.contracts import Question
    seen=[]
    class Provider:
        async def ainvoke(self,messages):
            seen.append(list(messages))
            key="c0.absent.mae" if first_invalid and len(seen)==1 else "c0.ridge.mae"
            return AIMessage(content=request_plan(EvidencePlan(**ASSESSMENT,
                operations=[{"kind":"values","fact_ids":[key]}])).model_dump_json())
    data=evidence()
    monkeypatch.setattr(workflow,"prepare_answer_context",lambda *args:(data,[]))
    agent=workflow.Assistant(None,provider=Provider(),response_mode="program")
    state={"question":Question(question="ridge MAE原值是多少？",contexts=[{"kind":"engie_import","id":IDENTITY}]),
        "evidence":data,"documents":[],"trace":{"model_calls":[]}}
    try:
        result=asyncio.run(agent.answer(state))
    finally:
        agent.close()
    assert len(seen)==(2 if first_invalid else 1) and result["result"]["status"]=="answered"
    assert seen[0] == workflow.response_messages(state["question"], data, [], agent.instructions,
        output_schema("program",data,[],[]), response_mode="program", layout=agent.prompt_layout)
    assert all('待回答的问题原文' in call[-1][1] and 'ridge MAE原值是多少？' in call[-1][1] for call in seen)
    assert "execution_review" not in state["trace"]


def test_values_include_count_meaning_without_model_prose():
    draft=execute_program(plan({"kind":"values","fact_ids":["c0.output_count","c0.scoreable"]}),evidence(),[])
    assert "评价标签可用" in draft.answer
    assert "不能仅凭未评分认定没有输出" in draft.answer
    assert "实际输出与可评分次数不相等" in draft.answer


def test_gap_explanation_cannot_invent_absence_of_all_existing_information():
    response=ProgramAnswer(**ASSESSMENT,status="insufficient_evidence",body={"kind":"gap","missing_evidence":["生产实测到达延迟分布"]})
    draft=execute_program(response,evidence(),[])
    assert validate_answer(draft,evidence(),[],"生产延迟分布？")["status"]=="insufficient_evidence"
    assert "生产实测到达延迟分布" in draft.answer
    assert "只有" not in draft.answer
    assert program_schema(evidence(),[])["properties"]["operations"]["anyOf"][0]["items"]["anyOf"][-1]=={"$ref":"#/$defs/UnavailableRequest"}


def test_generation_contract_omits_unavailable_time_and_stage_operations():
    schema=program_schema(evidence(),[])
    assert schema["$defs"]["ContextOperation"]["properties"]["kind"]["const"]=="coverage"
    assert "StageOperation" not in schema["$defs"]
    assert list(schema["properties"])==["request_summary","evidence_assessment","operations"]
    assert set(ASSESSMENT)<=set(schema["required"])


@pytest.mark.parametrize("value",[-15.,0.,5.])
def test_bias_interpretation_tracks_changed_values(value):
    data=evidence()
    fact=deepcopy(data["facts"][0]); fact.update(id="c0.ridge.bias",label="ridge BIAS",value=value)
    data["facts"].append(fact)
    draft=execute_program(plan({"kind":"values","fact_ids":[fact["id"]]}),data,[])
    assert ("原值不为零" in draft.answer)==(value!=0)
    assert ("原值为零" in draft.answer)==(value==0)


def test_interval_values_keep_applicability_boundary_without_generated_explanation():
    data=evidence()
    for name,value in (("gain_ci_low",-2.),("gain_ci_high",3.)):
        fact=deepcopy(data["facts"][0]); fact.update(id="c0."+name,label=name,value=value,unit="%")
        data["facts"].append(fact)
    draft=execute_program(plan({"kind":"values","fact_ids":["c0.gain_ci_low","c0.gain_ci_high"]}),data,[])
    assert "不能代表换场站" in draft.answer
    assert "改善一定" not in draft.answer
    assert len(validate_answer(draft,data,[])["facts"])==2


@pytest.mark.parametrize("real_gap",[False,True])
def test_audited_interval_resolution_preserves_original_plan_and_real_gap(real_gap):
    data=evidence()
    for name,value in (("gain_ci_low",-2.),("gain_ci_high",3.)):
        fact=deepcopy(data["facts"][0]); fact.update(id="c0."+name,value=value,unit="%")
        data["facts"].append(fact)
    request="七天配对时间块下共享收缩主MAE改善的区间是否能代表换场站"
    missing=[request]+(["其他风场的实际MAE"] if real_gap else [])
    response=partial({"kind":"program","operations":[{"kind":"values","fact_ids":["c0.gain_ci_low","c0.gain_ci_high"]}]},missing)
    before=response.model_dump_json()
    events=[]
    draft=execute_program(response,data,[],reconciliation_log=events)
    result=validate_answer(draft,data,[])
    assert result["status"]==("insufficient_evidence" if real_gap else "answered")
    assert "不能代表换场站" in result["answer"]
    assert draft.missing_evidence==(["其他风场的实际MAE"] if real_gap else [])
    assert events==[{"request":request,"rule":"selected_interval_scope","fact_ids":["c0.gain_ci_low","c0.gain_ci_high"],
        "reason":"已执行区间操作说明只对应当前对象，不能代表其他场站；不提供其他场站的实际指标。"}]
    assert response.model_dump_json()==before


@pytest.mark.parametrize("missing",[
    "区间是否能代表换场站，以及实际售电收入",
    "区间是否能代表换场站的实际MAE数值",
    "其他风场的真实指标",
])
def test_gap_resolution_never_discards_a_compound_or_actual_data_request(missing):
    data=evidence()
    for name,value in (("gain_ci_low",-2.),("gain_ci_high",3.)):
        fact=deepcopy(data["facts"][0]); fact.update(id="c0."+name,value=value,unit="%")
        data["facts"].append(fact)
    response=partial({"kind":"program","operations":[{"kind":"values","fact_ids":["c0.gain_ci_low","c0.gain_ci_high"]}]},[missing])
    events=[]
    try:
        draft=execute_program(response,data,[],reconciliation_log=events)
    except AssistantError as error:
        assert error.code=="answer_program_gap_conflict"
    else:
        assert draft.status=="insufficient_evidence" and draft.missing_evidence==[missing]
    assert events==[] and response.body.missing_evidence==[missing]


@pytest.mark.parametrize("gap_request,resolved",[("默认模型是否已被替换",True),("线上生产默认模型是否已被替换",False),
    ("默认模型是否已被替换以及实际收入",False)])
def test_default_resolution_uses_project_record_not_production_inference(gap_request,resolved):
    from tests.unit.assistant.test_stages import setup_question
    data,docs=setup_question("engie","开发改善和采用状态？")
    data["records"][0]["adoption_boundary"]="开发采用门未过，默认持续性。"
    operations=[{"kind":"stage","context":"c0","role":r["role"]} for r in data["stage_requirements"]]
    response=partial({"kind":"program","operations":operations},[gap_request])
    events=[]
    draft=execute_program(response,data,docs,reconciliation_log=events)
    assert validate_answer(draft,data,docs)["status"]==("answered" if resolved else "insufficient_evidence")
    assert bool(events)==resolved
    if resolved:
        assert events[0]["fact_ids"]==["c0.adopted"]
    else:
        assert draft.missing_evidence==[gap_request]


def test_workflow_records_actual_resolution_without_an_extra_model_call(monkeypatch):
    import asyncio
    from langchain_core.messages import AIMessage
    from power_forecast_service.assistant import workflow
    from power_forecast_service.assistant.contracts import Question
    from tests.unit.assistant.test_stages import setup_question
    data,docs=setup_question("engie","开发改善和采用状态？")
    data["records"][0]["adoption_boundary"]="开发采用门未过，默认持续性。"
    operations=[{"kind":"stage","context":"c0","role":r["role"]} for r in data["stage_requirements"]]
    raw=EvidencePlan(**ASSESSMENT,operations=[*operations,{"kind":"unknown","request":"默认模型是否已被替换"}])
    calls=[]
    class Provider:
        async def ainvoke(self,messages):
            calls.append(messages)
            return AIMessage(content=request_plan(raw).model_dump_json())
    monkeypatch.setattr(workflow,"prepare_answer_context",lambda *args:(data,docs))
    agent=workflow.Assistant(None,provider=Provider(),response_mode="program")
    state={"question":Question(question="开发改善和采用状态？",contexts=[{"kind":"engie_import","id":data["records"][0]["id"]}]),
        "evidence":data,"documents":docs,"trace":{"model_calls":[]}}
    try:
        result=asyncio.run(agent.answer(state))
    finally:
        agent.close()
    assert result["result"]["status"]=="answered" and len(calls)==1
    audit=state["trace"]["program_reconciliations"][0]
    assert audit["original_status"]=="insufficient_evidence" and audit["resolved_status"]=="answered"
    assert audit["items"][0]["rule"]=="registered_project_adoption"
    assert raw.operations[-1].kind=="unknown"


def test_adoption_boolean_alone_cannot_resolve_default_or_deployment_state():
    from tests.unit.assistant.test_stages import setup_question
    data,docs=setup_question("engie","开发改善和采用状态？")
    assert "adoption_boundary" not in data["records"][0]
    operations=[{"kind":"stage","context":"c0","role":r["role"]} for r in data["stage_requirements"]]
    response=partial({"kind":"program","operations":operations},["默认模型是否已被替换"])
    events=[]
    draft=execute_program(response,data,docs,reconciliation_log=events)
    assert draft.status=="insufficient_evidence" and events==[]


def test_comparison_feedback_names_both_operands_and_source_axes():
    data=evidence(); data["facts"][1]["stage"]="other_stage"
    with pytest.raises(AssistantError) as error:
        execute_program(plan({"kind":"compare","left":"c0.ridge.mae","right":"c0.persistence.mae"}),data,[])
    assert "compare(c0.ridge.mae, c0.persistence.mae)" in error.value.detail
    assert "stage: selected_result != other_stage" in error.value.detail
    assert "删除所有同类" in error.value.detail


def test_comparison_repair_preserves_inputs_and_restricts_both_failed_operands():
    from power_forecast_service.assistant.program import comparison_facts, comparison_repair_schema
    data = evidence()
    response = EvidencePlan(**ASSESSMENT, operations=[
        {"kind":"compare", "left":"c0.ridge.mae", "right":"c0.persistence.rmse"},
        {"kind":"unknown", "request":"实际售电凭证"}])
    schema = program_schema(data, [])
    before = deepcopy((data, schema, response.model_dump()))
    revised, invalid = comparison_repair_schema(schema, response, data)
    assert invalid == [{"index":0, "left":"c0.ridge.mae", "right":"c0.persistence.rmse"}]
    assert (data, schema, response.model_dump()) == before
    assert revised["$defs"]["UnavailableRequest"] == schema["$defs"]["UnavailableRequest"]
    assert revised["$defs"]["Values"] == schema["$defs"]["Values"]
    choices = revised["$defs"]["Compare"]["anyOf"]
    facts = {fact["id"]:fact for fact in data["facts"]}
    constrained = {item["properties"]["left"].get("const"):item for item in choices if "const" in item["properties"]["left"]}
    assert set(constrained) == {"c0.ridge.mae", "c0.persistence.rmse"}
    for left, item in constrained.items():
        rights = item["properties"]["right"]["enum"]
        assert rights
        for right in facts:
            try:
                comparison_facts(left, right, facts)
                valid = True
            except AssistantError:
                valid = False
            assert (right in rights) == valid
    assert "c0.ridge.mae" not in choices[0]["properties"]["left"]["enum"]
    assert "c0.persistence.rmse" not in choices[0]["properties"]["left"]["enum"]


def test_comparison_repair_keeps_valid_schema_unchanged():
    from power_forecast_service.assistant.program import comparison_repair_schema
    data = evidence()
    schema = program_schema(data, [])
    response = EvidencePlan(**ASSESSMENT, operations=[{"kind":"compare", "left":"c0.ridge.mae", "right":"c0.persistence.mae"}])
    revised, invalid = comparison_repair_schema(schema, response, data)
    assert revised is schema and invalid == []


def test_comparison_repair_without_legal_pair_retains_unknown_and_values():
    from power_forecast_service.assistant.program import comparison_repair_schema
    data = evidence()
    data["facts"] = data["facts"][:1]
    response = EvidencePlan(**ASSESSMENT, operations=[{"kind":"compare", "left":"c0.ridge.mae", "right":"c0.ridge.mae"}])
    revised, invalid = comparison_repair_schema(program_schema(data, []), response, data)
    assert len(invalid) == 1 and "Compare" not in revised["$defs"]
    assert "UnavailableRequest" in revised["$defs"] and "Values" in revised["$defs"]


def test_actual_workflow_supplies_constrained_repair_and_still_rejects_bad_output(monkeypatch):
    import asyncio
    from langchain_core.messages import AIMessage
    from power_forecast_service.assistant import workflow
    from power_forecast_service.assistant.contracts import Question
    data = evidence()
    invalid = EvidencePlan(**ASSESSMENT, operations=[{"kind":"compare", "left":"c0.ridge.mae", "right":"c0.persistence.rmse"}])
    monkeypatch.setattr(workflow, "prepare_answer_context", lambda *args:(data, []))
    for repair_succeeds in (True, False):
        calls = []
        class Provider:
            async def ainvoke_structured(self, messages, *, schema, call_id):
                calls.append(deepcopy(schema))
                response = EvidencePlan(**ASSESSMENT, operations=[{"kind":"unknown", "request":"实际售电凭证"}]) if repair_succeeds and len(calls) == 2 else invalid
                return AIMessage(content=request_plan(response).model_dump_json())
        agent = workflow.Assistant(None, provider=Provider(), response_mode="program")
        state = {"question":Question(question="有没有实际售电凭证？", contexts=[{"kind":"engie_import", "id":IDENTITY}]),
                 "evidence":data, "documents":[], "trace":{"model_calls":[]}}
        try:
            if repair_succeeds:
                result = asyncio.run(agent.answer(state))
                assert result["result"]["status"] == "insufficient_evidence"
            else:
                with pytest.raises(AssistantError, match="answer_program_comparison"):
                    asyncio.run(agent.answer(state))
        finally:
            agent.close()
        assert len(calls) == 2
        assert "anyOf" not in calls[0]["$defs"]["Compare"]
        assert "anyOf" in calls[1]["$defs"]["Compare"]
        assert state["trace"]["program_repair_constraints"]["invalid_operations"][0]["index"] == 0


def test_coverage_does_not_promote_generic_example_into_selected_period_cause():
    data = evidence()
    data["records"][0]["interpretation_constraints"]["coverage"] = "末六个越界只是其中一部分。"
    draft = execute_program(plan({"kind":"coverage", "context":"c0"}), data, [])
    assert "末六个" not in draft.answer
    assert draft.answer.count("合法输入与实际输出次数相等") == 1
    assert "评价标签可用" in draft.answer


@pytest.mark.parametrize("candidate,baseline", [(8.,10.), (12.,10.), (10.,10.)])
def test_derived_metric_includes_bound_operands_signed_difference_and_direction(candidate,baseline):
    data = evidence()
    data["facts"][0]["value"], data["facts"][1]["value"] = candidate,baseline
    for suffix,value in (("delta",candidate-baseline), ("gain",100*(1-candidate/baseline))):
        fact = deepcopy(data["facts"][0])
        fact.update(id="c0.ridge.mae_"+suffix, label=suffix, value=value,
                    unit="kW" if suffix == "delta" else "%", derived=True)
        data["facts"].append(fact)
    draft = execute_program(plan({"kind":"values", "fact_ids":["c0.ridge.mae_gain"]}),data,[])
    assert set(draft.fact_ids) == {"c0.ridge.mae", "c0.persistence.mae", "c0.ridge.mae_delta", "c0.ridge.mae_gain"}
    assert "前者"+("低于" if candidate<baseline else "高于" if candidate>baseline else "等于")+"后者" in draft.answer
    assert validate_answer(draft,data,[])["status"] == "answered"
    data["facts"][-1]["stage"] = "other-stage"
    with pytest.raises(AssistantError, match="answer_program_derivation"):
        execute_program(plan({"kind":"values", "fact_ids":["c0.ridge.mae_gain"]}),data,[])


@pytest.mark.parametrize("mode", ["values", "compare", "split_values"])
def test_baseline_self_deltas_preserve_originals(mode):
    data = evidence()
    for model,offset in (("ridge",0),("persistence",1)):
        for suffix,value in (("delta",2. if model=="ridge" else 0.), ("gain",-20. if model=="ridge" else 0.)):
            fact = deepcopy(data["facts"][offset])
            fact.update(id="c0."+model+".mae_"+suffix,label=model+suffix,value=value,
                        unit="kW" if suffix=="delta" else "%",derived=True)
            data["facts"].append(fact)
    left,right = "c0.ridge.mae_delta","c0.persistence.mae_delta"
    if mode == "split_values":
        response = plan({"kind":"values", "fact_ids":["c0.ridge.mae","c0.persistence.mae"]},
                        {"kind":"values", "fact_ids":["c0.ridge.mae_gain"]})
    else:
        operation = {"kind":"values", "fact_ids":[left,right]} if mode=="values" else {"kind":"compare", "left":left, "right":right}
        response = plan(operation)
    draft = execute_program(response,data,[])
    result = validate_answer(draft,data,[])
    assert result["status"] == "answered"
    assert {"c0.ridge.mae", "c0.persistence.mae"} <= {fact["id"] for fact in result["facts"]}
    expected_relation = "候选原值高于基线原值" if mode == "split_values" else "前者高于后者"
    assert expected_relation in result["answer"]
    if mode in {"values", "compare"}:
        assert "基线与自身比较" in result["answer"]


def test_stage_explanation_uses_typed_operations_without_losing_boundaries():
    from tests.unit.assistant.test_stages import setup_question
    data, docs = setup_question("engie", "开发采用门为什么不能改成最终改善的置信区间？这个区间是不是证明现在必须采用共享收缩作为默认？")
    data["records"][0]["adoption_boundary"] = "事前开发采用门未通过；区间也不是采用门。"
    claims = []
    for requirement in data["stage_requirements"]:
        option = next(o for o in data["stage_options"] if o["role"] == requirement["role"])
        keys = [key for key in option["required_fact_ids"] if key != "c0.adopted"]
        claims.append({"object_id":option["object_id"], "role":option["role"],
            "text":"当前阶段的记录如下。"+"；".join("{{"+key+"}}" for key in keys),
            "citations":[{"document_id":identity} for identity in option["required_citations"]]})
    response = ProgramAnswer(**ASSESSMENT, status="answered", body={"kind":"staged", "claims":claims})
    with pytest.raises(AssistantError, match="answer_program_structured_required"):
        execute_program(response,data,docs)
    typed = plan(*[{"kind":"stage", "context":"c0", "role":r["role"]} for r in data["stage_requirements"]])
    result = validate_answer(execute_program(typed,data,docs),data,docs)
    assert "c0.adopted" in {fact["id"] for fact in result["facts"]}
    assert "区间也不是采用门" in result["answer"]
    assert "不能" in result["answer"] or "不得" in result["answer"]


@pytest.mark.parametrize("question", [
    "开发改善和采用门槛是多少，正式结果能改写事前决定吗？",
    "22:10起报的输入截止和目标时刻是什么？另外实测延迟P91有没有？",
])
def test_structured_representation_domain_matches_runtime_and_keeps_unknown(question):
    from tests.unit.assistant.test_stages import setup_question
    from power_forecast_service.assistant.workflow import prepare_answer_context
    data, docs = setup_question("engie", "仅查询方法")
    data, docs = prepare_answer_context(data, docs, question, "program")
    before = deepcopy((data, docs))
    schema = program_schema(data, docs)
    assert len(schema["properties"]["operations"]["anyOf"]) == 1
    assert "PlainReferenceBody" not in schema["$defs"] and "StagedReferenceBody" not in schema["$defs"]
    assert "UnavailableRequest" in schema["$defs"]
    assert (data, docs) == before
    forged = EvidencePlan(**ASSESSMENT, operations=[{"kind":"plain", "text":"文档没有实测记录。", "citations":[]}])
    with pytest.raises(AssistantError, match="answer_program_structured_required"):
        execute_program(forged.as_answer(), data, docs, question)


def test_stage_fact_identity_cannot_be_relabelled_in_free_prose():
    from tests.unit.assistant.test_stages import setup_question
    data, docs = setup_question("engie", "开发改善和门槛以及最终结果有什么关系？")
    option = next(o for o in data["stage_options"] if o["role"] == "final_holdout_result")
    forged = plan({"kind":"staged", "claims":[{"object_id":option["object_id"], "role":option["role"],
        "text":"正式改善高于开发阈值{{c0.lightgbm_l1_shrink.mae_gain}}。",
        "citations":[{"document_id":d} for d in option["required_citations"]]}]})
    with pytest.raises(AssistantError, match="answer_program_structured_required"):
        execute_program(forged, data, docs)


def test_unstructured_method_fallback_and_reference_mode_remain_available():
    from power_forecast_service.assistant.workflow import output_schema
    data = evidence()
    docs = [{"id":"method", "text":"未来标签只能用于评分。", "source_sha256":"a"*64}]
    schema = program_schema(data, docs)
    assert len(schema["properties"]["operations"]["anyOf"]) == 2
    assert "PlainReferenceBody" in schema["$defs"]
    result = execute_program(plan({"kind":"plain", "text":"未来标签只能用于评分。", "citations":[{"document_id":"method"}]}), data, docs)
    assert result.status == "answered"
    assert output_schema("references", data, docs, []) != schema


def test_selected_stage_facts_are_formatted_without_requiring_redundant_stage_selection():
    from tests.unit.assistant.test_stages import setup_question
    question = "开发共享收缩改善和采用门是多少，通过了吗？实际售电收入是多少？"
    data, docs = setup_question("engie", question)
    response = EvidencePlan(**ASSESSMENT, operations=[
        {"kind":"values", "fact_ids":["c0.development.lightgbm_l1_shrink.mae_gain", "c0.development.mae_gain_gate", "c0.adopted"]},
        {"kind":"unknown", "request":"实际售电收入"}])
    before = deepcopy((data, docs, response.model_dump()))
    result = validate_answer(execute_program(response.as_answer(),data,docs,question),data,docs,question)
    assert result["status"] == "insufficient_evidence"
    assert "开发评价" in result["answer"] and "开发采用决定" in result["answer"]
    assert "实际售电收入" in result["answer"]
    assert (data, docs, response.model_dump()) == before


def test_stage_formatting_does_not_supply_unselected_required_facts():
    from tests.unit.assistant.test_stages import setup_question
    data, docs = setup_question("engie", "开发改善和采用门是多少，通过了吗？")
    draft = execute_program(plan({"kind":"values", "fact_ids":["c0.development.lightgbm_l1_shrink.mae_gain"]}),data,docs)
    assert "c0.adopted" not in draft.fact_ids and "c0.development.mae_gain_gate" not in draft.fact_ids
    with pytest.raises(AssistantError, match="answer_stage_incomplete"):
        validate_answer(draft,data,docs)


def test_program_hints_do_not_demand_an_unasked_final_metric():
    from power_forecast_service.assistant.program import prepare_program_evidence
    from tests.unit.assistant.test_stages import setup_question
    question = "请给开发共享收缩MAE改善、开发采用门槛和最终采用状态。正式留出结果能否事后把当时的未采用改成采用？"
    original, docs = setup_question("engie", question)
    before = deepcopy(original)
    data = prepare_program_evidence(original)
    draft = execute_program(plan({"kind":"values", "fact_ids":[
        "c0.development.lightgbm_l1_shrink.mae_gain", "c0.development.mae_gain_gate", "c0.adopted"]}), data, docs, question)
    result = validate_answer(draft, data, docs, question)
    assert result["status"] == "answered" and original["records"][0]["stage_boundary"] in result["answer"]
    assert "c0.lightgbm_l1_shrink.mae_gain" not in draft.fact_ids
    assert "开发评价" in result["answer"] and "开发采用决定" in result["answer"]
    assert original == before and data["stage_requirements"] == [] and data["stage_hints"] == original["stage_requirements"]
    assert data["facts"] == original["facts"]


def test_program_projection_exposes_source_roles_even_without_keyword_hints():
    from power_forecast_service.assistant.program import prepare_program_evidence
    from tests.unit.assistant.test_stages import setup_question
    original, docs = setup_question("engie", "开发改善和正式改善分别是多少？")
    original["stage_requirements"] = []
    data = prepare_program_evidence(original)
    schema = program_schema(data, docs)
    assert set(schema["$defs"]["StageOperation"]["properties"]["role"]["enum"]) == {o["role"] for o in data["stage_options"]}
    draft = execute_program(plan({"kind":"stage", "context":"c0", "role":"final_holdout_result", "fact_ids":[]}), data, docs)
    result = validate_answer(draft, data, docs)
    assert "不得反向" in result["answer"] and draft.fact_ids == []
    assert result["citations"]


def test_program_projection_keeps_cross_stage_fact_rejection():
    from power_forecast_service.assistant.program import prepare_program_evidence
    from tests.unit.assistant.test_stages import setup_question
    original, docs = setup_question("engie", "开发改善是多少？")
    data = prepare_program_evidence(original)
    with pytest.raises(AssistantError, match="answer_program_stage"):
        execute_program(plan({"kind":"stage", "context":"c0", "role":"development_result",
            "fact_ids":["c0.lightgbm_l1_shrink.mae_gain"]}), data, docs)


@pytest.mark.parametrize("stage_first", [False, True])
def test_empty_stage_selection_uses_actual_separate_values_before_rejecting(stage_first):
    from power_forecast_service.assistant.program import prepare_program_evidence
    from tests.unit.assistant.test_stages import setup_question
    original, docs = setup_question("engie", "开发改善是多少？")
    data = prepare_program_evidence(original)
    operations = [{"kind":"values", "fact_ids":["c0.development.lightgbm_l1_shrink.mae_gain"]},
        {"kind":"stage", "context":"c0", "role":"development_result", "fact_ids":[]}]
    if stage_first:
        operations.reverse()
    draft = execute_program(plan(*operations), data, docs)
    result = validate_answer(draft, data, docs)
    assert len(draft.stage_claims) == 1 and draft.fact_ids == ["c0.development.lightgbm_l1_shrink.mae_gain"]
    assert result["status"] == "answered" and "开发评价" in result["answer"]


def test_empty_stage_cannot_be_satisfied_by_other_stage_facts():
    from power_forecast_service.assistant.program import prepare_program_evidence
    from tests.unit.assistant.test_stages import setup_question
    original, docs = setup_question("engie", "开发改善是多少？")
    data = prepare_program_evidence(original)
    with pytest.raises(AssistantError, match="answer_program_stage"):
        execute_program(plan(
            {"kind":"stage", "context":"c0", "role":"development_result", "fact_ids":[]},
            {"kind":"values", "fact_ids":["c0.lightgbm_l1_shrink.mae_gain"]}), data, docs)


def test_program_context_demotion_does_not_change_reference_mode():
    from power_forecast_service.assistant.workflow import prepare_answer_context
    from tests.unit.assistant.test_stages import setup_question
    original, docs = setup_question("engie", "开发改善和采用门是多少？")
    program_data, _ = prepare_answer_context(original, docs, "开发改善和采用门是多少？", "program")
    reference_data, _ = prepare_answer_context(original, docs, "开发改善和采用门是多少？", "references")
    assert program_data["stage_requirements"] == [] and program_data["stage_hints"]
    assert reference_data["stage_requirements"] and "stage_hints" not in reference_data
    assert program_data["facts"] == reference_data["facts"]


def test_selected_facts_get_required_nonnumeric_stage_boundary():
    from tests.unit.assistant.test_stages import setup_question
    data, docs = setup_question("engie", "开发改善和采用门是多少？最终成绩能否改变事前决定？")
    draft = execute_program(plan({"kind":"values", "fact_ids":["c0.development.lightgbm_l1_shrink.mae_gain", "c0.development.mae_gain_gate", "c0.adopted"]}),data,docs)
    result = validate_answer(draft,data,docs)
    assert any(c.role == "final_holdout_result" for c in draft.stage_claims)
    assert "不得反向" in result["answer"]
    assert "c0.lightgbm_l1_shrink.mae_gain" not in draft.fact_ids


def test_implicitly_grouped_adoption_reuses_selected_gate_even_when_not_required():
    from tests.unit.assistant.test_stages import setup_question
    data, docs = setup_question("engie", "开发改善和采用阈值分别是多少？最终留出指标更好是否允许回头改写开发时的未采用决定？")
    assert "adoption_decision" not in {r["role"] for r in data["stage_requirements"]}
    response = plan({"kind":"values", "fact_ids":["c0.development.mae_gain_gate", "c0.development.lightgbm_l1_shrink.mae_gain", "c0.adopted"]},
        {"kind":"stage", "context":"c0", "role":"final_holdout_result"})
    result = validate_answer(execute_program(response,data,docs),data,docs)
    assert result["status"] == "answered"
    assert "开发采用门通过：否" in result["answer"] and "不得反向" in result["answer"]


@pytest.mark.parametrize("field,limit", [("request_summary",300), ("evidence_assessment",500)])
def test_assessment_limits_remain_runtime_enforced_without_decoder_counted_repetition(field,limit):
    schema = program_schema(evidence(), [])
    assert "maxLength" not in schema["properties"][field]
    for invalid in ("", "x"*(limit+1)):
        notes = {**ASSESSMENT, field:invalid}
        with pytest.raises(ValidationError):
            ProgramAnswer(**notes, status="answered", body={"kind":"program", "operations":[{"kind":"values", "fact_ids":["c0.ridge.mae"]}]})


def partial(supported, missing=None):
    return ProgramAnswer(**ASSESSMENT, status="insufficient_evidence", body={"kind":"partial",
        "supported":supported, "missing_evidence":missing or ["生产实测延迟分布"]})


def test_partial_known_metric_survives_missing_measurement_without_false_success():
    data=evidence()
    response=partial({"kind":"program", "operations":[{"kind":"values","fact_ids":["c0.ridge.mae"]}]})
    result=validate_answer(execute_program(response,data,[]),data,[],"ridge MAE原值是多少？还要生产实测延迟。")
    assert result["status"]=="insufficient_evidence"
    assert "12 kW" in result["answer"] and "生产实测延迟" in result["answer"]
    assert [f["id"] for f in result["facts"]]==["c0.ridge.mae"]


def test_partial_method_explanation_keeps_actual_citation_revision():
    data=evidence()
    doc={"id":"simulation", "title":"方法", "text":"到达延迟使用二十分钟的模拟假设，没有生产实测分布。", "source_sha256":"b"*64}
    response=partial({"kind":"plain","text":"现有方法是二十分钟的模拟到达延迟，并非生产测量。",
        "citations":[{"document_id":"simulation"}]})
    result=validate_answer(execute_program(response,data,[doc]),data,[doc],"生产实测延迟多少？模拟是什么？")
    assert "二十分钟" in result["answer"] and result["status"]=="insufficient_evidence"
    assert result["citations"][0]["revision"]=="b"*64 and result["citations"][0]["quote"]==doc["text"]


@pytest.mark.parametrize("supported,code",[
    ({"kind":"program","operations":[{"kind":"values","fact_ids":["c0.unknown.mae"]}]},"answer_program_fact"),
    ({"kind":"plain","text":"{{c0.ridge.mae}} MW", "citations":[]},"answer_program_fact_requires_operation"),
    ({"kind":"plain","text":"有一个无证据结论。", "citations":[]},"answer_evidence_missing"),
    ({"kind":"plain","text":"资料中的模拟规则。", "citations":[{"document_id":"missing-doc"}]},"answer_citation_invalid"),
])
def test_partial_cannot_use_refusal_to_bypass_supported_checks(supported,code):
    with pytest.raises(AssistantError,match=code):
        execute_program(partial(supported),evidence(),[])


def test_partial_cannot_label_a_present_fact_as_missing():
    data=evidence()
    response=partial({"kind":"program","operations":[{"kind":"values","fact_ids":["c0.ridge.mae"]}]},["c0.ridge.mae"])
    with pytest.raises(AssistantError,match="answer_refusal_present_fact"):
        validate_answer(execute_program(response,data,[]),data,[],"未知原因？")


def test_partial_checks_stage_before_flattening_to_insufficient_answer():
    from tests.unit.assistant.test_stages import setup_question
    data,docs=setup_question("engie","开发改善、门槛和采用状态是什么？")
    operations=[{"kind":"stage","context":"c0","role":r["role"]} for r in data["stage_requirements"]]
    result=validate_answer(execute_program(partial({"kind":"program","operations":operations},["实际售电收入"]),data,docs),data,docs,"实际售电收入与开发采用？")
    assert "开发采用" in result["answer"] and "不能" in result["answer"]
    assert any(f["id"]=="c0.adopted" and f["value"] is False for f in result["facts"])
    with pytest.raises(AssistantError,match="answer_stage_incomplete"):
        execute_program(partial({"kind":"program","operations":[{"kind":"values","fact_ids":["c0.persistence.mae"]}]}),data,docs)


def test_partial_schema_reuses_current_supported_domain_and_rejects_nesting():
    schema=program_schema(evidence(),[])
    branches=schema["properties"]["operations"]["anyOf"]
    choices=branches[0]["items"]["anyOf"]
    assert {"$ref":"#/$defs/PlainReferenceBody"} not in choices
    assert branches[1]["prefixItems"] == [{"$ref":"#/$defs/PlainReferenceBody"}]
    assert branches[1]["items"] == {"$ref":"#/$defs/UnavailableRequest"}
    assert {"$ref":"#/$defs/UnavailableRequest"} in choices
    assert {"$ref":"#/$defs/ProgramBody"} not in choices
    with pytest.raises(ValidationError):
        partial({"kind":"gap","missing_evidence":["unknown"]})
    with pytest.raises(ValidationError):
        ProgramAnswer(**ASSESSMENT,status="answered",body=partial({"kind":"plain","text":"已有规则","citations":[]}).body)


@pytest.mark.parametrize("addition", ["\n无依据数字2014。", "\n开发评价（2099三季度汇总）：未知。", "\n文字中的开发评价（2014三季度汇总）：未知。"])
def test_partial_stage_heading_does_not_whitelist_arbitrary_years(addition):
    from tests.unit.assistant.test_stages import setup_question
    data, docs = setup_question("engie", "开发改善、门槛和采用状态是什么？")
    operations = [{"kind":"stage", "context":"c0", "role":r["role"]} for r in data["stage_requirements"]]
    draft = execute_program(partial({"kind":"program", "operations":operations}, ["实际售电收入"]), data, docs)
    assert "开发评价（2014三季度汇总）" in draft.answer
    draft.answer += addition
    with pytest.raises(AssistantError, match="answer_number_unbound"):
        validate_answer(draft, data, docs)


def test_unrelated_supported_metric_does_not_evade_requested_metric_check():
    data=evidence()
    response=partial({"kind":"program","operations":[{"kind":"values","fact_ids":["c0.persistence.mae"]}]})
    with pytest.raises(AssistantError,match="answer_refusal_answerable"):
        validate_answer(execute_program(response,data,[]),data,[],"ridge MAE原值是多少？另需生产延迟。")


@pytest.mark.parametrize("missing,supported,status", [
    ([], {"kind":"program","operations":[{"kind":"values","fact_ids":["c0.ridge.mae"]}]}, "answered"),
    (["实际售电收入"], {"kind":"program","operations":[{"kind":"values","fact_ids":["c0.ridge.mae"]}]}, "insufficient_evidence"),
    (["实际售电收入"], None, "insufficient_evidence"),
])
def test_evidence_plan_derives_status_without_a_second_model_decision(missing,supported,status):
    response = EvidencePlan(**ASSESSMENT, operations=(supported["operations"] if supported else [])
        + [{"kind":"unknown","request":item} for item in missing])
    result = validate_answer(execute_program(response.as_answer(), evidence(), []), evidence(), [])
    assert result["status"] == status
    assert ("12 kW" in result["answer"]) == (supported is not None)
    with pytest.raises(ValidationError):
        EvidencePlan(**response.model_dump(), status="answered")


@pytest.mark.parametrize("missing", [[], ["x"], ["x"*161]])
def test_evidence_plan_rejects_empty_response_and_invalid_gap_lengths(missing):
    with pytest.raises(ValidationError):
        EvidencePlan(**ASSESSMENT, operations=[{"kind":"unknown","request":item} for item in missing])


def test_query_and_unknown_operations_are_peers_and_keep_runtime_bounds():
    operations = [{"kind":"values","fact_ids":[key]} for key in ("c0.ridge.mae","c0.persistence.mae")]
    response = EvidencePlan(**ASSESSMENT, operations=[*operations,{"kind":"unknown","request":"生产延迟分布"}])
    result = validate_answer(execute_program(response.as_answer(),evidence(),[]),evidence(),[])
    assert result["status"] == "insufficient_evidence" and len(result["facts"]) == 2
    with pytest.raises(ValidationError):
        EvidencePlan(**ASSESSMENT,operations=[{"kind":"exec","code":"unsafe"}])
    repeated=EvidencePlan(**ASSESSMENT,operations=operations*6)
    assert len(repeated.operations)==12 and len(repeated.as_answer().body.operations)==2
    keys=[f["id"] for f in evidence()["facts"]]
    distinct=[{"kind":"values","fact_ids":[a,b]} for a in keys[:2] for b in keys[2:]]
    with pytest.raises(ValidationError):
        EvidencePlan(**ASSESSMENT,operations=distinct)
    with pytest.raises(ValidationError):
        EvidencePlan(**ASSESSMENT,operations=[{"kind":"unknown","request":"生产延迟"}]*7)
    reordered = EvidencePlan(**ASSESSMENT,operations=list(reversed(response.operations)))
    assert reordered.as_answer().status == "insufficient_evidence"


def test_unknown_is_optional_item_not_mandatory_global_gap():
    schema = program_schema(evidence(),[])
    assert "missing_evidence" not in schema["properties"] and "supported" not in schema["properties"]
    assert "maxLength" not in schema["$defs"]["UnavailableRequest"]["properties"]["request"]
    with pytest.raises(ValidationError):
        EvidencePlan(**ASSESSMENT,operations=[{"kind":"unknown","request":"x"*161}])


def test_schema_fact_domain_is_shared_and_all_definitions_are_reachable():
    schema = program_schema(evidence(),[])
    assert "InsufficientReferenceBody" not in schema["$defs"]
    assert schema["$defs"]["Values"]["properties"]["fact_ids"]["items"] == {"$ref":"#/$defs/FactId"}
    assert schema["$defs"]["SortValues"]["properties"]["fact_ids"]["items"] == {"$ref":"#/$defs/FactId"}
    assert schema["$defs"]["FactId"]["enum"] == [f["id"] for f in evidence()["facts"]]


def test_duplicate_queries_preserve_raw_plan_unknown_items_and_answer():
    query={"kind":"values","fact_ids":["c0.ridge.mae"]}
    response=EvidencePlan(**ASSESSMENT,operations=[query]*12+[{"kind":"unknown","request":"实测延迟"}])
    assert len(response.model_dump()["operations"])==13
    draft=execute_program(response.as_answer(),evidence(),[])
    assert draft.status=="insufficient_evidence" and draft.missing_evidence==["实测延迟"]
    assert draft.answer.count("{{c0.ridge.mae}}")==1


def test_exact_request_duplicates_render_once_but_distinct_requests_remain():
    item = {"request":"查询ridge的MAE", "resolution":{"kind":"supported",
            "operations":[{"kind":"values","fact_ids":["c0.ridge.mae"]}]}}
    response = RequestPlan(requests=[deepcopy(item) for _ in range(12)])
    raw = response.model_dump()
    draft = execute_program(response, evidence(), [])
    result = validate_answer(draft, evidence(), [])
    assert len(response.requests) == 12 and response.model_dump() == raw
    assert len(draft.parts) == 1 and result["answer"].count("ridge.mae：12 kW") == 1
    distinct = deepcopy(item)
    distinct["request"] = "另一个合法子请求"
    distinct["resolution"]["operations"][0]["fact_ids"] = ["c0.ridge.rmse"]
    result = validate_answer(execute_program(RequestPlan(requests=[item, distinct]), evidence(), []), evidence(), [])
    assert {f["id"] for f in result["facts"]} == {"c0.ridge.mae", "c0.ridge.rmse"}


@pytest.mark.parametrize("missing,accepted", [("原始到达日志", True), ("实际收入凭证", False), ("训练日志", False)])
def test_measured_gap_uses_bound_request_and_evidence_subject(missing, accepted):
    from tests.unit.assistant.test_stages import setup_question
    from power_forecast_service.assistant.retrieval import corpus
    from power_forecast_service.assistant.workflow import prepare_answer_context
    question = "解释模拟到达延迟，同时报告真实延迟P98.5。"
    data, _ = setup_question("engie", "仅查询方法")
    docs = corpus()["chunks"]
    data, docs = prepare_answer_context(data, docs, question, "program")
    response = RequestPlan(requests=[
        {"request":"解释模拟到达延迟", "resolution":{"kind":"supported", "operations":[{"kind":"delay_policy","context":"c0"}]}},
        {"request":"报告真实延迟P98.5", "resolution":{"kind":"unavailable", "missing":missing}}])
    if accepted:
        result = validate_answer(execute_program(response, data, docs, question), data, docs, question)
        assert result["status"] == "insufficient_evidence" and "到达日志" in result["answer"] and "P98.5" in result["answer"]
    else:
        with pytest.raises(AssistantError, match="answer_program_measurement_gap"):
            execute_program(response, data, docs, question)


def test_one_named_missing_quantile_cannot_cover_two_requested_quantiles():
    from tests.unit.assistant.test_stages import setup_question
    from power_forecast_service.assistant.retrieval import corpus
    data, _ = setup_question("engie", "仅查询方法")
    question = "报告真实现场延迟P95和P99。"
    first = {"request":"真实现场延迟P95", "resolution":{"kind":"unavailable", "missing":"原始到达日志"}}
    second = {"request":"真实现场延迟P99", "resolution":{"kind":"unavailable", "missing":"原始到达日志"}}
    with pytest.raises(AssistantError, match="answer_program_measurement_gap"):
        execute_program(RequestPlan(requests=[first]), data, corpus()["chunks"], question)
    generic = {"request":"真实现场延迟", "resolution":{"kind":"unavailable", "missing":"原始到达日志"}}
    with pytest.raises(AssistantError, match="answer_program_measurement_gap"):
        execute_program(RequestPlan(requests=[generic]), data, corpus()["chunks"], question)
    assert execute_program(RequestPlan(requests=[first, second]), data, corpus()["chunks"], question).status == "insufficient_evidence"


@pytest.mark.parametrize("question,required", [
    ("实际售电收入是多少？", True),
    ("请提供真实收款凭证。", True),
    ("不需要实际收入，只查询MAE。", False),
    ("误差改善是否可以代表实际收入？", False),
])
def test_bound_revenue_obligation_detects_omission_without_inventing_excluded_request(question, required):
    from tests.unit.assistant.test_stages import setup_question
    from power_forecast_service.assistant.retrieval import corpus
    data, _ = setup_question("engie", "仅查询方法")
    docs = corpus()["chunks"]
    response = RequestPlan(requests=[{"request":"模拟规则", "resolution":{"kind":"supported",
        "operations":[{"kind":"delay_policy","context":"c0"}]}}])
    if required:
        with pytest.raises(AssistantError, match="answer_program_revenue_gap"):
            execute_program(response, data, docs, question)
        fixed = RequestPlan(requests=[*response.model_dump()["requests"],
            {"request":"实际收入", "resolution":{"kind":"unavailable", "missing":"售电结算收入凭证"}}])
        assert execute_program(fixed, data, docs, question).status == "insufficient_evidence"
    else:
        assert execute_program(response, data, docs, question).status == "answered"


def test_interval_endpoints_compare_only_with_matching_identity():
    data=evidence()
    for key,value in (("gain_ci_low",2.),("gain_ci_high",4.)):
        item=deepcopy(data["facts"][0])
        item.update(id="c0."+key,label=key,value=value,unit="%")
        data["facts"].append(item)
    response=plan({"kind":"compare","left":"c0.gain_ci_low","right":"c0.gain_ci_high"})
    assert "前者低于后者" in execute_program(response,data,[]).answer
    data["facts"][-1]["stage"]="different-stage"
    with pytest.raises(AssistantError):
        execute_program(response,data,[])


@pytest.mark.parametrize("keep_gap",[False,True])
def test_valid_plan_does_not_request_unproven_semantic_self_review(monkeypatch,keep_gap):
    import asyncio
    from langchain_core.messages import AIMessage
    from power_forecast_service.assistant import workflow
    from power_forecast_service.assistant.contracts import Question
    known = {"kind":"values","fact_ids":["c0.ridge.mae"]}
    gap = {"kind":"unknown","request":"生产实测延迟"}
    first = EvidencePlan(**ASSESSMENT,operations=[known,*([gap] if keep_gap else [])])
    seen=[]
    class Provider:
        async def ainvoke(self,messages):
            seen.append(list(messages))
            return AIMessage(content=request_plan(first).model_dump_json())
    monkeypatch.setattr(workflow,"prepare_answer_context",lambda *args:(evidence(),[]))
    agent=workflow.Assistant(None,provider=Provider(),response_mode="program")
    state={"question":Question(question="ridge MAE及生产延迟？",contexts=[{"kind":"engie_import","id":IDENTITY}]),
        "evidence":evidence(),"documents":[],"trace":{"model_calls":[]}}
    try:
        result=asyncio.run(agent.answer(state))["result"]
    finally:
        agent.close()
    assert len(seen)==1
    assert result["status"] == ("insufficient_evidence" if keep_gap else "answered")
    assert "execution_feedback" not in state["trace"] and not state["trace"].get("validation_errors")


def test_explanation_operations_cannot_evade_binding_validation():
    response=EvidencePlan(**ASSESSMENT,operations=[
        {"kind":"plain","text":"{{c0.ridge.mae}} MW","citations":[]},
        {"kind":"unknown","request":"实际收入"}])
    with pytest.raises(AssistantError,match="answer_program_fact_requires_operation"):
        execute_program(response.as_answer(),evidence(),[])


@pytest.mark.parametrize("operations", [
    [{"kind":"values","fact_ids":["c0.ridge.mae"]}, {"kind":"plain","text":"错误重复解释","citations":[]}],
    [{"kind":"plain","text":"原文解释","citations":[]}, {"kind":"plain","text":"第二种解释","citations":[]}],
    [{"kind":"unknown","request":"实际收入"}, {"kind":"plain","text":"原文解释","citations":[]}],
])
def test_plan_rejects_mixed_rendering_and_explanation_order(operations):
    with pytest.raises(ValidationError):
        EvidencePlan(**ASSESSMENT,operations=operations)


@pytest.mark.parametrize("missing,rejected",[
    ("为何有输出不一定可评分的详细原因",True),
    ("未评分的计划起报是否算作系统拒绝预测",True),
    ("实际输出次数和可评分次数的统计对象含义是否相同",True),
    ("哪些未评分起报是传感器故障造成的",False),
    ("停机记录与未评分起报的对应关系",False),
])
def test_computed_coverage_conflict_does_not_erase_real_diagnostic_gap(missing,rejected):
    response=partial({"kind":"program","operations":[{"kind":"coverage","context":"c0"}]},[missing])
    if rejected:
        with pytest.raises(AssistantError,match="answer_program_gap_conflict"):
            execute_program(response,evidence(),[])
    else:
        assert execute_program(response,evidence(),[]).status=="insufficient_evidence"


@pytest.mark.parametrize("with_gap",[False,True])
def test_plain_refusal_cannot_masquerade_as_supported_explanation(with_gap):
    body={"kind":"plain","text":"根据提供的证据，无法直接获取由传感器硬件故障导致的拒绝次数。","citations":[]}
    response=partial(body,["传感器硬件故障拒绝次数"]) if with_gap else ProgramAnswer(**ASSESSMENT,status="answered",body=body)
    with pytest.raises(AssistantError,match="answer_program_gap_in_explanation"):
        execute_program(response,evidence(),[])


def test_negative_method_statement_is_not_automatically_a_gap():
    data,docs=evidence(),[{"id":"method", "text":"输出后仍需合法标签才能评分。", "source_sha256":"a"*64}]
    body={"kind":"plain","text":"输出完成不代表可以评分。","citations":[{"document_id":"method"}]}
    response=ProgramAnswer(**ASSESSMENT,status="answered",body=body)
    assert execute_program(response,data,docs).status=="answered"


def test_selected_temporal_fact_brings_its_requested_identity_and_basis():
    from tests.unit.assistant.test_stages import setup_question
    from power_forecast_service.assistant.temporal import prepare_temporal_evidence,validate_temporal_coverage
    data,docs = setup_question("engie","仅查询方法")
    data,docs = prepare_temporal_evidence(data,docs,"02:00起报，输入截止、目标和延迟是什么？")
    draft=execute_program(plan({"kind":"values","fact_ids":["c0.time.input_cutoff"]}),data,docs)
    validate_temporal_coverage(draft,data)
    assert set(data["temporal_requirements"]) <= set(draft.fact_ids)
    assert validate_answer(draft,data,docs)["status"] == "answered"
    ordinary = next(f["id"] for f in data["facts"] if ".time." not in f["id"])
    unrelated=execute_program(plan({"kind":"values","fact_ids":[ordinary]}),data,docs)
    assert not any(".time." in key for key in unrelated.fact_ids)


def test_delay_policy_uses_bound_protocol_not_training_cutoff():
    from tests.unit.assistant.test_stages import setup_question
    from power_forecast_service.assistant.retrieval import corpus
    data, _ = setup_question("engie", "到达延迟的模拟假设是什么？")
    docs = corpus()["chunks"]
    response = partial({"kind":"program", "operations":[{"kind":"delay_policy", "context":"c0"}]})
    result = validate_answer(execute_program(response, data, docs), data, docs)
    assert "20分钟" in result["answer"] and "不晚于起报时刻减去" in result["answer"]
    assert "训练" not in result["answer"] and result["status"] == "insufficient_evidence"
    assert result["citations"] and not result["facts"]
    assert program_schema(data, docs)["$defs"]["DelayPolicy"]["properties"]["context"]["enum"] == ["c0"]
    data["records"][0]["protocol_sha256"] = "other"
    assert "DelayPolicy" not in program_schema(data, docs)["$defs"]
    with pytest.raises(AssistantError, match="answer_program_policy"):
        execute_program(response, data, docs)


def test_delay_policy_rejects_unmatched_source_and_absent_method_documents():
    from tests.unit.assistant.test_stages import setup_question
    data, _ = setup_question("engie")
    with pytest.raises(AssistantError, match="answer_program_policy"):
        execute_program(plan({"kind":"delay_policy", "context":"c0"}), data, [])
    assert "DelayPolicy" not in program_schema(evidence(),[])["$defs"]


def test_delay_policy_reuses_existing_temporal_fact_binding():
    from tests.unit.assistant.test_stages import setup_question
    from power_forecast_service.assistant.temporal import prepare_temporal_evidence
    data, docs = setup_question("engie","仅查询方法")
    data, docs = prepare_temporal_evidence(data,docs,"23:45起报的输入、目标时刻与延迟是多少？")
    response = plan({"kind":"time","context":"c0"},{"kind":"delay_policy","context":"c0"})
    result = validate_answer(execute_program(response,data,docs),data,docs)
    assert result["status"] == "answered" and "20 分钟" in result["answer"]
    assert "减去该延迟" in result["answer"]
    assert any(f["id"]=="c0.time.arrival_lag" for f in result["facts"])
    fact = next(f for f in data["facts"] if f["id"]=="c0.time.arrival_lag")
    fact["source_sha256"] = "wrong-source"
    with pytest.raises(AssistantError,match="answer_program_policy"):
        execute_program(response,data,docs)


@pytest.mark.parametrize("quantile", ["P97", "P91.25", "P98.7"])
def test_explicit_measured_quantile_cannot_disappear_behind_simulated_times(quantile):
    from tests.unit.assistant.test_stages import setup_question
    from power_forecast_service.assistant.workflow import prepare_answer_context
    question = "设01:10起报，输入截止和目标时刻是什么？再给出生产实测延迟" + quantile
    data, docs = setup_question("engie", "仅查询方法")
    data, docs = prepare_answer_context(data, docs, question, "program")
    response = plan({"kind":"time", "context":"c0"})
    before = deepcopy((data, docs, response.model_dump()))
    with pytest.raises(AssistantError, match="answer_program_measurement_gap"):
        execute_program(response, data, docs, question)
    assert (data, docs, response.model_dump()) == before
    fixed = partial(response.body.model_dump(), ["生产实测延迟" + quantile])
    result = validate_answer(execute_program(fixed, data, docs, question), data, docs, question)
    assert result["status"] == "insufficient_evidence"
    assert "00:50" in result["answer"] and quantile in result["answer"]
    wrong = partial(response.body.model_dump(), ["生产实测延迟P89"])
    with pytest.raises(AssistantError, match="answer_program_measurement_gap"):
        execute_program(wrong, data, docs, question)


@pytest.mark.parametrize("question", [
    "不要报告实测延迟P99，仅解释模拟。",
    "无需给出生产延迟P97；仅解释模拟。",
    "现有模拟参数能否推算真实延迟P99？",
    "是否可以把模拟延迟当作生产实测P99？",
    "只解释当前模拟输入规则。",
])
def test_excluded_or_inferability_question_does_not_create_missing_measurement(question):
    from tests.unit.assistant.test_stages import setup_question
    from power_forecast_service.assistant.retrieval import corpus
    data, _ = setup_question("engie", "仅查询方法")
    result = execute_program(plan({"kind":"delay_policy", "context":"c0"}), data, corpus()["chunks"], question)
    assert result.status == "answered" and not result.missing_evidence


def test_measurement_guard_requires_bound_source_and_specific_missing_subject():
    from power_forecast_service.assistant.program import validate_requested_measurement_gap
    from tests.unit.assistant.test_stages import setup_question
    from power_forecast_service.assistant.retrieval import corpus
    data, _ = setup_question("engie", "仅查询方法")
    docs = corpus()["chunks"]
    question = "给出生产实测延迟P95。"
    unrelated = ProgramAnswer(**ASSESSMENT, status="insufficient_evidence", body={"kind":"gap", "missing_evidence":["真实收入"]})
    with pytest.raises(AssistantError, match="answer_program_measurement_gap"):
        validate_requested_measurement_gap(unrelated, data, docs, question)
    validate_requested_measurement_gap(unrelated, data, [], question)
    data["records"][0]["protocol_sha256"] = "unmatched"
    validate_requested_measurement_gap(unrelated, data, docs, question)


def test_existing_plain_gap_diagnostic_precedes_new_measurement_coverage():
    from tests.unit.assistant.test_stages import setup_question
    from power_forecast_service.assistant.retrieval import corpus
    data, _ = setup_question("engie", "仅查询方法")
    response = ProgramAnswer(**ASSESSMENT, status="answered", body={"kind":"plain",
        "text":"文档没有生产到达延迟P95的测量记录。", "citations":[]})
    with pytest.raises(AssistantError, match="answer_program_gap_in_explanation"):
        execute_program(response, data, corpus()["chunks"], "报告生产到达延迟P95。")


@pytest.mark.parametrize("repaired", [True, False])
def test_workflow_preserves_mixed_answer_and_two_call_limit(monkeypatch, repaired):
    import asyncio
    from langchain_core.messages import AIMessage
    from power_forecast_service.assistant import workflow
    from power_forecast_service.assistant.contracts import Question
    from tests.unit.assistant.test_stages import setup_question
    question = "设01:10起报，最晚输入标签和目标时刻是什么？再给出生产实测延迟P97。"
    data, docs = setup_question("engie", "仅查询方法")
    data, docs = workflow.prepare_answer_context(data, docs, question, "program")
    monkeypatch.setattr(workflow, "prepare_answer_context", lambda *args:(data, docs))
    calls, captures = [], []
    class Provider:
        async def ainvoke_structured(self, messages, *, schema, call_id):
            calls.append(call_id)
            operations = [{"kind":"time", "context":"c0"}]
            if repaired and len(calls) == 2:
                operations.append({"kind":"unknown", "request":"生产实测延迟P97"})
            response = EvidencePlan(**ASSESSMENT, operations=operations)
            return AIMessage(content=request_plan(response).model_dump_json())
    agent = workflow.Assistant(None, provider=Provider(), response_mode="program", capture_draft=captures.append)
    state = {"question":Question(question=question, contexts=[{"kind":"engie_import", "id":data["records"][0]["id"]}]),
        "evidence":data, "documents":docs, "trace":{"model_calls":[]}}
    try:
        if repaired:
            result = asyncio.run(agent.answer(state))["result"]
            assert result["status"] == "insufficient_evidence" and "00:50" in result["answer"] and "P97" in result["answer"]
        else:
            with pytest.raises(AssistantError, match="answer_program_measurement_gap"):
                asyncio.run(agent.answer(state))
    finally:
        agent.close()
    assert len(calls) == len(captures) == 2
    assert all(item["resolution"]["kind"] == "supported" for item in captures[0]["requests"])
    assert state["trace"]["validation_errors"][0]["error"] == "answer_program_measurement_gap"


def test_development_gate_reports_registered_adoption_in_its_own_stage():
    from tests.unit.assistant.test_stages import setup_question
    question = "开发改善和采用阈值分别是多少？最终留出更好能否改写开发决定？"
    data, docs = setup_question("engie", question)
    before = deepcopy(data)
    operations = [{"kind":"stage", "context":"c0", "role":role}
                  for role in dict.fromkeys([*(o["role"] for o in data["stage_requirements"]), "adoption_decision"])]
    draft = execute_program(plan(*operations), data, docs, question)
    result = validate_answer(draft, data, docs, question)
    assert data == before
    assert any(f["id"] == "c0.adopted" and f["value"] is False for f in result["facts"])
    adoption = [c for c in draft.stage_claims if c.role == "adoption_decision"]
    assert len(adoption) == 1 and "{{c0.adopted}}" in adoption[0].text
    assert all("{{c0.adopted}}" not in c.text for c in draft.stage_claims if c.role != "adoption_decision")


@pytest.mark.parametrize("with_stage", [False, True])
def test_stage_values_are_rendered_once_not_once_per_projection(with_stage):
    from tests.unit.assistant.test_stages import setup_question
    question = "请把共享收缩在开发阶段和正式留出阶段的MAE改善各列一次，别把两个阶段混用。"
    data, docs = setup_question("engie", question)
    keys = ["c0.development.lightgbm_l1_shrink.mae_gain", "c0.lightgbm_l1_shrink.mae_gain"]
    operations = [{"kind":"values", "fact_ids":keys}, {"kind":"values", "fact_ids":keys[:1]}]
    if with_stage:
        operations.extend({"kind":"stage", "context":"c0", "role":item["role"]} for item in data["stage_requirements"])
    draft = execute_program(plan(*operations), data, docs, question)
    result = validate_answer(draft, data, docs, question)
    body = "\n".join(claim.text for claim in draft.stage_claims)
    assert all(body.count("{{"+key+"}}") == 1 for key in keys)
    assert "c0.adopted" not in {fact["id"] for fact in result["facts"]}
    assert "c0.development.mae_gain_gate" not in {fact["id"] for fact in result["facts"]}


def test_multiple_value_operations_share_count_relation_explanation():
    operations = [{"kind":"values", "fact_ids":["c0."+name]} for name in ("output_count", "scoreable")]
    result = validate_answer(execute_program(plan(*operations), evidence(), []), evidence(), [])
    assert "实际输出与可评分次数不相等" in result["answer"]
    assert result["answer"].count("90 次") == 1
    assert result["answer"].count("80 次") == 1


def test_comparison_already_presents_raw_values_before_extra_value_reads():
    data = evidence()
    response = plan({"kind":"compare", "left":"c0.output_count", "right":"c0.scoreable"},
                    {"kind":"values", "fact_ids":["c0.output_count", "c0.scoreable"]})
    draft = execute_program(response, data, [])
    assert draft.answer.count("{{c0.output_count}}") == 1
    assert draft.answer.count("{{c0.scoreable}}") == 1
    result = validate_answer(draft, data, [])
    assert "前者高于后者" in result["answer"]


def test_derived_values_do_not_reprint_selected_operands():
    from tests.unit.assistant.test_stages import setup_question
    data, docs = setup_question("engie", "共享收缩的MAE与持久性比较，给出原值、绝对差和改善率。")
    keys = ["c0.lightgbm_l1_shrink.mae", "c0.persistence.mae",
            "c0.lightgbm_l1_shrink.mae_delta", "c0.lightgbm_l1_shrink.mae_gain"]
    draft = execute_program(plan({"kind":"values", "fact_ids":keys}), data, docs)
    result = validate_answer(draft, data, docs)
    assert all(draft.answer.count("{{" + key + "}}") == 1 for key in keys)
    assert "候选原值低于基线原值" in result["answer"] and "。。" not in result["answer"]


@pytest.mark.parametrize("explicit_gain", [False, True])
def test_adoption_computation_renders_development_operand_in_its_own_stage(explicit_gain):
    from tests.unit.assistant.test_stages import setup_question
    question = "最终结果已有改善，为什么还不自动替换持续性默认模型？"
    data, docs = setup_question("engie", question)
    gain = "c0.development.lightgbm_l1_shrink.mae_gain"
    keys = ["c0.lightgbm_l1_shrink.mae_gain", "c0.adopted", "c0.development.mae_gain_gate"]
    if explicit_gain:
        keys.append(gain)
    response = plan({"kind":"values", "fact_ids":keys},
                    {"kind":"stage", "context":"c0", "role":"adoption_decision"})
    draft = execute_program(response, data, docs, question)
    result = validate_answer(draft, data, docs, question)
    development = next(c for c in draft.stage_claims if c.role == "development_result")
    assert "{{" + gain + "}}" in development.text
    assert sum(c.text.count("{{" + gain + "}}") for c in draft.stage_claims) == 1
    assert next(f for f in result["facts"] if f["id"] == "c0.adopted")["value"] is False
    assert "不由正式指标追认" in result["answer"]


@pytest.mark.parametrize("question", ["请解释模拟到达延迟的处理。", "到达延时按什么假设处理？"])
def test_explicit_policy_question_requires_consumption_not_vague_citation(question):
    from tests.unit.assistant.test_stages import setup_question
    from power_forecast_service.assistant.retrieval import corpus
    data, _ = setup_question("engie", "仅查询方法")
    docs = corpus()["chunks"]
    vague = partial({"kind":"plain", "text":"模拟是一种假设。", "citations":[{"document_id":"engie-1e630b893889c121"}]})
    with pytest.raises(AssistantError, match="answer_program_policy_required"):
        execute_program(vague, data, docs, question)
    valid = partial({"kind":"program", "operations":[{"kind":"delay_policy", "context":"c0"}]})
    assert "20分钟" in validate_answer(execute_program(valid,data,docs,question),data,docs)["answer"]


def test_policy_obligation_is_not_masked_by_generic_refusal_representation():
    from tests.unit.assistant.test_stages import setup_question
    from power_forecast_service.assistant.retrieval import corpus
    data,_=setup_question("engie","仅查询方法")
    response=ProgramAnswer(**ASSESSMENT,status="answered",body={"kind":"plain","text":"文档没有实测延迟。模拟不同于测量。","citations":[]})
    with pytest.raises(AssistantError,match="answer_program_policy_required"):
        execute_program(response,data,corpus()["chunks"],"实测生产P99有没有？请区分模拟到达延迟假设。")


@pytest.mark.parametrize("question", ["给生产实测延迟分布，不要用模拟假设代替。", "不要解释模拟到达延迟，仅给实测数据。", "没有实测时不需要说明模拟到达延迟。"])
def test_pure_measurement_refusal_does_not_force_unrequested_policy(question):
    from tests.unit.assistant.test_stages import setup_question
    from power_forecast_service.assistant.retrieval import corpus
    data, _ = setup_question("engie", "仅查询方法")
    gap = ProgramAnswer(**ASSESSMENT,status="insufficient_evidence",body={"kind":"gap","missing_evidence":["生产实测分布"]})
    assert execute_program(gap, data, corpus()["chunks"], question).status == "insufficient_evidence"


@pytest.mark.parametrize("counts", [None, ["planned"], ["output_count", "scoreable"]])
def test_time_projection_boundary_matches_generation_and_runtime(counts):
    from jsonschema import Draft202012Validator
    from pydantic import ValidationError
    from power_forecast_service.assistant.program import RequestPlan, request_program_schema
    from power_forecast_service.assistant.workflow import prepare_answer_context
    from tests.unit.assistant.test_stages import setup_question
    question = "23:45起报的输入截止和目标时刻是什么？"
    data, docs = setup_question("engie", "仅查询方法")
    data, docs = prepare_answer_context(data, docs, question, "program")
    payload = {"requests":[{"request":question, "resolution":{"kind":"supported",
        "operations":[{"kind":"time", "context":"c0", "counts":counts}]}}]}
    schema = Draft202012Validator(request_program_schema(data, docs))
    assert schema.is_valid(payload) is (counts is None)
    if counts is None:
        reply = validate_answer(execute_program(RequestPlan.model_validate(payload), data, docs, question), data, docs, question)
        assert "23:25" in reply["answer"] and "00:45" in reply["answer"]
    else:
        with pytest.raises(ValidationError, match="Time operations cannot"):
            RequestPlan.model_validate(payload)


@pytest.mark.parametrize("reverse", [False, True])
def test_requested_shared_stage_is_rendered_once_with_adoption_and_real_gap(reverse):
    from power_forecast_service.assistant.program import RequestedPlan
    from tests.unit.assistant.test_stages import setup_question
    question = "分别列出开发改善和采用状态，另给实际收入凭证。"
    data, docs = setup_question("engie", question)
    gain = "c0.development.lightgbm_l1_shrink.mae_gain"
    requests = [
        {"request":"开发改善", "resolution":{"kind":"supported", "operations":[
            {"kind":"stage", "context":"c0", "role":"development_result", "fact_ids":[gain]}]}},
        {"request":"采用状态", "resolution":{"kind":"supported", "operations":[
            {"kind":"stage", "context":"c0", "role":"adoption_decision",
             "fact_ids":["c0.adopted", "c0.development.mae_gain_gate"]}]}},
        {"request":"实际收入", "resolution":{"kind":"unavailable", "missing":"实际收入结算凭证"}},
    ]
    if reverse:
        requests.reverse()
    raw = {"requests":requests}
    response = RequestedPlan.model_validate(raw)
    before = response.model_dump()
    draft = execute_program(response, data, docs, question)
    result = validate_answer(draft, data, docs, question)
    assert response.model_dump() == before
    assert sum(c.text.count("{{" + gain + "}}") for part in draft.parts for c in part.draft.stage_claims) == 1
    assert {f["id"] for f in result["facts"]} >= {gain, "c0.adopted", "c0.development.mae_gain_gate"}
    assert result["status"] == "insufficient_evidence" and "实际收入" in result["answer"]
    assert "不由正式指标追认" in result["answer"]
    legacy = execute_program(RequestPlan.model_validate(raw), data, docs, question)
    assert sum(c.text.count("{{" + gain + "}}") for part in legacy.parts for c in part.draft.stage_claims) == 2


@pytest.mark.parametrize("difference", ["object", "role", "text", "citation", "quote", "body", "gap", None])
def test_requested_stage_dedup_preserves_distinct_evidence_and_original_parts(difference):
    from power_forecast_service.assistant.contracts import DraftAnswer, RequestAnswerPart, StageClaim
    from power_forecast_service.assistant.program import deduplicate_requested_stages
    claim = StageClaim(object_id=IDENTITY, role="development_result", text="{{c0.ridge.mae}}", citations=["doc-a"])
    first = RequestAnswerPart(request_index=0, draft=DraftAnswer(status="answered", stage_claims=[claim],
        fact_ids=["c0.ridge.mae"], citations=["doc-a"], quotes={"doc-a":"原文甲"}))
    second = first.model_copy(deep=True)
    second.request_index = 1
    target = second.draft.stage_claims[0]
    if difference == "object":
        from uuid import UUID
        target.object_id = UUID("00000000-0000-0000-0000-000000000002")
    elif difference == "role":
        target.role = "final_holdout_result"
    elif difference == "text":
        target.text += "。另有适用限制。"
    elif difference == "citation":
        target.citations = ["doc-b"]
        second.draft.citations = ["doc-b"]
        second.draft.quotes = {"doc-b":"原文甲"}
    elif difference == "quote":
        second.draft.quotes["doc-a"] = "原文乙"
    elif difference == "body":
        second.draft.answer = "保留非阶段正文"
    elif difference == "gap":
        second.draft.missing_evidence = ["现场记录"]
    parts = [first, second]
    before = deepcopy(parts)
    cleaned = deduplicate_requested_stages(parts)
    assert parts == before
    assert len(cleaned) == (1 if difference is None else 2)
