"""单一正文和版本引用的技术边界，不把手写回答当成模型成功率。"""

from copy import deepcopy
from uuid import uuid4
import re

import pytest
from pydantic import ValidationError

from power_forecast_service.assistant.contracts import AssistantError, ReferenceAnswer
from power_forecast_service.assistant.references import resolve_reference_answer, reference_schema
from power_forecast_service.assistant.validation import validate_answer


def document():
    return {"id": "independent-source", "title": "另一对象方法", "source_sha256": "a" * 64,
            "text": "这是完整的**连续证据**。采用二十分钟模拟延迟，不能称为真实测量。"}


def selected(document_id="independent-source"):
    return {"document_id": document_id}


def response(text="依据原文，这是模拟而非实测。", citations=None):
    return ReferenceAnswer(status="answered", body={"kind": "plain", "text": text,
        "citations": [selected()] if citations is None else citations})


def evidence():
    return {"facts": [{"id": "fresh_object.gain", "value": -8.75, "label": "另一对象改善", "unit": "%"}],
            "records": [{"models": {"new_candidate": {}}}]}


def test_program_uses_exact_full_source_and_preserves_model_text():
    item = document()
    original = deepcopy(item)
    value = response("真实事实为{{ fresh_object.gain }}；模拟不代表实测。")
    draft = resolve_reference_answer(value, [item])
    assert draft.quotes == {item["id"]: item["text"]}
    assert draft.fact_ids == ["fresh_object.gain"]
    result = validate_answer(draft, evidence(), [item])
    assert result["answer"] == "真实事实为另一对象改善：-8.75 %；模拟不代表实测。"
    assert result["citations"][0]["quote"] == item["text"]
    assert result["citations"][0]["revision"] == item["source_sha256"]
    assert item == original and "{{ fresh_object.gain }}" in value.body.text


def test_unknown_citation_is_not_guessed():
    with pytest.raises(AssistantError, match="answer_citation_invalid"):
        resolve_reference_answer(response(citations=[selected("missing")]), [document()])


@pytest.mark.parametrize("revision", ["a" * 64, "b" * 64])
def test_model_cannot_override_even_a_valid_looking_revision(revision):
    with pytest.raises(ValidationError, match="extra_forbidden"):
        response(citations=[{**selected(), "source_sha256": revision}])


def test_reused_document_id_binds_only_current_request_version():
    old = document()
    current = {**old, "source_sha256": "b" * 64, "text": "当前版本明确表示是模拟假设。"}
    for item in (old, current, old):
        draft = resolve_reference_answer(response(), [item])
        result = validate_answer(draft, evidence(), [item])
        assert result["citations"][0]["revision"] == item["source_sha256"]
        assert result["citations"][0]["quote"] == item["text"]


@pytest.mark.parametrize("key,value", [("text", "同ID不同原文。"), ("source_sha256", "b" * 64), ("title", "另一来源")])
def test_ambiguous_duplicate_document_is_rejected(key, value):
    first, second = document(), document()
    second[key] = value
    with pytest.raises(AssistantError, match="answer_citation_ambiguous"):
        resolve_reference_answer(response(), [first, second])


def test_identical_sources_and_repeated_selections_are_deduplicated():
    draft = resolve_reference_answer(response(citations=[selected(), selected()]), [document(), document()])
    assert draft.citations == ["independent-source"]
    assert len(draft.quotes) == 1


@pytest.mark.parametrize("extra", [{"quotes": {"independent-source": "伪造拼接原文"}},
                                  {"fact_ids": ["fresh_object.gain"]}, {"answer": "未绑定的另一份正文"}])
def test_reference_contract_cannot_smuggle_redundant_output(extra):
    payload = response().model_dump()
    with pytest.raises(ValidationError):
        ReferenceAnswer.model_validate({**payload, **extra})


@pytest.mark.parametrize("text,code", [
    ("未知{{another.gain}}", "answer_fact_invalid"),
    ("{{fresh_object.gain}} kW", "answer_fact_unit_conflict"),
    ("-{{fresh_object.gain}}", "answer_fact_sign_conflict"),
])
def test_resolved_references_do_not_bypass_original_numeric_contract(text, code):
    draft = resolve_reference_answer(response(text), [document()])
    with pytest.raises(AssistantError, match=code):
        validate_answer(draft, evidence(), [document()])


@pytest.mark.parametrize("wrong_stage", [False, True])
def test_staged_body_uses_existing_object_fact_and_source_rules(wrong_stage):
    identity = str(uuid4())
    data = evidence()
    data["facts"][0]["stage"] = "development_summary"
    data["facts"].append({"id": "fresh_object.final", "value": 1.25, "unit": "%", "label": "正式改善"})
    data.update(stage_requirements=[{"object_id": identity, "role": "development_result"}],
        stage_options=[{"object_id": identity, "role": "development_result", "label": "开发评价",
            "allowed_fact_ids": ["fresh_object.gain"], "required_fact_ids": ["fresh_object.gain"],
            "allowed_citations": ["independent-source"], "required_citations": ["independent-source"]}])
    value = ReferenceAnswer(status="answered", body={"kind": "staged", "claims": [{
        "object_id": identity, "role": "development_result",
        "text": "{{fresh_object.final}}" if wrong_stage else "{{fresh_object.gain}}；不反推最终采用。",
        "citations": [selected()]}]})
    draft = resolve_reference_answer(value, [document()])
    assert draft.answer == "" and len(draft.stage_claims) == 1
    if wrong_stage:
        with pytest.raises(AssistantError, match="answer_stage_fact_conflict"):
            validate_answer(draft, data, [document()])
    else:
        result = validate_answer(draft, data, [document()])
        assert result["answer"].startswith("开发评价：")
        assert result["answer"].count("另一对象改善") == 1


def test_valid_source_is_not_a_semantic_entailment_judge():
    # 格式/身份校验不能识别任意自然语言的反义关系，后续独立语义验收必须保留。
    draft = resolve_reference_answer(response("这些资料证明延迟是真实测量。"), [document()])
    assert validate_answer(draft, evidence(), [document()])["status"] == "answered"


def test_fact_count_after_resolution_keeps_original_delivery_limit():
    value = response(" ".join("{{fact." + str(index) + "}}" for index in range(16)), citations=[])
    with pytest.raises(AssistantError, match="answer_reference_limit"):
        resolve_reference_answer(value, [])


def test_individually_valid_stage_citations_cannot_overflow_aggregate_limit():
    documents = [{**document(), "id": "doc-" + str(index)} for index in range(9)]
    claims = [{"object_id": str(uuid4()), "role": "development_result", "text": "开发阶段。",
               "citations": [selected(doc["id"]) for doc in documents[start:start + 3]]}
              for start in range(0, 9, 3)]
    value = ReferenceAnswer(status="answered", body={"kind": "staged", "claims": claims})
    with pytest.raises(AssistantError, match="answer_reference_limit"):
        resolve_reference_answer(value, documents)


@pytest.mark.parametrize("text,valid", [
    ("实际改善{{fresh_object.gain}}，仅限所选结果。", True), ("改进为-8.75%。", False),
    ("{{abs(fresh_object.gain)}}", False), ("{{unknown.fact}}", False), ("七天区间不能推广。", True),
])
def test_plain_decoder_grammar_requires_real_fact_tokens(text, valid):
    schema = reference_schema(evidence(), [document()], ["fresh_object.gain"])
    assert schema["properties"]["body"] == {"$ref": "#/$defs/PlainReferenceBody"}
    assert "StagedReferenceBody" not in schema["$defs"]
    value = schema["$defs"]["PlainReferenceBody"]["properties"]["text"]
    assert bool(re.fullmatch(value["pattern"], text)) is valid
    assert "{{fresh_object.gain}}" in value["description"]


def test_compiled_stage_grammar_limits_each_role_without_choosing_conclusion():
    data = evidence()
    identity = str(uuid4())
    data["facts"].append({"id": "other.final", "label": "正式值", "value": 3.1, "unit": "%"})
    data.update(stage_requirements=[{"object_id": identity, "role": "development_result"}],
        stage_options=[{"object_id": identity, "role": "development_result", "allowed_fact_ids": ["fresh_object.gain"],
                        "required_fact_ids": ["fresh_object.gain"], "allowed_citations": ["independent-source"], "required_citations": ["independent-source"]}])
    original = deepcopy(data)
    schema = reference_schema(data, [document()])
    claims = schema["$defs"]["StagedReferenceBody"]["properties"]["claims"]
    assert claims["minItems"] == claims["maxItems"] == 1 and claims["items"] is False
    fields = claims["prefixItems"][0]["properties"]
    assert fields["object_id"]["const"] == identity and fields["role"]["const"] == "development_result"
    assert fields["text"]["maxLength"] == 1500
    assert re.fullmatch(fields["text"]["pattern"], "{{fresh_object.gain}}")
    assert not re.fullmatch(fields["text"]["pattern"], "{{other.final}}")
    assert "enum" not in fields["text"] and data == original


def test_decoder_allows_natural_order_and_leaves_missing_binding_to_validation():
    data = evidence()
    data["facts"].append({"id": "fresh_object.other", "label": "状态", "value": False, "unit": ""})
    schema = reference_schema(data, [document()], ["fresh_object.gain", "fresh_object.other"])
    pattern = schema["$defs"]["PlainReferenceBody"]["properties"]["text"]["pattern"]
    assert re.fullmatch(pattern, "已知{{fresh_object.gain}}，未通过。")
    assert re.fullmatch(pattern, "已知{{fresh_object.gain}}，状态为{{fresh_object.other}}。")
    assert re.fullmatch(pattern, "状态为{{fresh_object.other}}，已知{{fresh_object.gain}}。")
    assert not re.fullmatch(pattern, "已知{{fresh_object.gain}}，状态为{{unknown}}。")
    description = schema["$defs"]["PlainReferenceBody"]["properties"]["text"]["description"]
    assert "fresh_object.gain=另一对象改善（数值，%）" in description
    assert "fresh_object.other=状态（是否事实，）" in description


@pytest.mark.parametrize("unsafe", ['"', '\\', '\n', '\r', '\t', '\x00', '\x1f'])
def test_decoder_pattern_excludes_raw_json_string_delimiters(unsafe):
    schema = reference_schema(evidence(), [document()], ["fresh_object.gain"])
    pattern = schema["$defs"]["PlainReferenceBody"]["properties"]["text"]["pattern"]
    assert not re.fullmatch(pattern, "说明" + unsafe + "{{fresh_object.gain}}。")
    assert re.fullmatch(pattern, "说明“改善”{{fresh_object.gain}}。")
