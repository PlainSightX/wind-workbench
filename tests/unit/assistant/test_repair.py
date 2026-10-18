"""局部修复必须由真实工作流消费；mock仅证明边界，不代表模型质量。"""

import asyncio
from copy import deepcopy
import json
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage

from power_forecast_service.assistant import workflow
from power_forecast_service.assistant.contracts import AssistantError, Question, ReferenceAnswer
from power_forecast_service.assistant.repair import TextRepairs, apply_repairs, repair_targets, repair_schema, repair_feedback
from power_forecast_service.assistant.references import reference_schema


def test_local_repair_is_explicit_experimental_not_default():
    assistant = workflow.Assistant(None, response_mode="references")
    try:
        assert assistant.repair_mode == "full"
    finally:
        assistant.close()


def fixture():
    identity = str(uuid4())
    facts = [{"id": "x.gain", "value": 1.2, "unit": "%", "label": "开发改善", "object_id": identity},
             {"id": "x.adopted", "value": False, "unit": "", "label": "采用", "object_id": identity}]
    options = [{"object_id": identity, "role": role, "label": role, "allowed_fact_ids": allowed,
                "required_fact_ids": required, "allowed_citations": [], "required_citations": []}
               for role, allowed, required in [("development_result", ["x.gain"], ["x.gain"]),
                                               ("adoption_decision", ["x.gain", "x.adopted"], ["x.adopted"])]]
    evidence = {"facts": facts, "records": [{"id": identity, "models": {"candidate": {}}}], "scopes": ["test"],
                "stage_options": options, "stage_requirements": [{"object_id": identity, "role": o["role"]} for o in options]}
    response = ReferenceAnswer(status="answered", body={"kind": "staged", "claims": [
        {"object_id": identity, "role": "development_result", "text": "开发改善为{{x.gain}}。", "citations": []},
        {"object_id": identity, "role": "adoption_decision", "text": "开发门未过，默认保持原模型。", "citations": []}]})
    return evidence, response


def test_only_bad_fragment_selected_and_unchanged_fields_preserved():
    data, original = fixture()
    errors = [AssistantError("answer_stage_metric_missing")]
    targets = repair_targets(original, data, [], errors)
    assert [item["index"] for item in targets] == [1]
    assert targets[0]["missing"] == ["x.adopted"] and not targets[0]["problems"]
    before = original.model_dump()
    replacement = TextRepairs(patches=[{"index": 1, "text": "由于开发门未过，{{x.adopted}}，默认保持原模型。"}])
    result = apply_repairs(original, replacement, targets)
    assert original.model_dump() == before
    assert result.body.claims[0] == original.body.claims[0]
    assert result.body.claims[1].object_id == original.body.claims[1].object_id
    assert result.body.claims[1].citations == original.body.claims[1].citations
    schema = repair_schema(reference_schema(data, [], []), targets)
    item = schema["properties"]["patches"]["prefixItems"][0]
    assert item["properties"]["index"]["const"] == 1
    assert set(item["properties"]) == {"index", "text"}
    assert "{{x.adopted}}" in repair_feedback(targets)


@pytest.mark.parametrize("indices", [[0], [1, 1], [1, 0]])
def test_wrong_duplicate_or_extra_target_is_rejected(indices):
    _, response = fixture()
    patches = TextRepairs(patches=[{"index": index, "text": "越界文本"} for index in indices])
    with pytest.raises(AssistantError, match="answer_repair_target_invalid"):
        apply_repairs(response, patches, [{"index": 1}])


@pytest.mark.parametrize("code", ["answer_fact_invalid", "answer_citation_invalid", "answer_stage_identity_invalid", "answer_schema_invalid", "provider_result_unknown", "answer_adoption_contradiction", "answer_time_fact_missing"])
def test_unlocalized_or_transport_problem_never_uses_local_patch(code):
    data, response = fixture()
    assert repair_targets(response, data, [], [AssistantError(code)]) == []


class Provider:
    def __init__(self, responses):
        self.responses, self.calls = responses, []

    async def ainvoke_structured(self, messages, **kwargs):
        self.calls.append({"messages": deepcopy(messages), **kwargs})
        value = self.responses[len(self.calls) - 1]
        if isinstance(value, Exception):
            raise value
        return AIMessage(content=json.dumps(value, ensure_ascii=False))


def execute(monkeypatch, second):
    data, original = fixture()
    provider = Provider([original.model_dump(mode="json"), second])
    monkeypatch.setattr(workflow, "prepare_answer_context", lambda evidence, documents, *args: (evidence, documents))
    assistant = workflow.Assistant(None, provider=provider, response_mode="references", repair_mode="targeted")
    state = {"evidence": data, "documents": [], "trace": {"model_calls": []},
             "question": Question(question="为什么没有采用？", contexts=[{"kind": "engie_import", "id": data["records"][0]["id"]}])}
    try:
        return asyncio.run(assistant.answer(state)), provider, state
    finally:
        assistant.close()


def test_workflow_consumes_local_response_and_validates_whole_answer(monkeypatch):
    result, provider, state = execute(monkeypatch, {"patches": [{"index": 1, "text": "开发门未过，因此{{x.adopted}}，默认保持原模型。"}]})
    assert result["result"]["status"] == "answered"
    assert len(provider.calls) == 2
    assert "patches" in provider.calls[1]["schema"]["properties"]
    assert provider.calls[0]["messages"] == provider.calls[1]["messages"][:2]
    assert state["trace"]["repair_targets"][0]["index"] == 1
    assert "默认模型已替换" not in result["result"]["answer"]


@pytest.mark.parametrize("text,code", [
    ("{{x.adopted}}，默认模型已替换。", "answer_adoption_contradiction"),
    ("{{x.adopted}}，改善为 个百分点。", "answer_prose_incomplete"),
    ("只有事实列表，没有绑定。", "answer_stage_metric_missing"),
    ("引用另一个对象{{wrong.adopted}}。", "answer_fact_invalid"),
])
def test_bad_local_repair_is_rejected_without_third_call(monkeypatch, text, code):
    with pytest.raises(AssistantError, match=code):
        execute(monkeypatch, {"patches": [{"index": 1, "text": text}]})


def test_bad_local_target_does_not_fall_back_to_third_call(monkeypatch):
    with pytest.raises(AssistantError, match="answer_repair_target_invalid"):
        execute(monkeypatch, {"patches": [{"index": 0, "text": "覆盖正常段落"}]})


