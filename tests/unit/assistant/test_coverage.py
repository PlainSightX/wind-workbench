"""覆盖数量的业务边界与实际纠错消费；不使用独立评测的required_facts。"""

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
from uuid import UUID

import pytest

from power_forecast_service.assistant.contracts import AssistantError, DraftAnswer, Question
from power_forecast_service.assistant.coverage import coverage_requirements, validate_coverage
from power_forecast_service.assistant.workflow import Assistant


def evidence():
    identity = str(UUID(int=4))
    counts = {"planned": 100, "input_valid": 90, "output_count": 90, "scoreable": 80}
    return {"records": [{"id": identity, "models": {}, "coverage": counts}], "scopes": [],
        "facts": [{"id": "c0." + key, "label": key, "value": value, "unit": "次", "object_id": identity}
                  for key, value in counts.items()]}


@pytest.mark.parametrize("question,expected", [
    ("最终计划起报多少？合法输入是否全部输出？有多少可评分？", {"planned", "input_valid", "output_count", "scoreable"}),
    ("有效输入都产出了吗？", {"input_valid", "output_count"}),
    ("可评分起报有几次？", {"scoreable"}),
    ("计划起报与可评分是什么含义？", set()),
    ("输出模型的MAE是多少？", set()),
    ("实际输出多少次？", {"output_count"}),
    ("计划起报多少？合法输入是什么含义？", {"planned"}),
])
def test_requirements_come_from_question_and_selected_evidence(question, expected):
    assert set(coverage_requirements(evidence(), question)) == {"c0." + key for key in expected}


def test_unknown_or_other_object_counts_do_not_create_requirements():
    value = evidence()
    value["facts"][0]["object_id"] = str(UUID(int=5))
    assert coverage_requirements(value, "计划起报多少？") == []
    assert coverage_requirements({"records": [], "facts": []}, "计划起报多少？") == []


@pytest.mark.parametrize("status,body", [("answered", "全部输出。"), ("insufficient_evidence", "{{c0.output_count}}")])
def test_attachment_or_refusal_cannot_replace_requested_body(status, body):
    draft = DraftAnswer(status=status, answer=body, fact_ids=["c0.output_count"])
    with pytest.raises(AssistantError) as error:
        validate_coverage(draft, ["c0.output_count"])
    assert error.value.code == "answer_coverage_incomplete"


@pytest.mark.parametrize("unbound", [False, True])
def test_changed_object_and_new_wording_use_actual_two_call_loop(unbound):
    value = evidence()
    question = Question(question="有效输入都产出了吗？", contexts=[{"kind": "engie_import", "id": value["records"][0]["id"]}])
    received = []
    drafts = [
        {"status": "answered", "answer": "合法输入{{c0.input_valid}}全部输出。", "fact_ids": ["c0.input_valid", "c0.output_count"]},
        {"status": "answered", "answer": "输入{{c0.input_valid}}，实际输出{{c0.output_count}}，全部产出。", "fact_ids": ["c0.input_valid", "c0.output_count"]},
    ]
    if unbound:
        drafts[0]["answer"] += "另有999次。"
    class Provider:
        async def ainvoke(self, messages):
            received.append(deepcopy(messages))
            return SimpleNamespace(content=json.dumps(drafts[len(received)-1]), usage_metadata={}, response_metadata={})
    assistant = Assistant(None, provider=Provider())
    state = {"question": question, "evidence": value, "documents": [], "trace": {"model_calls": [], "tools": []}}
    try:
        result = asyncio.run(assistant.answer(state))["result"]
    finally:
        assistant.close()
    assert result["status"] == "answered" and len(received) == 2
    assert state["trace"]["repair_reason"] == ("answer_number_unbound" if unbound else "answer_coverage_incomplete")
    assert "c0.output_count" in received[1][-1][1]
    assert "answer_coverage_incomplete" in received[1][-1][1]
    if unbound:
        assert "answer_number_unbound" in received[1][-1][1]
    assert state["trace"]["coverage_requirements"] == ["c0.input_valid", "c0.output_count"]
    assert "required_facts" not in json.dumps(received, ensure_ascii=False)
