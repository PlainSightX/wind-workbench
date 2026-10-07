"""时间推导的真实协议来源、新时刻和拒绝猜测边界；不把fixture记作模型成绩。"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from uuid import UUID

import pytest

from power_forecast_service.assistant.contracts import AssistantError, DraftAnswer
from power_forecast_service.assistant.retrieval import corpus
from power_forecast_service.assistant.stages import stage_sources
from power_forecast_service.assistant.temporal import prepare_temporal_evidence, validate_temporal_coverage


def evidence():
    source = stage_sources()["engie"]
    return {"records": [{"id": str(UUID(int=17)), "scope": "engie_final", "models": {},
                         "result_sha256": source["result_sha256"], "protocol_sha256": source["protocol_sha256"]}],
            "facts": [], "scopes": ["engie_final"]}


def question(clock="10:00"):
    return clock + "起报用到多晚的输入？预测哪些时距？这个延迟是真实测量的吗？"


def test_time_projection_matches_original_protocol_and_method_sources():
    root = Path(__file__).resolve().parents[3]
    protocol = stage_sources()["engie"]["time_contract"]
    raw = (root / protocol["source"]["path"]).read_bytes()
    assert sha256(raw).hexdigest() == protocol["source"]["sha256"]
    actual = json.loads(raw)["protocol"]
    assert actual["arrival_lag_minutes"] == protocol["arrival_lag_minutes"]
    assert actual["target_minutes_after_issue"] == protocol["target_minutes_after_issue"]
    docs = {d["id"]: d for d in corpus()["chunks"]}
    for identity in protocol["citations"]:
        doc = docs[identity]
        assert doc["source_sha256"] == protocol["method_source_sha256"]
        assert sha256((root / doc["source"]).read_bytes()).hexdigest() == protocol["method_source_sha256"]
    assert "20分钟是模拟假设" in docs[protocol["citations"][0]]["text"]


@pytest.mark.parametrize("clock,cutoff,first,last", [
    ("10:00", "09:40", "10:10", "11:00"), ("13:25", "13:05", "13:35", "14:25"),
    ("00:05", "前一日 23:45", "00:15", "01:05"), ("23:50", "23:30", "次日 00:00", "次日 00:50"),
])
def test_protocol_arithmetic_for_unseen_clocks_and_day_rollover(clock, cutoff, first, last):
    original = evidence()
    before = deepcopy(original)
    data, docs = prepare_temporal_evidence(original, [], question(clock))
    facts = {f["id"]: f for f in data["facts"]}
    assert facts["c0.time.input_cutoff"]["value"] == cutoff
    targets = facts["c0.time.targets"]["value"].split("、")
    assert len(targets) == 6 and targets[0] == first and targets[-1] == last
    assert facts["c0.time.arrival_lag"]["value"] == 20
    assert facts["c0.time.delay_basis"]["value"] == "模拟假设，非真实到达延迟测量"
    assert facts["c0.time.targets"]["derived"] is True
    assert original == before
    assert prepare_temporal_evidence(data, docs, question(clock)) == (data, docs)


@pytest.mark.parametrize("text", [question("25:00"), question("10:70"), question("10:3"),
                                  question("110:00"), question("10:00:30"),
                                  "10:00或11:00起报用哪些输入？", "10:00训练标签截止吗？", "模型MAE是多少？"])
def test_ambiguous_malformed_or_non_issue_question_does_not_guess(text):
    original = evidence()
    assert prepare_temporal_evidence(original, [], text) == (original, [])


@pytest.mark.parametrize("key,value", [("scope", "q1"), ("result_sha256", "f" * 64), ("protocol_sha256", "f" * 64)])
def test_other_scope_or_version_cannot_inherit_time_contract(key, value):
    data = evidence()
    data["records"][0][key] = value
    assert prepare_temporal_evidence(data, [], question()) == (data, [])


def test_only_requested_time_components_are_added():
    data, _ = prepare_temporal_evidence(evidence(), [], "14:20起报输入截止到几点？")
    assert data["temporal_requirements"] == ["c0.time.issue_time", "c0.time.input_cutoff"]
    assert len(data["facts"]) == 2
    origin = data["facts"][0]
    assert origin["stage"] == "question_scenario"
    assert origin["pointer"] == "request:/question/issue_time" and origin["value"] == "14:20"
    assert origin["source_sha256"] == sha256("14:20起报输入截止到几点？".encode()).hexdigest()


def test_conflicting_derived_fact_cannot_be_silently_overwritten():
    data, docs = prepare_temporal_evidence(evidence(), [], question())
    next(f for f in data["facts"] if f["id"] == "c0.time.input_cutoff")["value"] = "10:00"
    with pytest.raises(AssistantError, match="context_time_conflict"):
        prepare_temporal_evidence(data, docs, question())


def test_facts_only_in_attachment_do_not_cover_time_question():
    data, docs = prepare_temporal_evidence(evidence(), [], question())
    draft = DraftAnswer(status="answered", answer="这是模拟。", fact_ids=data["temporal_requirements"])
    with pytest.raises(AssistantError, match="answer_time_fact_missing"):
        validate_temporal_coverage(draft, data)
    draft.answer = "；".join("{{" + identity + "}}" for identity in data["temporal_requirements"])
    validate_temporal_coverage(draft, data)
    from power_forecast_service.assistant.validation import validate_answer
    rendered = validate_answer(draft, data, docs, question())
    assert "09:40" in rendered["answer"] and "非真实" in rendered["answer"]
