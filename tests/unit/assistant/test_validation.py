"""回答保护只证明绑定/引用，不以这些单测冒充真实模型答题质量。"""

import pytest

from power_forecast_service.assistant.contracts import AssistantError, DraftAnswer
from power_forecast_service.assistant.validation import validate_answer
from power_forecast_service.assistant.retrieval import corpus, keyword_rank, expand_sections


@pytest.fixture
def evidence():
    return {"facts":[{"id":"c0.ridge.mae", "label":"ridge MAE", "value":12.5, "unit":"kW"}],
            "records":[{"models":{"ridge":{}}}]}


def test_numeric_identity_and_plain_output(evidence):
    answer = DraftAnswer(status="answered", answer="该指标为{{c0.ridge.mae}}。", fact_ids=["c0.ridge.mae"])
    result = validate_answer(answer, evidence, [])
    assert result["answer"] == "该指标为ridge MAE：12.5 kW。"


@pytest.mark.parametrize("text,code", [
    ("MAE=-{{c0.ridge.mae}}.", "answer_fact_sign_conflict"),
    ("MAE=−{{c0.ridge.mae}}。", "answer_fact_sign_conflict"),
    ("MAE=﹣ {{c0.ridge.mae}}。", "answer_fact_sign_conflict"),
    ("MAE=－{{c0.ridge.mae}}。", "answer_fact_sign_conflict"),
    ("MAE=+{{c0.ridge.mae}}。", "answer_fact_sign_conflict"),
    ("MAE=-**{{c0.ridge.mae}}**。", "answer_fact_sign_conflict"),
    ("MAE=±{{c0.ridge.mae}}。", "answer_fact_sign_conflict"),
    ("MAE=-({{c0.ridge.mae}})。", "answer_fact_sign_conflict"),
    ("MAE=-(**{{c0.ridge.mae}}**)。", "answer_fact_sign_conflict"),
    ("MAE=−（{{c0.ridge.mae}}）。", "answer_fact_sign_conflict"),
    ("MAE=-\n{{c0.ridge.mae}}。", "answer_fact_sign_conflict"),
    ("MAE=-[{{c0.ridge.mae}}]。", "answer_fact_sign_conflict"),
    ("{{c0.ridge.mae}} GW。", "answer_fact_unit_conflict"),
    ("{{c0.ridge.mae}} kg。", "answer_fact_unit_conflict"),
    ("{{c0.ridge.mae}}joules。", "answer_fact_unit_conflict"),
    ("{{c0.ridge.mae}} kWxyz。", "answer_fact_unit_conflict"),
    ("{{c0.ridge.mae}} / s。", "answer_fact_unit_conflict"),
    ("{{c0.ridge.mae}}²。", "answer_fact_unit_conflict"),
    ("**{{c0.ridge.mae}}** GW。", "answer_fact_unit_conflict"),
    ("**{{c0.ridge.mae}}** * MW。", "answer_fact_unit_conflict"),
    ("{{c0.ridge.mae}} kW GW。", "answer_fact_unit_conflict"),
    ("{{c0.ridge.mae}} (GW)。", "answer_fact_unit_conflict"),
    ("{{c0.ridge.mae}} kW (GW)。", "answer_fact_unit_conflict"),
    ("（{{c0.ridge.mae}}） GW。", "answer_fact_unit_conflict"),
    ("{{c0.ridge.mae}}\nGW。", "answer_fact_unit_conflict"),
])
def test_placeholder_modifiers_cannot_change_complete_quantity(evidence, text, code):
    draft = DraftAnswer(status="answered", answer=text, fact_ids=["c0.ridge.mae"])
    with pytest.raises(AssistantError, match=code):
        validate_answer(draft, evidence, [])


@pytest.mark.parametrize("text", [
    "- {{c0.ridge.mae}}。", "+ {{c0.ridge.mae}}。", "* {{c0.ridge.mae}}。",
    "当前结果：\n  - {{c0.ridge.mae}}。", "**{{c0.ridge.mae}}**，与基线可比。",
    "{{c0.ridge.mae}}，单位来自原记录。", "{{c0.ridge.mae}}，and the baseline remains selected.",
    "（{{c0.ridge.mae}}），单位来自原记录。", "{{c0.ridge.mae}}（与基线使用相同单位）。",
])
def test_placeholder_complete_quantity_keeps_normal_punctuation(evidence, text):
    draft = DraftAnswer(status="answered", answer=text, fact_ids=["c0.ridge.mae"])
    assert validate_answer(draft, evidence, [])["status"] == "answered"


def test_negative_fact_renders_its_own_sign_without_external_sign(evidence):
    evidence["facts"][0]["value"] = -12.5
    draft = DraftAnswer(status="answered", answer="偏差为{{c0.ridge.mae}}。", fact_ids=["c0.ridge.mae"])
    assert validate_answer(draft, evidence, [])["answer"] == "偏差为ridge MAE：-12.5 kW。"


@pytest.mark.parametrize("marker", ["-", "+"])
def test_plain_text_list_marker_cannot_look_like_fact_sign(evidence, marker):
    draft = DraftAnswer(status="answered", answer=marker + " {{c0.ridge.mae}}。", fact_ids=["c0.ridge.mae"])
    assert validate_answer(draft, evidence, [])["answer"] == "• ridge MAE：12.5 kW。"


@pytest.mark.parametrize("suffix", ["%", " MW", "分钟", " kW/h", " kW/分钟", " kW / h"])
def test_placeholder_cannot_be_relabelled_with_wrong_unit(evidence, suffix):
    draft = DraftAnswer(status="answered", answer="指标为{{c0.ridge.mae}}" + suffix, fact_ids=["c0.ridge.mae"])
    with pytest.raises(AssistantError, match="answer_fact_unit_conflict"):
        validate_answer(draft, evidence, [])


def test_placeholder_same_unit_suffix_is_redundant_but_valid(evidence):
    draft = DraftAnswer(status="answered", answer="指标为{{c0.ridge.mae}} kW。", fact_ids=["c0.ridge.mae"])
    assert validate_answer(draft, evidence, [])["answer"] == "指标为ridge MAE：12.5 kW。"


@pytest.mark.parametrize("literal", ["-12.5 kW", "﹣12.5 kW", "－12.5 kW", "12.5%", "12.5 MW", "12.5 kW/h", "12.5 kW/分钟", "12.5 kW / h", "12.5"])
def test_same_magnitude_requires_fact_sign_and_unit(evidence, literal):
    draft = DraftAnswer(status="answered", answer="{{c0.ridge.mae}}，另写为" + literal,
                        fact_ids=["c0.ridge.mae"])
    with pytest.raises(AssistantError, match="answer_number_unbound"):
        validate_answer(draft, evidence, [])


@pytest.mark.parametrize("value,literal", [(12.5, "12.50 kW"), (-12.5, "-12.5 kW")])
def test_bound_metric_literal_keeps_its_signed_unit(evidence, value, literal):
    evidence["facts"][0]["value"] = value
    draft = DraftAnswer(status="answered", answer="{{c0.ridge.mae}}，即" + literal,
                        fact_ids=["c0.ridge.mae"])
    assert validate_answer(draft, evidence, [])["status"] == "answered"


def test_method_numbers_metadata_names_and_question_clock_remain_valid(evidence):
    evidence["records"][0].update(quarter="2014-Q1", horizon_minutes=60, models={"ridge_0_1": {}})
    docs = [{"id": "d1", "text": "方法采用3%门槛和-0.5修正；使用20分钟观测。", "title": "方法", "source_sha256": "a"*64}]
    draft = DraftAnswer(status="answered", answer="2014 Q1用ridge_0_1、L1，60分钟时距；09:40起报。方法3%门槛、-0.5修正、20分钟观测。",
                        citations=["d1"], quotes={"d1": docs[0]["text"]})
    assert validate_answer(draft, evidence, docs, "09:40起报？")["status"] == "answered"


@pytest.mark.parametrize("literal", ["-3%", "3 kW"])
def test_document_number_cannot_lose_sign_or_unit(evidence, literal):
    docs = [{"id": "d1", "text": "门槛为3%，原始方法。", "title": "方法", "source_sha256": "a"*64}]
    draft = DraftAnswer(status="answered", answer="门槛为" + literal, citations=["d1"], quotes={"d1": docs[0]["text"]})
    with pytest.raises(AssistantError, match="answer_number_unbound"):
        validate_answer(draft, evidence, docs)


def test_unused_fact_cannot_pass_as_answered_value(evidence):
    answer = DraftAnswer(status="insufficient_evidence", answer="不能确定。", fact_ids=["c0.ridge.mae"])
    result = validate_answer(answer,evidence,[])
    assert result["facts"] == []


def test_model_names_across_contexts_are_not_numbers(evidence):
    evidence["records"].append({"models":{"ridge_0_1":{}}})
    draft=DraftAnswer(status="answered",answer="ridge_0_1对应{{c0.ridge.mae}}。",fact_ids=["c0.ridge.mae"])
    assert validate_answer(draft,evidence,[])["status"] == "answered"


def test_normalized_quote_still_requires_same_source(evidence):
    docs=[{"id":"d1","text":"这是**原始\n证据**，不能更改。","title":"source","source_sha256":"a"*64}]
    draft=DraftAnswer(status="answered",answer="原文不能更改。",citations=["d1"],quotes={"d1":"这是原始证据"})
    result=validate_answer(draft,evidence,docs)
    assert result["citations"][0]["quote"] == docs[0]["text"]


@pytest.mark.parametrize("draft,code", [
    ({"answer":"MAE为13.5", "fact_ids":["c0.ridge.mae"]}, "answer_number_unbound"),
    ({"answer":"MAE为{{c0.other.mae}}", "fact_ids":["c0.other.mae"]}, "answer_fact_invalid"),
    ({"answer":"MAE为13.5", "fact_ids":[]}, "answer_number_unbound"),
    ({"answer":"有依据", "citations":["missing"]}, "answer_citation_invalid"),
    ({"answer":"有依据", "citations":["d1"], "quotes":{"d1":"篡改原文"}}, "answer_quote_invalid"),
    ({"answer":"有依据", "citations":["d1"]}, "answer_quote_missing"),
])
def test_reject_unbound_or_fabricated(draft, code, evidence):
    with pytest.raises(AssistantError, match=code):
        validate_answer(DraftAnswer(status="answered", **draft), evidence,
                        [{"id":"d1", "text":"只能读取既有运行。"}])


def test_parent_context_keeps_table_time_contract():
    chunks = [c for c in corpus()["chunks"] if "engie_final" in c["scopes"]]
    ranked = keyword_rank("数据合同 时间 输入 目标", chunks)
    expanded = expand_sections(ranked, chunks)
    assert any("t-20分钟" in c["text"] for c in expanded)
    assert len({c["id"] for c in expanded}) == len(expanded)


def test_cancelled_await_does_not_allow_background_queue():
    import asyncio
    from threading import Event
    from power_forecast_service.assistant.retrieval import ReadExecutor

    async def check():
        executor = ReadExecutor()
        started, finish = Event(), Event()
        def slow_read():
            started.set()
            finish.wait(2)
        try:
            future = asyncio.get_running_loop().run_in_executor(executor, slow_read)
            await asyncio.to_thread(started.wait, 1)
            assert started.is_set()
            future.cancel()
            with pytest.raises(AssistantError, match="assistant_busy"):
                executor.submit(lambda: None)
        finally:
            finish.set()
            executor.shutdown(wait=True)
    asyncio.run(check())
