"""定向阶段保护及冻结来源复核；不是开放问答质量或新预测效果评估。"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import asyncio

import pytest

from power_forecast_service.assistant.contracts import AssistantError, DraftAnswer, Fact, StageClaim
from power_forecast_service.assistant.evidence import compare_results
from power_forecast_service.assistant.retrieval import corpus
from power_forecast_service.assistant.stages import bind_stages, prepare_stage_requirements, stage_sources, validate_stage_claims
from power_forecast_service.assistant.validation import validate_answer


@pytest.mark.parametrize("kind", ["q1", "engie"])
@pytest.mark.parametrize("repair", [False, True])
def test_valid_stage_draft_runs_through_provider_and_audit(monkeypatch, kind, repair):
    """真实来源绑定走完整workflow；provider和审计为注入，不称真实问答/PG。"""
    from langchain_core.messages import AIMessage
    from power_forecast_service.assistant import workflow
    from power_forecast_service.assistant.contracts import ContextRef, Question

    evidence, _ = setup_question(kind)
    draft = valid_draft(kind, evidence)
    incomplete = draft.model_copy(deep=True)
    incomplete.answer = "所选对象有开发依据。"
    incomplete.stage_claims = []

    class Provider:
        calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            value = incomplete if repair and self.calls == 1 else draft
            return AIMessage(content=value.model_dump_json())

    class Audit:
        rows = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def begin(self):
            return self

        def add(self, row):
            self.rows.append(row)

    async def get_results(*args):
        return evidence

    monkeypatch.setattr(workflow, "get_results", get_results)
    provider, audit = Provider(), Audit()
    assistant = workflow.Assistant(lambda: audit, provider=provider)
    question = Question(question=Q1_QUESTION if kind == "q1" else ENGIE_QUESTION,
        contexts=[ContextRef(kind=evidence["records"][0]["kind"], id=evidence["records"][0]["id"])])
    try:
        result = asyncio.run(assistant.run(question, direct=True))
    finally:
        assistant.close()
    assert result["status"] == "answered"
    assert provider.calls == (2 if repair else 1) and len(audit.rows) == 1
    if repair:
        assert result["trace"]["validation_errors"][0]["error"] == "answer_stage_incomplete"
        assert result["trace"]["repair_reason"] == "answer_stage_incomplete"
    else:
        assert not result["trace"].get("validation_errors")
    assert "事前开发选型" in result["answer"] if kind == "q1" else "开发采用决定" in result["answer"]

ROOT = Path(__file__).resolve().parents[3]
Q1_QUESTION = "Q1 推荐模型为什么没有采用 Transformer？"
ENGIE_QUESTION = "ENGIE 开发与最终评价改善分别是多少，为什么默认仍是持久性？"


def projected(kind):
    sources = stage_sources()
    if kind == "q1":
        data = {"id": sources["q1"]["object_id"], "kind": "q1_run", "scope": "q1",
            "final_protocol_id": sources["q1"]["final_protocol_id"], "unit": "source_reported_unit",
            "models": {"persistence": {"mae": 197.534}, "ridge_0_1": {"mae": 183.187}, "transformer_delta": {"mae": 206.826}}}
    else:
        data = {"id": "b9ae2524-9abb-5ba1-98c7-e80c066e15d9", "kind": "engie_import", "scope": "engie_final",
            "result_sha256": sources["engie"]["result_sha256"], "protocol_sha256": sources["engie"]["protocol_sha256"],
            "unit": "kW", "models": {"persistence": {"mae": 449.951515}, "lightgbm_l1_shrink": {"mae": 436.520208}},
            "evaluation": {"development_adoption_gate_passed": False}}
    facts = [Fact(id=f"c0.{m}.mae", label=f"{m} MAE", value=v["mae"], unit=data["unit"],
        object_id=data["id"], aggregation="正式测试fixture", source_sha256="a"*64, pointer="/metrics").model_dump()
        for m, v in data["models"].items()]
    if kind == "engie":
        facts.append(Fact(id="c0.adopted", label="开发采用门通过", value=False, object_id=data["id"],
            aggregation="开发采用门", source_sha256="a"*64, pointer="/evaluation/development_adoption_gate_passed").model_dump())
    return compare_results(bind_stages({"facts": facts, "records": [data], "scopes": [data["scope"]]}))


def setup_question(kind, question=None):
    return prepare_stage_requirements(projected(kind), [], question or (Q1_QUESTION if kind == "q1" else ENGIE_QUESTION))


def valid_draft(kind, evidence):
    identity = evidence["records"][0]["id"]
    if kind == "q1":
        claims = [{"object_id": identity, "role": "selection_basis",
            "text": "三个开发窗中Ridge均优于持久性，平均MAE为{{c0.development.ridge_0_1.mae}}，持久性为{{c0.development.persistence.mae}}，增量Transformer为{{c0.development.transformer_delta.mae}}；因此事前推荐Ridge，不采用Transformer。正式留出只是事后核验。",
            "citations": stage_sources()["q1"]["pre_selection_evidence"]}]
    else:
        cite = stage_sources()["engie"]["development_evidence"]
        claims = [{"object_id": identity, "role": "development_result",
            "text": "共享收缩的开发MAE改善为{{c0.development.lightgbm_l1_shrink.mae_gain}}。", "citations": cite},
            {"object_id": identity, "role": "adoption_decision",
            "text": "未达到{{c0.development.mae_gain_gate}}，开发采用门通过状态为{{c0.adopted}}，所以默认保持持久性。", "citations": cite},
            {"object_id": identity, "role": "final_holdout_result",
            "text": "正式MAE改善为{{c0.lightgbm_l1_shrink.mae_gain}}，不能反向追认开发采用。", "citations": stage_sources()["engie"]["final_holdout_evidence"]}]
    docs = {d["id"]: d for d in corpus()["chunks"]}
    citations = list(dict.fromkeys(c for claim in claims for c in claim["citations"]))
    import re
    answer = "\n".join(c["text"] for c in claims)
    return DraftAnswer(status="answered", answer=answer,
        fact_ids=list(dict.fromkeys(re.findall(r"\{\{([^{}]+)\}\}", answer))), citations=citations,
        quotes={c: docs[c]["text"] for c in citations}, stage_claims=claims)


def test_projection_matches_frozen_machine_sources():
    sources = stage_sources()
    def load(source):
        raw = (ROOT / source["path"]).read_bytes()
        assert sha256(raw).hexdigest() == source["sha256"]
        return json.loads(raw)
    selection = load(sources["q1"]["selection_source"])
    assert not selection["test_scored"]
    assert sources["q1"]["mean_mae"] == {k: selection["mean_mae"][k] for k in sources["q1"]["mean_mae"]}
    final_selection = json.loads((ROOT / "src/power_forecast_service/forecasting/final_selection.json").read_text(encoding="utf-8"))
    assert final_selection["development_selection_sha256"] == sources["q1"]["selection_source"]["sha256"]
    development = load(sources["engie"]["development_source"])
    adoption = development["summary"]["adoption"]["lightgbm_l1_shrink"]
    assert sources["engie"]["development_mae_gain_percent"] == adoption["equal_quarter_mae_gain_percent"]
    assert sources["engie"]["development_adopted"] is adoption["passed"]
    gate = load(sources["engie"]["gate_source"])
    assert sources["engie"]["minimum_mae_gain_percent"] == gate["protocol"]["adoption_gates"]["minimum_equal_quarter_mae_gain_percent"]
    final_protocol = (ROOT / "docs/results/wind-engie-a3-20260923/protocol.json").read_bytes()
    assert sha256(final_protocol).hexdigest() == sources["engie"]["protocol_sha256"]
    assert json.loads(final_protocol)["protocol"]["development_result"]["sha256"] == sources["engie"]["development_source"]["sha256"]


@pytest.mark.parametrize("kind", ["q1", "engie"])
def test_valid_stage_answer_has_visible_labels_and_sources(kind):
    evidence, docs = setup_question(kind)
    result = validate_answer(valid_draft(kind, evidence), evidence, docs)
    assert result["status"] == "answered"
    assert "事前开发选型" in result["answer"] if kind == "q1" else "开发评价（2014三季度汇总）" in result["answer"]
    if kind == "engie":
        fact = next(f for f in result["facts"] if f["id"] == "c0.development.lightgbm_l1_shrink.mae_gain")
        assert fact["value"] == pytest.approx(1.7999357758374024)
        assert fact["source_sha256"] == stage_sources()["engie"]["development_source"]["sha256"]
        assert "2014三个开发季度" in fact["aggregation"]
        assert fact["stage"] == "development_summary"


def test_original_w1_answers_now_require_body_stage_binding():
    rows = json.loads((ROOT / "tests/fixtures/assistant/w1-negative-answers.json").read_text(encoding="utf-8"))["answers"]
    assert len(rows) == 2
    for kind, row in zip(("q1", "engie"), rows):
        evidence, _ = setup_question(kind)
        old = row
        draft = DraftAnswer(status="answered", answer=old["answer"])
        with pytest.raises(AssistantError, match="answer_stage_incomplete"):
            validate_stage_claims(draft, evidence)


@pytest.mark.parametrize("kind", ["q1", "engie"])
def test_final_metric_cannot_be_relabelled_as_development(kind):
    evidence, _ = setup_question(kind)
    draft = valid_draft(kind, evidence)
    draft.stage_claims[0].text += "因为正式改善{{c0." + ("ridge_0_1" if kind == "q1" else "lightgbm_l1_shrink") + ".mae_gain}}。"
    draft.answer = "\n".join(c.text for c in draft.stage_claims)
    with pytest.raises(AssistantError, match="answer_stage_fact_conflict"):
        validate_stage_claims(draft, evidence)


def test_development_value_only_in_fact_list_is_not_coverage():
    evidence, _ = setup_question("engie")
    draft = valid_draft("engie", evidence)
    draft.stage_claims[0].text = "开发有改善。"
    draft.answer = "\n".join(c.text for c in draft.stage_claims)
    with pytest.raises(AssistantError, match="answer_stage_metric_missing"):
        validate_stage_claims(draft, evidence)


@pytest.mark.parametrize("defect,code", [("extra_body", "answer_stage_body_mismatch"), ("wrong_object", "answer_stage_identity_invalid"),
    ("wrong_citation", "answer_stage_citation_conflict"), ("missing_gate", "answer_stage_incomplete"), ("refusal", "answer_stage_incomplete")])
def test_stage_metadata_cannot_hide_bad_body(defect, code):
    evidence, _ = setup_question("engie")
    draft = valid_draft("engie", evidence)
    if defect == "extra_body":
        draft.answer += "最终结果支持事前采用。"
    elif defect == "wrong_object":
        from uuid import uuid4
        draft.stage_claims[0].object_id = uuid4()
    elif defect == "wrong_citation":
        draft.stage_claims[0].citations = stage_sources()["engie"]["final_holdout_evidence"]
    elif defect == "missing_gate":
        draft.stage_claims.pop(1)
        draft.answer = "\n".join(c.text for c in draft.stage_claims)
    else:
        draft.status = "insufficient_evidence"
    with pytest.raises(AssistantError, match=code):
        validate_stage_claims(draft, evidence)


def test_q1_may_explain_selection_and_separate_final_performance():
    evidence, docs = setup_question("q1", "Q1当时为何选Ridge，后来正式表现如何？")
    draft = valid_draft("q1", evidence)
    source = stage_sources()["q1"]["final_holdout_evidence"][0]
    draft.stage_claims.append(StageClaim(
        object_id=evidence["records"][0]["id"], role="final_holdout_result", text="事后正式Ridge改善为{{c0.ridge_0_1.mae_gain}}。", citations=[source]))
    draft.answer = "\n".join(c.text for c in draft.stage_claims)
    draft.fact_ids.append("c0.ridge_0_1.mae_gain")
    draft.citations.append(source)
    draft.quotes[source] = next(d["text"] for d in docs if d["id"] == source)
    result = validate_answer(draft, evidence, docs)
    assert "事前开发选型" in result["answer"] and "正式留出核验" in result["answer"]


@pytest.mark.parametrize("scope", ["engie_development", "q1_unbound"])
def test_unbound_or_single_quarter_does_not_inherit_frozen_summary(scope):
    data = deepcopy(projected("engie" if scope.startswith("engie") else "q1"))
    data["records"][0] = {k: v for k, v in data["records"][0].items() if k not in {"stage_boundary", "pre_selection_evidence", "development_evidence", "final_holdout_evidence"}}
    data["records"][0]["scope"] = scope
    data["facts"] = []
    result = bind_stages(data)
    assert not result["facts"]
    assert "stage_boundary" not in result["records"][0]


def test_wrong_final_result_hash_cannot_inherit_a2():
    data = projected("engie")
    data["facts"] = []
    data["records"][0].pop("stage_boundary")
    data["records"][0]["result_sha256"] = "0" * 64
    assert not bind_stages(data)["facts"]


def test_only_final_question_does_not_force_development_answer():
    evidence, _ = setup_question("engie", "ENGIE正式留出MAE改善多少？")
    assert [r["role"] for r in evidence["stage_requirements"]] == ["final_holdout_result"]
    evidence, _ = setup_question("q1", "Q1持久性MAE是多少？")
    assert not evidence["stage_requirements"]


@pytest.mark.parametrize("kind,question", [
    ("q1", "Q1训练标签为什么必须早于评分？"),
    ("q1", "Q1为什么采用MSE训练损失？"),
    ("engie", "为什么采用L1训练损失？"),
    ("engie", "ENGIE正式持久性RMSE结果是多少？"),
    ("engie", "ENGIE最终RMSE改善多少？"),
    ("engie", "默认预测步长是多少？"),
])
def test_unrelated_questions_keep_original_path(kind, question):
    evidence, _ = setup_question(kind, question)
    assert not evidence["stage_requirements"]


def test_same_object_two_models_does_not_replace_first_context_rules():
    data = projected("q1")
    data["records"][0]["selected_model"] = "ridge_0_1"
    data["records"].append({**data["records"][0], "selected_model": "transformer_delta"})
    data["facts"] += [{**f, "id": f["id"].replace("c0.", "c1.")} for f in data["facts"]]
    evidence, docs = prepare_stage_requirements(data, [], Q1_QUESTION)
    assert len(evidence["stage_requirements"]) == 1
    assert validate_answer(valid_draft("q1", evidence), evidence, docs)["status"] == "answered"


def test_stage_fact_tokens_share_plain_answer_normalization():
    evidence, docs = setup_question("engie")
    draft = valid_draft("engie", evidence)
    for claim in draft.stage_claims:
        claim.text = claim.text.replace("{{c0.", "{{ fact_id: c0.").replace("}}", " }}")
    draft.answer = "\n".join(c.text for c in draft.stage_claims)
    assert validate_answer(draft, evidence, docs)["status"] == "answered"


def test_stage_answer_is_rendered_from_single_body_source():
    evidence, docs = setup_question("engie")
    draft = valid_draft("engie", evidence)
    draft.answer = ""
    result = validate_answer(draft, evidence, docs)
    assert "1.799936" in result["answer"] and "正式留出核验" in result["answer"]
    assert len(result["facts"]) == 4
    adopted = next(f for f in result["facts"] if f["id"] == "c0.adopted")
    assert adopted["aggregation"] == "开发采用决定（2014三开发季度）"


def test_stage_empty_answer_still_rejects_unbound_text_number():
    evidence, docs = setup_question("engie")
    draft = valid_draft("engie", evidence)
    draft.answer = ""
    draft.stage_claims[0].text += "改善为99999。"
    with pytest.raises(AssistantError, match="answer_number_unbound"):
        validate_answer(draft, evidence, docs)
