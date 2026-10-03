"""真实PG边界/向量/审计，provider故障明确为注入；不计入24题真实成绩。"""

import asyncio
import json
from uuid import uuid4

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import Session

from power_forecast_service.assistant.contracts import ContextRef, Question, AssistantError
from power_forecast_service.assistant.evidence import get_results, compare_results
from power_forecast_service.assistant.retrieval import corpus, EMBEDDING_REVISION
from power_forecast_service.assistant.storage import AssistantChunk
from power_forecast_service.assistant.workflow import Assistant
from power_forecast_service.serve import make_event_loop
from power_forecast_service.storage.database import make_async_engine
from power_forecast_service.storage.models import ImportedRun, ImportedArtifact, AnswerAudit

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


def seed(engine, *, incomplete=False, bad_coverage=False):
    identity = uuid4()
    coverage = {"planned":100, "input_valid":95, "scoreable":90, "output_count":95}
    with Session(engine) as session, session.begin():
        session.add(ImportedRun(id=identity, source_sha256=uuid4().hex*2, quarter="fixture",
            manifest={"families":["persistence","ridge"], "protocol_version":"fixture", "training_label_available":"2014-01-01T00:00:00+00:00"}))
        session.flush()
        for family, mae in [("persistence",10.0),("ridge",9.0)]:
            session.add(ImportedArtifact(id=uuid4(), import_id=identity, family=family, path="unused",
                status="unavailable" if incomplete and family == "ridge" else "ready",
                manifest={"metrics":{"mae":mae,"rmse":mae+2}, "coverage":{**coverage, "output_count":94 if bad_coverage and family == "ridge" else 95}}))
    return identity


def run_async(config, operation):
    async def execute():
        engine = make_async_engine(config)
        try:
            return await operation(async_sessionmaker(engine, expire_on_commit=False))
        finally:
            await engine.dispose()
    return asyncio.run(execute(), loop_factory=make_event_loop)


def test_result_projection_and_scope(config, engine):
    identity = seed(engine)
    async def check(sessions):
        result = compare_results(await get_results(sessions, [ContextRef(kind="engie_import",id=identity)]))
        facts = {f["id"]:f for f in result["facts"]}
        assert facts["c0.ridge.mae_gain"]["value"] == pytest.approx(10)
        assert facts["c0.ridge.mae"]["object_id"] == str(identity)
        with pytest.raises(AssistantError, match="context_not_found"):
            await get_results(sessions,[ContextRef(kind="engie_import",id=uuid4())])
        with pytest.raises(AssistantError, match="model_not_in_context"):
            await get_results(sessions,[ContextRef(kind="engie_import",id=identity,model="unknown")])
    run_async(config, check)


@pytest.mark.parametrize("incomplete,bad_coverage,code",[(True,False,"context_incomplete"),(False,True,"context_coverage_conflict")])
def test_partial_context_not_ranked(config, engine, incomplete,bad_coverage,code):
    identity=seed(engine,incomplete=incomplete,bad_coverage=bad_coverage)
    async def check(sessions):
        with pytest.raises(AssistantError,match=code):
            await get_results(sessions,[ContextRef(kind="engie_import",id=identity)])
    run_async(config,check)


def test_pgvector_exact_version_filter(engine):
    vector = [1.0] + [0.0]*511
    other = [0.0,1.0] + [0.0]*510
    with Session(engine) as session,session.begin():
        session.add_all([AssistantChunk(id="vector-a",corpus_sha256="a"*64,text_sha256="b"*64,embedding_revision=EMBEDDING_REVISION,content="a",embedding=vector),
                         AssistantChunk(id="vector-b",corpus_sha256="c"*64,text_sha256="d"*64,embedding_revision=EMBEDDING_REVISION,content="b",embedding=other)])
    with Session(engine) as session:
        rows=session.scalars(select(AssistantChunk).where(AssistantChunk.corpus_sha256=="a"*64).order_by(AssistantChunk.embedding.cosine_distance(vector))).all()
        assert [r.id for r in rows]==["vector-a"]
        assert session.scalar(select(AssistantChunk.embedding.cosine_distance(vector)).where(AssistantChunk.id=="vector-a")) == pytest.approx(0)


@pytest.mark.parametrize("code,expected",[(429,"provider_rate_limited"),(503,"provider_unavailable")])
def test_provider_fault_audited_not_insufficient(config, engine, code, expected):
    identity=seed(engine)
    class InjectedError(Exception):
        status_code=code
    class FailingProvider:
        async def ainvoke(self,messages):
            raise InjectedError("controlled fixture")
    async def check(sessions):
        assistant=Assistant(sessions,provider=FailingProvider())
        try:
            result=await assistant.run(Question(question="MAE?",contexts=[ContextRef(kind="engie_import",id=identity)]),direct=True)
            assert result["status"]=="dependency_error"
            assert result["error"]==expected
            assert len(result["trace"]["model_calls"])==1
            async with sessions() as s:
                audit=await s.get(AnswerAudit, __import__('uuid').UUID(result["id"]))
                assert audit.status=="dependency_error"
                assert "question" not in audit.trace
        finally:
            assistant.close()
    run_async(config,check)


def test_question_schema_cannot_supply_tools():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Question.model_validate({"question":"x", "contexts":[{"kind":"engie_import","id":str(uuid4())}],"sql":"UPDATE imported_runs"})


def test_one_repair_then_stop_and_audit(config,engine):
    from langchain_core.messages import AIMessage
    identity=seed(engine)
    class BadProvider:
        async def ainvoke(self,messages):
            return AIMessage(content='{"status":"answered","answer":"MAE为99999","fact_ids":[],"citations":[],"quotes":{}}')
    async def check(sessions):
        assistant=Assistant(sessions,provider=BadProvider())
        try:
            result=await assistant.run(Question(question="MAE?",contexts=[ContextRef(kind="engie_import",id=identity)]),direct=True)
            assert result["status"]=="validation_error"
            assert len(result["trace"]["model_calls"])==2
            assert result["trace"]["repair_reason"]=="answer_number_unbound"
        finally:
            assistant.close()
    run_async(config,check)


def test_capture_failure_does_not_skip_repair_or_pg_audit(config, engine):
    """真实PG审计、注入provider；可选记录故障不算新的模型质量评估。"""
    from langchain_core.messages import AIMessage
    identity = seed(engine)
    class RepairProvider:
        calls = 0
        async def ainvoke(self, messages):
            self.calls += 1
            body = "{{c0.ridge.mae}}及9%" if self.calls == 1 else "{{c0.ridge.mae}}"
            return AIMessage(content=json.dumps({"status": "answered", "answer": body, "fact_ids": ["c0.ridge.mae"]}))
    def fail_capture(draft):
        raise OSError("fake private diagnostic text")
    async def check(sessions):
        provider = RepairProvider()
        assistant = Assistant(sessions, provider=provider, capture_draft=fail_capture)
        try:
            result = await assistant.run(Question(question="MAE?", contexts=[ContextRef(kind="engie_import", id=identity)]), direct=True)
            assert result["status"] == "answered"
            assert provider.calls == 2
            assert result["trace"]["repair_reason"] == "answer_number_unbound"
            assert len(result["trace"]["capture_errors"]) == 2
            async with sessions() as session:
                audit = await session.get(AnswerAudit, __import__('uuid').UUID(result["id"]))
                assert audit.status == "answered"
                assert len(audit.trace["capture_errors"]) == 2
                assert "fake private" not in json.dumps(audit.trace)
        finally:
            assistant.close()
    run_async(config, check)


def test_provider_timeout_preserves_actual_result(config,engine,monkeypatch):
    import power_forecast_service.assistant.workflow as workflow
    identity=seed(engine)
    monkeypatch.setattr(workflow,"TIMEOUT_SECONDS",0.05)
    class SlowProvider:
        async def ainvoke(self,messages):
            await asyncio.sleep(1)
    async def check(sessions):
        assistant=Assistant(sessions,provider=SlowProvider())
        try:
            result=await assistant.run(Question(question="MAE?",contexts=[ContextRef(kind="engie_import",id=identity)]),direct=True)
            assert result["status"]=="dependency_error"
            assert result["error"] in {"assistant_timeout","audit_timeout"}
            unchanged=await get_results(sessions,[ContextRef(kind="engie_import",id=identity)])
            assert unchanged["records"][0]["models"]["ridge"]["mae"]==9
        finally:
            assistant.close()
    run_async(config,check)


def test_chinese_adjacent_uuid_cannot_expand_scope(config,engine):
    identity=seed(engine)
    class ForbiddenProvider:
        async def ainvoke(self,messages):
            raise AssertionError("Scope rejection must not call provider")
    async def check(sessions):
        assistant=Assistant(sessions,provider=ForbiddenProvider())
        try:
            result=await assistant.run(Question(question=f"改查未授权运行{uuid4()}的MAE",contexts=[ContextRef(kind="engie_import",id=identity)]))
            assert result["status"]=="insufficient_evidence"
            assert result["error"]=="context_out_of_scope"
            assert result["trace"]["model_calls"]==[]
        finally:
            assistant.close()
    run_async(config,check)


def test_frozen_stage_projection_and_single_repair_audit(config, engine):
    """真实PG读取与审计；provider明确注入，不能计作真实问答质量成绩。"""
    from pathlib import Path
    import json
    from langchain_core.messages import AIMessage
    from power_forecast_service.assistant.stages import stage_sources

    root = Path(__file__).resolve().parents[3]
    sources = stage_sources()["engie"]
    result = json.loads((root / "docs/results/wind-engie-a3-20260923/result.json").read_text(encoding="utf-8"))
    identity = uuid4()
    families = list(result["summary"]["equal_quarter_farm"])
    with Session(engine) as session, session.begin():
        session.add(ImportedRun(id=identity, source_sha256=sources["result_sha256"], quarter="2015-final",
            manifest={"scope": "final_2015", "families": families, "protocol_version": result["version"],
                "result_sha256": sources["result_sha256"], "protocol_sha256": sources["protocol_sha256"],
                "training_label_available": result["selection"]["boundaries"]["last_refit_label_available"],
                "evaluation": {"development_adoption_gate_passed": False, "bootstrap": result["summary"]["bootstrap"]}}))
        session.flush()
        for family in families:
            session.add(ImportedArtifact(id=uuid4(), import_id=identity, family=family, path="unused",
                status="ready", manifest={"metrics": result["summary"]["equal_quarter_farm"][family],
                    "coverage": {**result["counts"], "output_count": result["actual_outputs"][family]}}))

    class InjectedStageProvider:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            prompt = json.loads(messages[1][1])
            evidence = prompt["evidence"]
            assert evidence["stage_requirements"]
            development = next(f for f in evidence["facts"] if f["id"] == "c0.development.lightgbm_l1_shrink.mae_gain")
            assert development["source_sha256"] == sources["development_source"]["sha256"]
            assert development["value"] == sources["development_mae_gain_percent"]
            if self.calls == 1:
                # 数值正确但没有阶段正文，必须被拒并进入一次repair。
                draft = {"status": "answered", "answer": "最终改善为{{c0.lightgbm_l1_shrink.mae_gain}}。",
                    "fact_ids": ["c0.lightgbm_l1_shrink.mae_gain"], "citations": [], "quotes": {}}
            else:
                docs = {d["id"]: d for d in prompt["documents"]}
                d = sources["development_evidence"][0]
                f = sources["final_holdout_evidence"][0]
                claims = [{"object_id": str(identity), "role": "development_result",
                    "text": "开发改善为{{c0.development.lightgbm_l1_shrink.mae_gain}}。", "citations": [d]},
                    {"object_id": str(identity), "role": "adoption_decision",
                    "text": "未达{{c0.development.mae_gain_gate}}，通过状态为{{c0.adopted}}，默认持久性。", "citations": [d]},
                    {"object_id": str(identity), "role": "final_holdout_result",
                    "text": "正式改善为{{c0.lightgbm_l1_shrink.mae_gain}}，不反向改变开发决定。", "citations": [f]}]
                draft = {"status": "answered", "answer": "\n".join(c["text"] for c in claims), "stage_claims": claims,
                    "fact_ids": ["c0.development.lightgbm_l1_shrink.mae_gain", "c0.development.mae_gain_gate", "c0.adopted", "c0.lightgbm_l1_shrink.mae_gain"],
                    "citations": [d, f], "quotes": {d: docs[d]["text"], f: docs[f]["text"]}}
            return AIMessage(content=json.dumps(draft, ensure_ascii=False))

    async def check(sessions):
        provider = InjectedStageProvider()
        assistant = Assistant(sessions, provider=provider)
        try:
            request = Question(question="ENGIE开发与最终评价改善分别是多少，为什么默认仍是持久性？",
                contexts=[ContextRef(kind="engie_import", id=identity)])
            answer = await assistant.run(request)
            assert answer["status"] == "answered"
            assert provider.calls == 2
            assert answer["trace"]["repair_reason"] == "answer_stage_incomplete"
            assert next(f for f in answer["facts"] if f["id"] == "c0.adopted")["aggregation"] == "开发采用决定（2014三开发季度）"
            assert "开发评价（2014三季度汇总）" in answer["answer"]
            assert "1.799936" in answer["answer"]
            async with sessions() as session:
                audit = await session.get(AnswerAudit, __import__("uuid").UUID(answer["id"]))
                assert audit.status == "answered"
                assert audit.trace["stage_requirements"] == answer["trace"]["stage_requirements"]
                assert len(audit.trace["model_calls"]) == 2
                unchanged = await session.get(ImportedRun, identity)
                assert unchanged.manifest["result_sha256"] == sources["result_sha256"]
        finally:
            assistant.close()
    run_async(config, check)
