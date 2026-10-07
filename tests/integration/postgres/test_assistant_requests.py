"""真实 PostgreSQL 请求登记与 API 生命周期；模型/SSE 明确使用故障注入。"""

import asyncio
import json
from dataclasses import replace
from datetime import timedelta
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from power_forecast_service.api.app import create_app
from power_forecast_service.assistant.contracts import Question, ContextRef
from power_forecast_service.assistant.requests import DuplicateRequest, RequestJournal
from power_forecast_service.assistant.storage import AssistantRequest
from power_forecast_service.assistant.evidence import digest
from power_forecast_service.storage.models import ImportedRun, ImportedArtifact, AnswerAudit
from power_forecast_service.serve import make_event_loop

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


def seed(engine):
    identity = uuid4()
    with Session(engine) as session, session.begin():
        session.add(ImportedRun(id=identity, source_sha256=uuid4().hex * 2, quarter="fixture",
            manifest={"families": ["persistence", "ridge"], "protocol_version": "fixture",
                      "training_label_available": "2014-01-01T00:00:00+00:00"}))
        session.flush()
        for family, mae in (("persistence", 10.0), ("ridge", 9.0)):
            session.add(ImportedArtifact(id=uuid4(), import_id=identity, family=family,
                path="unused", status="ready", manifest={"metrics": {"mae": mae, "rmse": mae + 2},
                    "coverage": {"planned": 100, "input_valid": 95, "output_count": 95, "scoreable": 90}}))
    return identity


def body(identity):
    return {"question": "ridge 的 MAE 是多少？", "contexts": [
        {"kind": "engie_import", "id": str(identity), "model": "ridge"}]}


def response(*, complete=True):
    text = json.dumps({"status": "answered", "body": {"kind": "plain",
        "text": "ridge 的 MAE 是{{c0.ridge.mae}}。", "citations": []}})
    chunks = [{"model": "candidate", "choices": [{"index": 0,
        "delta": {"content": text}, "finish_reason": "stop"}]},
        {"model": "candidate", "choices": [], "usage": {"prompt_tokens": 100,
            "completion_tokens": 10, "total_tokens": 110}}]
    return httpx.Response(200, text="".join("data: " + json.dumps(item) + "\n\n" for item in chunks)
        + ("data: [DONE]\n\n" if complete else ""))


async def clients(config, handler, operation):
    settings = replace(config, assistant_backend="vllm", assistant_vllm_url="http://127.0.0.1:18112",
                       assistant_vllm_model="candidate", assistant_capacity=1)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        provider = app.state.assistant.provider
        await provider.client.aclose()
        provider.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
            await operation(app, client)


def test_success_duplicate_conflict_and_no_raw_retention(config, engine):
    identity, key, calls = seed(engine), str(uuid4()), []
    def handler(request):
        calls.append(request.headers["x-request-id"])
        return response()
    async def check(app, client):
        answer = await client.post("/assistant/answers", json=body(identity), headers={"x-request-id": key})
        assert answer.status_code == 200 and answer.json()["status"] == "answered"
        assert answer.json()["id"] == key
        assert calls == [key + "-1"]
        duplicate = await client.post("/assistant/answers", json=body(identity), headers={"x-request-id": key})
        assert duplicate.status_code == 409
        changed = {**body(identity), "question": "另一个问题"}
        assert (await client.post("/assistant/answers", json=changed, headers={"x-request-id": key})).json()["detail"]["code"] == "request_identity_conflict"
        status = (await client.get("/assistant/requests/" + key)).json()
        assert status["status"] == "completed" and status["answer_retained"] is False
        assert len(calls) == 1
        async with app.state.sessions() as session:
            row = await session.get(AssistantRequest, UUID(key))
            audit = await session.get(AnswerAudit, UUID(key))
            assert row.answer_sha256 == audit.answer_sha256
            assert body(identity)["question"] not in json.dumps(row.calls, ensure_ascii=False)
            assert "answer" not in audit.trace
    asyncio.run(clients(config, handler, check), loop_factory=make_event_loop)


def test_unknown_stream_not_repaired_or_resent(config, engine):
    identity, key, calls = seed(engine), str(uuid4()), []
    def handler(request):
        calls.append(request)
        return response(complete=False)
    async def check(app, client):
        result = (await client.post("/assistant/answers", json=body(identity), headers={"x-request-id": key})).json()
        assert result["status"] == "dependency_error" and result["error"] == "provider_result_unknown"
        assert len(result["trace"]["model_calls"]) == 1
        assert (await client.get("/assistant/requests/" + key)).json()["status"] == "unknown"
        assert (await client.post("/assistant/answers", json=body(identity), headers={"x-request-id": key})).status_code == 409
        assert len(calls) == 1
    asyncio.run(clients(config, handler, check), loop_factory=make_event_loop)


def test_disconnect_keeps_slot_until_actual_completion(config, engine):
    identity = seed(engine)
    async def check(app, client):
        entered, release = asyncio.Event(), asyncio.Event()
        async def handler(request):
            entered.set()
            await release.wait()
            return response()
        await app.state.assistant.provider.client.aclose()
        app.state.assistant.provider.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        first = asyncio.create_task(client.post("/assistant/answers", json=body(identity)))
        await asyncio.wait_for(entered.wait(), 5)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        busy = await client.post("/assistant/answers", json=body(identity))
        assert busy.status_code == 429 and len(app.state.assistant_tasks) == 1
        tasks = list(app.state.assistant_tasks)
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 5)
        assert (await client.post("/assistant/answers", json=body(identity))).json()["status"] == "answered"
    asyncio.run(clients(config, lambda request: response(), check), loop_factory=make_event_loop)


def test_expired_dispatch_remains_unknown_after_new_journal(config, engine):
    identity, key = seed(engine), uuid4()
    async def check(app, client):
        journal = app.state.assistant.journal
        question = Question.model_validate(body(identity))
        await journal.reserve(key, question, seconds=75)
        await journal.dispatch(str(key), 1)
        with Session(engine) as session, session.begin():
            row = session.get(AssistantRequest, key)
            row.deadline_at = session.scalar(select(func.now())) - timedelta(seconds=1)
        recovered = RequestJournal(app.state.sessions, journal.identity)
        assert (await recovered.lookup(key))["status"] == "unknown"
        with pytest.raises(DuplicateRequest):
            await recovered.reserve(key, question, seconds=75)
        assert not await recovered.finish(key, {"status": "answered"})
    asyncio.run(clients(config, lambda request: response(), check), loop_factory=make_event_loop)


def test_concurrent_reservation_has_one_owner(config, engine):
    identity, key = seed(engine), uuid4()
    async def check(app, client):
        journal, question = app.state.assistant.journal, Question.model_validate(body(identity))
        results = await asyncio.gather(*(journal.reserve(key, question, seconds=75) for _ in range(4)), return_exceptions=True)
        assert sum(item is None for item in results) == 1
        assert sum(isinstance(item, DuplicateRequest) for item in results) == 3
        assert (await journal.lookup(key))["calls"] == []
    asyncio.run(clients(config, lambda request: response(), check), loop_factory=make_event_loop)


@pytest.mark.parametrize("code,expected", [(429, "provider_rate_limited"), (400, "provider_rejected")])
def test_acknowledged_rejection_is_known_failure_not_unknown(config, engine, code, expected):
    identity, key, calls = seed(engine), str(uuid4()), []
    def handler(request):
        calls.append(request)
        return httpx.Response(code)
    async def check(app, client):
        result = (await client.post("/assistant/answers", json=body(identity), headers={"x-request-id": key})).json()
        assert result["status"] == "dependency_error" and result["error"] == expected
        status = (await client.get("/assistant/requests/" + key)).json()
        assert status["status"] == "completed" and status["outcome"] == "dependency_error"
        assert status["calls"][0]["status"] == "rejected" and len(calls) == 1
    asyncio.run(clients(config, handler, check), loop_factory=make_event_loop)


def test_finalization_checks_deadline_without_needing_status_poll(config, engine):
    identity, key = seed(engine), uuid4()
    async def check(app, client):
        journal = app.state.assistant.journal
        await journal.reserve(key, Question.model_validate(body(identity)), seconds=75)
        call_id = await journal.dispatch(str(key), 1)
        await journal.returned(str(key), call_id)
        result = {"id": str(key), "status": "answered", "answer": "注入的边界样本", "facts": [], "citations": []}
        with Session(engine) as session, session.begin():
            row = session.get(AssistantRequest, key)
            row.deadline_at = session.scalar(select(func.now())) - timedelta(seconds=1)
            session.add(AnswerAudit(id=key, question_sha256="a" * 64,
                answer_sha256=digest(result), status="answered", trace={}))
        assert not await journal.finish(key, result)
        status = await journal.lookup(key)
        assert status["status"] == "unknown" and status["error"] == "request_deadline_elapsed"
    asyncio.run(clients(config, lambda request: response(), check), loop_factory=make_event_loop)
