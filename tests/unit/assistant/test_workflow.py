"""可选诊断不拥有回答流程；模拟审计只验证调用边界，不代替真实PG。"""

import asyncio
import json
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage

from power_forecast_service.assistant.contracts import ContextRef, Question
from power_forecast_service.assistant.workflow import Assistant
from power_forecast_service.assistant import workflow


class AuditSession:
    def __init__(self, fail=False):
        self.rows, self.fail = [], fail

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        if self.fail:
            raise RuntimeError("mandatory audit failed")

    def begin(self):
        return self

    def add(self, row):
        self.rows.append(row)


class Provider:
    def __init__(self, drafts):
        self.drafts, self.calls = drafts, 0

    async def ainvoke(self, messages):
        draft = self.drafts[self.calls]
        self.calls += 1
        return AIMessage(content=json.dumps(draft))


def run_answer(monkeypatch, provider, capture, *, stage=False, fail_audit=False):
    identity = str(uuid4())
    evidence = {"facts": [{"id": "c0.ridge.mae", "label": "MAE", "value": 12.5, "unit": "kW"}],
                "records": [{"models": {"ridge": {}}}], "scopes": []}
    if stage:
        evidence.update(stage_requirements=[{"object_id": identity, "role": "selection_basis"}], stage_options=[{
            "object_id": identity, "role": "selection_basis", "label": "开发依据",
            "allowed_fact_ids": ["c0.ridge.mae"], "required_fact_ids": ["c0.ridge.mae"],
            "allowed_citations": [], "required_citations": []}])
        provider.drafts[0]["stage_claims"] = [{"object_id": identity, "role": "selection_basis", "text": "{{c0.ridge.mae}}"}]
    else:
        evidence.update(stage_requirements=[], stage_options=[])
    async def get_results(*args):
        return evidence
    monkeypatch.setattr(workflow, "get_results", get_results)
    monkeypatch.setattr(workflow, "compare_results", lambda value: value)
    monkeypatch.setattr(workflow, "prepare_stage_requirements", lambda data, docs, question: (data, docs))
    monkeypatch.setattr(workflow, "corpus", lambda: {"sha256": "a"*64, "chunks": []})
    session = AuditSession(fail_audit)
    assistant = Assistant(lambda: session, provider=provider, capture_draft=capture)
    question = Question(question="MAE是多少？", contexts=[ContextRef(kind="engie_import", id=identity)])
    try:
        return asyncio.run(assistant.run(question, direct=True)), session
    finally:
        assistant.close()


def test_capture_receives_json_serializable_stage_uuid(monkeypatch):
    captured = []
    def capture(draft):
        json.dumps(draft)
        captured.append(draft)
    provider = Provider([{"status": "answered", "answer": "", "fact_ids": ["c0.ridge.mae"]}])
    result, audit = run_answer(monkeypatch, provider, capture, stage=True)
    assert result["status"] == "answered"
    assert isinstance(captured[0]["stage_claims"][0]["object_id"], str)
    assert provider.calls == 1 and len(audit.rows) == 1


def test_capture_failure_keeps_validation_repair_and_audit(monkeypatch):
    def capture(draft):
        raise OSError("private fake message sk-test-secret https://private.invalid")
    provider = Provider([{"status": "answered", "answer": "{{c0.ridge.mae}}及12.5%", "fact_ids": ["c0.ridge.mae"]},
                         {"status": "answered", "answer": "{{c0.ridge.mae}}", "fact_ids": ["c0.ridge.mae"]}])
    result, audit = run_answer(monkeypatch, provider, capture)
    assert result["status"] == "answered"
    assert result["trace"]["repair_reason"] == "answer_number_unbound"
    assert provider.calls == 2 and len(audit.rows) == 1
    assert len(result["trace"]["capture_errors"]) == 2
    serialized = json.dumps(result["trace"])
    assert "private fake" not in serialized and "sk-test-secret" not in serialized


def test_capture_failure_alone_never_adds_provider_request(monkeypatch):
    def capture(draft):
        raise RuntimeError("fake diagnostic failure")
    provider = Provider([{"status": "answered", "answer": "{{c0.ridge.mae}}", "fact_ids": ["c0.ridge.mae"]}])
    result, audit = run_answer(monkeypatch, provider, capture)
    assert result["status"] == "answered"
    assert provider.calls == 1 and len(audit.rows) == 1


@pytest.mark.parametrize("bad,code", [
    ("MAE=-{{c0.ridge.mae}}。", "answer_fact_sign_conflict"),
    ("MAE={{c0.ridge.mae}} GW。", "answer_fact_unit_conflict"),
])
@pytest.mark.parametrize("repaired", [False, True])
def test_fact_boundary_error_uses_one_repair_and_required_audit(monkeypatch, bad, code, repaired):
    first = {"status": "answered", "answer": bad, "fact_ids": ["c0.ridge.mae"]}
    second = {**first, "answer": "MAE为{{c0.ridge.mae}}。"} if repaired else first
    provider = Provider([first, second])
    result, audit = run_answer(monkeypatch, provider, None)
    assert result["status"] == ("answered" if repaired else "validation_error")
    assert result["trace"]["repair_reason"] == code
    assert provider.calls == 2 and len(audit.rows) == 1
    assert len(result["trace"]["validation_errors"]) == (1 if repaired else 2)
    if repaired:
        assert result["answer"] == "MAE为MAE：12.5 kW。"
    else:
        assert bad not in result["answer"]


def test_final_validation_detail_is_bounded_and_audited(monkeypatch):
    provider = Provider([{"status": "answered", "answer": "99999"}] * 2)
    result, audit = run_answer(monkeypatch, provider, None)
    assert result["status"] == "validation_error"
    assert "99999" in result["trace"]["validation_errors"][-1]["detail"]
    assert len(result["trace"]["validation_errors"][-1]["detail"]) <= 240
    assert len(audit.rows) == 1


def test_schema_failure_keeps_only_bounded_field_and_type_diagnostic(monkeypatch):
    provider = Provider([{"status": "answered", "answer": 123, "sk-test-private": "https://private.invalid"}])
    result, audit = run_answer(monkeypatch, provider, None)
    assert result["status"] == "validation_error"
    errors = [row for row in result["trace"]["model_calls"] if row["status"] == "schema_error"]
    assert len(errors) == 1
    assert all("answer:string_type" in row["schema_errors"] for row in errors)
    assert all(len(row["schema_errors"]) <= 240 for row in errors)
    assert "123" not in errors[0]["schema_errors"]
    assert "[key]:extra_forbidden" in errors[0]["schema_errors"]
    assert "sk-test-private" not in json.dumps(result["trace"])
    assert "private.invalid" not in json.dumps(result["trace"])
    assert provider.calls == 1
    assert len(audit.rows) == 1


def test_optional_capture_cannot_hide_mandatory_audit_failure(monkeypatch):
    provider = Provider([{"status": "answered", "answer": "{{c0.ridge.mae}}", "fact_ids": ["c0.ridge.mae"]}])
    with pytest.raises(RuntimeError, match="mandatory audit failed"):
        run_answer(monkeypatch, provider, None, fail_audit=True)
