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


def run_answer(monkeypatch, provider, capture, *, stage=False, fail_audit=False, response_mode="legacy", documents=None):
    identity = str(uuid4())
    evidence = {"facts": [{"id": "c0.ridge.mae", "label": "MAE", "value": 12.5, "unit": "kW"}],
                "records": [{"models": {"ridge": {}}}], "scopes": ["fixture"]}
    if stage:
        evidence.update(stage_requirements=[{"object_id": identity, "role": "selection_basis"}], stage_options=[{
            "object_id": identity, "role": "selection_basis", "label": "开发依据",
            "allowed_fact_ids": ["c0.ridge.mae"], "required_fact_ids": ["c0.ridge.mae"],
            "allowed_citations": [], "required_citations": []}])
        claim = {"object_id": identity, "role": "selection_basis", "text": "{{c0.ridge.mae}}"}
        if response_mode == "references":
            provider.drafts[0]["body"] = {"kind": "staged", "claims": [claim]}
        else:
            provider.drafts[0]["stage_claims"] = [claim]
    else:
        evidence.update(stage_requirements=[], stage_options=[])
    async def get_results(*args):
        return evidence
    monkeypatch.setattr(workflow, "get_results", get_results)
    monkeypatch.setattr(workflow, "compare_results", lambda value: value)
    monkeypatch.setattr(workflow, "prepare_stage_requirements", lambda data, docs, question: (data, docs))
    chunks = [{**doc, "scopes": ["fixture"]} for doc in documents or []]
    monkeypatch.setattr(workflow, "corpus", lambda: {"sha256": "a"*64, "chunks": chunks})
    session = AuditSession(fail_audit)
    assistant = Assistant(lambda: session, provider=provider, capture_draft=capture, response_mode=response_mode)
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
    ("MAE=-MAE：{{c0.ridge.mae}}。", "answer_fact_sign_conflict"),
    ("MAE=−MAE：MAE：{{c0.ridge.mae}}。", "answer_fact_sign_conflict"),
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
    provider = Provider([{"status": "answered", "answer": 123, "sk-test-private": "https://private.invalid"}] * 2)
    result, audit = run_answer(monkeypatch, provider, None)
    assert result["status"] == "validation_error"
    errors = [row for row in result["trace"]["model_calls"] if row["status"] == "schema_error"]
    assert len(errors) == 2
    assert all("answer:string_type" in row["schema_errors"] for row in errors)
    assert all(len(row["schema_errors"]) <= 240 for row in errors)
    assert "123" not in errors[0]["schema_errors"]
    assert "[key]:extra_forbidden" in errors[0]["schema_errors"]
    assert "sk-test-private" not in json.dumps(result["trace"])
    assert "private.invalid" not in json.dumps(result["trace"])
    assert provider.calls == 2
    assert len(audit.rows) == 1


def test_schema_feedback_repairs_without_echoing_invalid_private_fields(monkeypatch):
    class RecordingProvider(Provider):
        async def ainvoke(self, messages):
            self.messages = messages
            return await super().ainvoke(messages)
    provider = RecordingProvider([{"status": "answered", "answer": 123, "sk-test-secret": "https://private.invalid"},
        {"status": "answered", "answer": "{{c0.ridge.mae}}", "fact_ids": ["c0.ridge.mae"]}])
    result, audit = run_answer(monkeypatch, provider, None)
    assert result["status"] == "answered" and provider.calls == 2 and len(audit.rows) == 1
    feedback = json.dumps(provider.messages, ensure_ascii=False)
    assert "answer:string_type" in feedback
    assert "sk-test-secret" not in feedback and "private.invalid" not in feedback
    assert result["trace"]["repair_reason"] == "answer_schema_invalid"


@pytest.mark.parametrize("code", ["provider_timeout", "provider_unavailable", "provider_rate_limited"])
@pytest.mark.parametrize("mode", ["legacy", "references"])
def test_transport_failures_do_not_become_answer_retries(monkeypatch, code, mode):
    class FailedProvider:
        calls = 0
        async def ainvoke(self, messages):
            from power_forecast_service.assistant.contracts import AssistantError
            self.calls += 1
            raise AssistantError(code)
    provider = FailedProvider()
    result, audit = run_answer(monkeypatch, provider, None, response_mode=mode)
    assert result["status"] == "dependency_error" and result["error"] == code
    assert provider.calls == 1 and len(audit.rows) == 1
    assert "repair_reason" not in result["trace"]


def test_optional_capture_cannot_hide_mandatory_audit_failure(monkeypatch):
    provider = Provider([{"status": "answered", "answer": "{{c0.ridge.mae}}", "fact_ids": ["c0.ridge.mae"]}])
    with pytest.raises(RuntimeError, match="mandatory audit failed"):
        run_answer(monkeypatch, provider, None, fail_audit=True)


@pytest.mark.parametrize("stage", [False, True])
def test_reference_contract_runs_through_actual_answer_and_audit(monkeypatch, stage):
    drafts = [{"status": "answered", "body": {"kind": "plain", "text": "{{c0.ridge.mae}}"}}]
    provider, captures = Provider(drafts), []
    result, audit = run_answer(monkeypatch, provider, captures.append, response_mode="references", stage=stage)
    assert result["status"] == "answered" and provider.calls == 1 and len(audit.rows) == 1
    assert result["trace"]["response_mode"] == "references"
    assert result["trace"]["prompt_version"] == workflow.REFERENCE_PROMPT_VERSION
    assert result["facts"][0]["id"] == "c0.ridge.mae"
    assert set(captures[0]) == {"status", "body"}
    assert "body" in captures[0] and "fact_ids" not in captures[0]


@pytest.mark.parametrize("repair", [False, True])
def test_reference_unknown_id_gets_at_most_one_completed_response_repair(monkeypatch, repair):
    class RecordingProvider(Provider):
        async def ainvoke(self, messages):
            self.messages = messages
            return await super().ainvoke(messages)
    document = {"id": "doc", "text": "该对象有固定结果。", "title": "方法", "source_sha256": "a" * 64}
    def value(identity):
        return {"status": "answered", "body": {"kind": "plain", "text": "{{c0.ridge.mae}}",
            "citations": [{"document_id": identity}]}}
    provider = RecordingProvider([value("unknown"), value("doc" if repair else "unknown")])
    result, audit = run_answer(monkeypatch, provider, None, response_mode="references", documents=[document])
    assert result["status"] == ("answered" if repair else "validation_error")
    assert result["trace"]["repair_reason"] == "answer_citation_invalid"
    assert provider.calls == 2 and len(audit.rows) == 1
    assert set(json.loads(provider.messages[2][1])) == {"status", "body"}
    assert "source_sha256" in provider.messages[3][1]
    if repair:
        assert result["citations"][0]["quote"] == document["text"]


def test_reference_schema_error_is_repaired_without_echoing_invalid_data(monkeypatch):
    class RecordingProvider(Provider):
        async def ainvoke(self, messages):
            self.messages = messages
            return await super().ainvoke(messages)
    provider = RecordingProvider([{"status": "answered", "body": {"kind": "plain", "text": 123},
                                  "sk-private-secret": "https://private.invalid"},
        {"status": "answered", "body": {"kind": "plain", "text": "{{c0.ridge.mae}}"}}])
    result, audit = run_answer(monkeypatch, provider, None, response_mode="references")
    assert result["status"] == "answered" and provider.calls == 2 and len(audit.rows) == 1
    feedback = json.dumps(provider.messages)
    assert "body.plain.text:string_type" in feedback
    assert "sk-private-secret" not in feedback and "private.invalid" not in feedback


def test_aggregate_reference_limit_is_repairable_not_an_unclassified_exception(monkeypatch):
    first = " ".join("{{fact." + str(index) + "}}" for index in range(16))
    provider = Provider([{ "status": "answered", "body": {"kind": "plain", "text": first}},
                         {"status": "answered", "body": {"kind": "plain", "text": "{{c0.ridge.mae}}"}}])
    result, audit = run_answer(monkeypatch, provider, None, response_mode="references")
    assert result["status"] == "answered" and provider.calls == 2 and len(audit.rows) == 1
    assert result["trace"]["repair_reason"] == "answer_reference_limit"
