"""PG 请求身份与发送边界；不将结果未知解释成未执行，也不保存问答原文。"""

import asyncio
from datetime import timedelta
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError

from ..storage.models import AnswerAudit
from .contracts import AssistantError
from .evidence import digest
from .storage import AssistantRequest


class DuplicateRequest(Exception):
    def __init__(self, snapshot, *, conflict=False):
        self.snapshot = snapshot
        self.conflict = conflict


def snapshot(row):
    return {"id": str(row.id), "status": row.status, "outcome": row.outcome,
        "error": row.error, "calls": row.calls, "answer_sha256": row.answer_sha256,
        "deadline_at": row.deadline_at.isoformat(), "answer_retained": False}


class RequestJournal:
    def __init__(self, sessions, identity):
        self.sessions, self.identity = sessions, identity

    def fingerprint(self, question):
        return digest({"question": question.model_dump(mode="json"), "consumer": self.identity})

    async def reserve(self, request_id, question, *, seconds):
        fingerprint = self.fingerprint(question)
        async with asyncio.timeout(3), self.sessions() as session, session.begin():
            result = await session.execute(insert(AssistantRequest).values(
                id=request_id, fingerprint=fingerprint, status="pending", calls=[],
                deadline_at=func.now() + timedelta(seconds=seconds))
                .on_conflict_do_nothing(index_elements=[AssistantRequest.id]).returning(AssistantRequest.id))
            if result.scalar_one_or_none() is None:
                row = await session.get(AssistantRequest, request_id)
                raise DuplicateRequest(snapshot(row), conflict=row.fingerprint != fingerprint)

    async def lookup(self, request_id):
        async with asyncio.timeout(3), self.sessions() as session, session.begin():
            row = await session.get(AssistantRequest, request_id, with_for_update=True)
            if row is None:
                return None
            now = await session.scalar(select(func.clock_timestamp()))
            if row.status == "pending" and row.deadline_at <= now:
                # 过期是停止自动重发的信号，不是远端恰好执行一次的证明。
                row.status, row.error, row.finalized_at = "unknown", "request_deadline_elapsed", now
            return snapshot(row)

    async def dispatch(self, request_id, index):
        try:
            async with asyncio.timeout(3), self.sessions() as session, session.begin():
                row = await session.get(AssistantRequest, UUID(request_id), with_for_update=True)
                now = await session.scalar(select(func.clock_timestamp()))
                if row is None or row.status != "pending" or row.deadline_at <= now or len(row.calls) != index - 1:
                    raise AssistantError("provider_request_not_dispatchable")
                call_id = f"{request_id}-{index}"
                row.calls = [*row.calls, {"id": call_id, "status": "dispatching"}]
                return call_id
        except (TimeoutError, SQLAlchemyError) as exc:
            raise AssistantError("provider_request_registration_failed") from exc

    async def returned(self, request_id, call_id, *, status="returned"):
        if status not in {"returned", "rejected"}:
            raise ValueError("Invalid acknowledged provider outcome")
        async with asyncio.timeout(3), self.sessions() as session, session.begin():
            row = await session.get(AssistantRequest, UUID(request_id), with_for_update=True)
            if row is None or row.status != "pending" or not row.calls or row.calls[-1]["id"] != call_id:
                raise AssistantError("provider_request_not_dispatchable")
            row.calls = [*row.calls[:-1], {"id": call_id, "status": status}]

    async def finish(self, request_id, result=None):
        async with asyncio.timeout(3), self.sessions() as session, session.begin():
            row = await session.get(AssistantRequest, request_id, with_for_update=True)
            if row is None or row.status != "pending":
                return False
            audit = await session.get(AnswerAudit, request_id) if result else None
            now = await session.scalar(select(func.clock_timestamp()))
            complete = (audit is not None and result is not None
                and now <= row.deadline_at
                and audit.status == result["status"] and audit.answer_sha256 == digest(
                    {key: value for key, value in result.items() if key != "trace"})
                and all(call["status"] in {"returned", "rejected"} for call in row.calls))
            row.status = "completed" if complete else "unknown"
            row.outcome = result["status"] if result else None
            row.error = result.get("error") if result else "request_interrupted"
            if now > row.deadline_at:
                row.error = "request_deadline_elapsed"
            row.answer_sha256 = audit.answer_sha256 if complete else None
            row.finalized_at = now
            return complete
