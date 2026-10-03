"""不同HTTP请求身份也不能重复创建同一冻结协议的正式评分任务。"""

from uuid import uuid4

import pytest
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from power_forecast_service.experiments import contracts
from power_forecast_service.storage.models import Task, Outbox
from power_forecast_service.forecasting.sequence_protocol import CONFIG


def test_final_protocol_deduplication_and_database_guard(engine, config, submit, monkeypatch):
    selection = {"sequence_key": "transformer_delta", "config": CONFIG, "fixture_only": True}
    monkeypatch.setattr(contracts, "frozen_selection", lambda: (selection, "a" * 64))
    request = contracts.ExperimentRequest(purpose="final_evaluation", training_policy="fixed_iterations")
    first = submit(config, "final-one", request)
    second = submit(config, "final-two", request)
    assert first.task_id == second.task_id
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(Outbox)) == 1
        spec = session.get(Task, first.task_id).spec
    with pytest.raises(IntegrityError), Session(engine) as session, session.begin():
        session.add(Task(id=uuid4(), idempotency_key="bypass", request_fingerprint="b" * 64,
                         spec=spec, status="pending_dispatch", attempt_count=0))
        session.flush()
