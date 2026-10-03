"""真实连接失败后保留 outbox；成功投递的整链路由 e2e 覆盖。"""

from dataclasses import replace

from sqlalchemy import select
from sqlalchemy.orm import Session

from power_forecast_service.jobs.celery_app import make_celery
from power_forecast_service.jobs.dispatcher import dispatch_once
from power_forecast_service.storage.models import Outbox, Task


def test_real_broker_connection_failure_retains_outbox(config, engine, submit):
    task_id = submit(config).task_id
    unavailable = replace(config, broker_url="redis://127.0.0.1:1/0")
    assert dispatch_once(engine, make_celery(unavailable), unavailable) == 0
    with Session(engine) as session:
        assert session.get(Task, task_id).status == "pending_dispatch"
        message = session.scalar(select(Outbox))
        assert message.sent_at is None and message.publish_count == 0
        assert message.last_error is not None
