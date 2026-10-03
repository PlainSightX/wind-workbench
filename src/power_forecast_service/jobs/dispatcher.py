"""从 PG outbox 投递；Redis 失败不会删除已接受的任务。"""

import logging
import time
from datetime import timedelta

from celery import Celery
from kombu.exceptions import OperationalError
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..experiments.execution import db_now, recover_expired
from ..settings import Settings
from ..storage.database import make_sync_engine
from ..storage.models import Outbox, Task
from .celery_app import make_celery

log = logging.getLogger(__name__)


def dispatch_once(engine, celery: Celery, settings: Settings) -> int:
    sent = 0
    # 每条通知单独提交；短网络超时限制锁持有时间。
    for _ in range(20):
        with Session(engine) as session, session.begin():
            pair = session.execute(
                select(Task, Outbox)
                .join(Outbox, Outbox.task_id == Task.id)
                .where(
                    Task.status.in_(["pending_dispatch", "queued", "retry_wait"]),
                    Outbox.available_at <= func.clock_timestamp(),
                )
                .order_by(Outbox.available_at)
                .limit(1)
                .with_for_update(skip_locked=True, of=Task)
            ).first()
            if pair is None:
                break
            task, message = pair
            now = db_now(session)
            try:
                celery.send_task(
                    "wind.execute", args=[str(task.id)], task_id=str(message.id), retry=False
                )
            except (OperationalError, OSError) as exc:
                message.last_error = type(exc).__name__
                message.available_at = now + timedelta(seconds=settings.dispatch_interval_seconds)
                log.warning("dispatch deferred task=%s error_type=%s", task.id, type(exc).__name__)
                break
            message.sent_at = now
            message.publish_count += 1
            message.last_error = None
            # 消息被接受后仍可能丢失；未领取任务到期可重投，领取用 task 状态去重。
            message.available_at = now + timedelta(seconds=settings.redelivery_seconds)
            if task.status in {"pending_dispatch", "retry_wait"}:
                task.status = "queued"
                task.updated_at = now
            sent += 1
    return sent


def main():
    logging.basicConfig(level=logging.INFO)
    settings = Settings.from_environment()
    engine = make_sync_engine(settings)
    app = make_celery(settings)
    try:
        while True:
            try:
                recover_expired(engine, settings)
                dispatch_once(engine, app, settings)
            except SQLAlchemyError as exc:
                log.warning("database unavailable error_type=%s", type(exc).__name__)
            time.sleep(settings.dispatch_interval_seconds)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
