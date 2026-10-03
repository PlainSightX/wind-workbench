"""共享队列配置；导入配置不读取数据库密码、不建立连接。"""

import os

from celery import Celery

from ..settings import Settings


def make_celery(settings: Settings | None = None) -> Celery:
    broker = settings.broker_url if settings else os.getenv("WIND_BROKER_URL", Settings.broker_url)
    time_limit = settings.task_limit_seconds if settings else Settings.task_limit_seconds
    app = Celery("wind_experiments", broker=broker)
    app.conf.update(
        task_default_queue="wind-experiments",
        task_serializer="json",
        accept_content=["json"],
        task_ignore_result=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_reject_on_worker_lost=True,
        task_time_limit=time_limit,
        task_publish_retry=False,
        broker_connection_timeout=3,
        broker_transport_options={
            "visibility_timeout": 300,
            "socket_connect_timeout": 3,
            "socket_timeout": 3,
        },
    )
    return app
