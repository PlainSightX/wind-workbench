"""统一连接参数，连接池生命周期仍由各进程拥有。"""

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine

from ..settings import Settings


def make_sync_engine(settings: Settings):
    return create_engine(
        settings.database_url, pool_pre_ping=True, connect_args={"connect_timeout": 5}
    )


def make_async_engine(settings: Settings):
    return create_async_engine(
        settings.database_url, pool_pre_ping=True, connect_args={"connect_timeout": 5}
    )
