"""从进程配置读取凭据；不将真实 URL 写入 alembic.ini。"""

from alembic import context
from sqlalchemy import create_engine, pool

from power_forecast_service.settings import Settings
from power_forecast_service.storage.models import Base
from power_forecast_service.assistant.storage import AssistantChunk

if context.is_offline_mode():
    raise RuntimeError("Use an actual PostgreSQL database for migration verification")

engine = create_engine(Settings.from_environment().database_url, poolclass=pool.NullPool)
with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=Base.metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()
