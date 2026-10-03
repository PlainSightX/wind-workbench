"""真实 PG 测试的专属资源；收集测试不读密码，执行时才创建随机独立库。"""

import asyncio
import os
from pathlib import Path
from dataclasses import replace
from datetime import timedelta
from uuid import uuid4

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from psycopg import sql
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import Session

from power_forecast_service.experiments.contracts import ExperimentRequest
from power_forecast_service.experiments.execution import db_now
from power_forecast_service.experiments.service import submit_experiment
from power_forecast_service.serve import make_event_loop
from power_forecast_service.settings import ROOT, Settings
from power_forecast_service.storage.database import make_async_engine, make_sync_engine
from power_forecast_service.storage.models import Task
from power_forecast_service.storage.artifacts import sha256_file


@pytest.fixture
def legacy_spec(config):
    # 字面冻结的旧合同；不通过新freeze函数创建历史样本。
    return {
        "dataset_id": "wind-2019-q1", "purpose": "development",
        "input_sha256": sha256_file(config.data_path),
        "split_version": "temporal-70-15-15-label-isolated-v0.2",
        "feature_contract_version": "lag-only-source-time-v0.1",
        "model_set": ["persistence", "hist_gradient_boosting"],
        "horizon_steps": 12, "random_seed": 42,
        "hgb": {"learning_rate": 0.06, "max_iter": 180,
                "max_leaf_nodes": 31, "l2_regularization": 0.2},
    }


@pytest.fixture(scope="session")
def config(tmp_path_factory):
    settings = Settings.from_environment()
    admin_password = Path(os.getenv("WIND_DB_ADMIN_PASSWORD_FILE", str(ROOT / ".local/runtime/postgres-password.txt"))).read_text().strip()
    database = "wind_test_" + uuid4().hex
    # 不替换 SQLite、不自动跳过；环境不可用应明确失败。
    with psycopg.connect(
        host=settings.database_url.host,
        port=settings.database_url.port,
        dbname=settings.database_url.database,
        user="wind_admin",
        password=admin_password,
        autocommit=True,
        connect_timeout=5,
    ) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {} OWNER wind_app").format(sql.Identifier(database)))
        try:
            # 与正式基础设施一致：管理员安装vector，应用账户执行表迁移。
            with psycopg.connect(host=settings.database_url.host, port=settings.database_url.port,
                                dbname=database, user="wind_admin", password=admin_password,
                                autocommit=True) as setup:
                setup.execute("CREATE EXTENSION vector")
            settings = replace(
                settings,
                database_url=settings.database_url.set(database=database),
                artifact_root=tmp_path_factory.mktemp("task-artifacts"),
                retry_delay_seconds=0,
            )
            with pytest.MonkeyPatch.context() as patch:
                patch.setenv("WIND_DB_NAME", database)
                command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
                command.check(Config(str(ROOT / "alembic.ini")))
            yield settings
        finally:
            # 清理对象来自本夹具实际创建的随机库名，绝不使用调用者传入的库名。
            admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname=%s AND pid<>pg_backend_pid()",
                (database,),
            )
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database)))


@pytest.fixture
def engine(config):
    engine = make_sync_engine(config)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "TRUNCATE model_artifacts, experiment_runs, experiment_outbox, "
                    "experiment_attempts, experiment_tasks"
                )
            )
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def submit():
    def submit_task(config, key="request-1", request=None):
        async def call():
            engine = make_async_engine(config)
            try:
                async with async_sessionmaker(engine)() as session:
                    return await submit_experiment(
                        session, request=request or ExperimentRequest(),
                        idempotency_key=key, settings=config,
                    )
            finally:
                await engine.dispose()

        return asyncio.run(call(), loop_factory=make_event_loop)

    return submit_task


@pytest.fixture
def expire():
    def expire_lease(engine, task_id):
        with Session(engine) as session, session.begin():
            task = session.get(Task, task_id)
            task.lease_until = db_now(session) - timedelta(seconds=1)

    return expire_lease


@pytest.fixture
def publication(config, trained_product):
    """事务测试复用一次拟合，但每次建立真实独立包；不以测试清单伪装可加载工件。"""
    from copy import deepcopy
    from power_forecast_service.forecasting.pipeline import ExperimentProduct
    from power_forecast_service.forecasting.bundles import save_packages, verify_packages
    from power_forecast_service.storage.artifacts import execution_provenance

    def prepare(task_id, attempt_id, spec):
        result = deepcopy(trained_product.result)
        result.update(run_id=str(uuid4()), frozen_spec=spec, execution=execution_provenance())
        product = ExperimentProduct(result, trained_product.models, trained_product.frame)
        packages = save_packages(product, config.artifact_root, task_id, attempt_id)
        result["model_verification"] = verify_packages(config.artifact_root, packages)
        return result, packages

    return prepare
