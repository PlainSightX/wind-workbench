"""仅在fixture拥有的随机PG库中演练迁移；旧结果不被自动变成可用模型。"""

from uuid import uuid4

from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from power_forecast_service.experiments.execution import claim
from power_forecast_service.settings import ROOT
from power_forecast_service.storage.models import Attempt, ModelArtifact, Run, Task


def test_upgrade_and_downgrade_preserve_legacy_rows(config, engine, submit, monkeypatch):
    assert config.database_url.database.startswith("wind_test_")
    monkeypatch.setenv("WIND_DB_NAME", config.database_url.database)
    migration = Config(str(ROOT / "alembic.ini"))
    tables = ("experiment_tasks", "experiment_attempts", "experiment_outbox", "experiment_runs")

    def snapshot():
        with engine.connect() as connection:
            return {name: connection.execute(text(f"SELECT * FROM {name} ORDER BY id")).mappings().all()
                    for name in tables}

    command.downgrade(migration, "0001_experiments")
    try:
        task_id = submit(config).task_id
        attempt_id, _ = claim(engine, task_id, config)
        run_id = uuid4()
        with Session(engine) as session, session.begin():
            task = session.get(Task, task_id)
            task.status, task.lease_until = "succeeded", None
            session.get(Attempt, attempt_id).status = "succeeded"
            session.add(Run(id=run_id, task_id=task_id, attempt_id=attempt_id,
                            result={"legacy": True}, artifact_path="historic.json",
                            artifact_sha256="a" * 64))
        before = snapshot()
        for revision in ("head", "0001_experiments", "head"):
            if revision == "head":
                command.upgrade(migration, revision)
                with Session(engine) as session:
                    assert session.scalars(select(ModelArtifact)).all() == []
                command.check(migration)
            else:
                command.downgrade(migration, revision)
            assert snapshot() == before
    finally:
        command.upgrade(migration, "head")
