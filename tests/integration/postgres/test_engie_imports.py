"""真实PG：原始九包导入、事务失败重入、迁移和HTTP回放，不训练。"""

from copy import deepcopy
import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import event, text

from power_forecast_service.api.app import create_app
from power_forecast_service.settings import ROOT
from power_forecast_service.serve import make_event_loop
from power_forecast_service.storage.engie_imports import prepare_import, publish_import

pytestmark = pytest.mark.original_replay


@pytest.fixture(scope="module")
def imported(config, tmp_path_factory):
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    value = os.getenv("WIND_ENGIE_DEVELOPMENT_SOURCE")
    if not value:
        pytest.fail("Set WIND_ENGIE_DEVELOPMENT_SOURCE to the prepared development-2014 attachment")
    staging = Path(value)
    with pytest.MonkeyPatch.context() as patch:
        def forbidden(*args, **kwargs):
            raise AssertionError("integration_must_not_fit")
        for cls in (LGBMRegressor, Ridge, StandardScaler):
            patch.setattr(cls, "fit", forbidden)
        return prepare_import(staging, config.artifact_root)


def test_import_atomic_retry_and_real_http(config, engine, imported):
    prepared, _ = imported
    with engine.begin() as connection:
        connection.execute(text("TRUNCATE engie_deliveries, imported_artifacts, imported_runs"))
    calls = 0

    def interrupt(conn, cursor, statement, parameters, context, executemany):
        nonlocal calls
        if statement.startswith("INSERT INTO imported_artifacts"):
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated_import_interruption")

    event.listen(engine, "before_cursor_execute", interrupt)
    try:
        with pytest.raises(RuntimeError, match="interruption"):
            publish_import(engine, prepared)
    finally:
        event.remove(engine, "before_cursor_execute", interrupt)
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM imported_runs")) == 0
        assert connection.scalar(text("SELECT count(*) FROM imported_artifacts")) == 0
    ids = publish_import(engine, prepared)
    assert publish_import(engine, prepared) == ids
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM imported_artifacts")) == 9
        assert connection.scalar(text("SELECT count(*) FROM experiment_tasks")) == 0
    with TestClient(create_app(config), base_url="http://127.0.0.1", backend_options={"loop_factory": make_event_loop}) as client:
        rows = client.get("/engie/imports").json()
        assert len(rows) == 3
        quarter = rows[-1]
        model = next(item for item in quarter["models"] if item["family"] == "persistence")
        windows = client.get(f"/engie/artifacts/{model['artifact_id']}/windows").json()
        # 最后起报可预测，但2015事后实况完全不开放。
        response = client.post("/engie/replays", json={"artifact_id": model["artifact_id"], "issue_time": windows["issue_times"][-1]})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["actual"] == [[None] * 6] * 4
        assert len(result["forecast"]["farm_predictions"]) == 6
        body = {"artifact_id": model["artifact_id"], "issue_time": windows["issue_times"][-1], "history": result["history"]}
        prediction = client.post("/engie/forecasts", json=body)
        assert prediction.status_code == 200
        assert prediction.json() == result["forecast"]
        oversized = deepcopy(body)
        for turbine in oversized["history"][-1]["turbines"].values():
            turbine["power_kw"] = 1e308
        rejected = client.post("/engie/forecasts", json=oversized)
        assert rejected.status_code == 409
        assert rejected.json()["detail"] == "engie_model_invalid_output"
        for problem in ("missing", "gap", "stale", "holdout"):
            bad = deepcopy(body)
            if problem == "missing":
                bad["history"][0]["turbines"].pop("R80711")
            elif problem == "gap":
                bad["history"][1]["timestamp"] = bad["history"][0]["timestamp"]
            elif problem == "stale":
                bad["issue_time"] = "2014-12-31T23:40:00Z"
            else:
                assert client.post("/engie/replays", json={"artifact_id": model["artifact_id"], "issue_time": "2015-01-01T00:00:00Z"}).status_code == 409
                continue
            assert client.post("/engie/forecasts", json=bad).status_code == 422
        file = config.artifact_root / prepared[-1][1][0]["path"]
        original = file.read_bytes()
        try:
            file.write_bytes(b"corrupted")
            corrupted = client.post("/engie/replays", json={"artifact_id": str(prepared[-1][1][0]["id"]), "issue_time": windows["issue_times"][0]})
            assert corrupted.status_code == 409
            assert corrupted.json()["detail"] == "engie_package_integrity_failed"
        finally:
            file.write_bytes(original)
    with TestClient(create_app(config), base_url="http://127.0.0.1", backend_options={"loop_factory": make_event_loop}) as restarted:
        assert restarted.get("/engie/imports").json() == rows


def test_migration_preserves_old_task(config, engine, submit, monkeypatch):
    monkeypatch.setenv("WIND_DB_NAME", config.database_url.database)
    submit(config, key="engie-migration-preserve")
    migration = Config(str(ROOT / "alembic.ini"))
    with engine.connect() as connection:
        original = connection.execute(text("SELECT * FROM experiment_tasks ORDER BY id")).mappings().all()
    command.downgrade(migration, "0003_final_protocol")
    try:
        command.upgrade(migration, "head")
        command.check(migration)
        with engine.connect() as connection:
            assert connection.execute(text("SELECT * FROM experiment_tasks ORDER BY id")).mappings().all() == original
            assert connection.scalar(text("SELECT count(*) FROM imported_runs")) == 0
    finally:
        command.upgrade(migration, "head")
