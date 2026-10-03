"""真实PG最终导入、截止时间和结果发布；不以重复请求冒充重复结果测试。"""

import asyncio
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import timedelta
from threading import Event
from time import sleep
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from power_forecast_service.api.app import create_app
from power_forecast_service.api.forecasts import bounded_forecast
from power_forecast_service.experiments.engie_delivery import reserve, finalize, read_delivery, deliver
from power_forecast_service.forecasting.engie_predictor import replay_input, forecast
from power_forecast_service.forecasting.engie_service_contract import EngieDeliveryRequest
from power_forecast_service.serve import make_event_loop
from power_forecast_service.settings import ROOT
from power_forecast_service.storage.engie_final_imports import prepare_import
from power_forecast_service.storage.engie_imports import publish_import
from power_forecast_service.storage.models import EngieDelivery
from power_forecast_service.storage.model_packages import PackageError

pytestmark = pytest.mark.original_replay


@pytest.fixture(scope="module")
def final_prepared(config, tmp_path_factory):
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    value = os.getenv("WIND_ENGIE_FINAL_SOURCE")
    if not value:
        pytest.fail("Set WIND_ENGIE_FINAL_SOURCE to the prepared final-2015 attachment")
    source = Path(value)
    with pytest.MonkeyPatch.context() as patch:
        def forbidden(*args, **kwargs):
            raise AssertionError("final_import_must_not_fit")
        for cls in (LGBMRegressor, Ridge, StandardScaler):
            patch.setattr(cls, "fit", forbidden)
        return prepare_import(source, config.artifact_root)[0]


@pytest.fixture
def case(engine, final_prepared, config):
    publish_import(engine, final_prepared)
    run, artifacts = final_prepared[0]
    artifact = next(item for item in artifacts if item["family"] == "lightgbm_l1_shrink")
    registration = {**artifact, "run": run["manifest"]}
    from datetime import datetime
    body, _ = replay_input(config.artifact_root, registration, datetime.fromisoformat("2015-01-01T00:00:00+00:00"))
    result = forecast(config.artifact_root, registration, body)
    request = EngieDeliveryRequest(**body.model_dump(), request_key=uuid4().hex)
    return registration, request, result


def test_request_race_immutable_deadline_and_result_duplicates(engine, case):
    _, body, result = case
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(lambda _: reserve(engine, body), range(2)))
    assert sum(created for _, created in records) == 1
    identity = records[0][0]["id"]
    assert identity == records[1][0]["id"]
    first = finalize(engine, identity, result)
    assert first["status"] == "published"
    assert finalize(engine, identity, result) == first
    assert reserve(engine, body)[0] == first
    with pytest.raises(PackageError, match="request_conflict"):
        reserve(engine, body.model_copy(update={"budget_ms": 2000}))
    bad = deepcopy(result)
    bad["predictions"][0][0] += 1
    bad["farm_predictions"][0] += 1
    with pytest.raises(PackageError, match="result_conflict"):
        finalize(engine, identity, bad)
    assert read_delivery(engine, identity) == first


@pytest.mark.parametrize("field", ["roster", "target_times", "unit", "model_version", "source_cutoff"])
def test_result_contract_rejects_wrong_envelope(engine, case, field):
    _, body, result = case
    record, _ = reserve(engine, body)
    bad = deepcopy(result)
    bad[field] = [] if isinstance(bad[field], list) else "invalid"
    with pytest.raises(PackageError):
        finalize(engine, record["id"], bad)
    assert read_delivery(engine, record["id"])["status"] == "pending"


def test_waiting_for_lock_checks_actual_pg_time_and_late_digest(engine, case):
    _, body, result = case
    body = body.model_copy(update={"budget_ms": 150})
    record, _ = reserve(engine, body)
    with engine.connect() as lock:
        tx = lock.begin()
        lock.execute(text("SELECT id FROM engie_deliveries WHERE id=:id FOR UPDATE"), {"id": record["id"]})
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(finalize, engine, record["id"], result)
            sleep(.3)
            tx.commit()
            late = future.result(timeout=5)
    assert late["status"] == "expired" and late["result"] is None
    assert late["completed_at"] > late["deadline_at"] and late["result_sha256"]
    assert finalize(engine, record["id"], result) == late


def test_get_expiry_and_published_history_remain_terminal(engine, case):
    _, body, result = case
    record, _ = reserve(engine, body.model_copy(update={"budget_ms": 1}))
    sleep(.02)
    assert read_delivery(engine, record["id"])["status"] == "expired"
    late = finalize(engine, record["id"], result)
    assert late["result"] is None and late["completed_at"] is not None
    record, _ = reserve(engine, body.model_copy(update={"request_key": uuid4().hex}))
    published = finalize(engine, record["id"], result)
    # 只改测试库时间，验证过去截止时间不会让已发布历史失效。
    with Session(engine) as session, session.begin():
        row = session.get(EngieDelivery, record["id"])
        row.deadline_at = row.created_at - timedelta(seconds=1)
    assert read_delivery(engine, record["id"])["result"] == published["result"]


def test_cancelled_waiter_does_not_skip_finalization(config, engine, case, monkeypatch):
    registration, body, result = case
    entered, release = Event(), Event()
    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return result
    monkeypatch.setattr("power_forecast_service.forecasting.engie_predictor.forecast", blocked)
    async def scenario():
        state = SimpleNamespace(forecast_gate=asyncio.Semaphore(1), forecast_tasks=set())
        request = SimpleNamespace(app=SimpleNamespace(state=state))
        waiter = asyncio.create_task(bounded_forecast(request, deliver, engine, config.artifact_root, registration, body))
        assert await asyncio.to_thread(entered.wait, 5)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert state.forecast_gate.locked()
        active = list(state.forecast_tasks)
        release.set()
        records = await asyncio.gather(*active)
        assert records[0]["status"] == "published" and not state.forecast_gate.locked()
    asyncio.run(scenario(), loop_factory=make_event_loop)
    assert reserve(engine, body)[0]["status"] == "published"


def test_final_real_http_and_budget_validation(config, engine, final_prepared, case):
    _, body, result = case
    assert publish_import(engine, final_prepared) == publish_import(engine, final_prepared)
    with TestClient(create_app(config), base_url="http://127.0.0.1", backend_options={"loop_factory": make_event_loop}) as client:
        rows = client.get("/engie/imports").json()
        row = next(v for v in rows if v["scope"] == "final_2015")
        assert len(row["models"]) == 5
        response = client.post("/engie/deliveries", json=body.model_dump(mode="json"))
        assert response.status_code == 200, response.text
        saved = response.json()
        assert saved["status"] == "published" and saved["result"] == result
        assert client.get(f"/engie/deliveries/{saved['id']}").json() == saved
        for value in (True, 1.5, 0, 120001):
            invalid = {**body.model_dump(mode="json"), "budget_ms": value}
            assert client.post("/engie/deliveries", json=invalid).status_code == 422
