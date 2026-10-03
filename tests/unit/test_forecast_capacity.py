"""取消HTTP等待不等于线程结束；容量必须绑定真实计算生命周期。"""

import asyncio
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from power_forecast_service.api import forecasts


def test_cancelled_request_keeps_capacity_until_thread_finishes(monkeypatch):
    started, release = threading.Event(), threading.Event()
    identity = uuid4()
    item = SimpleNamespace(id=identity, model_key="persistence", path="unused",
                           manifest_sha256="a" * 64, manifest={}, status="ready")

    async def get(*args):
        return item

    @asynccontextmanager
    async def sessions():
        yield SimpleNamespace(get=get)

    def blocked(*args):
        started.set()
        assert release.wait(5)
        return "prediction"

    monkeypatch.setattr(forecasts, "forecast_from_package", blocked)

    async def check():
        state = SimpleNamespace(sessions=sessions, forecast_gate=asyncio.Semaphore(1),
                                forecast_tasks=set(), settings=SimpleNamespace(artifact_root=None))
        request = SimpleNamespace(app=SimpleNamespace(state=state))
        body = SimpleNamespace(artifact_id=identity)
        first = asyncio.create_task(forecasts.forecast(request, body))
        try:
            assert await asyncio.to_thread(started.wait, 3)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert state.forecast_gate.locked()
            with pytest.raises(HTTPException) as failure:
                await forecasts.forecast(request, body)
            assert failure.value.status_code == 503
        finally:
            release.set()
            await asyncio.gather(*state.forecast_tasks)
        assert not state.forecast_gate.locked()
        assert await forecasts.forecast(request, body) == "prediction"

    asyncio.run(check())
