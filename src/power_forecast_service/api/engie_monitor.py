"""监测普通入口：登记、有限步推进、迟到实况与报告读取。"""

from uuid import UUID

from fastapi import APIRouter, Request

from ..forecasting.engie_monitor_contract import MonitorAdvance, MonitorLabel, MonitorRequest
from .forecasts import bounded_forecast

router = APIRouter(prefix="/engie/monitors", tags=["ENGIE delayed monitoring"])


def operation(settings, kind, *args):
    from ..experiments.engie_monitor import (advance_monitor, create_monitor, ingest_label, read_monitor)
    from ..storage.database import make_sync_engine

    engine = make_sync_engine(settings)
    try:
        if kind == "create":
            identity = create_monitor(engine, *args)
            return read_monitor(engine, identity)
        if kind == "advance":
            return advance_monitor(engine, settings.artifact_root, *args)
        if kind == "label":
            return ingest_label(engine, *args)
        return read_monitor(engine, *args)
    finally:
        engine.dispose()


@router.post("")
async def create(request: Request, body: MonitorRequest):
    return await bounded_forecast(request, operation, request.app.state.settings, "create", body)


@router.post("/{monitor_id}/advance")
async def advance(request: Request, monitor_id: UUID, body: MonitorAdvance):
    return await bounded_forecast(request, operation, request.app.state.settings, "advance", monitor_id, body)


@router.post("/{monitor_id}/labels")
async def label(request: Request, monitor_id: UUID, body: MonitorLabel):
    return await bounded_forecast(request, operation, request.app.state.settings, "label", monitor_id, body)


@router.get("/{monitor_id}")
async def read(request: Request, monitor_id: UUID):
    return await bounded_forecast(request, operation, request.app.state.settings, "read", monitor_id)
