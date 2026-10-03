"""ENGIE独立HTTP合同；复用预测容量，不改变Q1的运行与评分对象。"""

from uuid import UUID
import asyncio

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select

from ..forecasting.engie_service_contract import EngieForecastRequest, EngieReplayRequest, EngieDeliveryRequest
from ..storage.models import ImportedArtifact, ImportedRun
from .forecasts import bounded_forecast

router = APIRouter(prefix="/engie")


@router.get("/imports")
async def imports(request: Request):
    async with request.app.state.sessions() as session:
        rows = (await session.execute(select(ImportedRun, ImportedArtifact).join(
            ImportedArtifact, ImportedArtifact.import_id == ImportedRun.id
        ).where(ImportedArtifact.status == "ready").order_by(ImportedRun.quarter, ImportedArtifact.family))).all()
        grouped = {}
        for run, item in rows:
            if run.id not in grouped:
                grouped[run.id] = {"import_id": str(run.id), "quarter": run.quarter,
                    **{key: run.manifest[key] for key in ("protocol_version", "source_sha256", "training_label_available")},
                    "origin": "imported_offline", "models": [],
                    "families": run.manifest.get("families", ["persistence", "ridge", "lightgbm"]),
                    "scope": run.manifest.get("scope", "development"),
                    "evaluation": run.manifest.get("evaluation")}
            grouped[run.id]["models"].append({"artifact_id": str(item.id), "family": item.family,
                **{key: item.manifest[key] for key in ("model_version", "metrics", "coverage")}})
        return [value for value in grouped.values()
                if {model["family"] for model in value["models"]} == set(value["families"])]


async def registration(request, artifact_id):
    async with request.app.state.sessions() as session:
        item = await session.get(ImportedArtifact, artifact_id)
        if item is None:
            raise HTTPException(404, "engie_artifact_not_found")
        if item.status != "ready":
            raise HTTPException(409, "engie_artifact_not_ready")
        run = await session.get(ImportedRun, item.import_id)
        return {"id": item.id, "import_id": run.id, "family": item.family,
                "path": item.path, "manifest": item.manifest, "run": run.manifest}


def operate(kind, root, item, value=None):
    # 数值库只在受限线程使用，不在API导入时连接服务或加载模型。
    from ..forecasting import engie_predictor

    if kind == "windows":
        return engie_predictor.windows(root, item)
    if kind == "forecast":
        return engie_predictor.forecast(root, item, value)
    return engie_predictor.replay(root, item, value)


@router.get("/artifacts/{artifact_id}/windows")
async def windows(request: Request, artifact_id: UUID):
    item = await registration(request, artifact_id)
    return await bounded_forecast(request, operate, "windows", request.app.state.settings.artifact_root, item)


@router.post("/forecasts")
async def forecast(request: Request, body: EngieForecastRequest):
    item = await registration(request, body.artifact_id)
    return await bounded_forecast(request, operate, "forecast", request.app.state.settings.artifact_root, item, body)


@router.post("/replays")
async def replay(request: Request, body: EngieReplayRequest):
    item = await registration(request, body.artifact_id)
    return await bounded_forecast(request, operate, "replay", request.app.state.settings.artifact_root, item, body.issue_time)


def delivery_operation(settings, kind, *args):
    from ..storage.database import make_sync_engine
    from ..experiments.engie_delivery import deliver, read_delivery

    engine = make_sync_engine(settings)
    try:
        return deliver(engine, settings.artifact_root, *args) if kind == "deliver" else read_delivery(engine, *args)
    finally:
        engine.dispose()


@router.post("/deliveries")
async def delivery(request: Request, body: EngieDeliveryRequest):
    item = await registration(request, body.artifact_id)
    # 整段登记、推理、终结属于受生命周期管理的任务；请求断开不会跳过终结。
    return await bounded_forecast(request, delivery_operation, request.app.state.settings, "deliver", item, body)


@router.get("/deliveries/{delivery_id}")
async def delivery_read(request: Request, delivery_id: UUID):
    result = await asyncio.to_thread(delivery_operation, request.app.state.settings, "read", delivery_id)
    if result is None:
        raise HTTPException(404, "engie_delivery_not_found")
    return result
