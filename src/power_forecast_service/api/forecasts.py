"""预测与工件查询；只选数据库已登记版本，重计算/加载不占用异步事件循环。"""

import asyncio
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy import select

from ..forecasting.contracts import ForecastRequest, ForecastResponse
from ..storage.model_packages import PackageError, PackageRegistration
from ..storage.models import ModelArtifact

router = APIRouter()


def public_artifact(item: ModelArtifact) -> dict:
    manifest = item.manifest
    return {
        "artifact_id": item.id, "run_id": item.run_id, "model_key": item.model_key,
        "status": item.status, "created_at": item.created_at,
        "model_version": manifest["model_version"], "horizon_minutes": manifest["horizon_minutes"],
        "input_contract": manifest["input_contract"], "unit": manifest["unit"],
        "clock": manifest["clock"], "training_label_end": manifest["training_label_end"],
        "feature_contract_version": manifest["feature_contract_version"],
        "manifest_sha256": item.manifest_sha256,
    }


@router.get("/artifacts")
async def artifacts(request: Request, run_id: UUID | None = None,
                    limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    query = select(ModelArtifact).where(ModelArtifact.status == "ready")
    if run_id is not None:
        query = query.where(ModelArtifact.run_id == run_id)
    async with request.app.state.sessions() as session:
        rows = (await session.scalars(query.order_by(
            ModelArtifact.created_at.desc(), ModelArtifact.id).limit(limit).offset(offset))).all()
        return [public_artifact(item) for item in rows]


def forecast_from_package(root, registration, body):
    # 训练库和MLflow只在实际预测线程需要时导入，不在API启动时加载/拟合。
    from ..forecasting.bundles import predict_package

    manifest, prediction = predict_package(root, registration, body)
    cutoff = body.observations[-1].timestamp
    return ForecastResponse(
        artifact_id=registration.artifact_id, run_id=manifest.run_id,
        model_key=manifest.model_key, model_version=manifest.model_version,
        cutoff=cutoff, target_time=prediction["target_time"], prediction=prediction["prediction"],
        unit=manifest.unit, clock=manifest.clock, training_label_end=manifest.training_label_end,
        after_training_cutoff=cutoff > manifest.training_label_end,
    )


@router.post("/forecasts", response_model=ForecastResponse)
async def forecast(request: Request, body: ForecastRequest):
    async with request.app.state.sessions() as session:
        item = await session.get(ModelArtifact, body.artifact_id)
        if item is None:
            raise HTTPException(404, "artifact_not_found")
        if item.status != "ready":
            raise HTTPException(409, "artifact_not_ready")
        registration = PackageRegistration(
            artifact_id=item.id, model_key=item.model_key, path=item.path,
            manifest_sha256=item.manifest_sha256, manifest=item.manifest,
        )
    return await bounded_forecast(request, forecast_from_package,
                                  request.app.state.settings.artifact_root, registration, body)


async def bounded_forecast(request: Request, operation, *args):
    """显式预测和历史回放共享容量；请求断开不能释放仍在工作的线程额度。"""
    # 单机有界计算：占满时明确拒绝，不让任意数量加载同时耗尽线程/内存。
    gate = request.app.state.forecast_gate
    if gate.locked():
        raise HTTPException(503, "forecast_capacity_busy")
    await gate.acquire()

    async def run_prediction():
        try:
            return await asyncio.to_thread(operation, *args)
        except PackageError as exc:
            raise HTTPException(409, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(503, "artifact_storage_unavailable") from exc
        finally:
            gate.release()

    # 请求取消只停止等待，不释放仍在运行的线程所占额度；生命周期保留任务强引用。
    work = asyncio.create_task(run_prediction())
    request.app.state.forecast_tasks.add(work)

    def finished(task):
        request.app.state.forecast_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    work.add_done_callback(finished)
    return await asyncio.shield(work)
