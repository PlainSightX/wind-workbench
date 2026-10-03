"""面向页面的浏览与回放；已有业务仍由原实验服务和预测函数负责。"""

import asyncio
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy import select

from ..experiments.comparison import compare_results
from ..forecasting.replay import ReplayRequest, prepare_replay, replay_windows
from ..storage.model_packages import PackageRegistration
from ..storage.models import ModelArtifact, Run, Task
from .forecasts import bounded_forecast, forecast_from_package

router = APIRouter()


@router.get("/runs")
async def runs(request: Request, limit: int = Query(20, ge=1, le=100),
               offset: int = Query(0, ge=0)):
    async with request.app.state.sessions() as session:
        # 分页摘要不搬运每个run数千行评分数组；仅投影界面需要的字段。
        rows = (await session.execute(select(
            Run.id, Run.task_id, Run.created_at,
            Task.spec["training_policy"].astext,
            Run.result["model_set"], Run.result["metrics"],
            Run.result.has_key("scoring"),
            Run.result["purpose"].astext,
        ).join(Task, Task.id == Run.task_id).order_by(Run.created_at.desc(), Run.id)
            .limit(limit).offset(offset))).all()
        return [{"run_id": row[0], "task_id": row[1], "created_at": row[2],
                 "training_policy": row[3] or "auto_early_stopping",
                 "models": row[4] or list(row[5] or {}), "has_scoring": row[6],
                 "purpose": row[7] or "development"} for row in rows]


@router.get("/runs/compare-series")
async def compare_series(request: Request, left_run_id: UUID, right_run_id: UUID,
                         left_model: str = "persistence", right_model: str = "hist_gradient_boosting",
                         limit: int = Query(288, ge=1, le=288), offset: int = Query(0, ge=0)):
    async with request.app.state.sessions() as session:
        left, right = await session.get(Run, left_run_id), await session.get(Run, right_run_id)
        if left is None or right is None:
            raise HTTPException(404, "run_not_found")
    return await asyncio.to_thread(comparison_series, left.id, left.result, left_model,
                                   right.id, right.result, right_model, limit, offset)


def comparison_series(left_id, left_result, left_model, right_id, right_result, right_model,
                      limit, offset):
    comparison = compare_results(left_id, left_result, left_model, right_id, right_result, right_model)
    data = {"comparison": comparison, "rows": [], "total": 0, "offset": offset}
    if comparison.status == "comparable":
        a, b = left_result["scoring"]["rows"], right_result["scoring"]["rows"]
        data.update(total=len(a), rows=[
            {"cutoff": x["cutoff"], "target_time": x["target_time"], "actual": x["actual"],
             "left_prediction": x["predictions"][left_model],
             "right_prediction": y["predictions"][right_model]}
            for x, y in zip(a[offset:offset + limit], b[offset:offset + limit], strict=True)
        ])
    return data


async def registered_run(request, artifact_id):
    async with request.app.state.sessions() as session:
        item = await session.get(ModelArtifact, artifact_id)
        if item is None:
            raise HTTPException(404, "artifact_not_found")
        if item.status != "ready":
            raise HTTPException(409, "artifact_not_ready")
        run = await session.get(Run, item.run_id)
        if run is None:
            raise HTTPException(409, "replay_run_missing")
        return PackageRegistration(artifact_id=item.id, model_key=item.model_key, path=item.path,
                                   manifest_sha256=item.manifest_sha256, manifest=item.manifest), {
            "run_id": run.id, "task_id": run.task_id, "attempt_id": run.attempt_id,
            "result": run.result,
        }


@router.get("/artifacts/{artifact_id}/replay-windows")
async def windows(request: Request, artifact_id: UUID):
    registration, run = await registered_run(request, artifact_id)
    return await bounded_forecast(request, replay_windows,
                                  request.app.state.settings.artifact_root, registration, run)


def replay_prediction(root, registration, run, cutoff):
    body, actual = prepare_replay(root, registration, run, cutoff)
    forecast = forecast_from_package(root, registration, body)
    return {"mode": "final_historical_replay" if run["result"]["purpose"] == "final_evaluation" else "development_historical_replay",
            "forecast": forecast, "actual": actual,
            "history": [row.model_dump() for row in body.observations]}


@router.post("/replays")
async def replay(request: Request, body: ReplayRequest):
    registration, run = await registered_run(request, body.artifact_id)
    return await bounded_forecast(request, replay_prediction,
                                  request.app.state.settings.artifact_root,
                                  registration, run, body.cutoff)
