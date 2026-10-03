"""当前实验接口；写操作交给实验服务，查询直接读取持久对象。"""

from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query, Request
from sqlalchemy import select, text

from ..experiments.contracts import DATASET_ID, ExperimentRequest, TaskReceipt
from ..experiments.comparison import ComparisonResult, compare_results
from ..experiments.service import receipt, submit_experiment
from ..storage.models import Attempt, Run, Task

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    async with request.app.state.sessions() as session:
        await session.execute(text("SELECT 1"))
    return {"status": "ok", "service": "experiment-tasks", "training_on_read": False}


@router.get("/datasets")
async def datasets():
    return [
        {
            "dataset_id": DATASET_ID,
            "purpose": "development",
            "test_usage": "historically_exposed_not_blind",
        }
    ]


@router.post("/experiments", response_model=TaskReceipt, status_code=202)
async def submit(
    request: Request,
    body: ExperimentRequest,
    idempotency_key: str = Header(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$"),
):
    async with request.app.state.sessions() as session:
        try:
            return await submit_experiment(
                session,
                request=body,
                idempotency_key=idempotency_key,
                settings=request.app.state.settings,
            )
        except OSError as exc:
            raise HTTPException(409, "dataset_unavailable") from exc


@router.get("/tasks")
async def tasks(
    request: Request, limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)
):
    async with request.app.state.sessions() as session:
        found = (await session.execute(
            select(Task, Run.id).outerjoin(Run, Run.task_id == Task.id)
            .order_by(Task.created_at.desc(), Task.id).limit(limit).offset(offset)
        )).all()
        return [{**receipt(task).model_dump(), "created_at": task.created_at,
                 "updated_at": task.updated_at, "error_code": task.error_code,
                 "training_policy": task.spec.get("training_policy", "auto_early_stopping"),
                 "run_id": run_id} for task, run_id in found]


@router.get("/tasks/{task_id}")
async def task_detail(request: Request, task_id: UUID):
    async with request.app.state.sessions() as session:
        task = await session.get(Task, task_id)
        if task is None:
            raise HTTPException(404, "task_not_found")
        attempts = (
            await session.scalars(
                select(Attempt).where(Attempt.task_id == task_id).order_by(Attempt.number)
            )
        ).all()
        run = await session.scalar(select(Run).where(Run.task_id == task_id))
        return {
            **receipt(task).model_dump(),
            "spec": task.spec,
            "attempt_count": task.attempt_count,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "error_code": task.error_code,
            "attempts": [
                {
                    "attempt_id": item.id,
                    "number": item.number,
                    "status": item.status,
                    "error_code": item.error_code,
                }
                for item in attempts
            ],
            "result_url": f"/runs/{run.id}" if run else None,
        }


@router.get("/runs/compare", response_model=ComparisonResult)
async def compare_runs(
    request: Request, left_run_id: UUID, right_run_id: UUID,
    left_model: str = Query("persistence", min_length=1, max_length=100),
    right_model: str = Query("hist_gradient_boosting", min_length=1, max_length=100),
):
    """选择运行与模型，只读已有结果；历史证据不足时返回不可比原因。"""
    async with request.app.state.sessions() as session:
        left = await session.get(Run, left_run_id)
        right = await session.get(Run, right_run_id)
        if left is None or right is None:
            raise HTTPException(404, "run_not_found")
        return compare_results(left.id, left.result, left_model, right.id, right.result, right_model)


@router.get("/runs/{run_id}")
async def run_detail(request: Request, run_id: UUID):
    async with request.app.state.sessions() as session:
        run = await session.get(Run, run_id)
        if run is None:
            raise HTTPException(404, "run_not_found")
        return {
            "run_id": run.id,
            "task_id": run.task_id,
            "attempt_id": run.attempt_id,
            "result": run.result,
            "artifact_sha256": run.artifact_sha256,
        }
