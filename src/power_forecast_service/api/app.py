"""组装 HTTP 应用；连接池归应用生命周期，不在导入或启动时训练。"""

from contextlib import asynccontextmanager
import asyncio
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ..experiments.contracts import TaskConflict
from ..settings import Settings
from ..storage.database import make_async_engine
from .routes import router
from .forecasts import router as forecast_router
from .workbench import router as workbench_router
from .engie import router as engie_router
from .engie_monitor import router as monitor_router
from .assistant import router as assistant_router


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings or Settings.from_environment()
        engine = make_async_engine(app.state.settings)
        app.state.sessions = async_sessionmaker(engine, expire_on_commit=False)
        app.state.forecast_gate = asyncio.Semaphore(1)
        app.state.forecast_tasks = set()
        from ..assistant.workflow import Assistant
        app.state.assistant = Assistant(app.state.sessions)
        app.state.assistant_tasks = set()
        try:
            yield
        finally:
            # 客户端断开不代表预测线程结束；正常停机等待这些只读工作完成。
            await asyncio.gather(*app.state.forecast_tasks, return_exceptions=True)
            await asyncio.gather(*app.state.assistant_tasks, return_exceptions=True)
            await asyncio.to_thread(app.state.assistant.close)
            await engine.dispose()

    app = FastAPI(title="Wind Experiment Workbench", version="0.2.0", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and request.method not in {"GET", "HEAD", "OPTIONS"}:
            expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
            if origin != expected:
                return JSONResponse(
                    status_code=403, content={"detail": "cross_origin_write_forbidden"}
                )
        return await call_next(request)

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request: Request, exc: SQLAlchemyError):
        # DB 异常中可能含连接信息，不原样回传给浏览器。
        return JSONResponse(status_code=503, content={"detail": "database_unavailable"})

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # 不回显原始NaN/Infinity输入，否则错误响应本身可能再次JSON序列化失败。
        return JSONResponse(status_code=422, content={"detail": [
            {"loc": list(item["loc"]), "type": item["type"], "msg": item["msg"]}
            for item in exc.errors()
        ]})

    @app.exception_handler(TaskConflict)
    async def conflict(request: Request, exc: TaskConflict):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    # 具体浏览路由先于 /runs/{run_id} 注册，避免把 compare-series 误当UUID。
    app.include_router(workbench_router)
    app.include_router(router)
    app.include_router(forecast_router)
    app.include_router(engie_router)
    app.include_router(monitor_router)
    app.include_router(assistant_router)
    static = Path(__file__).resolve().parents[1] / "web_static"
    app.mount("/assets", StaticFiles(directory=static / "assets", check_dir=False), name="assets")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(static / "index.html", headers={"Cache-Control": "no-cache"})

    return app
