"""服务进程配置；本机和容器使用同一应用账户，不把密码写进日志。"""

import os
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import URL

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    database_url: URL
    data_path: Path
    artifact_root: Path
    broker_url: str = "redis://redis:6379/0"
    lease_seconds: int = 60
    heartbeat_seconds: int = 10
    task_limit_seconds: int = 180
    retry_delay_seconds: int = 10
    dispatch_interval_seconds: int = 3
    redelivery_seconds: int = 30
    max_attempts: int = 3
    assistant_backend: str = "default"
    assistant_vllm_url: str = ""
    assistant_vllm_model: str = ""
    assistant_capacity: int = 1

    @classmethod
    def from_environment(cls) -> "Settings":
        secret = Path(
            os.getenv("WIND_DB_PASSWORD_FILE", str(ROOT / ".local/runtime/app-password.txt"))
        )
        password = secret.read_text(encoding="utf-8").strip()
        if not password:
            raise ValueError("Database password file is empty")
        return cls(
            database_url=URL.create(
                "postgresql+psycopg",
                username="wind_app",
                password=password,
                host=os.getenv("WIND_DB_HOST", "127.0.0.1"),
                port=int(os.getenv("WIND_DB_PORT", "5433")),
                database=os.getenv("WIND_DB_NAME", "wind_workbench"),
            ),
            data_path=Path(
                os.getenv("POWER_FORECAST_DATA", str(ROOT / "data/sample/wind_2019_q1.csv"))
            ),
            artifact_root=Path(os.getenv("WIND_ARTIFACT_ROOT", str(ROOT / "artifacts/tasks"))),
            broker_url=os.getenv("WIND_BROKER_URL", "redis://redis:6379/0"),
            assistant_backend=os.getenv("WIND_ASSISTANT_BACKEND", "default"),
            assistant_vllm_url=os.getenv("WIND_ASSISTANT_VLLM_URL", ""),
            assistant_vllm_model=os.getenv("WIND_ASSISTANT_VLLM_MODEL", ""),
            assistant_capacity=int(os.getenv("WIND_ASSISTANT_CAPACITY", "1")),
        )
