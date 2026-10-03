"""各进程入口的导入边界；新进程避免其他测试已加载模块掩盖隐式依赖。"""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("entry", ["api", "worker"])
def test_definitions_import_without_credentials_or_service_connections(tmp_path, entry):
    script = """
import os
import socket
import sys
from pathlib import Path

def forbidden(*args, **kwargs):
    raise AssertionError("Import must not read credentials or connect to a service")

read_text = Path.read_text
def guarded_read(path, *args, **kwargs):
    if path == Path(os.environ["WIND_DB_PASSWORD_FILE"]):
        forbidden()
    return read_text(path, *args, **kwargs)

Path.read_text = guarded_read
socket.socket.connect = forbidden
from power_forecast_service.settings import Settings
Settings.from_environment = classmethod(forbidden)

if sys.argv[1] == "api":
    from power_forecast_service.main import app
    assert set(app.openapi()["paths"]) == {
        "/health", "/datasets", "/experiments", "/tasks", "/tasks/{task_id}",
        "/runs/{run_id}", "/runs/compare", "/artifacts", "/forecasts",
        "/runs", "/runs/compare-series", "/artifacts/{artifact_id}/replay-windows", "/replays",
        "/engie/imports", "/engie/artifacts/{artifact_id}/windows", "/engie/forecasts", "/engie/replays",
        "/engie/deliveries", "/engie/deliveries/{delivery_id}",
        "/assistant/answers", "/assistant/documents/{document_id}"
    }
    for dependency in ("numpy", "pandas", "sklearn", "power_forecast_service.forecasting.pipeline"):
        assert dependency not in sys.modules, dependency
else:
    from power_forecast_service.jobs.worker import app, execute_experiment
    from power_forecast_service.jobs.dispatcher import dispatch_once
    assert "wind.execute" in app.tasks
    assert callable(execute_experiment) and callable(dispatch_once)
print("isolated import passed")
"""
    env = {**os.environ, "WIND_DB_PASSWORD_FILE": str(tmp_path / "missing-password.txt")}
    result = subprocess.run(
        [sys.executable, "-c", script, entry],
        check=False,
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "isolated import passed" in result.stdout
