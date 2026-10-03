"""配置消费回归：使用合成配置，不读密码、不启动Docker、不创建实验。"""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
PWSH = shutil.which("pwsh")
CONFIG = {
    "name": "wind-stage-20260922",
    "secrets": {
        "app_password": {"file": "D:/fixture-current/app-password.txt"},
        "postgres_password": {"file": "D:/fixture-current/postgres-password.txt"},
    },
    "services": {
        "postgres": {"ports": [{"target": 5432, "published": "15433"}]},
        "api": {"ports": [{"target": 8000, "published": "18000"}], "volumes": [
            {"type": "bind", "target": "/app/assistant-runtime", "source": "D:/fixture-current/assistant"}]},
    },
    "volumes": {"postgres_data": {"name": "fixture_existing_volume"}},
}


def powershell(code):
    if PWSH is None:
        pytest.skip("Windows PowerShell entrypoint requires pwsh")
    result = subprocess.run(
        [PWSH, "-NoProfile", "-Command", code], cwd=ROOT,
        capture_output=True, text=True, encoding="utf-8",
    )
    return result


def config_code(config=CONFIG):
    return "$cfg = '" + json.dumps(config) + "' | ConvertFrom-Json; "


def test_host_child_consumes_compose_values_instead_of_stale_environment():
    result = powershell(
        ". ./tools/dev/compose-context.ps1; " + config_code()
        + "$env:WIND_DB_PORT='5433'; $env:WIND_RUNTIME_ROOT='D:/old'; "
        + "Set-WindHostEnvironment $cfg; "
        + "& pwsh -NoProfile -Command '[pscustomobject]@{port=$env:WIND_DB_PORT; "
        + "secret=$env:WIND_DB_PASSWORD_FILE; admin=$env:WIND_DB_ADMIN_PASSWORD_FILE; "
        + "http=$env:WIND_TEST_BASE_URL; assistant=$env:WIND_ASSISTANT_RUNTIME} | ConvertTo-Json -Compress'"
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["port"] == "15433"
    assert data["http"] == "http://127.0.0.1:18000"
    assert Path(data["assistant"]) == Path(CONFIG["services"]["api"]["volumes"][0]["source"])
    assert Path(data["secret"]) == Path(CONFIG["secrets"]["app_password"]["file"])
    assert Path(data["admin"]) == Path(CONFIG["secrets"]["postgres_password"]["file"])


def fake_docker(config):
    return config_code(config) + """
function docker {
    $global:LASTEXITCODE = 0
    $command = $args -join ' '
    if ($command -like '*config --format json*') { $cfg | ConvertTo-Json -Depth 8; return }
    if ($command -like 'ps -a*') { return }
    if ($command -like 'volume ls*') { 'fixture_existing_volume'; return }
    if ($command -like '*up *') { 'UNSAFE_START_MARKER'; return }
    throw "Unexpected Docker operation: $command"
}
function Get-NetTCPConnection { return }
function Test-Path { return $false }
"""


def test_retired_identity_never_reaches_startup_command():
    config = dict(CONFIG, name="wind-workbench")
    result = powershell(fake_docker(config) + "& ./tools/dev/start-infra.ps1")
    assert result.returncode != 0
    assert "retired" in result.stderr
    assert "UNSAFE_START_MARKER" not in result.stdout


def test_existing_volume_missing_actual_password_never_generates_or_starts():
    result = powershell(fake_docker(CONFIG) + "& ./tools/dev/start-infra.ps1")
    assert result.returncode != 0
    assert "Existing database volume requires" in result.stderr
    assert "UNSAFE_START_MARKER" not in result.stdout


def test_check_service_missing_uv_fails_before_any_docker_command():
    result = powershell("""
function Get-Command { return $null }
function docker { 'UNSAFE_DOCKER_MARKER'; throw 'Docker must not run without uv' }
& ./tools/dev/check-service.ps1 -TestPath tests/unit
""")
    assert result.returncode != 0
    assert "uv is required" in result.stderr
    assert "UNSAFE_DOCKER_MARKER" not in result.stdout


def diagnostic_module():
    spec = importlib.util.spec_from_file_location("wind_infra_probe", ROOT / "tools/diagnostics/check_infra.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_diagnostic_uses_effective_config(monkeypatch):
    module = diagnostic_module()
    monkeypatch.setenv("WIND_DB_PORT", "5433")
    monkeypatch.setenv("WIND_RUNTIME_ROOT", "D:/old")
    monkeypatch.setattr(module.shutil, "which", lambda _: "docker")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps(CONFIG)))
    secret, port = module.compose_connection()
    assert secret == Path(CONFIG["secrets"]["app_password"]["file"])
    assert port == 15433


def test_direct_diagnostic_rejects_retired_identity(monkeypatch):
    module = diagnostic_module()
    monkeypatch.setattr(module.shutil, "which", lambda _: "docker")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps(dict(CONFIG, name="wind-workbench"))))
    with pytest.raises(RuntimeError, match="Retired"):
        module.compose_connection()


@pytest.mark.parametrize("source, expected", [("D:/fixture-old/app-password.txt", "False"), ("D:/fixture-current/app-password.txt", "True")])
@pytest.mark.parametrize("target", ["app_password", "/run/secrets/app_password"])
def test_actual_mount_must_match_effective_secret_path(source, expected, target):
    config = json.loads(json.dumps(CONFIG))
    config["services"]["api"]["secrets"] = [{"source": "app_password", "target": target}]
    mount_json = json.dumps([{"Source": source, "Destination": "/run/secrets/app_password"}])
    result = powershell(
        ". ./tools/dev/compose-context.ps1; " + config_code(config)
        + "$mounts = '" + mount_json + "' | ConvertFrom-Json; Test-WindSecretMounts $cfg 'api' $mounts"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected
