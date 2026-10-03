"""外部目录初始化与凭据冲突；测试不触发下载或 provider。"""

import runpy

import pytest

from power_forecast_service.assistant.configuration import provider_key
from power_forecast_service.assistant.contracts import AssistantError
from power_forecast_service.settings import ROOT


def test_configure_uses_explicit_runtime_and_survives_env_removal(tmp_path, monkeypatch):
    monkeypatch.setenv("WIND_ASSISTANT_RUNTIME", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "local-test-placeholder")
    configure = runpy.run_path(str(ROOT / "tools/diagnostics/assistant_probe.py"))["save_key"]
    assert configure() == {"provider_configured": True, "network_requests": 0}
    assert not (tmp_path / "embedding").exists()
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    assert provider_key() == "local-test-placeholder"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "different-test-placeholder")
    with pytest.raises(ValueError, match="provider_key_conflict"):
        configure()
    assert (tmp_path / "provider-key.txt").read_text() == "local-test-placeholder"


def test_missing_configuration_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("WIND_ASSISTANT_RUNTIME", str(tmp_path))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(AssistantError, match="provider_not_configured"):
        provider_key()


@pytest.mark.parametrize("operation", ["chmod", "fsync", "link"])
def test_key_prepare_failure_never_publishes_partial_key(tmp_path, monkeypatch, operation):
    monkeypatch.setenv("WIND_ASSISTANT_RUNTIME", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-atomic-test-key")
    configure = runpy.run_path(str(ROOT / "tools/diagnostics/assistant_probe.py"))["save_key"]
    os = configure.__globals__["os"]
    original = getattr(os, operation)
    def fail(*args, **kwargs):
        raise OSError("fake preparation failure")
    monkeypatch.setattr(os, operation, fail)
    with pytest.raises(OSError, match="fake preparation failure"):
        configure()
    assert not (tmp_path / "provider-key.txt").exists()
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(os, operation, original)
    assert configure()["provider_configured"]
    assert (tmp_path / "provider-key.txt").read_text() == "fake-atomic-test-key"
