"""可选宿主入口的只读检查、显式配置与安全失败。"""

import asyncio
from dataclasses import replace
import importlib.util
from pathlib import Path
import socket

import httpx
import pytest
from sqlalchemy import URL

from power_forecast_service.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("serve_inference_unit", ROOT / "tools/dev/serve_inference.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


@pytest.fixture
def settings():
    return Settings(URL.create("postgresql+psycopg", database="unit_only"), Path("sample"), Path("artifacts"),
        assistant_backend="vllm", assistant_vllm_url="http://127.0.0.1:18110", assistant_vllm_model="test-model")


def inspect(monkeypatch, settings, metadata=None, *, code=200):
    seen = []
    def handle(request):
        seen.append((request.method, request.url.path))
        return httpx.Response(code, json=metadata if metadata is not None else {
            "data": [{"id": "test-model", "max_model_len": 32768}]})
    monkeypatch.setattr(entry, "database_revision", lambda config: ["current-head"])
    client = httpx.AsyncClient(transport=httpx.MockTransport(handle), trust_env=False)
    result = asyncio.run(entry.check_configuration(settings, client=client))
    assert client.is_closed
    return result, seen


@pytest.mark.parametrize("mode", ["references", "program", "program_requested"])
def test_check_only_reads_metadata(monkeypatch, settings, mode):
    settings = replace(settings, assistant_response_mode=mode)
    result, seen = inspect(monkeypatch, settings)
    assert seen == [("GET", "/v1/models")]
    assert result["generation_sent"] is False and result["database_revisions"] == ["current-head"]
    assert result["response_mode"] == mode


def test_lora_uses_unique_parent_context(monkeypatch, settings):
    metadata = {"data": [
        {"id": "base", "parent": None, "root": "/weights", "max_model_len": 32768},
        {"id": "test-model", "parent": "base", "root": "/adapter", "max_model_len": None},
    ]}
    result, seen = inspect(monkeypatch, settings, metadata)
    assert result["model"] == "test-model" and result["context_model"] == "base"
    assert result["max_model_len"] == 32768 and seen == [("GET", "/v1/models")]


@pytest.mark.parametrize("cards", [
    [{"id": "test-model", "parent": "missing"}],
    [{"id": "test-model", "parent": "test-model"}],
    [{"id": "test-model", "parent": 1}],
    [{"id": "test-model", "parent": "base"}, {"id": "base", "max_model_len": 32768},
     {"id": "base", "max_model_len": 32768}],
    [{"id": "test-model", "parent": "base"}, {"id": "base", "parent": "other", "max_model_len": 32768}],
    [{"id": "test-model", "parent": "base"}, {"id": "base", "max_model_len": 8192}],
    [{"id": "test-model", "parent": "base", "max_model_len": 8192},
     {"id": "base", "max_model_len": 32768}],
])
def test_lora_cannot_borrow_unverified_or_larger_capacity(monkeypatch, settings, cards):
    with pytest.raises(entry.EntryError, match="inference_(model_identity_mismatch|context_insufficient)"):
        inspect(monkeypatch, settings, {"data": cards})


def test_research_entry_propagates_explicit_profile(monkeypatch, settings):
    monkeypatch.setattr(entry.Settings, "from_environment", lambda: settings)
    monkeypatch.setattr(entry, "check_port", lambda port, url: None)
    seen = []
    async def check(config):
        seen.append(config)
        return {"generation_sent": False}
    monkeypatch.setattr(entry, "check_configuration", check)
    entry.main(["--provider-url", "http://127.0.0.1:18112", "--model", "candidate",
        "--port", "18113", "--response-mode", "program", "--generation-profile", "offline-greedy-v1",
        "--provider-timeout", "60", "--check"])
    assert seen[0].inference_generation_profile == "offline-greedy-v1"
    assert seen[0].inference_provider_timeout == 60
    assert settings.assistant_response_mode == "references"
    assert seen[0].inference_sort_schema_properties is False


def test_research_entry_propagates_explicit_schema_property_order(monkeypatch, settings):
    monkeypatch.setattr(entry.Settings, "from_environment", lambda: settings)
    monkeypatch.setattr(entry, "check_port", lambda port, url: None)
    seen = []
    async def check(config):
        seen.append(config)
        return {"generation_sent": False}
    monkeypatch.setattr(entry, "check_configuration", check)
    entry.main(["--provider-url", "http://127.0.0.1:18112", "--model", "candidate",
        "--port", "18113", "--response-mode", "program_requested", "--sort-schema-properties", "--check"])
    assert seen[0].inference_sort_schema_properties is True


@pytest.mark.parametrize("metadata,code,error", [
    ({"data": [{"id": "wrong", "max_model_len": 32768}]}, 200, "inference_model_identity_mismatch"),
    ({"data": [{"id": "test-model", "max_model_len": 8192}]}, 200, "inference_context_insufficient"),
    ({"data": [{"id": "test-model"}]}, 200, "inference_context_insufficient"),
    ({"data": ["bad"]}, 200, "inference_metadata_invalid"),
    ({"data": []}, 503, "inference_metadata_unavailable"),
])
def test_metadata_failure_is_not_success(monkeypatch, settings, metadata, code, error):
    with pytest.raises(entry.EntryError, match=error):
        inspect(monkeypatch, settings, metadata, code=code)


@pytest.mark.parametrize("url", ["https://127.0.0.1:18110", "http://example.invalid", "http://user:secret@127.0.0.1:18110"])
def test_unapproved_address_fails_before_database(monkeypatch, settings, url):
    monkeypatch.setattr(entry, "database_revision", lambda config: pytest.fail("must not inspect database"))
    with pytest.raises(entry.EntryError, match="inference_configuration_invalid"):
        asyncio.run(entry.check_configuration(replace(settings, assistant_vllm_url=url)))


def test_port_never_reuses_ordinary_api_or_provider(monkeypatch):
    monkeypatch.setenv("WIND_API_PORT", "18000")
    for port in (18000, 18110):
        with pytest.raises(entry.EntryError, match="inference_port_conflict"):
            entry.check_port(port, "http://127.0.0.1:18110")
    with socket.socket() as owned:
        owned.bind(("127.0.0.1", 0))
        owned.listen()
        with pytest.raises(entry.EntryError, match="inference_port_in_use"):
            entry.check_port(owned.getsockname()[1], "http://127.0.0.1:18110")


@pytest.mark.parametrize("check_only", [True, False])
@pytest.mark.parametrize("response_mode", ["references", "program", "program_requested"])
def test_cli_consumes_explicit_provider_and_port(monkeypatch, settings, capsys, check_only, response_mode):
    original = replace(settings, assistant_backend="default")
    monkeypatch.setattr(entry.Settings, "from_environment", lambda: original)
    consumed, started = [], []
    monkeypatch.setattr(entry, "check_port", lambda port, url: None)
    async def check(config):
        consumed.append(config)
        return {"generation_sent": False}
    monkeypatch.setattr(entry, "check_configuration", check)
    monkeypatch.setattr(entry, "serve", lambda config, port: started.append((config, port)))
    args = ["--provider-url", "http://127.0.0.1:18210/v1", "--model", "explicit-model", "--port", "18213", "--capacity", "2"]
    entry.main(args + ["--response-mode", response_mode] + (["--check"] if check_only else []))
    assert consumed[0].assistant_vllm_model == "explicit-model"
    assert consumed[0].assistant_vllm_url == "http://127.0.0.1:18210/v1"
    assert consumed[0].assistant_capacity == 2 and original.assistant_backend == "default"
    assert consumed[0].assistant_response_mode == response_mode
    assert started == ([] if check_only else [(consumed[0], 18213)])
    assert '"port": 18213' in capsys.readouterr().out
