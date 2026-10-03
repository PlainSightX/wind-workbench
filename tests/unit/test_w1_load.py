"""W1测量工具的失败与证据边界；这里不启动服务、训练或真实provider。"""

import asyncio
import importlib.util
from pathlib import Path

import httpx
import pytest

MODULE = Path(__file__).resolve().parents[2] / "tools/diagnostics/run_w1_load.py"
spec = importlib.util.spec_from_file_location("w1_load", MODULE)
w1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w1)


def test_original_numeric_tolerance_and_time():
    assert not w1.same(0.0, 2e-7)
    assert w1.same(0.0, 2e-7, atol=1e-6)
    assert w1.same("2015-10-04T04:40:00+08:00", "2015-10-03T20:40:00Z")
    assert not w1.same(0.0, float("nan"))


def test_series_must_check_values_not_only_count():
    endpoint = {"expected_comparison": {"status": "comparable"}, "expected_total": 1,
                "expected_rows": [{"actual": 4.0, "left_prediction": 3.0}]}
    body = {"comparison": {"status": "comparable"}, "total": 1, "offset": 0,
            "rows": [{"actual": 4.0, "left_prediction": 9.0}]}
    assert not w1.verify_response(endpoint, body)
    body["rows"][0]["left_prediction"] = 3.0
    assert w1.verify_response(endpoint, body)


def test_engie_oracle_checks_nested_forecast():
    endpoint = {"name": "engie_replay", "expected": {"predictions": [[0.0]]}, "atol": 1e-6}
    assert w1.verify_response(endpoint, {"forecast": {"predictions": [[2e-7]]}, "actual": []})
    assert not w1.verify_response(endpoint, {"forecast": {"predictions": [[2.0]]}})


def test_conservative_overlap():
    observation = {"host_begin": "2026-09-30T00:00:00+00:00", "host_end": "2026-09-30T00:00:02+00:00",
                   "db_now": "2026-09-30T00:00:01+00:00", "tasks": [{"attempt_id": "a",
                   "started_at": "2026-09-30T00:00:04+00:00", "finished_at": "2026-09-30T00:00:10+00:00"}]}
    row = {"started_at": "2026-09-30T00:00:05+00:00", "finished_at": "2026-09-30T00:00:09+00:00"}
    assert w1.overlap(row, [observation]) == "task_execution"
    row["started_at"] = "2026-09-30T00:00:04+00:00"
    assert w1.overlap(row, [observation]) == "boundary_or_clock_uncertain"
    assert w1.overlap(row, []) == "unobserved"


@pytest.mark.parametrize("status,payload,category", [
    (503, {"detail": "forecast_capacity_busy"}, "capacity_rejected"),
    (429, {"detail": "assistant_busy"}, "assistant_rejected"),
    (200, {"value": 2}, "wrong_output"),
])
def test_response_classification(tmp_path, status, payload, category):
    async def check():
        measure = w1.Measurement(tmp_path, {})
        transport = httpx.MockTransport(lambda request: httpx.Response(status, json=payload))
        async with httpx.AsyncClient(base_url="http://fixture", transport=transport) as client:
            row, _ = await measure.request(client, {"name": "probe", "method": "GET", "path": "/", "expected": {"value": 1}}, "idle_warm")
        assert row["category"] == category
        assert bool(measure.stop) is (category == "wrong_output")
        assert len(w1.lines(tmp_path / "requests.jsonl")) == 1
        assert len(w1.lines(tmp_path / "request-intents.jsonl")) == 1
    asyncio.run(check())


def test_malformed_response_still_records_attempt(tmp_path):
    async def check():
        measure = w1.Measurement(tmp_path, {})
        transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"bad-json"))
        async with httpx.AsyncClient(base_url="http://fixture", transport=transport) as client:
            row, _ = await measure.request(client, {"name": "probe", "method": "GET", "path": "/"}, "idle_warm")
        assert row["category"] == "measurement_error" and measure.stop
        assert w1.lines(tmp_path / "requests.jsonl")[0]["finished_at"]
    asyncio.run(check())


def test_failed_initial_database_observer_prevents_submission(tmp_path, monkeypatch):
    async def unavailable():
        raise ConnectionError()
    monkeypatch.setattr(w1, "connect", unavailable)
    async def check():
        measure = w1.Measurement(tmp_path, {"keys": []})
        await measure.observe()
        assert measure.stop == "observer_unavailable" and measure.observer_ready.is_set()
    asyncio.run(check())


def test_unknown_first_answer_blocks_second_question(tmp_path, monkeypatch):
    async def check():
        protocol = {"questions": [{}, {}], "assistant_oracles": [{}, {}], "endpoints": [{}, {}, {}]}
        measure = w1.Measurement(tmp_path, protocol)
        sent = []
        async def unknown(client, endpoint, purpose, **kwargs):
            sent.append(endpoint["name"])
            measure.halt("unknown_inflight_after_transport")
            return {"category": "timeout_or_transport"}, None
        monkeypatch.setattr(measure, "request", unknown)
        await measure.assistant(None)
        assert sent == ["assistant_1"]
    asyncio.run(check())


def test_frozen_protocol_tampering_rejected(tmp_path):
    w1.save(tmp_path / "protocol.json", {"spec": w1.SPEC})
    w1.save(tmp_path / "freeze.json", {"protocol.json": w1.digest({"spec": w1.SPEC})})
    w1.save(tmp_path / "protocol.json", {"spec": {"purpose": "final_evaluation"}})
    with pytest.raises(ValueError, match="frozen_inputs_changed"):
        w1.validate_frozen(tmp_path)
