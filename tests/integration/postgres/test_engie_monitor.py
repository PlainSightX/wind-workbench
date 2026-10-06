"""真实PG、原五模型与普通API：延迟评分、回滚、重复和恢复。"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from time import monotonic, sleep
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from power_forecast_service.api.app import create_app
from power_forecast_service.experiments.engie_monitor import (advance_monitor, create_monitor,
                                                             ingest_label, read_monitor)
from power_forecast_service.forecasting.engie_monitor_contract import MonitorAdvance, MonitorLabel, MonitorRequest
from power_forecast_service.forecasting.engie_predictor import MonitorReplaySource
from power_forecast_service.forecasting.engie_service_contract import HORIZONS, ROSTER
from power_forecast_service.serve import make_event_loop
from power_forecast_service.settings import ROOT
from power_forecast_service.storage.engie_final_imports import prepare_import
from power_forecast_service.storage.engie_imports import publish_import
from power_forecast_service.storage.model_packages import PackageError
from power_forecast_service.storage.models import (EngieMonitor, EngieMonitorIssue,
                                                   EngieMonitorObservation, EngieMonitorResidual)

pytestmark = pytest.mark.original_replay
START = datetime(2015, 1, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def prepared(config):
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    source = os.getenv("WIND_ENGIE_FINAL_SOURCE")
    if not source:
        pytest.fail("Set WIND_ENGIE_FINAL_SOURCE to prepared final-2015 attachment")
    with pytest.MonkeyPatch.context() as patch:
        def forbidden(*args, **kwargs):
            raise AssertionError("monitor_must_not_train")
        for cls in (LGBMRegressor, Ridge, StandardScaler):
            patch.setattr(cls, "fit", forbidden)
        return prepare_import(Path(source), config.artifact_root)[0]


@pytest.fixture
def case(engine, prepared):
    publish_import(engine, prepared)
    run, artifacts = prepared[0]
    c = next(item for item in artifacts if item["family"] == "persistence")
    s = next(item for item in artifacts if item["family"] == "lightgbm_l1_shrink")
    body = MonitorRequest(request_key=uuid4().hex, champion_id=c["id"], shadow_id=s["id"],
                          start=START, end=START + timedelta(hours=2))
    return body, {**c, "run": run["manifest"]}


def progress(engine, config, identity, through, max_steps=144):
    return advance_monitor(engine, config.artifact_root, identity,
                           MonitorAdvance(through=through, max_steps=max_steps))


def api(config):
    return TestClient(create_app(config), base_url="http://127.0.0.1",
                      backend_options={"loop_factory": make_event_loop})


def test_prediction_before_truth_and_first_arrival(engine, config, case, monkeypatch):
    body, registration = case
    identity = create_monitor(engine, body)
    original = MonitorReplaySource.observation
    calls = []
    def inspected(source, target, *, clock):
        assert target + timedelta(minutes=20) <= clock
        calls.append((target, clock))
        return original(source, target, clock=clock)
    monkeypatch.setattr(MonitorReplaySource, "observation", inspected)
    report = progress(engine, config, identity, START + timedelta(minutes=20))
    assert not calls and report["counts"]["attempted_issues"] == 3
    assert all(item["scored_pairs"] == 0 for item in report["horizons"])
    report = progress(engine, config, identity, START + timedelta(minutes=30))
    assert calls == [(START + timedelta(minutes=10), START + timedelta(minutes=30))]
    assert [item["scored_pairs"] for item in report["horizons"]] == [1, 0, 0, 0, 0, 0]
    with pytest.raises(PackageError, match="not_arrived"):
        original(MonitorReplaySource(config.artifact_root, registration), START, clock=START)
    assert create_monitor(engine, body) == identity
    with pytest.raises(PackageError, match="request_conflict"):
        create_monitor(engine, body.model_copy(update={"window_issues": 72}))


def test_actual_model_http_restart_and_cli_consumer(config, engine, case):
    body, _ = case
    spec = importlib.util.spec_from_file_location("monitor_cli", ROOT / "tools/dev/replay_engie_monitor.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    with api(config) as client:
        result = client.post("/engie/monitors", json=body.model_dump(mode="json"))
        assert result.status_code == 200, result.text
        identity = result.json()["id"]
        first = client.post(f"/engie/monitors/{identity}/advance", json={
            "through": result.json()["finish_time"], "max_steps": 3})
        assert first.status_code == 200 and first.json()["counts"]["attempted_issues"] == 3
    # 新应用、新连接池，不保留累计量；普通CLI消费者由同一个HTTP合同继续。
    with api(config) as client:
        report = cli.run(client, champion=str(body.champion_id), shadow=str(body.shadow_id),
                         start=body.start.isoformat(), end=body.end.isoformat(), key=body.request_key,
                         step_limit=4)
        saved = client.get(f"/engie/monitors/{identity}").json()
        assert {key: value for key, value in report.items() if key != "steps_committed"} == saved
        retry = cli.run(client, champion=str(body.champion_id), shadow=str(body.shadow_id),
                        start=body.start.isoformat(), end=body.end.isoformat(), key=body.request_key)
        assert retry == saved
    assert saved["replay_complete"] and saved["counts"]["planned_issues"] == 12
    assert saved["counts"]["outputs"] == {"champion": 12, "shadow": 12}
    assert all(item["scored_pairs"] == 12 and item["pending_labels"] == 0 for item in saved["horizons"])
    assert all(item["rolling_pairs"] == 12 for item in saved["horizons"])
    # 用保存的预测和实况独立重算，不只检查响应字段有值。
    with Session(engine) as session:
        issues = session.scalars(select(EngieMonitorIssue).where(EngieMonitorIssue.monitor_id == UUID(identity))).all()
        observations = {item.target_time: item for item in session.scalars(select(EngieMonitorObservation).where(
            EngieMonitorObservation.monitor_id == UUID(identity)))}
        for index, horizon in enumerate(HORIZONS):
            for role in ("champion", "shadow"):
                errors = [item.predictions[role]["farm_predictions"][index] - sum(
                    observations[item.issue_time + timedelta(minutes=horizon)].turbines.values()) for item in issues]
                assert saved["horizons"][index][role]["mae_kw"] == pytest.approx(sum(abs(e) for e in errors) / 12)
                assert saved["horizons"][index][role]["bias_kw"] == pytest.approx(sum(errors) / 12)


def test_missing_truth_late_out_of_order_duplicate_conflict(engine, config, case, monkeypatch):
    body, registration = case
    identity = create_monitor(engine, body)
    source = MonitorReplaySource(config.artifact_root, registration)
    original = MonitorReplaySource.observation
    missing = {START + timedelta(minutes=10), START + timedelta(minutes=20)}
    def partial(source, target, *, clock):
        truth = original(source, target, clock=clock)
        if target in missing:
            truth[ROSTER[0]] = None
        return truth
    monkeypatch.setattr(MonitorReplaySource, "observation", partial)
    report = progress(engine, config, identity, START + timedelta(minutes=50))
    assert report["horizons"][0]["due_missing_labels"] == 2
    before = report["horizons"][0]["scored_pairs"]
    for target in sorted(missing, reverse=True):
        label = MonitorLabel(target_time=target, available_at=START + timedelta(minutes=50),
                             turbines=original(source, target, clock=START + timedelta(minutes=50)))
        report = ingest_label(engine, identity, label)
        assert ingest_label(engine, identity, label) == report
        bad = dict(label.turbines)
        bad[ROSTER[0]] += 1
        with pytest.raises(PackageError, match="truth_conflict"):
            ingest_label(engine, identity, label.model_copy(update={"turbines": bad}))
        assert read_monitor(engine, identity) == report
    assert report["horizons"][0]["scored_pairs"] == before + 2
    assert report["horizons"][0]["due_missing_labels"] == 0
    early = MonitorLabel(target_time=START + timedelta(minutes=60),
                         available_at=START + timedelta(minutes=80), turbines=dict.fromkeys(ROSTER, 1.))
    with pytest.raises(PackageError, match="not_arrived"):
        ingest_label(engine, identity, early)


def test_mid_step_failure_rolls_back_and_restart_recomputes_same_metrics(engine, config, case, monkeypatch):
    body, _ = case
    identity = create_monitor(engine, body)
    original = MonitorReplaySource.observation
    def failure(source, target, *, clock):
        raise RuntimeError("injected_after_prediction_before_step_commit")
    progress(engine, config, identity, START + timedelta(minutes=20))
    monkeypatch.setattr(MonitorReplaySource, "observation", failure)
    with pytest.raises(RuntimeError, match="injected_after_prediction"):
        progress(engine, config, identity, START + timedelta(minutes=30))
    report = read_monitor(engine, identity)
    assert report["processed_until"] == (START + timedelta(minutes=20)).isoformat()
    assert report["counts"]["attempted_issues"] == 3
    monkeypatch.setattr(MonitorReplaySource, "observation", original)
    resumed = progress(engine, config, identity, START + timedelta(minutes=80))
    uninterrupted = create_monitor(engine, body.model_copy(update={"request_key": uuid4().hex}))
    clean = progress(engine, config, uninterrupted, START + timedelta(minutes=80))
    assert resumed["counts"] == clean["counts"] and resumed["horizons"] == clean["horizons"]


def test_concurrent_advances_and_residual_identity(engine, config, case):
    body, _ = case
    identity = create_monitor(engine, body)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(progress, engine, config, identity, START + timedelta(minutes=80)) for _ in range(2)]
        reports = [future.result(timeout=60) for future in futures]
    assert sum(report["steps_committed"] for report in reports) == 9
    with Session(engine) as session:
        points = session.scalars(select(EngieMonitorResidual).where(EngieMonitorResidual.monitor_id == identity)).all()
        assert all(point.target_time == point.issue_time + timedelta(minutes=point.horizon_minutes) for point in points)
        assert len(points) == len({(point.monitor_id, point.issue_time, point.horizon_minutes) for point in points})


@pytest.mark.parametrize("field", ["model_version", "target_times"])
def test_invalid_input_partial_output_and_wrong_model_are_visible(engine, config, case, monkeypatch, field):
    from power_forecast_service.forecasting import engie_predictor

    body, _ = case
    identity = create_monitor(engine, body)
    original_inputs, original_forecast = MonitorReplaySource.inputs, engie_predictor.forecast
    def inputs(source, issue, artifact):
        if issue == START:
            raise PackageError("engie_history_unavailable")
        return original_inputs(source, issue, artifact)
    def wrong_model(root, registration, request):
        result = original_forecast(root, registration, request)
        if registration["family"] != "persistence" and request.issue_time == START + timedelta(minutes=10):
            result[field] = "wrong-model-version" if field == "model_version" else list(reversed(result[field]))
        return result
    monkeypatch.setattr(MonitorReplaySource, "inputs", inputs)
    monkeypatch.setattr(engie_predictor, "forecast", wrong_model)
    report = progress(engine, config, identity, START + timedelta(minutes=90))
    assert report["counts"]["attempted_issues"] == 10
    assert report["counts"]["invalid_inputs"] == report["counts"]["failed_issues"] == 1
    assert report["counts"]["outputs"] == {"champion": 9, "shadow": 8}
    assert all(item["rolling_invalid_inputs"] == item["rolling_failed_issues"] == 1
               and item["window_status"] == "incomplete" for item in report["horizons"])
    with Session(engine) as session:
        assert session.get(EngieMonitorIssue, (identity, START)).status == "invalid_input"
        assert session.get(EngieMonitorIssue, (identity, START + timedelta(minutes=10))).reason == "engie_delivery_result_contract_conflict"
        assert not session.scalars(select(EngieMonitorResidual).where(
            EngieMonitorResidual.monitor_id == identity,
            EngieMonitorResidual.issue_time.in_([START, START + timedelta(minutes=10)]))).all()


def test_database_rejects_wrong_target_horizon_association(engine, config, case):
    body, _ = case
    identity = create_monitor(engine, body)
    progress(engine, config, identity, START + timedelta(minutes=40))
    with Session(engine) as session:
        point = session.get(EngieMonitorResidual, (identity, START, 10))
        # 指向一个真实存在的实况对象，FK仍满足，但不能串到错误时距。
        point.target_time = START + timedelta(minutes=20)
        with pytest.raises(IntegrityError, match="engie_monitor_residual_target"):
            session.commit()
        session.rollback()
        assert session.get(EngieMonitorResidual, (identity, START, 10)).target_time == START + timedelta(minutes=10)


def test_clock_can_move_after_nominal_finish_to_accept_late_truth(engine, config, case, monkeypatch):
    body, registration = case
    body = body.model_copy(update={"end": START + timedelta(minutes=10)})
    identity = create_monitor(engine, body)
    original = MonitorReplaySource.observation
    monkeypatch.setattr(MonitorReplaySource, "observation", lambda *args, **kwargs: dict.fromkeys(ROSTER, None))
    report = progress(engine, config, identity, START + timedelta(minutes=80))
    assert report["replay_complete"] and report["horizons"][0]["due_missing_labels"] == 1
    late = START + timedelta(minutes=100)
    progress(engine, config, identity, late)
    target = START + timedelta(minutes=10)
    label = MonitorLabel(target_time=target, available_at=late,
                         turbines=original(MonitorReplaySource(config.artifact_root, registration), target, clock=late))
    result = ingest_label(engine, identity, label)
    assert result["replay_complete"] and result["horizons"][0]["scored_pairs"] == 1


def test_http_window_support_late_truth_and_restart(config, engine, case, monkeypatch, capsys):
    body, registration = case
    body = body.model_copy(update={"end": START + timedelta(hours=4), "window_issues": 12})
    original = MonitorReplaySource.observation
    outside, inside = START + timedelta(minutes=10), START + timedelta(minutes=130)
    source = MonitorReplaySource(config.artifact_root, registration)

    def delayed(source, target, *, clock):
        truth = original(source, target, clock=clock)
        if target in (outside, inside):
            truth[ROSTER[0]] = None
        return truth

    monkeypatch.setattr(MonitorReplaySource, "observation", delayed)
    spec = importlib.util.spec_from_file_location("monitor_cli_support", ROOT / "tools/dev/replay_engie_monitor.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    with api(config) as client:
        response = client.post("/engie/monitors", json=body.model_dump(mode="json"))
        response.raise_for_status()
        identity, through = response.json()["id"], response.json()["finish_time"]
        response = client.post(f"/engie/monitors/{identity}/advance", json={"through": through, "max_steps": 144})
        response.raise_for_status()
        before = client.get(f"/engie/monitors/{identity}").json()
        short = before["horizons"][0]
        assert (short["scored_pairs"], short["rolling_pairs"], short["rolling_scheduled_issues"]) == (22, 11, 12)
        assert (short["due_missing_labels"], short["rolling_due_missing_labels"]) == (2, 1)
        assert short["window_status"] == "incomplete" and short["alert"] == "insufficient_pairs"
        cli.display(before)
        visible = capsys.readouterr().out
        assert "11/12" in visible and "0/1/0/0" in visible and "窗口不完整" in visible
        for target in (outside, inside):
            label = MonitorLabel(target_time=target, available_at=datetime.fromisoformat(through),
                                 turbines=original(source, target, clock=datetime.fromisoformat(through)))
            response = client.post(f"/engie/monitors/{identity}/labels", json=label.model_dump(mode="json"))
            response.raise_for_status()
            saved = response.json()
            retry = client.post(f"/engie/monitors/{identity}/labels", json=label.model_dump(mode="json"))
            assert retry.status_code == 200 and retry.json() == saved
            short = saved["horizons"][0]
            if target == outside:
                assert short["scored_pairs"] == 23
                assert short["rolling_pairs"] == 11 and short["rolling_due_missing_labels"] == 1
                assert short["champion"] == before["horizons"][0]["champion"]
                assert short["shadow"] == before["horizons"][0]["shadow"]
                assert short["window_status"] == "incomplete"
                after_outside = saved
            else:
                assert short["scored_pairs"] == 24 and short["rolling_pairs"] == 12
                assert short["rolling_due_missing_labels"] == 0 and short["window_status"] == "complete"
    # 换应用和连接池后由账本重算，不把“窗口完整”误当成模型采用结论。
    with api(config) as client:
        response = client.get(f"/engie/monitors/{identity}")
        response.raise_for_status()
        assert response.json() == saved
    for item in saved["horizons"]:
        assert sum(item[key] for key in ("rolling_pairs", "rolling_pending_labels",
                                        "rolling_due_missing_labels", "rolling_invalid_inputs",
                                        "rolling_failed_issues")) == item["rolling_scheduled_issues"]
    destination = os.getenv("WIND_MONITOR_WINDOW_REPORT")
    if destination:
        Path(destination).write_text(json.dumps({"before": before, "after_outside": after_outside, "after": saved,
            "verification": {"database": "real_owned_PostgreSQL", "models": "original_fixed",
                             "missing_labels": "controlled_injection", "new_training": False,
                             "client": "ordinary_HTTP_and_CLI_display", "restart": "new_app_and_pool"}},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@contextmanager
def owned_http_server(config, tmp_path):
    # 只停止这里创建的API进程；随机测试库与空闲端口不接管18000普通服务。
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "WIND_DB_NAME": config.database_url.database,
           "WIND_ARTIFACT_ROOT": str(config.artifact_root), "WIND_API_PORT": str(port)}
    with (tmp_path / f"server-{port}.log").open("wb") as output:
        process = subprocess.Popen([sys.executable, "-B", "-m", "power_forecast_service.serve"],
                                   cwd=ROOT, env=env, stdout=output, stderr=output,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=180) as client:
                deadline = monotonic() + 20
                while True:
                    assert process.poll() is None, "owned_monitor_api_exited_during_start"
                    try:
                        response = client.get("/engie/imports", timeout=1)
                        response.raise_for_status()
                        break
                    except httpx.TransportError:
                        assert monotonic() < deadline, "owned_monitor_api_start_timeout"
                        sleep(.1)
                yield client
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def test_actual_tcp_process_restart_and_cli(engine, config, case, tmp_path):
    body, _ = case
    with owned_http_server(config, tmp_path) as client:
        response = client.post("/engie/monitors", json=body.model_dump(mode="json"))
        response.raise_for_status()
        identity = response.json()["id"]
        response = client.post(f"/engie/monitors/{identity}/advance", json={
            "through": (START + timedelta(minutes=20)).isoformat(), "max_steps": 3})
        response.raise_for_status()
        before = response.json()
    with owned_http_server(config, tmp_path) as client:
        saved = client.get(f"/engie/monitors/{identity}")
        saved.raise_for_status()
        assert saved.json() == {key: value for key, value in before.items() if key != "steps_committed"}
        command = subprocess.run([sys.executable, "-B", str(ROOT / "tools/dev/replay_engie_monitor.py"),
                                  "--base-url", str(client.base_url).rstrip("/"),
                                  "--champion", str(body.champion_id), "--shadow", str(body.shadow_id),
                                  "--start", body.start.isoformat(), "--end", body.end.isoformat(),
                                  "--key", body.request_key], cwd=ROOT, capture_output=True, timeout=60)
        assert command.returncode == 0, command.stderr.decode("utf-8", "replace")
        result = client.get(f"/engie/monitors/{identity}").json()
        assert result["replay_complete"] and result["counts"]["attempted_issues"] == 12
        assert [item["scored_pairs"] for item in result["horizons"]] == [12] * 6
        assert all(item["rolling_pairs"] == item["rolling_scheduled_issues"] == 12
                   and item["window_status"] == "complete" for item in result["horizons"])


def test_natural_one_day_report(config, engine, case):
    body, _ = case
    body = body.model_copy(update={"end": START + timedelta(days=1), "request_key": uuid4().hex})
    spec = importlib.util.spec_from_file_location("monitor_cli_natural", ROOT / "tools/dev/replay_engie_monitor.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    with api(config) as client:
        report = cli.run(client, champion=str(body.champion_id), shadow=str(body.shadow_id),
                         start=body.start.isoformat(), end=body.end.isoformat(), key=body.request_key)
    assert report["replay_complete"]
    assert report["counts"]["attempted_issues"] == report["counts"]["planned_issues"] == 144
    with Session(engine) as session:
        identity = UUID(report["id"])
        totals = {table.__tablename__: session.scalar(select(func.count()).select_from(table).where(
            table.monitor_id == identity)) for table in (EngieMonitorIssue, EngieMonitorObservation, EngieMonitorResidual)}
    report["verification"] = {"database": "real_owned_PostgreSQL", "HTTP": "FastAPI_TestClient_and_normal_CLI_consumer",
                              "training": "none", "arrival_delay": "simulated", "seen_period": True, "ledger_rows": totals}
    # 私有实测结果只写到本轮明确给定的证据目录，不进入默认公开树。
    destination = os.getenv("WIND_MONITOR_REPORT")
    if destination:
        Path(destination).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
