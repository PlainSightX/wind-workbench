"""时间防泄漏、固定分母、成对评分与告警的离线规则。"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from pathlib import Path
import importlib.util
from uuid import uuid4

import pytest
from pydantic import ValidationError

from power_forecast_service.experiments.engie_monitor import summarize
from power_forecast_service.forecasting.engie_monitor_contract import (MonitorAdvance, MonitorLabel,
                                                                      MonitorRequest, finish_time)
from power_forecast_service.forecasting.engie_service_contract import ROSTER

pytestmark = pytest.mark.unit
START = datetime(2015, 1, 1, tzinfo=timezone.utc)


def test_time_contract_and_bounded_progress():
    body = MonitorRequest(request_key="x", champion_id=uuid4(), shadow_id=uuid4(),
                          start=START, end=START + timedelta(hours=1))
    assert finish_time(body.end) == START + timedelta(minutes=130)
    for values in ({"end": START}, {"start": START.replace(tzinfo=None)},
                   {"shadow_id": body.champion_id}, {"window_issues": True}):
        with pytest.raises(ValidationError):
            MonitorRequest(**{**body.model_dump(), **values})
    for steps in (0, True, 145, 1.5):
        with pytest.raises(ValidationError):
            MonitorAdvance(through=START, max_steps=steps)


def test_label_availability_roster_and_finite_values():
    base = dict(target_time=START, available_at=START + timedelta(minutes=20),
                turbines=dict.fromkeys(ROSTER, None))
    assert MonitorLabel(**base).turbines[ROSTER[0]] is None
    for values in ({"available_at": START}, {"turbines": {ROSTER[0]: 1}},
                   {"turbines": dict.fromkeys(ROSTER, True)},
                   {"turbines": dict.fromkeys(ROSTER, float("nan"))}):
        with pytest.raises(ValidationError):
            MonitorLabel(**{**base, **values})


def test_missing_and_invalid_do_not_disappear_from_report():
    issues = [SimpleNamespace(issue_time=START + timedelta(minutes=10 * i),
                              status="predicted" if i < 2 else "invalid_input",
                              predictions={"champion": {}, "shadow": {}} if i < 2 else None)
              for i in range(3)]
    residuals = [SimpleNamespace(issue_time=START, horizon_minutes=10,
                                champion_error_kw=-2., shadow_error_kw=3.)]
    report = summarize(issues, residuals, planned=6, clock=START + timedelta(minutes=30),
                       window=6, margin=.10, minimum=12)
    assert report["counts"] == {"planned_issues": 6, "attempted_issues": 3, "valid_inputs": 2,
                                "invalid_inputs": 1, "failed_issues": 0,
                                "outputs": {"champion": 2, "shadow": 2}}
    short, long = report["horizons"][0], report["horizons"][-1]
    assert (short["scored_pairs"], short["pending_labels"], short["due_missing_labels"]) == (1, 1, 0)
    assert short["champion"] == {"mae_kw": 2., "bias_kw": -2.}
    assert short["alert"] == "insufficient_pairs"
    assert long["pending_labels"] == 2 and long["champion"]["mae_kw"] is None


def test_rolling_window_is_scheduled_not_successful_and_zero_mae_is_defined():
    issues = [SimpleNamespace(issue_time=START + timedelta(minutes=10 * i), status="predicted",
                              predictions={"champion": {}, "shadow": {}}) for i in range(18)]
    points = [SimpleNamespace(issue_time=item.issue_time, horizon_minutes=10,
                              champion_error_kw=0., shadow_error_kw=1.) for item in issues[:12]]
    report = summarize(issues, points, planned=18, clock=START + timedelta(days=1),
                       window=12, margin=.10, minimum=12)
    assert report["horizons"][0]["rolling_pairs"] == 6
    assert report["horizons"][0]["alert"] == "insufficient_pairs"
    report = summarize(issues[:12], points, planned=12, clock=START + timedelta(days=1),
                       window=12, margin=.10, minimum=12)
    assert report["horizons"][0]["alert"] == "shadow_worse_review"


def test_cli_displays_missing_truth_separately_from_model_quality(capsys):
    path = Path(__file__).resolve().parents[2] / "tools/dev/replay_engie_monitor.py"
    spec = importlib.util.spec_from_file_location("monitor_cli_display", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = summarize([], [], planned=12, clock=START, window=36, margin=.10, minimum=12)
    report["contract"] = {"champion": {"model_version": "persistence-fixed"},
                          "shadow": {"model_version": "candidate-fixed"}}
    module.display(report)
    text = capsys.readouterr().out
    assert "计划起报 12" in text and "到期缺失" in text and "样本不足" in text
    assert "模型不会自动切换" in text


def window_support_case():
    # 40次处理、窗口36次：24已评分、4未到达、4缺测、2无效、2失败。
    issues = [SimpleNamespace(issue_time=START + timedelta(minutes=10 * i),
                              status="invalid_input" if i in (36, 37) else
                                     "failed" if i in (38, 39) else "predicted",
                              predictions=None if i in (36, 37, 38, 39) else
                                          {"champion": {}, "shadow": {}})
              for i in range(40)]
    points = [SimpleNamespace(issue_time=item.issue_time, horizon_minutes=10,
                              champion_error_kw=-1., shadow_error_kw=1.05)
              for item in issues[:28]]
    return summarize(issues, points, planned=40, clock=START + timedelta(minutes=340),
                     window=36, margin=.10, minimum=12)


def test_window_support_partitions_current_window_without_changing_relative_alert():
    item = window_support_case()["horizons"][0]
    assert (item["scored_pairs"], item["rolling_pairs"], item["rolling_scheduled_issues"]) == (28, 24, 36)
    assert item["rolling_pending_labels"] == 4
    assert item["rolling_due_missing_labels"] == 4
    assert item["rolling_invalid_inputs"] == 2
    assert item["rolling_failed_issues"] == 2
    assert item["window_status"] == "incomplete"
    assert item["alert"] == "within_margin"
    assert item["champion"]["mae_kw"] == 1.
    assert item["shadow"]["mae_kw"] == pytest.approx(1.05)
    assert sum(item[key] for key in ("rolling_pairs", "rolling_pending_labels",
                                    "rolling_due_missing_labels", "rolling_invalid_inputs",
                                    "rolling_failed_issues")) == item["rolling_scheduled_issues"]


def test_cli_distinguishes_cumulative_count_from_current_mae_support(capsys):
    path = Path(__file__).resolve().parents[2] / "tools/dev/replay_engie_monitor.py"
    spec = importlib.util.spec_from_file_location("monitor_cli_window", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = window_support_case()
    report["contract"] = {"champion": {"model_version": "persistence-fixed"},
                          "shadow": {"model_version": "candidate-fixed"}}
    module.display(report)
    text = capsys.readouterr().out
    assert "累计已评分" in text and "窗口成对/已处理" in text
    assert "28/4/4" in text and "24/36" in text and "4/4/2/2" in text
    assert "窗口不完整" in text and "未超复核线" in text
    assert "缺测/无效/失败不代表模型健康" in text


@pytest.mark.parametrize("status, errors, clock, expected", [
    (None, [], START, "not_started"),
    ("predicted", [], START, "awaiting_labels"),
    ("predicted", [], START + timedelta(hours=2), "incomplete"),
    ("invalid_input", [], START + timedelta(hours=2), "incomplete"),
    ("failed", [], START + timedelta(hours=2), "incomplete"),
    ("predicted", [1.], START + timedelta(hours=2), "complete"),
])
def test_window_readiness_does_not_claim_health_or_statistical_confidence(status, errors, clock, expected):
    issues = [] if status is None else [SimpleNamespace(issue_time=START, status=status,
                                                       predictions={"champion": {}, "shadow": {}})]
    points = [SimpleNamespace(issue_time=START, horizon_minutes=10,
                              champion_error_kw=value, shadow_error_kw=value) for value in errors]
    report = summarize(issues, points, planned=12, clock=clock, window=36, margin=.10, minimum=12)
    item = report["horizons"][0]
    assert item["window_status"] == expected
    assert item["alert"] == "insufficient_pairs"
    for item in report["horizons"]:
        assert sum(item[key] for key in ("rolling_pairs", "rolling_pending_labels",
                                        "rolling_due_missing_labels", "rolling_invalid_inputs",
                                        "rolling_failed_issues")) == item["rolling_scheduled_issues"]
