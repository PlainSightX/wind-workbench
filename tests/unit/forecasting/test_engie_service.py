"""HTTP合同拒绝未来/缺失输入，特征顺序与原冻结算法一致。"""

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from uuid import uuid4

import numpy as np
import pytest
from pydantic import ValidationError

from power_forecast_service.forecasting.engie_service_contract import EngieForecastRequest, ROSTER
from power_forecast_service.forecasting.engie_predictor import check_issue, feature_array
from power_forecast_service.storage.engie_packages import checked_bytes, write_verified
from power_forecast_service.storage.model_packages import PackageError


def request_data():
    issue = datetime(2014, 4, 1, tzinfo=timezone.utc)
    return {"artifact_id": str(uuid4()), "issue_time": issue.isoformat(), "history": [
        {"timestamp": (issue - timedelta(minutes=130 - 10 * index)).isoformat(),
         "turbines": {name: {"power_kw": -2 + index + t, "wind_speed": 5, "direction_degrees": 90,
                             "temperature": 10} for t, name in enumerate(ROSTER)}} for index in range(12)]}


@pytest.mark.parametrize("problem", ["missing_turbine", "unknown_turbine", "gap", "stale", "future", "naive", "offset", "off_grid", "nan", "extra_actual"])
def test_reject_incomplete_or_wrong_clock(problem):
    body = request_data()
    if problem == "missing_turbine":
        body["history"][0]["turbines"].pop(ROSTER[0])
    elif problem == "unknown_turbine":
        body["history"][0]["turbines"]["other"] = body["history"][0]["turbines"][ROSTER[0]]
    elif problem == "gap":
        body["history"][1]["timestamp"] = body["history"][0]["timestamp"]
    elif problem in {"stale", "future"}:
        shift = -10 if problem == "stale" else 10
        for row in body["history"]:
            row["timestamp"] = (datetime.fromisoformat(row["timestamp"]) + timedelta(minutes=shift)).isoformat()
    elif problem == "naive":
        body["issue_time"] = "2014-04-01T00:00:00"
    elif problem == "offset":
        body["issue_time"] = "2014-04-01T08:00:00+08:00"
    elif problem == "off_grid":
        body["issue_time"] = "2014-04-01T00:01:00Z"
    elif problem == "nan":
        body["history"][0]["turbines"][ROSTER[0]]["power_kw"] = float("nan")
    else:
        body["actual"] = [1, 2, 3]
    with pytest.raises(ValidationError):
        EngieForecastRequest(**body)


def test_features_keep_negative_power_and_order():
    body = EngieForecastRequest(**request_data())
    x = feature_array(body)
    assert x.shape == (1, 4, 60)
    np.testing.assert_allclose(x[0, 0, :5], [-2, 5, 1, 0, 10], atol=1e-15)
    assert x[0, 3, -5] == 12


def test_open_quarter_is_issue_time_not_last_input():
    body = EngieForecastRequest(**request_data())
    registration = {"run": {"start": "2014-04-01T00:00:00Z", "end": "2014-07-01T00:00:00Z",
                             "training_label_available": "2014-03-31T23:50:00Z"}}
    check_issue(registration, body.issue_time)
    for stamp in ("2014-03-31T23:50:00Z", "2014-07-01T00:00:00Z", "2015-01-01T00:00:00Z"):
        with pytest.raises(PackageError):
            check_issue(registration, datetime.fromisoformat(stamp))


def test_bad_hash_and_escape_rejected_before_deserialization(tmp_path):
    payload = b"not a pickle"
    (tmp_path / "model.joblib").write_bytes(payload)
    assert checked_bytes(tmp_path, "model.joblib", sha256(payload).hexdigest()) == payload
    for name, digest in [("model.joblib", "0" * 64), ("../model.joblib", sha256(payload).hexdigest())]:
        with pytest.raises(PackageError, match="integrity"):
            checked_bytes(tmp_path, name, digest)


def test_import_refuses_symlink_parent_before_writing(tmp_path, monkeypatch):
    from pathlib import Path

    redirected = tmp_path / "redirected"
    monkeypatch.setattr(Path, "is_symlink", lambda self: self == redirected)
    with pytest.raises(PackageError, match="destination_conflict"):
        write_verified(redirected / "model.joblib", b"data")
    assert not redirected.exists()
