"""推理输入边界：历史数据不是任意形状的特征数组。"""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from power_forecast_service.forecasting.contracts import ForecastRequest


def observations():
    return [{"timestamp": datetime(2026, 1, 1) + timedelta(minutes=5 * n),
             "wind_power": 100, "wind_speed": 4, "humidity": 50, "temperature": 20}
            for n in range(13)]


def test_minimum_history_has_no_future_target():
    body = ForecastRequest(artifact_id=uuid4(), observations=observations())
    assert len(body.observations) == 13
    assert body.observations[-1].wind_power == 100.0


@pytest.mark.parametrize("mutation", ["short", "long", "gap", "order", "duplicate", "nan",
                                     "inf", "negative", "humidity", "timezone", "seconds", "extra"])
def test_bad_history_is_rejected(mutation):
    rows = observations()
    if mutation == "short":
        rows.pop()
    elif mutation == "long":
        rows *= 23
    elif mutation == "gap":
        rows[-1]["timestamp"] += timedelta(minutes=5)
    elif mutation == "order":
        rows.reverse()
    elif mutation == "duplicate":
        rows[-1] = rows[-2]
    elif mutation in {"nan", "inf"}:
        rows[-1]["temperature"] = float(mutation)
    elif mutation == "negative":
        rows[-1]["wind_power"] = -1
    elif mutation == "humidity":
        rows[-1]["humidity"] = 101
    elif mutation == "timezone":
        rows[-1]["timestamp"] = rows[-1]["timestamp"].replace(tzinfo=timezone.utc)
    elif mutation == "seconds":
        rows[-1]["timestamp"] += timedelta(seconds=1)
    else:
        rows[-1]["target_power"] = 999
    with pytest.raises(ValidationError):
        ForecastRequest(artifact_id=uuid4(), observations=rows)
