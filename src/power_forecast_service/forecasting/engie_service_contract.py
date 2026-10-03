"""ENGIE HTTP 输入：时间与固定机组先验明确，未来实况不属于预测请求。"""

from datetime import datetime, timedelta
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator

ROSTER = ("R80711", "R80721", "R80736", "R80790")
HORIZONS = (10, 20, 30, 40, 50, 60)


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


def require_utc(value):
    if value.utcoffset() != timedelta(0):
        raise ValueError("engie_utc_required")
    if value.minute % 10 or value.second or value.microsecond:
        raise ValueError("engie_time_off_grid")
    return value


class TurbineObservation(StrictInput):
    power_kw: FiniteFloat
    wind_speed: FiniteFloat
    direction_degrees: FiniteFloat
    temperature: FiniteFloat


class HistoryRow(StrictInput):
    timestamp: datetime
    turbines: dict[str, TurbineObservation]

    _utc = field_validator("timestamp")(require_utc)

    @field_validator("turbines")
    @classmethod
    def fixed_roster(cls, value):
        if set(value) != set(ROSTER):
            raise ValueError("engie_fixed_roster_required")
        return value


class EngieReplayRequest(StrictInput):
    artifact_id: UUID
    issue_time: datetime

    _utc = field_validator("issue_time")(require_utc)


class EngieForecastRequest(EngieReplayRequest):
    history: list[HistoryRow] = Field(min_length=12, max_length=12)

    @model_validator(mode="after")
    def complete_history(self):
        for index, row in enumerate(self.history):
            if row.timestamp != self.issue_time - timedelta(minutes=130 - 10 * index):
                raise ValueError("engie_history_gap_or_stale_input")
        return self


class EngieDeliveryRequest(EngieForecastRequest):
    request_key: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    budget_ms: int = Field(default=60000, ge=1, le=120000, strict=True)
