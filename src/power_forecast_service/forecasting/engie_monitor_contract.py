"""固定模型历史监测合同；模拟时钟不是机器时钟，也不是实际采集日志。"""

from datetime import datetime, timedelta
from typing import Annotated
from uuid import UUID

from pydantic import Field, FiniteFloat, field_validator, model_validator

from .engie_service_contract import ROSTER, StrictInput, require_utc

STEP = timedelta(minutes=10)
LABEL_DELAY = timedelta(minutes=20)
CONTRACT = "engie-delayed-fixed-models-v1"


class MonitorRequest(StrictInput):
    request_key: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    champion_id: UUID
    shadow_id: UUID
    start: datetime
    end: datetime
    window_issues: int = Field(default=36, ge=12, le=144, strict=True)

    _utc = field_validator("start", "end")(require_utc)

    @model_validator(mode="after")
    def interval(self):
        if self.end <= self.start:
            raise ValueError("engie_monitor_empty_interval")
        if self.champion_id == self.shadow_id:
            raise ValueError("engie_monitor_distinct_models_required")
        return self


class MonitorAdvance(StrictInput):
    through: datetime
    max_steps: int = Field(default=12, ge=1, le=144, strict=True)

    _utc = field_validator("through")(require_utc)


class MonitorLabel(StrictInput):
    target_time: datetime
    available_at: datetime
    turbines: dict[str, Annotated[FiniteFloat, Field(strict=True)] | None]

    _utc = field_validator("target_time", "available_at")(require_utc)

    @model_validator(mode="after")
    def label_time_and_roster(self):
        if self.available_at < self.target_time + LABEL_DELAY:
            raise ValueError("engie_monitor_label_before_availability")
        if set(self.turbines) != set(ROSTER):
            raise ValueError("engie_fixed_roster_required")
        return self


def finish_time(end):
    return end - STEP + timedelta(minutes=60) + LABEL_DELAY
