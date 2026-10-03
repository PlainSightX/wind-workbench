"""轻量预测输入/输出；API导入不加载训练库或读取模型文件。"""

from datetime import datetime, timedelta
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    timestamp: datetime
    wind_power: float = Field(ge=0)
    wind_speed: float = Field(ge=0)
    humidity: float = Field(ge=0, le=100)
    temperature: float

    @field_validator("timestamp")
    @classmethod
    def source_clock(cls, value: datetime) -> datetime:
        if value.tzinfo is not None or value.second or value.microsecond or value.minute % 5:
            raise ValueError("timestamp_must_use_naive_five_minute_source_time")
        return value


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_id: UUID
    observations: list[Observation] = Field(min_length=13, max_length=288)

    @model_validator(mode="after")
    def continuous_history(self):
        for previous, current in zip(self.observations, self.observations[1:]):
            if current.timestamp - previous.timestamp != timedelta(minutes=5):
                raise ValueError("history_must_be_ordered_continuous_five_minute_observations")
        return self


class ForecastResponse(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    artifact_id: UUID
    run_id: UUID
    model_key: str
    model_version: str
    cutoff: datetime
    target_time: datetime
    prediction: float
    unit: str
    clock: str
    training_label_end: datetime
    after_training_cutoff: bool
