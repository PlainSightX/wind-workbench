"""轻量评分证据合同；API 可核验结果，不导入训练库或重新训练。"""

import hashlib
import json
from datetime import datetime, timedelta
from math import fsum, sqrt
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


class ScoreRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cutoff: datetime
    target_time: datetime
    actual: FiniteFloat
    predictions: dict[str, FiniteFloat] = Field(min_length=1)


class ScoringEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal["scoring-v1"]
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target: Literal["wind_power_single_point"]
    unit: Literal["source_reported_unit"]
    clock: Literal["source_time_timezone_unknown"]
    evaluation_split: Literal["validation", "test"]
    horizon_minutes: int = Field(gt=0)
    split_version: str = Field(min_length=1)
    metric_version: Literal["unweighted-mae-rmse-v1"]
    samples_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rows: list[ScoreRow] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_rows(self):
        names = set(self.rows[0].predictions)
        previous = None
        for row in self.rows:
            # 源数据无已知时区，不能与 UTC 审计时间混同；一个 cutoff 只评分一次。
            if row.cutoff.tzinfo is not None or row.target_time.tzinfo is not None:
                raise ValueError("Scoring timestamps must use the unlabelled source clock")
            if previous is not None and row.cutoff <= previous:
                raise ValueError("Scoring cutoffs must be unique and ordered")
            if row.target_time - row.cutoff != timedelta(minutes=self.horizon_minutes):
                raise ValueError("Scoring target does not match horizon")
            if set(row.predictions) != names:
                raise ValueError("Each model must score every row")
            previous = row.cutoff
        if sample_fingerprint(self.rows) != self.samples_sha256:
            raise ValueError("Scoring sample fingerprint mismatch")
        return self


def sample_fingerprint(rows: list[ScoreRow]) -> str:
    """预测值允许不同；样本身份绑定时间、顺序和真实标签，不能只 hash 行数。"""
    values = [
        [row.cutoff.isoformat(), row.target_time.isoformat(), float(row.actual)] for row in rows
    ]
    canonical = json.dumps(values, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def metrics_for_rows(rows: list[ScoreRow], model: str) -> dict[str, float | int]:
    """比较时直接由留存预测重算，避免把失配的历史汇总数字当作同条件结果。"""
    errors = [row.actual - row.predictions[model] for row in rows]
    return {
        "mae": fsum(abs(error) for error in errors) / len(errors),
        "rmse": sqrt(fsum(error * error for error in errors) / len(errors)),
        "samples": len(errors),
    }
