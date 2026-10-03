"""风电 CSV 的字段、时间顺序与数值质量检查。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "Time",
    "Wind_speed",
    "Humidity",
    "Temperature",
    "Wind_production",
}


class DataContractError(ValueError):
    """输入数据不满足当前实验契约。"""


@dataclass(frozen=True)
class QualityReport:
    rows: int
    start: str
    end: str
    duplicate_timestamps: int
    missing_values: int
    non_five_minute_gaps: int
    negative_target_rows: int
    out_of_order_timestamps: int = 0
    infinite_values: int = 0
    invalid_weather_rows: int = 0

    def as_dict(self) -> dict[str, int | str]:
        return {
            "rows": self.rows,
            "start": self.start,
            "end": self.end,
            "duplicate_timestamps": self.duplicate_timestamps,
            "missing_values": self.missing_values,
            "non_five_minute_gaps": self.non_five_minute_gaps,
            "negative_target_rows": self.negative_target_rows,
            "out_of_order_timestamps": self.out_of_order_timestamps,
            "infinite_values": self.infinite_values,
            "invalid_weather_rows": self.invalid_weather_rows,
        }


def _parse_time(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(
        values.astype("string").str.replace("-T", " ", regex=False),
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )
    if parsed.isna().any():
        bad_count = int(parsed.isna().sum())
        raise DataContractError(f"Time contains {bad_count} unparseable values")
    if getattr(parsed.dt, "tz", None) is not None:
        raise DataContractError("Time must remain timezone-naive for this source snapshot")
    return parsed


def load_wind_frame(path: Path, *, strict: bool = True) -> tuple[pd.DataFrame, QualityReport]:
    """读取并检查源数据；不在读取阶段静默修剪异常行。"""
    if not path.exists():
        raise FileNotFoundError(path)

    raw = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(raw.columns)
    if missing_columns:
        raise DataContractError(f"Missing required columns: {sorted(missing_columns)}")

    frame = pd.DataFrame(
        {
            "timestamp": _parse_time(raw["Time"]),
            "wind_speed": pd.to_numeric(raw["Wind_speed"], errors="coerce"),
            "humidity": pd.to_numeric(raw["Humidity"], errors="coerce"),
            "temperature": pd.to_numeric(raw["Temperature"], errors="coerce"),
            "wind_power": pd.to_numeric(raw["Wind_production"], errors="coerce"),
        }
    )

    if frame.empty:
        raise DataContractError("Input data has no rows")

    missing_values = int(frame.isna().sum().sum())
    duplicate_timestamps = int(frame["timestamp"].duplicated().sum())
    deltas = frame["timestamp"].diff().dropna()
    non_five_minute_gaps = int((deltas != pd.Timedelta(5, unit="min")).sum())
    negative_target_rows = int((frame["wind_power"] < 0).sum())
    # 保留原顺序供审查；排序会掩盖乱序输入，inf 也不会被 isna 捕捉。
    out_of_order_timestamps = int((deltas < pd.Timedelta(0)).sum())
    values = frame[["wind_speed", "humidity", "temperature", "wind_power"]].to_numpy(dtype=float)
    infinite_values = int(np.isinf(values).sum())
    invalid_weather_rows = int(
        ((frame["wind_speed"] < 0) | ~frame["humidity"].between(0, 100)).sum()
    )
    report = QualityReport(
        rows=len(frame),
        start=frame["timestamp"].iloc[0].isoformat(),
        end=frame["timestamp"].iloc[-1].isoformat(),
        duplicate_timestamps=duplicate_timestamps,
        missing_values=missing_values,
        non_five_minute_gaps=non_five_minute_gaps,
        negative_target_rows=negative_target_rows,
        out_of_order_timestamps=out_of_order_timestamps,
        infinite_values=infinite_values,
        invalid_weather_rows=invalid_weather_rows,
    )

    hard_failures = {
        "missing_values": missing_values,
        "duplicate_timestamps": duplicate_timestamps,
        "non_five_minute_gaps": non_five_minute_gaps,
        "out_of_order_timestamps": out_of_order_timestamps,
        "infinite_values": infinite_values,
        "invalid_weather_rows": invalid_weather_rows,
    }
    if strict and any(hard_failures.values()):
        raise DataContractError(f"Quality checks failed: {hard_failures}")
    if strict and negative_target_rows:
        raise DataContractError(
            "wind_power contains negative values; retain the raw rows and define a cleaning rule"
        )
    return frame.reset_index(drop=True), report
