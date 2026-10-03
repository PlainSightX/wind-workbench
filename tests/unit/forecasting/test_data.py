from pathlib import Path

import pandas as pd
import pytest

from power_forecast_service.forecasting.data import DataContractError, load_wind_frame


def write_csv(tmp_path: Path, frame: pd.DataFrame) -> Path:
    path = tmp_path / "input.csv"
    frame.to_csv(path, index=False)
    return path


def base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": ["2019-01-01-T00:00", "2019-01-01-T00:05", "2019-01-01-T00:10"],
            "Wind_speed": [2.0, 2.1, 2.2],
            "Humidity": [50.0, 51.0, 52.0],
            "Temperature": [1.0, 1.1, 1.2],
            "Wind_production": [100.0, 110.0, 120.0],
        }
    )


def test_loader_normalizes_source_columns(tmp_path: Path) -> None:
    frame, report = load_wind_frame(write_csv(tmp_path, base_frame()))
    assert list(frame.columns) == [
        "timestamp",
        "wind_speed",
        "humidity",
        "temperature",
        "wind_power",
    ]
    assert report.rows == 3
    assert frame["timestamp"].iloc[0] == pd.Timestamp("2019-01-01 00:00")


def test_duplicate_timestamp_is_rejected(tmp_path: Path) -> None:
    frame = base_frame()
    frame.loc[2, "Time"] = frame.loc[1, "Time"]
    with pytest.raises(DataContractError, match="duplicate_timestamps"):
        load_wind_frame(write_csv(tmp_path, frame))


def test_negative_target_is_rejected_in_strict_mode(tmp_path: Path) -> None:
    frame = base_frame()
    frame.loc[1, "Wind_production"] = -1
    with pytest.raises(DataContractError, match="negative values"):
        load_wind_frame(write_csv(tmp_path, frame))


def test_non_strict_mode_preserves_negative_target_for_audit(tmp_path: Path) -> None:
    frame = base_frame()
    frame.loc[1, "Wind_production"] = -1
    loaded, report = load_wind_frame(write_csv(tmp_path, frame), strict=False)
    assert loaded.loc[1, "wind_power"] == -1
    assert report.negative_target_rows == 1


def test_empty_input_is_rejected(tmp_path: Path) -> None:
    frame = base_frame().iloc[0:0]
    with pytest.raises(DataContractError, match="no rows"):
        load_wind_frame(write_csv(tmp_path, frame))


def test_gapped_source_observations_are_rejected(tmp_path: Path) -> None:
    frame = base_frame()
    frame.loc[2, "Time"] = "2019-01-01-T00:15"
    with pytest.raises(DataContractError, match="non_five_minute_gaps"):
        load_wind_frame(write_csv(tmp_path, frame))


def test_source_time_stays_timezone_naive(tmp_path: Path) -> None:
    loaded, _ = load_wind_frame(write_csv(tmp_path, base_frame()))
    assert loaded["timestamp"].dt.tz is None
    frame = base_frame()
    frame["Time"] = [
        "2019-01-01T00:00:00+08:00",
        "2019-01-01T00:05:00+08:00",
        "2019-01-01T00:10:00+08:00",
    ]
    with pytest.raises(DataContractError):
        load_wind_frame(write_csv(tmp_path, frame))


def test_unsorted_input_is_rejected_and_audit_preserves_order(tmp_path: Path) -> None:
    path = write_csv(tmp_path, base_frame().iloc[[1, 0, 2]])
    with pytest.raises(DataContractError, match="out_of_order_timestamps"):
        load_wind_frame(path)
    frame, report = load_wind_frame(path, strict=False)
    assert frame["timestamp"].iloc[0] == pd.Timestamp("2019-01-01 00:05")
    assert report.out_of_order_timestamps == 1


@pytest.mark.parametrize(
    "column,value",
    [
        ("Temperature", float("inf")),
        ("Wind_speed", float("-inf")),
        ("Humidity", 101.0),
        ("Wind_speed", -0.1),
    ],
)
def test_invalid_values_cannot_reach_training(tmp_path: Path, column: str, value: float) -> None:
    frame = base_frame()
    frame.loc[1, column] = value
    with pytest.raises(DataContractError, match="Quality checks failed"):
        load_wind_frame(write_csv(tmp_path, frame))
