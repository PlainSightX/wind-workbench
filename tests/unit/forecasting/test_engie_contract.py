"""新合同的时间/对象/缺测边界；不访问真实源或服务。"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from power_forecast_service.forecasting.engie_contract import (
    ROSTER, before_training_boundary, make_batch, persistence, source_from_frame,
)
from power_forecast_service.forecasting.engie_baselines import raw_metrics, score_curves

pytestmark = pytest.mark.unit


def raw_frame(start="2014-01-01", periods=160):
    times = pd.date_range(start, periods=periods, freq="10min", tz="UTC")
    rows = [{"Wind_turbine_name": turbine, "Date_time": time.tz_convert("Europe/Paris").isoformat(),
             "P_avg": index - 100 + j, "Ws_avg": 5, "Wa_avg": 359 if j == 0 else 1,
             "Ot_avg": 10} for index, time in enumerate(times) for j, turbine in enumerate(ROSTER)]
    return pd.DataFrame(rows), times


def source_fixture():
    raw, times = raw_frame()
    return source_from_frame(raw, start=times[0], end=times[-1] + pd.Timedelta(minutes=10))


def test_issue_information_delay_and_horizon_are_distinct():
    source = source_fixture()
    issue = pd.DatetimeIndex([pd.Timestamp("2014-01-01T03:00Z")])
    batch = make_batch(source, issue)
    assert batch.features.shape == (1, 4, 60)
    assert batch.scoreable[0]
    assert batch.features[0, 0, -5] == -84  # 02:40，不能取03:00。
    np.testing.assert_array_equal(batch.targets[0, 0], np.arange(19, 25) - 100)
    np.testing.assert_array_equal(persistence(batch.features)[0, 0], [-84] * 6)
    assert abs(batch.features[0, 0, -4 + 1]) < 0.02  # sin(359 degrees)
    changed = source.values.copy()
    changed[17:19] = 999999
    replay = make_batch(replace(source, values=changed), issue)
    np.testing.assert_array_equal(batch.features, replay.features)


def test_missing_future_label_never_filters_issue_or_input():
    source = source_fixture()
    issue = pd.DatetimeIndex([pd.Timestamp("2014-01-01T03:00Z")])
    changed = source.values.copy()
    changed[20, :, 0] = np.nan
    batch = make_batch(replace(source, values=changed), issue)
    assert len(batch.issues) == 1 and batch.input_valid[0]
    assert not batch.label_valid[0] and not batch.scoreable[0]
    assert np.isfinite(persistence(batch.features)).all()


@pytest.mark.parametrize("problem", ["missing", "conflicting", "nonfinite"])
def test_fixed_roster_window_rejects_bad_key_without_zero_fill(problem):
    raw, times = raw_frame()
    selected = (raw.Wind_turbine_name == ROSTER[0]) & (raw.Date_time == times[16].tz_convert("Europe/Paris").isoformat())
    if problem == "missing":
        raw = raw.loc[~selected]
    elif problem == "conflicting":
        duplicate = raw.loc[selected].copy()
        duplicate["P_avg"] = 9999
        raw = pd.concat([raw, duplicate], ignore_index=True)
    else:
        raw.loc[selected, "Ws_avg"] = np.inf
    source = source_from_frame(raw, start=times[0], end=times[-1] + pd.Timedelta(minutes=10))
    batch = make_batch(source, pd.DatetimeIndex([times[18]]))
    assert not batch.input_valid[0]
    assert batch.label_valid[0]
    assert batch.counts([True])["planned"] == 1
    assert batch.counts([True])["scoreable"] == 0
    if problem == "conflicting":
        assert batch.conflicting_input[0] and source.quality["per_turbine"][0]["duplicate_keys"] == 1


def test_real_offset_transition_is_parsed_on_utc_grid():
    raw, times = raw_frame("2014-03-30", periods=50)
    source = source_from_frame(raw, start=times[0], end=times[-1] + pd.Timedelta(minutes=10))
    assert set(source.quality["source_offsets"]) == {"+01:00", "+02:00"}
    assert not source.conflicting.any() and not source.missing.any()
    assert make_batch(source, pd.DatetimeIndex([times[20]])).scoreable[0]


def test_training_label_arrival_is_strictly_before_boundary():
    source = source_fixture()
    batch = make_batch(source, pd.date_range("2014-01-01T02:00Z", "2014-01-01T09:00Z", freq="10min"))
    mask = before_training_boundary(batch, "2014-01-01T10:00Z")
    assert batch.issues[mask][-1] == pd.Timestamp("2014-01-01T08:30Z")
    assert not mask[batch.issues.get_loc("2014-01-01T08:40Z")]


def test_holdout_targets_remain_unread_even_when_present():
    raw, times = raw_frame("2014-12-31", periods=160)
    source = source_from_frame(raw, start=times[0], end=times[-1] + pd.Timedelta(minutes=10))
    issues = pd.date_range("2014-12-31T23:00Z", periods=6, freq="10min")
    a = make_batch(source, issues)
    changed = source.values.copy()
    changed[source.times >= pd.Timestamp("2015-01-01T00:00Z")] = 123456789
    b = make_batch(replace(source, values=changed), issues)
    np.testing.assert_array_equal(a.targets, b.targets)
    assert a.input_valid.all() and not a.boundary_valid.any()
    assert np.isnan(a.targets[-1]).all()


def test_offsetless_source_and_wrong_roster_are_rejected():
    raw, times = raw_frame()
    raw["Date_time"] = "2014-01-01 00:00"
    with pytest.raises(ValueError, match="offset_required"):
        source_from_frame(raw, start=times[0], end=times[-1] + pd.Timedelta(minutes=10))
    with pytest.raises(ValueError, match="roster_mismatch"):
        source_from_frame(raw[raw.Wind_turbine_name != ROSTER[0]], start=times[0], end=times[-1])


def test_ranking_requires_complete_curves_and_uses_true_sum():
    actual = np.full((2, 4, 6), -10.0)
    predicted = actual - 2
    result = score_curves(actual, predicted)
    assert result["farm"]["mae"] == 8 and result["farm"]["bias"] == -8
    assert result["all_turbine_values"]["mae"] == 2
    predicted[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="incomplete_predictions"):
        score_curves(actual, predicted)
    with pytest.raises(ValueError, match="shape"):
        raw_metrics(np.ones(2), np.ones((2, 1)))
