"""共同样本、选择与回执门禁；不以构造分数声称真实模型表现。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from power_forecast_service.forecasting.method_comparison.data import inner_split, prepare_fit_data, window_fit_data
from power_forecast_service.forecasting.method_comparison.neural import Checkpoints, prediction_metrics
from power_forecast_service.forecasting.method_comparison.protocol import EXPECTED_COUNTS, recipes
from power_forecast_service.forecasting.method_comparison import runner
from power_forecast_service.forecasting.method_comparison.tabular import clipped_mae
from power_forecast_service.forecasting.sequence_experiment import source_data
from power_forecast_service.forecasting.fixed_evaluation import fixed_window_split
from power_forecast_service.forecasting.features import build_supervised_frame
from power_forecast_service.forecasting.sequence_data import common_history_start
from power_forecast_service.storage.artifacts import sha256_file


def synthetic_data():
    t = np.arange(300)
    frame = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=300, freq="5min"),
                          "wind_power": 100 + 30 * np.sin(t / 12), "wind_speed": 5 + np.sin(t / 20),
                          "humidity": 60 + np.cos(t / 15), "temperature": 20 + np.sin(t / 18)})
    supervised = common_history_start(frame, build_supervised_frame(frame))
    train, val = inner_split(supervised.iloc[:220])
    return frame, train, val


def test_real_common_splits_and_labels():
    frame, supervised, _ = source_data(Path("data/sample/wind_2019_q1.csv"))
    for window, counts in EXPECTED_COUNTS.items():
        data = window_fit_data(frame, supervised, window)
        outer_train, outer = fixed_window_split(supervised, window)
        assert (len(data.train), len(data.validation)) == counts
        assert len(outer_train) - sum(counts) == 12
        assert len(outer) == 3872
        assert data.train.target_timestamp.max() < data.validation.timestamp.min()
        assert (outer.target_timestamp - outer.timestamp == pd.Timedelta(hours=1)).all()
        positions = pd.Index(frame.timestamp).get_indexer(data.train.timestamp)
        np.testing.assert_array_equal(data.train.target_power, frame.wind_power.to_numpy()[positions + 12])


def test_future_changes_do_not_change_training_scaler_or_windows():
    frame, train, val = synthetic_data()
    first = prepare_fit_data(frame, train, val)
    mutated = frame.copy()
    mutated.loc[mutated.timestamp > train.timestamp.max(), ["wind_power", "wind_speed"]] = 1e9
    second = prepare_fit_data(mutated, train, val)
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.scale, second.scale)
    np.testing.assert_array_equal(first.train_windows, second.train_windows)
    expected = frame.loc[frame.timestamp <= train.timestamp.max(), "wind_power"].mean()
    assert first.mean[0] == pytest.approx(expected)


def test_checkpoint_selection_copies_weights_and_preserves_ties():
    net, checkpoints = torch.nn.Linear(1, 1), Checkpoints()
    for epoch in range(1, 61):
        net.weight.data.fill_(epoch)
        score = 1 if epoch in (5, 6) else (0 if epoch == 30 else 10)
        checkpoints.observe(epoch, score, net)
    assert checkpoints.epochs == {"best60": 30, "best20": 5, "epoch20": 20}
    assert checkpoints.states["best20"]["weight"].item() == 5
    assert checkpoints.states["epoch20"]["weight"].item() == 20
    assert checkpoints.states["best60"]["weight"].item() == 30


def test_raw_metrics_are_retained_and_selection_metric_is_clipped():
    actual, predicted = np.array([0., 2.]), np.array([-10., 3.])
    metrics = prediction_metrics(actual, predicted)
    assert metrics["raw"]["mae"] == 5.5
    assert metrics["clipped"]["mae"] == 0.5
    assert metrics["clipped_fraction"] == 0.5
    assert clipped_mae(actual, predicted) == ("clipped_mae", 0.5, False)
    with pytest.raises(ValueError, match="shape_or_finiteness"):
        prediction_metrics(actual, predicted[:, None])


def test_exact_recipe_budget_and_tie_order():
    items = recipes()
    assert len(items) * 3 == 39
    assert sum(x["family"].startswith("ridge") for x in items) == 5
    assert [x["alpha"] for x in items if x["family"] == "ridge_history"] == [100., 10., 1., .1]


def test_outer_gate_requires_all_terminal_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "load_frozen", lambda *args: {})
    (tmp_path / "protocol.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="39_terminal"):
        runner.lock_selection(tmp_path, Path("unused"))
    assert not (tmp_path / "selection.json").exists()


def test_selection_ties_failures_and_receipt_integrity(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "load_frozen", lambda *args: {})
    (tmp_path / "protocol.json").write_text("{}", encoding="utf-8")
    protocol_hash = sha256_file(tmp_path / "protocol.json")
    for window in EXPECTED_COUNTS:
        for recipe in recipes():
            path = tmp_path / "attempts" / window / recipe["id"] / "attempt-01" / "receipt.json"
            path.parent.mkdir(parents=True)
            failed = recipe["id"] == "ridge_history_100"
            value = {"protocol_sha256": protocol_hash, "recipe": recipe, "window": window,
                     "status": "failed" if failed else "succeeded", "files": [],
                     "details": {"validation": {"clipped": {"mae": 10.}}}}
            path.write_text(json.dumps(value), encoding="utf-8")
    selected = runner.lock_selection(tmp_path, Path("unused"))
    assert len(selected["attempts"]) == 39
    assert selected["selected"]["main"]["ridge_history"]["recipe_id"] == "ridge_history_10"
    assert selected["selected"]["main"]["encoder"]["recipe_id"] == "encoder_direct"
    runner.verified_selection(tmp_path, Path("unused"))
    selection_path = tmp_path / "selection.json"
    original = selection_path.read_text(encoding="utf-8")
    duplicate = json.loads(original)
    duplicate["attempts"] = [duplicate["attempts"][0]] * 39
    selection_path.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="exact_inner_winners"):
        runner.verified_selection(tmp_path, Path("unused"))
    changed = json.loads(original)
    changed["selected"]["main"]["encoder"]["recipe_id"] = "encoder_delta"
    selection_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="exact_inner_winners"):
        runner.verified_selection(tmp_path, Path("unused"))
    selection_path.write_text(original, encoding="utf-8")
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="receipt_changed"):
        runner.verified_selection(tmp_path, Path("unused"))


def test_evaluate_cannot_access_outer_data_without_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "load_frozen", lambda *args: {})
    monkeypatch.setattr(runner, "source_data", lambda *args: pytest.fail("outer data consumed before lock"))
    with pytest.raises(FileNotFoundError):
        runner.evaluate(Path("unused"), tmp_path)
