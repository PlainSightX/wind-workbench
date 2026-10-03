"""最终冻结评价的时间、配对抽样和中断边界。"""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from power_forecast_service.forecasting.engie_contract import SiteSource, make_batch, HOLDOUT_START
from power_forecast_service.forecasting.engie_final import FAMILIES, paired_blocks

pytestmark = pytest.mark.unit


@pytest.fixture
def runner():
    directory = Path(__file__).resolve().parents[3] / "tools/diagnostics"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("a3_test_runner", directory / "run_engie_a3.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(directory))


def test_final_purge_and_unavailable_future_do_not_remove_prediction(runner):
    times = pd.date_range("2014-11-01", "2015-01-02", freq="10min", tz="UTC")
    source = SiteSource(times, np.ones((len(times), 4, 4)),
                        np.zeros((len(times), 4), bool), np.zeros((len(times), 4), bool), {})
    issues = pd.date_range("2014-11-02", "2015-01-01", freq="10min", tz="UTC", inclusive="left")
    batch = make_batch(source, issues)
    train, validation, refit, _ = runner.split_masks(batch, HOLDOUT_START)
    assert batch.issues[train][-1] == pd.Timestamp("2014-12-03T22:30Z")
    assert batch.issues[refit][-1] == batch.issues[validation][-1] == pd.Timestamp("2014-12-31T22:30Z")
    assert batch.input_valid[-6:].all() and not batch.boundary_valid[-6:].any()


def test_paired_bootstrap_matches_explicit_calendar_sampling_with_gaps():
    # 长度不能整除块长；保留缺测洞，同时检验末块截断和等季度加权。
    quarters = {}
    for q, n in (("q1", 31), ("q2", 47)):
        ref = np.arange(n, dtype=float) + 10
        ref[9:12] = np.nan
        quarters[q] = {f: ref + i * 2 for i, f in enumerate(FAMILIES)}
    actual = paired_blocks(quarters, block_steps=7, samples=120, seed=6)
    rng, means = np.random.default_rng(6), {f: [] for f in FAMILIES}
    for losses in quarters.values():
        n = len(losses["persistence"])
        starts = rng.integers(0, n - 7 + 1, size=(120, int(np.ceil(n / 7))))
        indices = (starts[..., None] + np.arange(7)).reshape(120, -1)[:, :n]
        for f in FAMILIES:
            means[f].append(np.nanmean(losses[f][indices], axis=1))
    means = {f: np.mean(v, axis=0) for f, v in means.items()}
    for f in FAMILIES:
        np.testing.assert_allclose(actual["families"][f]["delta_mae_kw_95ci"],
                                   np.quantile(means[f] - means["persistence"], [.025, .975]))
        np.testing.assert_allclose(actual["families"][f]["gain_percent_95ci"],
                                   np.quantile(100 * (1 - means[f] / means["persistence"]), [.025, .975]))


def test_unpaired_mask_rejected():
    losses = {f: np.ones(30) for f in FAMILIES}
    losses["ridge"][10] = np.nan
    with pytest.raises(ValueError, match="unpaired"):
        paired_blocks({"q": losses}, block_steps=3)


def test_exposure_blocks_before_training_and_partial_stage_requires_diagnosis(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    runner.start("selection", "hash")
    with pytest.raises(ValueError, match="partial_stage"):
        runner.start("selection", "hash")
    runner.start("holdout-exposure", "hash")
    with pytest.raises(ValueError, match="already_exposed"):
        runner.training(None, None, "hash")


def test_freeze_detects_changed_identity(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    runner.write_json(tmp_path / "protocol.json", {"protocol": {"a": 1}})
    monkeypatch.setattr(runner, "protocol", lambda: {"a": 2})
    with pytest.raises(ValueError, match="frozen_protocol_changed"):
        runner.freeze_hash()
