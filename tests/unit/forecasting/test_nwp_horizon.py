"""气象时距研究的时间规则：起报可得性、时效/有效时刻、轮毂高度换算与泄漏守卫。"""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.unit

TOOLS = Path(__file__).resolve().parents[3] / "tools/research/nwp_horizon"


@pytest.fixture(scope="module")
def modules():
    sys.path.insert(0, str(TOOLS))
    try:
        import study
        import timing
        yield SimpleNamespace(study=study, timing=timing)
    finally:
        sys.path.remove(str(TOOLS))


def ts(text):
    return pd.Timestamp(text, tz="UTC")


def test_latest_run_respects_publication_delay(modules):
    t = modules.timing
    # 11:00 起报：06Z + 5h = 11:00 恰好可用；10:50 起报只能用 00Z
    assert t.latest_qualifying_run(ts("2015-03-01 11:00")) == ts("2015-03-01 06:00")
    assert t.latest_qualifying_run(ts("2015-03-01 10:50")) == ts("2015-03-01 00:00")
    # 跨日：03:00 起报只能用前一日 18Z
    assert t.latest_qualifying_run(ts("2015-03-02 03:00")) == ts("2015-03-01 18:00")


def test_no_run_before_archive_start_and_no_future_runs(modules):
    t = modules.timing
    assert t.latest_qualifying_run(ts("2015-01-15 04:50")) is None
    for issue in pd.date_range(ts("2015-01-16"), periods=300, freq="10min"):
        runs = t.qualifying_runs(issue)
        assert runs and all(run + t.PUBLICATION_DELAY <= issue for run in runs)
        assert runs == sorted(runs, reverse=True) and len(runs) == t.MAX_FALLBACK_RUNS + 1


def test_leakage_guard_rejects_unpublished_run(modules):
    t = modules.timing
    t.assert_no_future_information(ts("2015-03-01 11:00"), ts("2015-03-01 06:00"))
    with pytest.raises(ValueError, match="not_published"):
        t.assert_no_future_information(ts("2015-03-01 10:50"), ts("2015-03-01 06:00"))


def test_lead_valid_arithmetic_and_step_interpolation(modules):
    t = modules.timing
    b = t.bracket(ts("2015-03-01 06:00"), ts("2015-03-01 13:10"))
    assert b.reftime + b.lead == ts("2015-03-01 13:10")
    assert (b.lower, b.upper) == (6, 9) and b.weight == pytest.approx(70 / 180)
    exact = t.bracket(ts("2015-03-01 06:00"), ts("2015-03-01 15:00"))
    assert (exact.lower, exact.upper, exact.weight) == (9, 9, 0.0)
    with pytest.raises(ValueError, match="positive"):
        t.bracket(ts("2015-03-01 06:00"), ts("2015-03-01 06:00"))


def test_hub_height_power_law(modules):
    t = modules.timing
    low = np.full(500, 5.0)
    high = low * (100 / 10) ** 0.2
    alpha = t.fit_power_law_alpha(low, high)
    assert alpha == pytest.approx(0.2)
    assert t.hub_speed(10.0, alpha) == pytest.approx(10.0 * 0.8 ** 0.2)
    with pytest.raises(ValueError):
        t.fit_power_law_alpha(low[:50], high[:50])


def test_bilinear_weights_sum_to_one_and_reject_outside(modules):
    t = modules.timing
    w = t.bilinear_weights()
    assert w.sum() == pytest.approx(1.0) and (w >= 0).all()
    with pytest.raises(ValueError):
        t.bilinear_weights(lat=49.0)


def _row(value):
    return SimpleNamespace(u10=value, v10=0.0, u80=value, v80=0.0, gust=value, t2m=280.0, sp=1e5)


def test_gfs_features_choose_latest_run_and_fall_back(modules):
    s = modules.study
    issue = ts("2015-03-01 11:00")
    # 06Z 运行的值为 6，00Z 运行的值为 0；先完整提供两个运行
    table = {(ts("2015-03-01 06:00"), lead): _row(6.0) for lead in range(3, 16, 3)}
    table.update({(ts("2015-03-01 00:00"), lead): _row(1.0) for lead in range(3, 28, 3)})
    features, ok, lead, depth = s.gfs_features(pd.DatetimeIndex([issue]), (10, 240), table)
    assert ok.all() and (depth == 0).all()
    assert lead[0].tolist() == [310.0, 540.0]  # 11:10-06:00、15:00-06:00
    assert features[0, :, 0] == pytest.approx([6.0, 6.0])
    # 06Z 缺少 f006：10 分钟时距回退到 00Z（lead 11h10m），240 分钟仍用 06Z 的 f009
    del table[(ts("2015-03-01 06:00"), 6)]
    features, ok, lead, depth = s.gfs_features(pd.DatetimeIndex([issue]), (10, 240), table)
    assert depth[0].tolist() == [1, 0] and lead[0, 0] == 670.0
    assert features[0, 0, 0] == pytest.approx(1.0)


def test_gfs_features_never_use_runs_published_after_issue(modules):
    s = modules.study
    issue = ts("2015-03-01 10:50")
    # 只提供 06Z（10:50 时尚未发布）：必须判为不可用，不能借用
    table = {(ts("2015-03-01 06:00"), lead): _row(6.0) for lead in range(3, 16, 3)}
    _, ok, _, depth = s.gfs_features(pd.DatetimeIndex([issue]), (10,), table)
    assert not ok.any() and depth[0, 0] == -1


def test_training_labels_do_not_read_test_period(modules):
    s = modules.study
    times = pd.date_range(ts("2015-08-31 20:00"), ts("2015-09-01 06:00"), freq="10min")
    source = SimpleNamespace(times=times, values=np.ones((len(times), 4, 4)))
    issues = pd.DatetimeIndex([ts("2015-08-31 22:00")])
    y = s.targets(source, issues, (60, 120, 180), label_end=s.TEST_START)
    # 23:00 可读；00:00 和 01:00 已属测试期，封存前保持 NaN
    assert np.isfinite(y[0, :, 0]).all() and np.isnan(y[0, :, 1:]).all()
